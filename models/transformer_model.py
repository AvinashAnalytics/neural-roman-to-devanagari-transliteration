# models/transformer_model.py
"""
Transformer Model with Local Attention for Hindi Transliteration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import yaml
import logging
from typing import Optional, Tuple, List, Dict, Union
from pathlib import Path
from collections import OrderedDict

logger = logging.getLogger(__name__)


# ========================
# Positional Encoding
# ========================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but part of state_dict)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            x with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding (alternative to sinusoidal)"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Embedding(max_len, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.pe(positions)
        return self.dropout(x)


# ========================
# Local Attention
# ========================

class LocalAttention(nn.Module):
    """
    Local attention mechanism with sliding window.
    Each position attends to positions within ±window_size.
    
    FIXED: Improved cache management with OrderedDict LRU
    """
    
    def __init__(self, d_model: int, n_heads: int, window_size: int = 5, dropout: float = 0.1):
        super().__init__()
        
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.window_size = window_size
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # FIXED: Use OrderedDict for proper LRU cache
        self._cached_local_masks = OrderedDict()
        self._max_cache_size = 50
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query, key, value: [batch, seq_len, d_model]
            key_padding_mask: [batch, seq_len] - True to ignore
            attn_mask: [seq_len, seq_len] - True to ignore (e.g., causal mask)
        
        Returns:
            output: [batch, seq_len, d_model]
            attention_weights: None (for API compatibility)
        """
        batch_size, seq_len, _ = query.size()
        
        # Linear projections and reshape to [B, H, L, D_k]
        Q = self.W_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B, H, L, L]
        
        # Create or retrieve cached local mask
        local_mask = self._get_local_mask(seq_len, query.device)  # [L, L] boolean
        
        # Apply local mask
        # MASKING CONVENTION: local_mask[i, j] = True means "position i CAN attend to j"
        #                     ~local_mask inverts: True means "CANNOT attend" → fill with -inf
        mask_value = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~local_mask.unsqueeze(0).unsqueeze(0), mask_value)
        
        # Apply causal mask if provided
        # CONVENTION: attn_mask[i, j] = True means "position i CANNOT attend to j"
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), mask_value)
        
        # Apply padding mask if provided
        # CONVENTION: key_padding_mask[b, j] = True means "position j is padding"
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                mask_value
            )
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)  # [B, H, L, D_k]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        output = self.W_o(context)
        
        return output, None  # Return None for attention weights (API compatibility)
    
    def _get_local_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Get or create cached local attention mask with LRU eviction.
        
        FIXED: OrderedDict-based LRU cache for better memory management
        """
        key = (seq_len, str(device))
        
        if key in self._cached_local_masks:
            # Move to end (mark as recently used)
            self._cached_local_masks.move_to_end(key)
            return self._cached_local_masks[key]
        
        # Create new mask: True = can attend, False = cannot attend
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        for i in range(seq_len):
            start = max(0, i - self.window_size)
            end = min(seq_len, i + self.window_size + 1)
            mask[i, start:end] = True
        
        # Add to cache
        self._cached_local_masks[key] = mask
        
        # Evict oldest if cache too large (LRU)
        if len(self._cached_local_masks) > self._max_cache_size:
            self._cached_local_masks.popitem(last=False)  # Remove oldest (FIFO order)
        
        return mask


# ========================
# Transformer Block
# ========================

class TransformerBlock(nn.Module):
    """
    Transformer encoder/decoder block with optional local attention and cross-attention.
    Supports both pre-norm and post-norm variants.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_local_attention: bool = False,
        window_size: int = 5,
        cross_attention: bool = False,
        pre_norm: bool = True
    ):
        super().__init__()
        
        self.pre_norm = pre_norm
        self.cross_attention_enabled = cross_attention
        
        # Self-attention
        if use_local_attention:
            self.self_attention = LocalAttention(d_model, n_heads, window_size, dropout)
        else:
            self.self_attention = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True
            )
        
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross-attention (decoder only)
        if cross_attention:
            self.cross_attention = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True
            )
            self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm_ff = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            memory: [batch, src_len, d_model] (for cross-attention)
            self_attn_mask: [seq_len, seq_len] (e.g., causal mask)
            self_key_padding_mask: [batch, seq_len]
            memory_key_padding_mask: [batch, src_len]
        """
        
        # Self-attention
        if self.pre_norm:
            # Pre-norm: normalize before attention
            x_norm = self.norm1(x)
            if isinstance(self.self_attention, LocalAttention):
                attn_output, _ = self.self_attention(
                    x_norm, x_norm, x_norm,
                    key_padding_mask=self_key_padding_mask,
                    attn_mask=self_attn_mask
                )
            else:
                attn_output, _ = self.self_attention(
                    x_norm, x_norm, x_norm,
                    key_padding_mask=self_key_padding_mask,
                    attn_mask=self_attn_mask
                )
            x = x + self.dropout(attn_output)
        else:
            # Post-norm: normalize after attention
            if isinstance(self.self_attention, LocalAttention):
                attn_output, _ = self.self_attention(
                    x, x, x,
                    key_padding_mask=self_key_padding_mask,
                    attn_mask=self_attn_mask
                )
            else:
                attn_output, _ = self.self_attention(
                    x, x, x,
                    key_padding_mask=self_key_padding_mask,
                    attn_mask=self_attn_mask
                )
            x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention (decoder only)
        if self.cross_attention_enabled and memory is not None:
            if self.pre_norm:
                x_norm = self.norm2(x)
                attn_output, _ = self.cross_attention(
                    x_norm, memory, memory,
                    key_padding_mask=memory_key_padding_mask
                )
                x = x + self.dropout(attn_output)
            else:
                attn_output, _ = self.cross_attention(
                    x, memory, memory,
                    key_padding_mask=memory_key_padding_mask
                )
                x = self.norm2(x + self.dropout(attn_output))
        
        # Feed-forward network
        if self.pre_norm:
            x_norm = self.norm_ff(x)
            ff_output = self.feed_forward(x_norm)
            x = x + self.dropout(ff_output)
        else:
            ff_output = self.feed_forward(x)
            x = self.norm_ff(x + self.dropout(ff_output))
        
        return x


# ========================
# Transformer Seq2Seq
# ========================

class TransformerSeq2Seq(nn.Module):
    """
    Transformer sequence-to-sequence model with local attention.
    Config-driven and assignment-compliant (≤2 layers).
    
    FIXES APPLIED:
    - Top-p sampling corrected
    - Temperature=0 handling
    - Beam search memory indexing
    - Empty sequence validation
    """
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        config: Optional[Dict] = None,
        config_path: Optional[str] = None,
        src_pad_idx: int = 0,
        tgt_pad_idx: int = 0,
        tgt_sos_idx: int = 1,
        tgt_eos_idx: int = 2,
    ):
        super().__init__()
        
        # Load config
        if config is None:
            if config_path is None:
                config_path = "config/config.yaml"
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            except Exception as e:
                logger.warning(f"Could not load config: {e}, using defaults")
                config = {}
        
        self.config = config
        transformer_config = config.get('transformer', {})
        
        # Extract hyperparameters
        d_model = transformer_config.get('d_model', 256)
        n_heads = transformer_config.get('n_heads', 8)
        num_layers = min(transformer_config.get('num_layers', 2), 2)  # Assignment constraint
        d_ff = transformer_config.get('d_ff', 1024)
        dropout = transformer_config.get('dropout', 0.1)
        use_local_attention = transformer_config.get('use_local_attention', True)
        window_size = transformer_config.get('local_attention_window', 5)
        
        # FIXED: Use max_positional_encoding instead of max_seq_length
        max_pos_encoding = transformer_config.get('max_positional_encoding', 100)
        
        pre_norm = transformer_config.get('pre_norm', True)
        pos_encoding_type = transformer_config.get('positional_encoding_type', 'sinusoidal')
        
        # Validate num_layers (assignment compliance)
        if transformer_config.get('num_layers', 2) > 2:
            logger.warning(f"Assignment requires ≤2 layers, got {transformer_config.get('num_layers')}. Using 2.")
        
        # Store architecture info
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.tgt_sos_idx = tgt_sos_idx
        self.tgt_eos_idx = tgt_eos_idx
        self.d_model = d_model
        self.num_layers = num_layers
        self.use_local_attention = use_local_attention
        self.window_size = window_size
        self.max_seq_length = config.get('data', {}).get('max_seq_length', 50)
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=src_pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=tgt_pad_idx)
        
        # Positional encoding
        if pos_encoding_type == 'learned':
            self.pos_encoding = LearnedPositionalEncoding(d_model, max_pos_encoding, dropout)
        else:  # sinusoidal
            self.pos_encoding = PositionalEncoding(d_model, max_pos_encoding, dropout)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads, d_ff, dropout,
                use_local_attention=use_local_attention,
                window_size=window_size,
                cross_attention=False,
                pre_norm=pre_norm
            )
            for _ in range(num_layers)
        ])
        
        # Decoder layers with cross-attention
        self.decoder_layers = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads, d_ff, dropout,
                use_local_attention=use_local_attention,
                window_size=window_size,
                cross_attention=True,
                pre_norm=pre_norm
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # Initialize parameters
        self._init_parameters()
        
        logger.info(f"Initialized Transformer: {self.count_parameters():,} parameters")
        logger.info(f"  Local attention: {use_local_attention} (window: {window_size})")
        logger.info(f"  Pre-norm: {pre_norm}, Layers: {num_layers}")
    
    def _init_parameters(self):
        """Initialize parameters using Xavier uniform"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _create_padding_mask(self, x: torch.Tensor, pad_idx: int) -> torch.Tensor:
        """Create padding mask: True for padding tokens"""
        return (x == pad_idx)
    
    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generate causal (subsequent) mask: True to ignore"""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask
    
    def encode(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode source sequence.
        
        Args:
            src: [batch, src_len]
            src_key_padding_mask: [batch, src_len]
        
        Returns:
            memory: [batch, src_len, d_model]
        """
        # FIXED: Validate non-empty sequence
        if src.size(1) == 0:
            raise ValueError("Source sequence cannot be empty (src_len=0)")
        
        # Embed and add positional encoding
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x, self_key_padding_mask=src_key_padding_mask)
        
        return x
    
    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode target sequence.
        
        Args:
            tgt: [batch, tgt_len]
            memory: [batch, src_len, d_model]
            tgt_key_padding_mask: [batch, tgt_len]
            memory_key_padding_mask: [batch, src_len]
        
        Returns:
            logits: [batch, tgt_len, tgt_vocab_size]
        """
        # Embed and add positional encoding
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # Create causal mask
        tgt_len = tgt.size(1)
        causal_mask = self._generate_causal_mask(tgt_len, tgt.device)
        
        # Pass through decoder layers
        for layer in self.decoder_layers:
            x = layer(
                x,
                memory=memory,
                self_attn_mask=causal_mask,
                self_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
        
        # Project to vocabulary
        logits = self.output_projection(x)
        return logits
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        teacher_forcing_ratio: float = 1.0,  # Not used in Transformer, kept for API compatibility
        src_lengths: Optional[torch.Tensor] = None  # Not used, kept for API compatibility
    ) -> torch.Tensor:
        """
        Forward pass for training.
        
        Args:
            src: [batch, src_len]
            tgt: [batch, tgt_len]
            teacher_forcing_ratio: Ignored (Transformer always uses teacher forcing)
            src_lengths: Ignored (Transformer uses padding masks)
        
        Returns:
            logits: [batch, tgt_len, tgt_vocab_size]
        """
        # Create padding masks
        src_padding_mask = self._create_padding_mask(src, self.src_pad_idx)
        tgt_padding_mask = self._create_padding_mask(tgt, self.tgt_pad_idx)
        
        # Encode
        memory = self.encode(src, src_padding_mask)
        
        # Decode
        logits = self.decode(tgt, memory, tgt_padding_mask, src_padding_mask)
        
        return logits
    
    def generate(
        self,
        src: torch.Tensor,
        max_length: Optional[int] = None,
        beam_size: int = 1,
        temperature: float = 1.0,
        top_p: float = 1.0,
        src_lengths: Optional[torch.Tensor] = None  # Ignored, for API compatibility
    ) -> Union[torch.Tensor, List[List[torch.Tensor]]]:
        """
        Generate translations using greedy or beam search.
        
        Args:
            src: [batch, src_len]
            max_length: Maximum output length
            beam_size: Beam size (1 = greedy)
            temperature: Temperature for sampling (1.0 = no effect, 0.0 = greedy)
            top_p: Top-p sampling threshold (1.0 = no effect)
            src_lengths: Ignored (for API compatibility with LSTM)
        
        Returns:
            If beam_size == 1: [batch, out_len] tensor
            If beam_size > 1: List[List[Tensor]] - [batch_size][beam_size]
        """
        self.eval()
        
        if max_length is None:
            max_length = self.max_seq_length
        
        with torch.no_grad():
            if beam_size == 1:
                return self._greedy_decode(src, max_length, temperature, top_p)
            else:
                return self._beam_search(src, max_length, beam_size)
    
    def _greedy_decode(
        self, 
        src: torch.Tensor, 
        max_length: int, 
        temperature: float = 1.0, 
        top_p: float = 1.0
    ) -> torch.Tensor:
        """Greedy decoding with temperature and top-p sampling"""
        batch_size = src.size(0)
        device = src.device
        
        # Encode source
        src_padding_mask = self._create_padding_mask(src, self.src_pad_idx)
        memory = self.encode(src, src_padding_mask)
        
        # Start with SOS token
        output = torch.full(
            (batch_size, 1), self.tgt_sos_idx, dtype=torch.long, device=device
        )
        
        # Generate token by token
        for _ in range(max_length - 1):
            # Decode
            tgt_padding_mask = self._create_padding_mask(output, self.tgt_pad_idx)
            logits = self.decode(output, memory, tgt_padding_mask, src_padding_mask)
            
            # FIXED: Handle temperature=0 case
            if temperature == 0.0:
                next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            else:
                # Apply temperature
                probs = F.softmax(logits[:, -1] / temperature, dim=-1)
                
                # FIXED: Apply top-p sampling if needed
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    # Remove tokens with cumulative probability above top_p
                    sorted_indices_to_remove = cumulative_probs > top_p
                    
                    # Shift right to keep at least one token
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # FIXED: Scatter back to original indices correctly
                    indices_to_remove = torch.zeros_like(probs, dtype=torch.bool)
                    indices_to_remove.scatter_(
                        dim=-1,
                        index=sorted_indices,
                        src=sorted_indices_to_remove
                    )
                    probs = probs.masked_fill(indices_to_remove, 0.0)
                    
                    # Renormalize
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                
                # Sample next token (greedy if temperature=1.0 and top_p=1.0)
                if temperature == 1.0 and top_p == 1.0:
                    next_token = probs.argmax(dim=-1, keepdim=True)
                else:
                    next_token = torch.multinomial(probs, 1)
            
            output = torch.cat([output, next_token], dim=1)
            
            # Check if all sequences finished
            if (next_token == self.tgt_eos_idx).all():
                break
        
        return output
    
    def _beam_search(
        self, src: torch.Tensor, max_length: int, beam_size: int
    ) -> List[List[torch.Tensor]]:
        """
        Beam search decoding.
        Returns List[List[Tensor]] for consistency with LSTM API.
        
        FIXED: Memory indexing corrected
        """
        batch_size = src.size(0)
        device = src.device
        
        # Encode source (do this once for all beams)
        src_padding_mask = self._create_padding_mask(src, self.src_pad_idx)
        memory = self.encode(src, src_padding_mask)
        
        # Initialize beams for each batch item
        # Each beam: (sequence, score)
        beams = []
        for b in range(batch_size):
            initial_seq = torch.tensor([[self.tgt_sos_idx]], device=device)
            beams.append([(initial_seq, 0.0)])
        
        # Beam search for each timestep
        for step in range(max_length - 1):
            all_finished = True
            new_beams = []
            
            for b in range(batch_size):
                # Check if all beams finished
                if all(seq[0, -1].item() == self.tgt_eos_idx for seq, _ in beams[b]):
                    new_beams.append(beams[b])
                    continue
                
                all_finished = False
                candidates = []
                
                for seq, score in beams[b]:
                    # Skip if already finished
                    if seq[0, -1].item() == self.tgt_eos_idx:
                        candidates.append((seq, score))
                        continue
                    
                    # FIXED: Proper memory indexing without redundant expand
                    mem_b = memory[b:b+1]  # Already [1, src_len, d_model]
                    src_mask_b = src_padding_mask[b:b+1] if src_padding_mask is not None else None
                    tgt_mask_b = self._create_padding_mask(seq, self.tgt_pad_idx)
                    
                    logits = self.decode(seq, mem_b, tgt_mask_b, src_mask_b)
                    
                    # Get log probabilities
                    log_probs = F.log_softmax(logits[:, -1], dim=-1)
                    topk_log_probs, topk_tokens = log_probs.topk(beam_size)
                    
                    for k in range(beam_size):
                        token = topk_tokens[0, k].item()
                        token_log_prob = topk_log_probs[0, k].item()
                        
                        # Calculate new score (normalized by length)
                        new_score = (score * len(seq[0]) + token_log_prob) / (len(seq[0]) + 1)
                        
                        new_seq = torch.cat(
                            [seq, torch.tensor([[token]], device=device)], dim=1
                        )
                        candidates.append((new_seq, new_score))
                
                # Keep top beam_size candidates
                candidates.sort(key=lambda x: x[1], reverse=True)
                new_beams.append(candidates[:beam_size])
            
            beams = new_beams
            
            if all_finished:
                break
        
        # Extract beam sequences for each batch item
        results = []
        for b in range(batch_size):
            batch_results = [seq.squeeze(0) for seq, _ in beams[b]]
            results.append(batch_results)
        
        return results
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_checkpoint(self, filepath: Union[str, Path], **kwargs) -> None:
        """Save model checkpoint with config"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'src_vocab_size': self.src_vocab_size,
            'tgt_vocab_size': self.tgt_vocab_size,
            'src_pad_idx': self.src_pad_idx,
            'tgt_pad_idx': self.tgt_pad_idx,
            'tgt_sos_idx': self.tgt_sos_idx,
            'tgt_eos_idx': self.tgt_eos_idx,
            **kwargs
        }
        
        # Atomic write
        temp_path = filepath.with_suffix('.tmp')
        torch.save(checkpoint, temp_path)
        temp_path.replace(filepath)
        
        logger.info(f"💾 Checkpoint saved: {filepath}")
    
    @classmethod
    def load_checkpoint(
        cls, filepath: Union[str, Path], device: Optional[torch.device] = None
    ):
        """Load model from checkpoint"""
        filepath = Path(filepath)
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(filepath, map_location=device)
        
        model = cls(
            src_vocab_size=checkpoint['src_vocab_size'],
            tgt_vocab_size=checkpoint['tgt_vocab_size'],
            config=checkpoint['config'],
            src_pad_idx=checkpoint.get('src_pad_idx', 0),
            tgt_pad_idx=checkpoint.get('tgt_pad_idx', 0),
            tgt_sos_idx=checkpoint.get('tgt_sos_idx', 1),
            tgt_eos_idx=checkpoint.get('tgt_eos_idx', 2),
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        logger.info(f"📂 Checkpoint loaded: {filepath}")
        return model, checkpoint
    
    def get_device(self) -> torch.device:
        """Get current device"""
        return next(self.parameters()).device


def main():
    """Test Transformer model"""
    print("🧪 Testing Transformer Seq2Seq model...\n")
    
    # Test with dummy vocab sizes
    src_vocab_size = 100
    tgt_vocab_size = 200
    
    # Create model
    model = TransformerSeq2Seq(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        src_pad_idx=0,
        tgt_pad_idx=0,
        tgt_sos_idx=1,
        tgt_eos_idx=2,
    )
    
    print(f"✅ Model created: {model.count_parameters():,} parameters")
    print(f"   Local attention: {model.use_local_attention}")
    print(f"   Window size: {model.window_size}")
    print(f"   Device: {model.get_device()}\n")
    
    # Test forward pass
    batch_size = 4
    src_len = 10
    tgt_len = 12
    
    src = torch.randint(3, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(3, tgt_vocab_size, (batch_size, tgt_len))
    
    outputs = model(src, tgt)
    print(f"✅ Forward pass: {outputs.shape} (expected: [{batch_size}, {tgt_len}, {tgt_vocab_size}])")
    
    # Test greedy generation
    generated = model.generate(src, beam_size=1)
    print(f"✅ Greedy generation: {generated.shape}")
    
    # Test temperature=0
    generated_temp0 = model.generate(src, beam_size=1, temperature=0.0)
    print(f"✅ Temperature=0: {generated_temp0.shape}")
    
    # Test beam search
    beam_results = model.generate(src[:2], beam_size=3)
    print(f"✅ Beam search: {len(beam_results)} batch items, {len(beam_results[0])} beams each")
    
    # Test save/load
    checkpoint_path = Path("outputs/checkpoints/test_transformer.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(checkpoint_path, epoch=1, loss=0.5)
    loaded_model, ckpt = TransformerSeq2Seq.load_checkpoint(checkpoint_path)
    print(f"✅ Save/load checkpoint: epoch={ckpt.get('epoch')}, loss={ckpt.get('loss')}")
    
    # Clean up
    checkpoint_path.unlink()
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    main()