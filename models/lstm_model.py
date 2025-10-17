# models/lstm_model.py
"""
LSTM-based Sequence-to-Sequence Model for Hindi Transliteration
CS772 Assignment 2 - Roman → Devanagari
FIXES APPLIED:
✅ Bidirectional LSTM hidden state reshaping for decoder compatibility
✅ Attention mechanism implementation (Bahdanau and Luong)
✅ Decoder input handling with/without attention
✅ Greedy decoding with temperature and top-p sampling fixes
✅ Beam search hidden state cloning for independence
✅ Empty sequence validation in encoder

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List, Union
from pathlib import Path
import random
import logging
import yaml

logger = logging.getLogger(__name__)


class LSTMEncoder(nn.Module):
    """
    Bidirectional LSTM Encoder with padding mask support.
    Returns encoder outputs and final hidden states for decoder initialization.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        pad_idx: int = 0,
    ):
        super().__init__()
        
        # Validate assignment compliance (≤2 layers)
        if num_layers > 2:
            logger.warning(f"Assignment requires ≤2 layers, got {num_layers}. Setting to 2.")
            num_layers = 2
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.pad_idx = pad_idx
        
    def forward(
        self, 
        src: torch.Tensor, 
        src_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            src: [batch, src_len] - source token indices
            src_lengths: [batch] - actual lengths (for packing)
        
        Returns:
            outputs: [batch, src_len, hidden_dim * num_directions]
            (hidden, cell): each [num_layers, batch, hidden_dim * num_directions]
        """
        # FIXED: Validate non-empty sequence
        if src.size(1) == 0:
            raise ValueError("Source sequence cannot be empty (src_len=0)")
        
        batch_size = src.size(0)
        
        # Embed with dropout
        embedded = self.dropout(self.embedding(src))  # [B, L, E]
        
        # Pack padded sequences for efficiency (if lengths provided)
        if src_lengths is not None:
            # Move lengths to CPU for pack_padded_sequence
            src_lengths_cpu = src_lengths.cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, src_lengths_cpu, batch_first=True, enforce_sorted=False
            )
            packed_outputs, (hidden, cell) = self.lstm(packed)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        else:
            outputs, (hidden, cell) = self.lstm(embedded)
        
        # Reshape bidirectional hidden states for decoder compatibility
        # hidden/cell: [num_layers * num_directions, batch, hidden_dim]
        if self.bidirectional:
            # Concatenate forward and backward for each layer
            hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_dim)
            hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=-1)  # [num_layers, B, H*2]
            
            cell = cell.view(self.num_layers, 2, batch_size, self.hidden_dim)
            cell = torch.cat([cell[:, 0], cell[:, 1]], dim=-1)
        
        return outputs, (hidden, cell)


class Attention(nn.Module):
    """
    Attention mechanism for LSTM decoder.
    Supports Bahdanau (additive) and Luong (multiplicative) attention.
    """
    
    def __init__(
        self,
        encoder_hidden_dim: int,
        decoder_hidden_dim: int,
        attention_type: str = 'luong',  # 'luong' or 'bahdanau'
    ):
        super().__init__()
        self.attention_type = attention_type.lower()
        
        if self.attention_type == 'bahdanau':
            # Additive attention: score(h_t, h_s) = v^T tanh(W1 h_t + W2 h_s)
            self.W1 = nn.Linear(decoder_hidden_dim, decoder_hidden_dim)
            self.W2 = nn.Linear(encoder_hidden_dim, decoder_hidden_dim)
            self.v = nn.Linear(decoder_hidden_dim, 1, bias=False)
        elif self.attention_type == 'luong':
            # Multiplicative attention: score(h_t, h_s) = h_t^T W h_s
            self.W = nn.Linear(encoder_hidden_dim, decoder_hidden_dim, bias=False)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}. Use 'luong' or 'bahdanau'")
    
    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            decoder_hidden: [batch, decoder_hidden_dim] - current decoder hidden state
            encoder_outputs: [batch, src_len, encoder_hidden_dim] - all encoder outputs
            src_mask: [batch, src_len] - MASKING CONVENTION: 1=valid token, 0=padding
        
        Returns:
            context: [batch, encoder_hidden_dim] - attended context vector
            attention_weights: [batch, src_len] - attention distribution
        """
        batch_size, src_len, encoder_hidden_dim = encoder_outputs.size()
        
        if self.attention_type == 'bahdanau':
            # decoder_hidden: [B, D] → [B, 1, D] → [B, L, D]
            decoder_proj = self.W1(decoder_hidden).unsqueeze(1).expand(-1, src_len, -1)
            encoder_proj = self.W2(encoder_outputs)
            scores = self.v(torch.tanh(decoder_proj + encoder_proj)).squeeze(-1)  # [B, L]
        
        else:  # luong
            # decoder_hidden: [B, D] → [B, D, 1]
            # encoder_outputs: [B, L, E] → [B, L, D] (after W)
            encoder_proj = self.W(encoder_outputs)  # [B, L, D]
            scores = torch.bmm(encoder_proj, decoder_hidden.unsqueeze(-1)).squeeze(-1)  # [B, L]
        
        # Apply mask if provided
        # MASKING CONVENTION: src_mask[b, i] = 0 means position i is padding → set score to -inf
        if src_mask is not None:
            scores = scores.masked_fill(src_mask == 0, float('-inf'))
        
        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)  # [B, L]
        
        # Compute context vector as weighted sum of encoder outputs
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # [B, E]
        
        return context, attention_weights


class LSTMDecoder(nn.Module):
    """
    LSTM Decoder with optional attention mechanism.
    Supports both standard (no attention) and attention-based decoding.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        encoder_bidirectional: bool = True,
        use_attention: bool = True,
        attention_type: str = 'luong',
        pad_idx: int = 0,
    ):
        super().__init__()
        
        self.use_attention = use_attention
        self.decoder_hidden_dim = hidden_dim * 2 if encoder_bidirectional else hidden_dim
        self.encoder_output_dim = hidden_dim * 2 if encoder_bidirectional else hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # LSTM input: embedding + context (if attention enabled)
        lstm_input_dim = embedding_dim + self.encoder_output_dim if use_attention else embedding_dim
        
        self.lstm = nn.LSTM(
            lstm_input_dim,
            self.decoder_hidden_dim,
            num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        
        # Attention mechanism
        if use_attention:
            self.attention = Attention(
                self.encoder_output_dim, 
                self.decoder_hidden_dim, 
                attention_type
            )
        
        # Output projection
        output_input_dim = self.decoder_hidden_dim + self.encoder_output_dim if use_attention else self.decoder_hidden_dim
        self.output_projection = nn.Linear(output_input_dim, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        input_token: torch.Tensor, 
        hidden: Tuple[torch.Tensor, torch.Tensor],
        encoder_outputs: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
        """
        Single decoder step.
        
        Args:
            input_token: [batch] or [batch, 1] - current input token
            hidden: (h, c) each [num_layers, batch, decoder_hidden_dim]
            encoder_outputs: [batch, src_len, encoder_hidden_dim] - for attention
            src_mask: [batch, src_len] - padding mask (1=valid, 0=padding)
        
        Returns:
            prediction: [batch, 1, vocab_size] - output logits
            hidden: updated (h, c)
            attention_weights: [batch, src_len] or None
        """
        if input_token.dim() == 1:
            input_token = input_token.unsqueeze(1)
        
        # Embed
        embedded = self.dropout(self.embedding(input_token))  # [B, 1, E]
        
        attention_weights = None
        
        if self.use_attention and encoder_outputs is not None:
            # Compute attention using previous hidden state
            # Use top layer hidden state for attention
            decoder_hidden = hidden[0][-1]  # [B, D]
            context, attention_weights = self.attention(decoder_hidden, encoder_outputs, src_mask)
            
            # Concatenate embedding with context
            lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)  # [B, 1, E+H]
        else:
            lstm_input = embedded
        
        # LSTM step
        output, hidden = self.lstm(lstm_input, hidden)  # output: [B, 1, D]
        
        # Compute prediction
        if self.use_attention and encoder_outputs is not None:
            # Concatenate LSTM output with context
            prediction_input = torch.cat([output, context.unsqueeze(1)], dim=-1)  # [B, 1, D+H]
        else:
            prediction_input = output
        
        prediction = self.output_projection(prediction_input)  # [B, 1, vocab]
        
        return prediction, hidden, attention_weights


class Seq2SeqLSTM(nn.Module):
    """
    Sequence-to-Sequence LSTM Model with optional attention.
    Config-driven and assignment-compliant (≤2 layers).
    
    FIXES APPLIED:
    - Top-p sampling corrected
    - Temperature=0 handling
    - Beam search hidden state cloning
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
        lstm_config = config.get('lstm', {})
        
        # Extract hyperparameters from config
        embedding_dim = lstm_config.get('embedding_dim', 256)
        hidden_dim = lstm_config.get('hidden_dim', 512)
        num_layers = min(lstm_config.get('num_layers', 2), 2)  # Assignment constraint
        dropout = lstm_config.get('dropout', 0.3)
        bidirectional = lstm_config.get('bidirectional', True)
        use_attention = lstm_config.get('use_attention', True)
        attention_type = lstm_config.get('attention_type', 'luong')
        
        # Get max length from config
        max_length = config.get('data', {}).get('max_seq_length', 100)
        
        # Store vocab info
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.tgt_sos_idx = tgt_sos_idx
        self.tgt_eos_idx = tgt_eos_idx
        self.max_length = max_length
        
        # Build encoder and decoder
        self.encoder = LSTMEncoder(
            vocab_size=src_vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            pad_idx=src_pad_idx,
        )
        
        self.decoder = LSTMDecoder(
            vocab_size=tgt_vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            encoder_bidirectional=bidirectional,
            use_attention=use_attention,
            attention_type=attention_type,
            pad_idx=tgt_pad_idx,
        )
        
        # Store architecture info
        self.use_attention = use_attention
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        
        # Device (will be set on first forward pass)
        self._device = None
        
        logger.info(f"Initialized LSTM Seq2Seq: {self.count_parameters():,} parameters")
        logger.info(f"  Attention: {use_attention} ({attention_type if use_attention else 'N/A'})")
        logger.info(f"  Bidirectional: {bidirectional}, Layers: {num_layers}")
    
    def create_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        Create mask for source sequence.
        MASKING CONVENTION: 1 for valid tokens, 0 for padding
        """
        return (src != self.src_pad_idx).long()
    
    def forward(
        self, 
        src: torch.Tensor, 
        tgt: torch.Tensor, 
        teacher_forcing_ratio: float = 0.5,
        src_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Training forward pass with teacher forcing.
        
        Args:
            src: [batch, src_len] - source sequences
            tgt: [batch, tgt_len] - target sequences (including SOS)
            teacher_forcing_ratio: probability of using teacher forcing
            src_lengths: [batch] - actual source lengths
        
        Returns:
            outputs: [batch, tgt_len, vocab_size] - logits for each position
        """
        batch_size, tgt_len = tgt.size()
        device = src.device
        
        # Initialize outputs tensor
        outputs = torch.zeros(batch_size, tgt_len, self.tgt_vocab_size, device=device)
        
        # Encode source
        encoder_outputs, (hidden, cell) = self.encoder(src, src_lengths)
        
        # Create source mask for attention
        src_mask = self.create_src_mask(src) if self.use_attention else None
        
        # First input is SOS token (tgt[:, 0])
        input_token = tgt[:, 0]  # [B]
        
        # Decode step by step
        for t in range(tgt_len):
            # Decoder step
            prediction, (hidden, cell), _ = self.decoder(
                input_token, (hidden, cell), encoder_outputs, src_mask
            )
            
            # Store output
            outputs[:, t] = prediction.squeeze(1)
            
            # Decide next input: teacher forcing or predicted token
            if t < tgt_len - 1:  # Don't need next input for last step
                use_teacher_force = random.random() < teacher_forcing_ratio
                if use_teacher_force:
                    input_token = tgt[:, t + 1]
                else:
                    input_token = prediction.argmax(dim=-1).squeeze(1)
        
        return outputs
    
    def generate(
        self, 
        src: torch.Tensor, 
        max_length: Optional[int] = None,
        beam_size: int = 1,
        temperature: float = 1.0,
        top_p: float = 1.0,
        src_lengths: Optional[torch.Tensor] = None,
        return_scores: bool = False,
    ) -> Union[torch.Tensor, List[List[torch.Tensor]]]:
        """
        Generate translation using greedy or beam search.
        
        Args:
            src: [batch, src_len] - source sequences
            max_length: maximum output length (default: self.max_length)
            beam_size: beam size (1 = greedy)
            temperature: Temperature for sampling (1.0 = no effect, 0.0 = greedy)
            top_p: Top-p sampling threshold (1.0 = no effect)
            src_lengths: [batch] - actual source lengths
        
        Returns:
            If beam_size == 1: [batch, out_len] - token indices
            If beam_size > 1: List of List of sequences (top beam_size per batch item)
        """
        self.eval()
        with torch.no_grad():
            if max_length is None:
                max_length = self.max_length
            
            if beam_size == 1:
                return self._greedy_decode(src, max_length, temperature, top_p, src_lengths)
            else:
                return self._beam_search(src, max_length, beam_size, src_lengths, return_scores=return_scores)
    
    def _greedy_decode(
        self, 
        src: torch.Tensor, 
        max_length: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        src_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Greedy decoding (beam_size=1) with temperature and top-p sampling"""
        batch_size = src.size(0)
        device = src.device
        
        # Encode
        encoder_outputs, (hidden, cell) = self.encoder(src, src_lengths)
        src_mask = self.create_src_mask(src) if self.use_attention else None
        
        # Start with SOS token
        input_token = torch.full((batch_size,), self.tgt_sos_idx, dtype=torch.long, device=device)
        
        # Collect outputs
        outputs = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_length):
            # Decoder step
            prediction, (hidden, cell), _ = self.decoder(
                input_token, (hidden, cell), encoder_outputs, src_mask
            )
            
            # FIXED: Handle temperature=0 case
            if temperature == 0.0:
                predicted_token = prediction.squeeze(1).argmax(dim=-1)
            else:
                # Apply temperature
                probs = F.softmax(prediction.squeeze(1) / temperature, dim=-1)
                
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
                    predicted_token = probs.argmax(dim=-1)
                else:
                    predicted_token = torch.multinomial(probs, 1).squeeze(1)
            
            outputs.append(predicted_token.unsqueeze(1))
            
            # Check for EOS
            is_eos = (predicted_token == self.tgt_eos_idx)
            finished = finished | is_eos
            
            # Stop if all sequences finished
            if finished.all():
                break
            
            # Next input
            input_token = predicted_token
        
        if len(outputs) == 0:
            return torch.full((batch_size, 1), self.tgt_eos_idx, dtype=torch.long, device=device)
        
        return torch.cat(outputs, dim=1)  # [B, L]
    
    def _beam_search(
        self, 
        src: torch.Tensor, 
        max_length: int, 
        beam_size: int,
        src_lengths: Optional[torch.Tensor] = None,
        return_scores: bool = False
    ) -> List[List[torch.Tensor]]:
        """
        Beam search decoding.
        Returns List[List[Tensor]] of shape [batch][beam_size] with token sequences.
        
        FIXED: Hidden states are cloned for independence
        """
        device = src.device
        batch_size = src.size(0)
        
        # Encode
        encoder_outputs, (hidden, cell) = self.encoder(src, src_lengths)
        src_mask = self.create_src_mask(src) if self.use_attention else None
        
        # Initialize beams for each batch item
        # Each beam: (sequence, score, (hidden_state, cell_state))
        beams = []
        for b in range(batch_size):
            # FIXED: Clone hidden states for independence across beams
            h_b = hidden[:, b:b+1, :].clone().contiguous()  # [num_layers, 1, H]
            c_b = cell[:, b:b+1, :].clone().contiguous()
            
            # Initial beam: SOS token with score 0.0
            initial_seq = torch.tensor([[self.tgt_sos_idx]], device=device)
            beams.append([(initial_seq, 0.0, (h_b, c_b))])
        
        # Beam search for each timestep
        for step in range(max_length):
            all_finished = True
            new_beams = []
            
            for b in range(batch_size):
                # Check if all beams for this batch item have finished
                if all(seq[0, -1].item() == self.tgt_eos_idx for seq, _, _ in beams[b]):
                    new_beams.append(beams[b])
                    continue
                
                all_finished = False
                candidates = []
                
                for seq, score, (h_state, c_state) in beams[b]:
                    # Skip if this beam already finished
                    if seq[0, -1].item() == self.tgt_eos_idx:
                        candidates.append((seq, score, (h_state, c_state)))
                        continue
                    
                    # Get last token
                    input_token = seq[:, -1]  # [1]
                    
                    # Decoder step
                    enc_out_b = encoder_outputs[b:b+1] if encoder_outputs is not None else None
                    src_mask_b = src_mask[b:b+1] if src_mask is not None else None
                    
                    pred, (new_h, new_c), _ = self.decoder(
                        input_token, (h_state, c_state), enc_out_b, src_mask_b
                    )
                    
                    # Get log probabilities
                    log_probs = F.log_softmax(pred[0, 0], dim=-1)
                    topk_log_probs, topk_indices = log_probs.topk(beam_size)
                    
                    for k in range(beam_size):
                        token = topk_indices[k].item()
                        token_log_prob = topk_log_probs[k].item()
                        
                        # Normalize score by sequence length
                        new_score = (score * len(seq[0]) + token_log_prob) / (len(seq[0]) + 1)
                        new_seq = torch.cat([seq, torch.tensor([[token]], device=device)], dim=1)
                        
                        # FIXED: Clone hidden states for each new beam
                        candidates.append((new_seq, new_score, (new_h.clone(), new_c.clone())))
                
                # Keep top beam_size candidates
                candidates.sort(key=lambda x: x[1], reverse=True)
                new_beams.append(candidates[:beam_size])
            
            beams = new_beams
            
            if all_finished:
                break
        
        # Extract top sequence (and optionally scores) from each batch item's beams
        results = []
        for b in range(batch_size):
            if return_scores:
                batch_results = [(seq.squeeze(0), float(score)) for seq, score, _ in beams[b]]
            else:
                batch_results = [seq.squeeze(0) for seq, _, _ in beams[b]]
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
        
        torch.save(checkpoint, filepath)
        logger.info(f"💾 Checkpoint saved: {filepath}")
    
    @classmethod
    def load_checkpoint(cls, filepath: Union[str, Path], device: Optional[torch.device] = None):
        """Load model from checkpoint"""
        filepath = Path(filepath)
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        
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
    """Test LSTM model"""
    print("🧪 Testing LSTM Seq2Seq model...\n")
    
    # Test with dummy vocab sizes
    src_vocab_size = 100
    tgt_vocab_size = 200
    
    # Create model with default config
    model = Seq2SeqLSTM(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        src_pad_idx=0,
        tgt_pad_idx=0,
        tgt_sos_idx=1,
        tgt_eos_idx=2,
    )
    
    print(f"✅ Model created: {model.count_parameters():,} parameters")
    print(f"   Attention: {model.use_attention}")
    print(f"   Device: {model.get_device()}\n")
    
    # Test forward pass
    batch_size = 4
    src_len = 10
    tgt_len = 12
    
    src = torch.randint(3, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(3, tgt_vocab_size, (batch_size, tgt_len))
    src_lengths = torch.tensor([10, 8, 9, 7])
    
    outputs = model(src, tgt, teacher_forcing_ratio=0.5, src_lengths=src_lengths)
    print(f"✅ Forward pass: {outputs.shape} (expected: [{batch_size}, {tgt_len}, {tgt_vocab_size}])")
    
    # Test greedy generation
    generated = model.generate(src, beam_size=1, src_lengths=src_lengths)
    print(f"✅ Greedy generation: {generated.shape}")
    
    # Test temperature=0
    generated_temp0 = model.generate(src, beam_size=1, temperature=0.0, src_lengths=src_lengths)
    print(f"✅ Temperature=0: {generated_temp0.shape}")
    
    # Test beam search
    beam_results = model.generate(src[:2], beam_size=3, src_lengths=src_lengths[:2])
    print(f"✅ Beam search: {len(beam_results)} batch items, {len(beam_results[0])} beams each")
    
    # Test save/load
    checkpoint_path = Path("outputs/checkpoints/test_lstm.pt")
    model.save_checkpoint(checkpoint_path, epoch=1, loss=0.5)
    loaded_model, ckpt = Seq2SeqLSTM.load_checkpoint(checkpoint_path)
    print(f"✅ Save/load checkpoint: epoch={ckpt.get('epoch')}, loss={ckpt.get('loss')}")
    
    # Clean up
    checkpoint_path.unlink()
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    main()