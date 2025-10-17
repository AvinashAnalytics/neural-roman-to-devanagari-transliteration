# scripts/train_transformer.py
"""
Transformer Training Script for Hindi Transliteration
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import yaml
import os
import json
import random
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple, List, Optional
from pathlib import Path
from datetime import datetime
import logging
import shutil

from models.transformer_model import TransformerSeq2Seq
from utils.data_loader import DataManager
from utils.evaluation import Evaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Make W&B optional
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("⚠️  W&B not available, continuing without it")


def set_seed(seed: int):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class TransformerTrainer:
    """Transformer Trainer with full config integration and assignment compliance"""
    
    def __init__(self, config_path: str = "config/config.yaml", use_wandb: bool = False):
        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.config_path = config_path
        
        # Set reproducibility seeds
        seed = self.config['global']['seed']
        set_seed(seed)
        logger.info(f"🎲 Random seed set to: {seed}")
        
        # Device setup
        device_config = self.config['global'].get('device', 'auto')
        if device_config == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device_config)
        
        logger.info(f"🔧 Device: {self.device}")
        if self.device.type == 'cuda':
            logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"   CUDA Version: {torch.version.cuda}")
        
        # Initialize data manager
        logger.info("📚 Loading data...")
        self.data_manager = DataManager(config_path)
        self.train_loader, self.valid_loader, self.test_loader = self.data_manager.prepare_data()
        
        # Load test data with grouped references (ACL W15-3902 format)
        test_file = Path(self.config['paths']['processed_data_dir']) / 'test.json'
        self.test_sources, self.test_references = self.data_manager.load_test_data_for_eval(
            str(test_file)
        )
        
        logger.info(f"✅ Data loaded:")
        logger.info(f"   Train: {len(self.train_loader.dataset):,} examples, {len(self.train_loader)} batches")
        logger.info(f"   Valid: {len(self.valid_loader.dataset):,} examples, {len(self.valid_loader)} batches")
        logger.info(f"   Test:  {len(self.test_sources):,} unique source words")
        
        # Initialize model with vocab indices and config
        logger.info("🤖 Initializing Transformer model...")
        self.model = TransformerSeq2Seq(
            src_vocab_size=len(self.data_manager.src_vocab),
            tgt_vocab_size=len(self.data_manager.tgt_vocab),
            config=self.config,
            src_pad_idx=self.data_manager.src_vocab.pad_idx,
            tgt_pad_idx=self.data_manager.tgt_vocab.pad_idx,
            tgt_sos_idx=self.data_manager.tgt_vocab.sos_idx,
            tgt_eos_idx=self.data_manager.tgt_vocab.eos_idx,
        ).to(self.device)
        
        transformer_config = self.config['transformer']
        logger.info(f"✅ Transformer Model initialized:")
        logger.info(f"   Parameters: {self.model.count_parameters():,}")
        logger.info(f"   Layers: {self.model.num_layers}")
        logger.info(f"   Local Attention: {self.model.use_local_attention}")
        logger.info(f"   Window Size: {self.model.window_size}")
        logger.info(f"   d_model: {transformer_config['d_model']}, n_heads: {transformer_config['n_heads']}")
        
        # FIXED: Use model-specific training config
        train_config = self.config['training']
        model_config = train_config.get('transformer_specific', {})
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.data_manager.tgt_vocab.pad_idx,
            label_smoothing=train_config.get('label_smoothing', 0.0)
        )
        
        # FIXED: Get optimizer params from transformer_specific
        optimizer_type = model_config.get('optimizer', 'adam').lower()
        learning_rate = model_config.get('learning_rate', 0.0005)
        weight_decay = model_config.get('weight_decay', 0.0001)
        betas = model_config.get('betas', [0.9, 0.98])
        epsilon = model_config.get('epsilon', 1e-9)
        
        # Optimizer
        if optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=betas,
                eps=epsilon
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=betas,
                eps=epsilon
            )
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision training
        self.use_amp = train_config.get('use_amp', False) and self.device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        if self.use_amp:
            logger.info("⚡ Mixed precision training enabled (AMP)")
        
        # Gradient accumulation
        self.gradient_accumulation_steps = train_config.get('gradient_accumulation_steps', 1)
        if self.gradient_accumulation_steps > 1:
            logger.info(f"📦 Gradient accumulation: {self.gradient_accumulation_steps} steps")
        
        # Evaluator
        self.evaluator = Evaluator(config=self.config, verbose=False)
        
        # Setup W&B if requested
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project=self.config.get('wandb', {}).get('project', 'transliteration-cs772'),
                name=f"transformer-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=self.config,
                mode="online" if use_wandb else "offline"
            )
            wandb.watch(self.model, log='all', log_freq=100)
        
        # Create output directories
        self.checkpoint_dir = Path(self.config['paths']['checkpoint_dir'])
        self.results_dir = Path(self.config['paths']['results_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0  # word_accuracy
        self.patience_counter = 0
        
        # Training history
        self.history = {
            'train_loss': [],
            'valid_loss': [],
            'word_accuracy': [],
            'char_f1': [],
            'char_precision': [],
            'char_recall': [],
            'learning_rate': []
        }
        
        # Get training config
        self.num_epochs = train_config['epochs']
        self.early_stopping_patience = train_config.get('early_stopping_patience', 5)
        self.gradient_clip = train_config.get('gradient_clip_norm', 1.0)
        self.log_every_n_steps = train_config.get('log_every_n_steps', 100)
        
        # Checkpoint config
        self.save_every_n_epochs = train_config.get('save_every_n_epochs', 1)
        self.keep_last_n_checkpoints = train_config.get('keep_last_n_checkpoints', 3)
        
        logger.info(f"🎯 Training configuration:")
        logger.info(f"   Epochs: {self.num_epochs}")
        logger.info(f"   Early stopping patience: {self.early_stopping_patience}")
        logger.info(f"   Initial LR: {learning_rate}")
        logger.info(f"   Gradient clip: {self.gradient_clip}")
    
    def _create_scheduler(self):
        """Create LR scheduler based on config"""
        train_config = self.config['training']
        model_config = train_config.get('transformer_specific', {})
        
        if not model_config.get('use_lr_scheduler', True):
            return None
        
        scheduler_type = model_config.get('scheduler_type', 'cosine_with_warmup').lower()
        
        if scheduler_type == 'cosine_with_warmup':
            # Simple warmup + cosine decay (manual implementation)
            warmup_steps = model_config.get('warmup_steps', 4000)
            min_lr = model_config.get('min_lr', 1e-6)
            
            # Use LambdaLR for custom warmup
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                else:
                    # Cosine decay after warmup
                    progress = (step - warmup_steps) / max(1, self.num_epochs * len(self.train_loader) - warmup_steps)
                    return max(min_lr / self.optimizer.param_groups[0]['lr'], 0.5 * (1 + np.cos(np.pi * progress)))
            
            return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        elif scheduler_type == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                patience=model_config.get('lr_decay_patience', 3),
                factor=model_config.get('lr_decay_factor', 0.5),
                min_lr=model_config.get('min_lr', 1e-6),
                verbose=True
            )
        else:
            logger.warning(f"Unknown scheduler type: {scheduler_type}, using cosine with warmup")
            return optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lambda step: min(1.0, step / 4000)
            )
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f'Epoch {epoch}/{self.num_epochs}',
            ncols=120,
            leave=True
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(src, tgt)  # [B, L, V]
                    
                    # Loss: predict next token (shift by 1)
                    # outputs[:, :-1] predicts tgt[:, 1:]
                    loss = self.criterion(
                        outputs[:, :-1].contiguous().view(-1, self.model.tgt_vocab_size),
                        tgt[:, 1:].contiguous().view(-1)
                    )
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    if self.gradient_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.gradient_clip
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    # Step scheduler (if not ReduceLROnPlateau)
                    if self.scheduler is not None and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step()
            else:
                outputs = self.model(src, tgt)
                loss = self.criterion(
                    outputs[:, :-1].contiguous().view(-1, self.model.tgt_vocab_size),
                    tgt[:, 1:].contiguous().view(-1)
                )
                loss = loss / self.gradient_accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    if self.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.gradient_clip
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    # Step scheduler
                    if self.scheduler is not None and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Log to W&B
            if self.use_wandb and self.global_step % self.log_every_n_steps == 0:
                wandb.log({
                    'train/batch_loss': loss.item() * self.gradient_accumulation_steps,
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'global_step': self.global_step
                })
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def validate(self) -> Tuple[float, Dict]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        all_predictions = []
        all_references = []
        
        with torch.no_grad():
            for batch in tqdm(self.valid_loader, desc='Validating', ncols=100, leave=False):
                src = batch['src'].to(self.device)
                tgt = batch['tgt'].to(self.device)
                
                # Compute loss
                outputs = self.model(src, tgt)
                loss = self.criterion(
                    outputs[:, :-1].contiguous().view(-1, self.model.tgt_vocab_size),
                    tgt[:, 1:].contiguous().view(-1)
                )
                # FIXED: Don't apply gradient accumulation to validation loss
                total_loss += loss.item()
                num_batches += 1
                
                # Generate predictions (greedy for speed during validation)
                pred_indices = self.model.generate(src, beam_size=1)
                
                # Decode predictions
                for i in range(len(batch['src_text'])):
                    pred_text = self.data_manager.tgt_vocab.decode(
                        pred_indices[i].cpu().tolist(),
                        remove_special=True
                    )
                    all_predictions.append(pred_text)
                    all_references.append([batch['tgt_text'][i]])
        
        # Calculate metrics
        metrics = self.evaluator.calculate_metrics(
            all_predictions, all_references, beam_search=False
        )
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss, metrics
    
    def train(self):
        """Main training loop"""
        logger.info(f"\n{'='*80}")
        logger.info(f"🚀 STARTING TRAINING")
        logger.info(f"{'='*80}\n")
        
        # Save config to checkpoint directory
        config_save_path = self.checkpoint_dir / 'config.yaml'
        shutil.copy(self.config_path, config_save_path)
        logger.info(f"📋 Config saved to: {config_save_path}")
        
        for epoch in range(1, self.num_epochs + 1):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            valid_loss, metrics = self.validate()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['valid_loss'].append(valid_loss)
            self.history['word_accuracy'].append(metrics['word_accuracy'])
            self.history['char_f1'].append(metrics['char_f1'])
            self.history['char_precision'].append(metrics['char_precision'])
            self.history['char_recall'].append(metrics['char_recall'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Print epoch summary
            logger.info(f"\n{'='*80}")
            logger.info(f"Epoch {epoch}/{self.num_epochs} Summary")
            logger.info(f"{'='*80}")
            logger.info(f"  Train Loss:      {train_loss:.4f}")
            logger.info(f"  Valid Loss:      {valid_loss:.4f}")
            logger.info(f"  Word Accuracy:   {metrics['word_accuracy']:.2%}")
            logger.info(f"  Char F1:         {metrics['char_f1']:.2%}")
            logger.info(f"  Char Precision:  {metrics['char_precision']:.2%}")
            logger.info(f"  Char Recall:     {metrics['char_recall']:.2%}")
            logger.info(f"  Learning Rate:   {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Log to W&B
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train/loss': train_loss,
                    'valid/loss': valid_loss,
                    'valid/word_accuracy': metrics['word_accuracy'],
                    'valid/char_f1': metrics['char_f1'],
                    'valid/char_precision': metrics['char_precision'],
                    'valid/char_recall': metrics['char_recall'],
                    'lr': self.optimizer.param_groups[0]['lr']
                })
            
            # Update learning rate scheduler (ReduceLROnPlateau only)
            if self.scheduler is not None and isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(metrics['word_accuracy'])
            
            # Check for best model
            current_metric = metrics['word_accuracy']
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True, metrics=metrics)
                logger.info(f"  ✨ New best model! Word Accuracy: {current_metric:.2%}")
            else:
                self.patience_counter += 1
                logger.info(f"  Patience: {self.patience_counter}/{self.early_stopping_patience}")
            
            # Regular checkpoint
            if self.save_every_n_epochs > 0 and epoch % self.save_every_n_epochs == 0:
                self.save_checkpoint(epoch, is_best=False, metrics=metrics)
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"\n⚠️  Early stopping triggered after {epoch} epochs")
                logger.info(f"   Best Word Accuracy: {self.best_metric:.2%}")
                break
            
            logger.info(f"{'='*80}\n")
        
        # Save training history
        history_path = self.results_dir / 'transformer_training_history.json'
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"📊 Training history saved: {history_path}")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ TRAINING COMPLETE!")
        logger.info(f"   Best Word Accuracy: {self.best_metric:.2%}")
        logger.info(f"   Total Epochs: {self.current_epoch}")
        logger.info(f"{'='*80}\n")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False, metrics: Optional[Dict] = None):
        """Save model checkpoint (atomic write)"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_metric': self.best_metric,
            'metrics': metrics,
            'config': self.config,
            'history': self.history
        }
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'transformer_best.pt'
            temp_path = best_path.with_suffix('.tmp')
            torch.save(checkpoint, temp_path)
            temp_path.replace(best_path)
            logger.info(f"  💾 Saved best checkpoint: {best_path.name}")
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'transformer_epoch{epoch}.pt'
        temp_path = checkpoint_path.with_suffix('.tmp')
        torch.save(checkpoint, temp_path)
        temp_path.replace(checkpoint_path)
        
        # Manage checkpoint history
        if self.keep_last_n_checkpoints > 0:
            checkpoints = sorted(
                self.checkpoint_dir.glob('transformer_epoch*.pt'),
                key=lambda p: p.stat().st_mtime
            )
            if len(checkpoints) > self.keep_last_n_checkpoints:
                for old_ckpt in checkpoints[:-self.keep_last_n_checkpoints]:
                    old_ckpt.unlink()
    
    def compare_decoding_methods(self, sample_size: int = None):
        """Compare greedy vs beam search decoding (assignment requirement)"""
        if sample_size is None:
            sample_size = min(1000, len(self.valid_loader.dataset))
        
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 COMPARING DECODING METHODS (sample_size={sample_size})")
        logger.info(f"{'='*80}\n")
        
        self.model.eval()
        beam_sizes = self.config['evaluation'].get('beam_sizes', [1, 3, 5, 10])
        comparison_results = {}
        
        for beam_size in beam_sizes:
            logger.info(f"Testing beam_size={beam_size}...")
            
            predictions = []
            references = []
            sample_count = 0
            
            with torch.no_grad():
                for batch in self.valid_loader:
                    if sample_count >= sample_size:
                        break
                    
                    src = batch['src'].to(self.device)
                    
                    # Generate with current beam size
                    if beam_size == 1:
                        output = self.model.generate(src, beam_size=1)
                        # output: [batch, seq_len]
                        for i in range(min(len(batch['src_text']), sample_size - sample_count)):
                            pred = self.data_manager.tgt_vocab.decode(
                                output[i].cpu().tolist(), remove_special=True
                            )
                            predictions.append(pred)
                            references.append([batch['tgt_text'][i]])
                            sample_count += 1
                    else:
                        beam_output = self.model.generate(src, beam_size=beam_size)
                        # beam_output: List[List[Tensor]] - [batch_size][beam_size]
                        for i in range(min(len(batch['src_text']), sample_size - sample_count)):
                            # Take top-1 from beam
                            pred = self.data_manager.tgt_vocab.decode(
                                beam_output[i][0].cpu().tolist(), remove_special=True
                            )
                            predictions.append(pred)
                            references.append([batch['tgt_text'][i]])
                            sample_count += 1
                    
                    if sample_count >= sample_size:
                        break
            
            # Calculate metrics
            metrics = self.evaluator.calculate_metrics(predictions, references, beam_search=False)
            method = "greedy" if beam_size == 1 else f"beam_{beam_size}"
            comparison_results[method] = metrics
            
            logger.info(f"  {method:12s}: Word Acc: {metrics['word_accuracy']:.4f}, "
                       f"Char F1: {metrics['char_f1']:.4f}, "
                       f"Char Prec: {metrics['char_precision']:.4f}, "
                       f"Char Rec: {metrics['char_recall']:.4f}")
        
        # Save comparison results
        comparison_path = self.results_dir / 'transformer_decoding_comparison.json'
        with open(comparison_path, 'w', encoding='utf-8') as f:
            json.dump(comparison_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n✅ Comparison saved: {comparison_path}")
        logger.info(f"{'='*80}\n")
        
        return comparison_results
    
    def test(self):
        """Final test evaluation with ACL W15-3902 compliance"""
        logger.info(f"\n{'='*80}")
        logger.info(f"🧪 FINAL TEST EVALUATION (ACL W15-3902 COMPLIANT)")
        logger.info(f"{'='*80}\n")
        
        # Load best checkpoint if available
        best_checkpoint_path = self.checkpoint_dir / 'transformer_best.pt'
        if best_checkpoint_path.exists():
            checkpoint = torch.load(best_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"📂 Loaded best checkpoint from epoch {checkpoint['epoch']}")
            logger.info(f"   Best validation Word Acc: {checkpoint['best_metric']:.2%}\n")
        else:
            logger.warning(f"Best checkpoint not found, using current model state")
        
        self.model.eval()
        
        beam_sizes = self.config['evaluation'].get('beam_sizes', [1, 5])
        results = {}
        
        for beam_size in beam_sizes:
            logger.info(f"Testing with beam_size={beam_size}...")
            
            predictions = []
            decoding_records = []
            
            with torch.no_grad():
                for src_word in tqdm(self.test_sources, desc=f'Beam-{beam_size}', ncols=100):
                    # Encode source
                    src_indices = self.data_manager.src_vocab.encode(src_word, add_special=True)
                    src_tensor = torch.tensor([src_indices], device=self.device)
                    
                    # Generate
                    if beam_size == 1:
                        output = self.model.generate(src_tensor, beam_size=1)
                        pred_text = self.data_manager.tgt_vocab.decode(
                            output[0].cpu().tolist(), remove_special=True
                        )
                        predictions.append(pred_text)
                        # Record decoding (single hypothesis)
                        decoding_records.append({
                            'source': src_word,
                            'beams': [
                                {'text': pred_text, 'score': None}
                            ]
                        })
                    else:
                        # Request per-beam scores when available
                        beam_output = self.model.generate(src_tensor, beam_size=beam_size, return_scores=True)
                        # FIXED: Collect all beam candidates as list
                        beam_preds = []
                        beams_for_record = []
                        for beam_entry in beam_output[0]:  # First batch item
                            if isinstance(beam_entry, tuple) or isinstance(beam_entry, list):
                                seq_tensor, score = beam_entry
                                pred_text = self.data_manager.tgt_vocab.decode(
                                    seq_tensor.cpu().tolist(), remove_special=True
                                )
                                beams_for_record.append({'text': pred_text, 'score': score})
                                beam_preds.append(pred_text)
                            else:
                                # Older format: tensor only
                                pred_text = self.data_manager.tgt_vocab.decode(
                                    beam_entry.cpu().tolist(), remove_special=True
                                )
                                beams_for_record.append({'text': pred_text, 'score': None})
                                beam_preds.append(pred_text)
                        predictions.append(beam_preds)
                        decoding_records.append({'source': src_word, 'beams': beams_for_record})
            
            # Calculate metrics
            beam_search = beam_size > 1
            metrics = self.evaluator.calculate_metrics(
                predictions, self.test_references, beam_search=beam_search
            )
            
            method = "greedy" if beam_size == 1 else f"beam_{beam_size}"
            results[method] = metrics
            
            logger.info(f"\n{method} results:")
            logger.info(f"  Word Accuracy:   {metrics['word_accuracy']:.4f}")
            logger.info(f"  Char F1:         {metrics['char_f1']:.4f}")
            logger.info(f"  Char Precision:  {metrics['char_precision']:.4f}")
            logger.info(f"  Char Recall:     {metrics['char_recall']:.4f}")
            if 'mrr' in metrics:
                logger.info(f"  MRR:             {metrics['mrr']:.4f}")
                logger.info(f"  Top-5 Accuracy:  {metrics['top5_accuracy']:.4f}")
        
        # Error analysis
        logger.info(f"\n{'='*80}")
        logger.info(f"🔍 ERROR ANALYSIS (Difficult Sequences)")
        logger.info(f"{'='*80}\n")
        
        # Use greedy predictions for error analysis
        greedy_preds = [p if isinstance(p, str) else p[0] for p in predictions]
        error_analysis = self.evaluator.analyze_difficult_sequences(
            greedy_preds,
            self.test_references,
            sources=self.test_sources,
            max_errors=20
        )
        
        logger.info("Difficult character patterns:")
        for pattern_name, pattern_data in error_analysis.items():
            if pattern_name != 'common_errors' and pattern_data:
                logger.info(f"\n  {pattern_name}:")
                for char, count in list(pattern_data.items())[:5]:
                    logger.info(f"    {char}: {count} errors")
        
        # Save results
        results_path = self.results_dir / 'transformer_test_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump({
                'results': results,
                'decoding': decoding_records if len(decoding_records) > 0 else None,
                'error_analysis': error_analysis
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n✅ Test results saved: {results_path}")
        logger.info(f"{'='*80}\n")
        
        # Generate comparison table
        table = self.evaluator.generate_comparison_table(results, output_format='markdown')
        logger.info("\n" + table + "\n")
        
        # Save comparison table
        table_path = self.results_dir / 'transformer_comparison_table.md'
        with open(table_path, 'w', encoding='utf-8') as f:
            f.write("# Transformer Transliteration Results\n\n")
            f.write(table)
        logger.info(f"📋 Comparison table saved: {table_path}\n")
        
        if self.use_wandb:
            wandb.log({'test_results': results})
            wandb.finish()
        
        return results


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Transformer for Hindi Transliteration')
    parser.add_argument('--config', default='config/config.yaml', help='Config file path')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--test-only', action='store_true', help='Only run testing (load best checkpoint)')
    parser.add_argument('--compare-only', action='store_true', help='Only run decoding comparison')
    
    args = parser.parse_args()
    
    try:
        trainer = TransformerTrainer(config_path=args.config, use_wandb=args.wandb)
        
        if args.test_only:
            trainer.test()
        elif args.compare_only:
            trainer.compare_decoding_methods()
        else:
            trainer.train()
            trainer.compare_decoding_methods()
            trainer.test()
    
    except KeyboardInterrupt:
        logger.info("\n⚠️  Training interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())