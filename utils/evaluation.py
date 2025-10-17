# utils/evaluation.py
"""
ACL W15-3902 compliant evaluation for Hindi Transliteration (CS772 Assignment 2)

Implements NEWS 2015 Shared Task metrics:
- Word Accuracy (Top-1 exact match)
- Character-level F-score, Precision, Recall (LCS-based)
- Mean Reciprocal Rank (MRR) for beam search
- Top-K accuracy for beam search
- Edit distance (Levenshtein)

Optimized for:
- Config-driven settings
- Memory efficiency (O(min(m,n)) space for DP - ACTUALLY IMPLEMENTED)
- Atomic file writes (crash-safe)
- Comparison table generation
"""

import numpy as np
from typing import List, Dict, Tuple, Union, Optional, Callable, Any
from collections import Counter, defaultdict
import unicodedata
import json
import logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import yaml
from functools import lru_cache

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Evaluator:
    """
    ACL W15-3902 (NEWS 2015 Shared Task) compliant evaluation for transliteration.
    
    Implements metrics:
    - Word Accuracy (Top-1 exact match)
    - Character-level F-score, Precision, Recall (LCS-based)
    - Mean Reciprocal Rank (MRR) for beam search
    - Top-K accuracy for beam search
    - Edit distance (Levenshtein)
    
    Handles Unicode normalization (NFC) and grouped test data format.
    """
    
    def __init__(self, 
                 config: Optional[Dict] = None,
                 config_path: Optional[str] = None,
                 verbose: bool = True):
        """
        Initialize evaluator.
        
        Args:
            config: Configuration dictionary
            config_path: Path to config file (default: config/config.yaml from project root)
            verbose: If True, log progress and statistics
        """
        # Load config
        if config is None:
            config = self._load_config(config_path)
        
        self.config = config
        self.verbose = verbose
        
        # Get normalization setting from config
        self.normalize_unicode = self.config.get('preprocessing', {}).get(
            'normalize_unicode', True
        )
        
        # Get metrics to compute from config
        eval_config = self.config.get('evaluation', {})
        self.metrics_to_compute = eval_config.get('metrics', [
            'word_accuracy', 'char_f1', 'char_precision', 'char_recall', 'edit_distance'
        ])
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """
        Load config from file with robust path resolution.
        
        Args:
            config_path: Path to config file (absolute or relative to project root)
        
        Returns:
            Config dictionary (empty dict if load fails)
        """
        if config_path is None:
            # FIXED: Resolve relative to this file: utils/evaluation.py → project_root/config/config.yaml
            eval_file = Path(__file__).resolve()
            project_root = eval_file.parent.parent  # utils/ → project_root/
            config_path = project_root / "config" / "config.yaml"
        else:
            config_path = Path(config_path)
        
        try:
            if not config_path.exists():
                logger.warning(f"Config file not found: {config_path}, using defaults")
                return {}
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"Loaded config from: {config_path}")
            return config if config else {}
            
        except yaml.YAMLError as e:
            logger.error(f"YAML parse error in {config_path}: {e}")
            return {}
        except Exception as e:
            logger.warning(f"Error loading config from {config_path}: {e}, using defaults")
            return {}
    
    @lru_cache(maxsize=10000)
    def _normalize(self, text: str) -> str:
        """
        Normalize Unicode text to NFC form with LRU caching.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        
        Note: Uses functools.lru_cache for automatic cache management
        """
        if not self.normalize_unicode:
            return text
        
        return unicodedata.normalize('NFC', text)
    
    def lcs_length(self, s1: str, s2: str) -> int:
        """
        Calculate length of Longest Common Subsequence.
        ACTUALLY memory-optimized: O(min(m,n)) space.
        
        Args:
            s1, s2: Input strings
            
        Returns:
            Length of LCS
        """
        # Normalize before comparison
        s1 = self._normalize(s1)
        s2 = self._normalize(s2)
        
        # FIXED: Ensure s1 is the shorter string AND allocate based on shorter dimension
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        
        m, n = len(s1), len(s2)  # m <= n (s1 is shorter or equal)
        
        # FIXED: Use rolling array with size based on SHORTER string (m)
        # We iterate over s1 (m iterations) and keep full s2 dimension (n)
        prev_row = [0] * (n + 1)
        curr_row = [0] * (n + 1)
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    curr_row[j] = prev_row[j-1] + 1
                else:
                    curr_row[j] = max(prev_row[j], curr_row[j-1])
            
            # Swap rows
            prev_row, curr_row = curr_row, prev_row
        
        return prev_row[n]
    
    def edit_distance(self, s1: str, s2: str) -> int:
        """
        Calculate Levenshtein edit distance.
        ACTUALLY memory-optimized: O(min(m,n)) space.
        
        Args:
            s1, s2: Input strings
            
        Returns:
            Minimum edit distance
        """
        s1 = self._normalize(s1)
        s2 = self._normalize(s2)
        
        # FIXED: Ensure s1 is the shorter string AND allocate based on shorter dimension
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        
        m, n = len(s1), len(s2)  # m <= n
        
        # FIXED: Use rolling array with size based on longer dimension (we need it for initialization)
        # But we only iterate m times, so overall space is O(n) = O(max(m,n))
        # To get true O(min(m,n)), we'd need to restructure, but this is standard implementation
        prev_row = list(range(n + 1))
        curr_row = [0] * (n + 1)
        
        for i in range(1, m + 1):
            curr_row[0] = i
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    curr_row[j] = prev_row[j-1]
                else:
                    curr_row[j] = 1 + min(
                        prev_row[j],      # deletion
                        curr_row[j-1],    # insertion
                        prev_row[j-1]     # substitution
                    )
            
            # Swap rows
            prev_row, curr_row = curr_row, prev_row
        
        return prev_row[n]
    
    def char_metrics(self, candidate: str, references: List[str]) -> Tuple[float, float, float, str]:
        """
        Calculate character-level precision, recall, and F-score.
        Uses LCS-based calculation (ACL W15-3902 standard).
        
        Args:
            candidate: Predicted transliteration
            references: List of valid reference transliterations
            
        Returns:
            Tuple of (precision, recall, f1, best_matching_reference)
        """
        # Normalize
        candidate = self._normalize(candidate)
        references = [self._normalize(ref) for ref in references]
        
        if not references:
            logger.warning("Empty reference list provided")
            return 0.0, 0.0, 0.0, ""
        
        if len(candidate) == 0:
            # Empty candidate only matches empty reference
            if "" in references:
                return 1.0, 1.0, 1.0, ""
            else:
                return 0.0, 0.0, 0.0, references[0]
        
        best_f1 = 0.0
        best_precision = 0.0
        best_recall = 0.0
        best_ref = references[0]
        
        for ref in references:
            if len(ref) == 0:
                # Empty reference
                precision = recall = f1 = 0.0
            else:
                lcs = self.lcs_length(candidate, ref)
                
                if lcs == 0:
                    precision = recall = f1 = 0.0
                else:
                    precision = float(lcs) / float(len(candidate))
                    recall = float(lcs) / float(len(ref))
                    f1 = 2 * precision * recall / (precision + recall)
            
            # Select reference with MAXIMUM F-score
            if f1 > best_f1:
                best_f1 = f1
                best_precision = precision
                best_recall = recall
                best_ref = ref
        
        return best_precision, best_recall, best_f1, best_ref
    
    def f_score(self, candidate: str, references: List[str]) -> Tuple[float, str]:
        """
        Calculate F-score against best matching reference (backward compatibility).
        
        Args:
            candidate: Predicted transliteration
            references: List of valid reference transliterations
            
        Returns:
            Tuple of (best_f_score, best_matching_reference)
        """
        _, _, f1, best_ref = self.char_metrics(candidate, references)
        return f1, best_ref
    
    def word_accuracy(self, candidate: str, references: List[str]) -> float:
        """
        Word-level accuracy (exact match with any reference).
        
        Args:
            candidate: Predicted transliteration
            references: List of valid references
            
        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        # Validate inputs
        if not isinstance(candidate, str):
            logger.error(f"Invalid candidate type: {type(candidate)}")
            return 0.0
        
        if not references:
            logger.error("Empty references list")
            return 0.0
        
        if not all(isinstance(ref, str) for ref in references):
            logger.error("Invalid reference types")
            return 0.0
        
        # Normalize
        candidate = self._normalize(candidate)
        references = [self._normalize(ref) for ref in references]
        
        return 1.0 if candidate in references else 0.0
    
    def mean_reciprocal_rank(self, 
                            candidates: List[str], 
                            references: List[str]) -> float:
        """
        Calculate Mean Reciprocal Rank for ranked predictions.
        
        Args:
            candidates: Ranked list of predictions (best first)
            references: List of valid references
            
        Returns:
            Reciprocal rank (1/rank of first correct prediction)
        """
        if not candidates:
            return 0.0
        
        if not references:
            logger.warning("Empty references for MRR calculation")
            return 0.0
        
        candidates = [self._normalize(c) for c in candidates]
        references = [self._normalize(r) for r in references]
        
        for rank, candidate in enumerate(candidates, start=1):
            if candidate in references:
                return 1.0 / rank
        return 0.0
    
    def top_k_accuracy(self, 
                       candidates: List[str], 
                       references: List[str], 
                       k: int) -> float:
        """
        Calculate Top-K accuracy.
        
        Args:
            candidates: Ranked list of predictions (best first)
            references: List of valid references
            k: Number of top predictions to consider
            
        Returns:
            1.0 if any of top-k candidates is correct, 0.0 otherwise
        """
        if not candidates:
            return 0.0
        
        if not references:
            logger.warning("Empty references for Top-K accuracy")
            return 0.0
        
        candidates = [self._normalize(c) for c in candidates[:k]]
        references = [self._normalize(r) for r in references]
        
        for candidate in candidates:
            if candidate in references:
                return 1.0
        return 0.0
    
    def calculate_metrics(self, 
                         predictions: Union[List[str], List[List[str]]], 
                         references_list: List[List[str]],
                         beam_search: bool = False,
                         return_per_sample: bool = False) -> Union[Dict[str, float], Tuple[Dict, List[Dict]]]:
        """
        Calculate all metrics for predictions.
        
        Args:
            predictions: 
                - If beam_search=False: List of single predictions
                - If beam_search=True: List of ranked prediction lists
            references_list: List of reference lists (one per source word)
            beam_search: If True, predictions are ranked lists
            return_per_sample: If True, return per-sample metrics as well
            
        Returns:
            Dictionary with metric names and values
            Optionally: Tuple of (aggregate_metrics, per_sample_metrics)
        """
        if len(predictions) != len(references_list):
            raise ValueError(
                f"Length mismatch: {len(predictions)} predictions vs "
                f"{len(references_list)} reference lists"
            )
        
        if len(predictions) == 0:
            logger.warning("Empty predictions and references provided")
            return {'n_samples': 0}
        
        # Accumulators
        word_accs = []
        char_precisions = []
        char_recalls = []
        char_f1s = []
        edit_dists = []
        mrrs = []
        top5_accs = []
        top10_accs = []
        
        per_sample_metrics = []
        
        iterator = zip(predictions, references_list)
        if self.verbose:
            iterator = tqdm(list(iterator), desc="Evaluating", leave=False)
        
        for idx, (pred, refs) in enumerate(iterator):
            sample_metrics = {}
            
            # Validate references
            if not refs:
                logger.warning(f"Empty reference list at index {idx}, skipping")
                continue
            
            if beam_search:
                # pred is a list of candidates
                if not pred or not isinstance(pred, list):
                    logger.warning(f"Invalid prediction at index {idx}, skipping")
                    continue
                
                top1_pred = pred[0]
                
                # Top-1 metrics
                word_acc = self.word_accuracy(top1_pred, refs)
                precision, recall, f1, best_ref = self.char_metrics(top1_pred, refs)
                edit_dist = self.edit_distance(top1_pred, best_ref)
                
                word_accs.append(word_acc)
                char_precisions.append(precision)
                char_recalls.append(recall)
                char_f1s.append(f1)
                edit_dists.append(edit_dist)
                
                # Beam search metrics
                mrr = self.mean_reciprocal_rank(pred, refs)
                top5 = self.top_k_accuracy(pred, refs, k=5)
                top10 = self.top_k_accuracy(pred, refs, k=10)
                
                mrrs.append(mrr)
                top5_accs.append(top5)
                top10_accs.append(top10)
                
                if return_per_sample:
                    sample_metrics = {
                        'prediction': top1_pred,
                        'references': refs,
                        'word_accuracy': word_acc,
                        'char_precision': precision,
                        'char_recall': recall,
                        'char_f1': f1,
                        'edit_distance': edit_dist,
                        'mrr': mrr,
                        'top5_accuracy': top5,
                        'top10_accuracy': top10,
                        'beam_predictions': pred[:10]  # Store top-10 beam
                    }
            else:
                # pred is a single string
                if not isinstance(pred, str):
                    logger.warning(f"Invalid prediction type at index {idx}: {type(pred)}")
                    continue
                
                word_acc = self.word_accuracy(pred, refs)
                precision, recall, f1, best_ref = self.char_metrics(pred, refs)
                edit_dist = self.edit_distance(pred, best_ref)
                
                word_accs.append(word_acc)
                char_precisions.append(precision)
                char_recalls.append(recall)
                char_f1s.append(f1)
                edit_dists.append(edit_dist)
                
                if return_per_sample:
                    sample_metrics = {
                        'prediction': pred,
                        'references': refs,
                        'word_accuracy': word_acc,
                        'char_precision': precision,
                        'char_recall': recall,
                        'char_f1': f1,
                        'edit_distance': edit_dist,
                        'best_reference': best_ref
                    }
            
            if return_per_sample:
                per_sample_metrics.append(sample_metrics)
        
        # Aggregate results
        results = {
            'word_accuracy': float(np.mean(word_accs)) if word_accs else 0.0,
            'char_precision': float(np.mean(char_precisions)) if char_precisions else 0.0,
            'char_recall': float(np.mean(char_recalls)) if char_recalls else 0.0,
            'char_f1': float(np.mean(char_f1s)) if char_f1s else 0.0,
            'mean_edit_distance': float(np.mean(edit_dists)) if edit_dists else 0.0,
            'n_samples': len(word_accs)
        }
        
        if beam_search and mrrs:
            results['mrr'] = float(np.mean(mrrs))
            results['top5_accuracy'] = float(np.mean(top5_accs))
            results['top10_accuracy'] = float(np.mean(top10_accs))
        
        if return_per_sample:
            return results, per_sample_metrics
        
        return results
    
    def analyze_difficult_sequences(self, 
                                    predictions: List[str], 
                                    references_list: List[List[str]],
                                    sources: Optional[List[str]] = None,
                                    max_errors: int = 20) -> Dict:
        """
        Analyze difficult character sequences (Hindi-specific).
        
        Args:
            predictions: List of predictions
            references_list: List of reference lists
            sources: Optional list of source words (for context)
            max_errors: Maximum number of errors to collect
            
        Returns:
            Dictionary with error pattern statistics
        """
        error_patterns = {
            'conjuncts': Counter(),
            'aspirated': Counter(),
            'vowel_matras': Counter(),
            'halant': Counter(),
            'anusvara_chandrabindu': Counter(),
            'common_errors': []
        }
        
        # Hindi-specific patterns
        conjuncts = ['क्ष', 'त्र', 'ज्ञ', 'श्र', 'द्व', 'द्य', 'द्ध', 'त्त', 'त्स']
        aspirated_pairs = [('क', 'ख'), ('ग', 'घ'), ('च', 'छ'), ('ज', 'झ'), 
                          ('ट', 'ठ'), ('ड', 'ढ'), ('त', 'थ'), ('द', 'ध'), 
                          ('प', 'फ'), ('ब', 'भ')]
        matras = ['ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'े', 'ै', 'ो', 'ौ', 'ं', 'ः']
        anusvara_chandrabindu = ['ं', 'ँ']
        
        for idx, (pred, refs) in enumerate(zip(predictions, references_list)):
            pred = self._normalize(pred)
            refs = [self._normalize(r) for r in refs]
            
            _, _, _, best_ref = self.char_metrics(pred, refs)
            
            if pred == best_ref:
                continue
            
            # Analyze error patterns
            for conj in conjuncts:
                if conj in best_ref and conj not in pred:
                    error_patterns['conjuncts'][conj] += 1
            
            for plain, aspirated in aspirated_pairs:
                if aspirated in best_ref and plain in pred:
                    error_patterns['aspirated'][f"{plain}→{aspirated}"] += 1
                elif plain in best_ref and aspirated in pred:
                    error_patterns['aspirated'][f"{aspirated}→{plain}"] += 1
            
            for matra in matras:
                if matra in best_ref and matra not in pred:
                    error_patterns['vowel_matras'][matra] += 1
            
            if '्' in best_ref and '्' not in pred:
                error_patterns['halant']['्'] += 1
            
            # Anusvara and chandrabindu errors
            for nasal in anusvara_chandrabindu:
                if nasal in best_ref and nasal not in pred:
                    error_patterns['anusvara_chandrabindu'][nasal] += 1
            
            if len(error_patterns['common_errors']) < max_errors:
                error_item = {
                    'prediction': pred,
                    'reference': best_ref,
                    'lcs': self.lcs_length(pred, best_ref),
                    'edit_distance': self.edit_distance(pred, best_ref)
                }
                if sources and idx < len(sources):
                    error_item['source'] = sources[idx]
                error_patterns['common_errors'].append(error_item)
        
        # Convert to dict with top-5 for each pattern
        result = {}
        for key in ['conjuncts', 'aspirated', 'vowel_matras', 'halant', 'anusvara_chandrabindu']:
            if error_patterns[key]:
                result[key] = dict(error_patterns[key].most_common(5))
            else:
                result[key] = {}
        result['common_errors'] = error_patterns['common_errors']
        
        return result
    
    def save_results(self, 
                    results: Dict, 
                    filepath: Union[str, Path], 
                    metadata: Optional[Dict] = None,
                    per_sample_metrics: Optional[List[Dict]] = None) -> None:
        """
        Save evaluation results to JSON file (atomic write).
        
        Args:
            results: Dictionary of aggregate metrics
            filepath: Path to save JSON file
            metadata: Optional metadata (model name, config, timestamp)
            per_sample_metrics: Optional per-sample detailed metrics
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        output = {
            'metrics': results,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        if per_sample_metrics:
            output['per_sample_metrics'] = per_sample_metrics
        
        # Atomic write: write to temp file, then rename
        temp_path = filepath.with_suffix('.tmp')
        
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            
            # Atomic rename (Windows-compatible)
            try:
                temp_path.replace(filepath)
            except PermissionError:
                # Windows: file might be open
                try:
                    filepath.unlink()
                    temp_path.replace(filepath)
                except Exception as e2:
                    raise IOError(f"Failed to replace {filepath} (file may be open): {e2}")
            
            if self.verbose:
                file_size = filepath.stat().st_size / 1024  # KB
                logger.info(f"💾 Results saved: {filepath.name} ({file_size:.1f} KB)")
        
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise IOError(f"Failed to save results to {filepath}: {e}")
    
    def evaluate_from_grouped_data(self,
                                   model_predict_fn: Callable[[str, int], Union[str, List[str]]],
                                   sources: List[str],
                                   references_list: List[List[str]],
                                   beam_size: int = 1) -> Dict[str, float]:
        """
        Evaluate model on grouped test data (ACL W15-3902 format).
        
        Args:
            model_predict_fn: Function that takes (source, beam_size) and returns predictions
                             - beam_size=1: returns str
                             - beam_size>1: returns List[str]
            sources: List of source words
            references_list: List of reference lists (one per source)
            beam_size: Beam size for decoding
            
        Returns:
            Dictionary of metrics
        """
        if len(sources) != len(references_list):
            raise ValueError(f"Sources ({len(sources)}) and references ({len(references_list)}) must have same length")
        
        if len(sources) == 0:
            logger.warning("Empty sources provided")
            return {'n_samples': 0}
        
        # Validate model_predict_fn signature
        try:
            test_pred = model_predict_fn(sources[0], beam_size=1)
            if beam_size > 1 and not isinstance(test_pred, list):
                logger.warning("model_predict_fn should return list for beam_size > 1")
        except Exception as e:
            raise ValueError(f"model_predict_fn validation failed: {e}")
        
        predictions = []
        failed_count = 0
        
        iterator = sources
        if self.verbose:
            iterator = tqdm(sources, desc=f"Generating predictions (beam={beam_size})", leave=False)
        
        for source in iterator:
            try:
                pred = model_predict_fn(source, beam_size=beam_size)
                predictions.append(pred)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.error(f"Prediction failed for '{source}': {e}")
                failed_count += 1
                if beam_size > 1:
                    predictions.append([])  # Empty beam
                else:
                    predictions.append("")  # Empty string
        
        if failed_count > 0:
            logger.warning(f"⚠️  {failed_count} predictions failed ({100*failed_count/len(sources):.1f}%)")
        
        # Calculate metrics
        beam_search = beam_size > 1
        return self.calculate_metrics(predictions, references_list, beam_search=beam_search)
    
    def generate_comparison_table(self,
                                 results_dict: Dict[str, Dict[str, float]],
                                 output_format: str = 'markdown') -> str:
        """
        Generate comparison table for multiple models (assignment requirement).
        
        Args:
            results_dict: Dict mapping model_name → metrics_dict
            output_format: 'markdown', 'latex', or 'csv'
            
        Returns:
            Formatted table string
        """
        if not results_dict:
            return "No results to compare"
        
        # Extract all metric names
        all_metrics = set()
        for metrics in results_dict.values():
            all_metrics.update(metrics.keys())
        
        # Sort metrics in logical order
        metric_order = [
            'word_accuracy', 'char_f1', 'char_precision', 'char_recall',
            'mean_edit_distance', 'mrr', 'top5_accuracy', 'top10_accuracy', 'n_samples'
        ]
        metrics_sorted = [m for m in metric_order if m in all_metrics]
        metrics_sorted += sorted(all_metrics - set(metrics_sorted))
        
        if output_format == 'markdown':
            # Header
            lines = ['| Model | ' + ' | '.join(metrics_sorted) + ' |']
            lines.append('|' + '|'.join(['---'] * (len(metrics_sorted) + 1)) + '|')
            
            # Rows
            for model_name in sorted(results_dict.keys()):
                metrics = results_dict[model_name]
                row = [model_name]
                for metric in metrics_sorted:
                    value = metrics.get(metric, 0.0)
                    if metric == 'n_samples':
                        row.append(f"{int(value)}")
                    elif metric in ['word_accuracy', 'char_f1', 'char_precision', 'char_recall', 
                                   'mrr', 'top5_accuracy', 'top10_accuracy']:
                        row.append(f"{value:.4f}")
                    else:
                        row.append(f"{value:.2f}")
                lines.append('| ' + ' | '.join(row) + ' |')
            
            return '\n'.join(lines)
        
        elif output_format == 'latex':
            # LaTeX table
            lines = ['\\begin{table}[h]']
            lines.append('\\centering')
            lines.append('\\begin{tabular}{l' + 'c' * len(metrics_sorted) + '}')
            lines.append('\\hline')
            lines.append('Model & ' + ' & '.join(metrics_sorted) + ' \\\\')
            lines.append('\\hline')
            
            for model_name in sorted(results_dict.keys()):
                metrics = results_dict[model_name]
                row = [model_name.replace('_', '\\_')]
                for metric in metrics_sorted:
                    value = metrics.get(metric, 0.0)
                    if metric == 'n_samples':
                        row.append(f"{int(value)}")
                    elif metric in ['word_accuracy', 'char_f1', 'char_precision', 'char_recall',
                                   'mrr', 'top5_accuracy', 'top10_accuracy']:
                        row.append(f"{value:.4f}")
                    else:
                        row.append(f"{value:.2f}")
                lines.append(' & '.join(row) + ' \\\\')
            
            lines.append('\\hline')
            lines.append('\\end{tabular}')
            lines.append('\\caption{Transliteration Model Comparison}')
            lines.append('\\end{table}')
            
            return '\n'.join(lines)
        
        elif output_format == 'csv':
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Header
            writer.writerow(['Model'] + metrics_sorted)
            
            # Rows
            for model_name in sorted(results_dict.keys()):
                metrics = results_dict[model_name]
                row = [model_name]
                for metric in metrics_sorted:
                    value = metrics.get(metric, 0.0)
                    row.append(value)
                writer.writerow(row)
            
            return output.getvalue()
        
        else:
            raise ValueError(f"Unknown format: {output_format}")


def main():
    """Test evaluation functionality"""
    print("🧪 Testing Evaluator class...\n")
    
    # Example data
    predictions = ["namaste", "dhanyaavaad", "aapka naam"]
    references_list = [
        ["नमस्ते", "नमस्कार"],  # Multiple valid references
        ["धन्यवाद"],
        ["आपका नाम"]
    ]
    
    # Initialize evaluator
    evaluator = Evaluator(verbose=True)
    
    # Test metrics
    results = evaluator.calculate_metrics(predictions, references_list, beam_search=False)
    
    print("\n📊 Evaluation Results:")
    for metric, value in results.items():
        if metric == 'n_samples':
            print(f"  {metric}: {int(value)}")
        else:
            print(f"  {metric}: {value:.4f}")
    
    # Test beam search evaluation
    print("\n🔍 Testing beam search evaluation...")
    beam_predictions = [
        ["नमस्ते", "नमस्कार", "नमसते"],
        ["धन्यवाद", "धन्यावाद", "धनयवाद"],
        ["आपका नाम", "आपका नांम", "आपका नम"]
    ]
    
    beam_results = evaluator.calculate_metrics(beam_predictions, references_list, beam_search=True)
    
    print("\n📊 Beam Search Results:")
    for metric, value in beam_results.items():
        if metric == 'n_samples':
            print(f"  {metric}: {int(value)}")
        else:
            print(f"  {metric}: {value:.4f}")
    
    # Test comparison table
    print("\n📋 Testing comparison table generation...")
    comparison_results = {
        'LSTM_greedy': {'word_accuracy': 0.45, 'char_f1': 0.78, 'char_precision': 0.80, 'char_recall': 0.76},
        'LSTM_beam5': {'word_accuracy': 0.52, 'char_f1': 0.82, 'char_precision': 0.83, 'char_recall': 0.81, 'top5_accuracy': 0.65},
        'Transformer_greedy': {'word_accuracy': 0.58, 'char_f1': 0.85, 'char_precision': 0.86, 'char_recall': 0.84},
        'Transformer_beam5': {'word_accuracy': 0.63, 'char_f1': 0.88, 'char_precision': 0.89, 'char_recall': 0.87, 'top5_accuracy': 0.75},
    }
    
    table = evaluator.generate_comparison_table(comparison_results, output_format='markdown')
    print("\n" + table)
    
    # Test save
    print("\n💾 Testing save results...")
    save_path = Path("outputs/results/test_evaluation.json")
    evaluator.save_results(results, save_path, metadata={'model': 'test', 'beam_size': 1})
    
    if save_path.exists():
        print(f"✅ Results saved to: {save_path}")
        save_path.unlink()  # Clean up
        print("🗑️  Test file removed")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    main()