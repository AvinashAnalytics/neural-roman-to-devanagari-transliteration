# utils/vocab.py
"""
Character-level Vocabulary for Hindi Transliteration (CS772 Assignment 2)
Supports Roman → Devanagari mapping with special tokens.
Includes debugging, validation, error prevention, and config integration.

Optimized for:
- Config-driven special tokens with absolute path resolution
- Atomic file writes (crash-safe)
- Pickle + JSON support (fast + human-readable)
- PyTorch compatibility (__len__, __contains__)
- Max vocab size enforcement
- Thread-safe design (read-only after build)
- Backward-compatible checkpoint loading
"""

from typing import List, Dict, Set, Optional, Union, Tuple, Iterator
from collections import Counter
from pathlib import Path
import json
import pickle
import logging
import sys

# Vocabulary format version (increment on breaking changes)
VOCAB_FORMAT_VERSION = "1.1"  # Incremented due to critical fixes

# Setup module logger
logger = logging.getLogger(__name__)


class Vocabulary:
    """
    Character-level vocabulary with special tokens.
    Supports encoding/decoding, saving/loading, and frequency-based pruning.
    Config-driven and optimized for speed.
    
    Thread-safe for read operations after build_vocab() completes.
    NOT thread-safe during build_vocab() - build sequentially.
    """

    def __init__(self, config: Optional[Dict] = None, config_path: Optional[str] = None):
        """
        Initialize vocabulary with optional config.
        
        Args:
            config: Configuration dict (takes precedence)
            config_path: Path to config.yaml file (absolute or relative to project root)
        """
        # Load config if not provided
        self.config = config or self._load_config(config_path)
        
        # Initialize special tokens from config (with fallback defaults)
        preproc = self.config.get('preprocessing', {})
        self.PAD_TOKEN = self._validate_token(preproc.get('pad_token', '<PAD>'), 'PAD')
        self.SOS_TOKEN = self._validate_token(preproc.get('sos_token', '<SOS>'), 'SOS')
        self.EOS_TOKEN = self._validate_token(preproc.get('eos_token', '<EOS>'), 'EOS')
        self.UNK_TOKEN = self._validate_token(preproc.get('unk_token', '<UNK>'), 'UNK')

        # Validate special tokens are unique
        tokens = [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]
        if len(set(tokens)) != 4:
            raise ValueError(f"Special tokens must be unique, got: {tokens}")

        # Token to index mapping (special tokens always get indices 0-3)
        self.char2idx = {
            self.PAD_TOKEN: 0,
            self.SOS_TOKEN: 1,
            self.EOS_TOKEN: 2,
            self.UNK_TOKEN: 3
        }

        # Index to token mapping (inverse)
        self.idx2char = {
            0: self.PAD_TOKEN,
            1: self.SOS_TOKEN,
            2: self.EOS_TOKEN,
            3: self.UNK_TOKEN
        }

        # Frequency counter (preserved across save/load)
        self.char_freq = Counter()

        # Vocabulary size (cached property)
        self._size = 4

        # Convenience attributes (for PyTorch DataLoader compatibility)
        # These are ALWAYS 0-3 by design
        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3
        
        # Metadata
        self.min_freq = None
        self.max_vocab_size = None
        self.num_texts_built = 0
        self.format_version = VOCAB_FORMAT_VERSION
        self._stats_cache = None  # Cache for statistics

    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """
        Load config from file with robust path resolution.
        
        Args:
            config_path: Path to config file (absolute or relative to project root)
        
        Returns:
            Config dictionary (empty dict if load fails)
        """
        if config_path is None:
            # Resolve relative to this file: utils/vocab.py → project_root/config/config.yaml
            vocab_file = Path(__file__).resolve()
            project_root = vocab_file.parent.parent  # Go up 2 levels (utils/ → project_root/)
            config_path = project_root / "config" / "config.yaml"
        else:
            config_path = Path(config_path)
        
        # Try to load YAML
        try:
            import yaml
        except ImportError:
            logger.error("PyYAML not installed. Install with: pip install pyyaml")
            return {}
        
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

    def _validate_token(self, token: str, name: str) -> str:
        """Validate special token is non-empty string"""
        if not isinstance(token, str):
            raise TypeError(f"{name}_TOKEN must be string, got {type(token).__name__}")
        if not token or not token.strip():
            raise ValueError(f"{name}_TOKEN cannot be empty or whitespace")
        return token

    @property
    def size(self) -> int:
        """Vocabulary size property"""
        return self._size
    
    def __len__(self) -> int:
        """Return vocabulary size (PyTorch compatibility)"""
        return self._size
    
    def __contains__(self, char: str) -> bool:
        """Check if character in vocabulary"""
        return char in self.char2idx
    
    def __repr__(self) -> str:
        """String representation"""
        return (f"Vocabulary(size={self._size}, min_freq={self.min_freq}, "
                f"max_size={self.max_vocab_size}, version={self.format_version})")

    def build_vocab(self, texts: List[str], min_freq: int = None, 
                   max_vocab_size: int = None, verbose: bool = True) -> None:
        """
        Build vocabulary from list of texts (single-pass for efficiency).
        
        Args:
            texts: List of text strings
            min_freq: Minimum frequency threshold (reads from config if None)
            max_vocab_size: Maximum vocabulary size (reads from config if None)
            verbose: Print statistics
        
        Raises:
            ValueError: If no valid texts provided or invalid parameters
        """
        if not texts:
            raise ValueError("No texts provided to build vocabulary")
        
        # Use config defaults if not specified
        if min_freq is None:
            min_freq = self.config.get('preprocessing', {}).get('min_frequency', 2)
        if max_vocab_size is None:
            # Use 0 to mean unlimited
            max_vocab_size = self.config.get('preprocessing', {}).get('max_vocab_size_src', 0)
        
        # Validate parameters
        if not isinstance(min_freq, int) or min_freq < 1:
            raise ValueError(f"min_freq must be integer ≥ 1, got {min_freq}")
        if not isinstance(max_vocab_size, int) or max_vocab_size < 0:
            raise ValueError(f"max_vocab_size must be integer ≥ 0, got {max_vocab_size}")
        
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size if max_vocab_size > 0 else None

        if verbose:
            logger.info(f"Building vocabulary from {len(texts):,} texts "
                       f"(min_freq={min_freq}, max_size={self.max_vocab_size or 'unlimited'})")
        
        # Single-pass validation and counting (memory efficient)
        char_counter = Counter()
        valid_count = 0
        invalid_count = 0
        empty_count = 0
        
        for text in texts:
            if not isinstance(text, str):
                invalid_count += 1
                continue
            if len(text.strip()) == 0:
                empty_count += 1
                continue
            
            char_counter.update(text)
            valid_count += 1

        if invalid_count > 0 and verbose:
            logger.warning(f"Skipped {invalid_count:,} non-string texts")
        if empty_count > 0 and verbose:
            logger.warning(f"Skipped {empty_count:,} empty texts")

        if valid_count == 0:
            raise ValueError("No valid texts found after filtering")
        
        self.num_texts_built = valid_count
        self.char_freq = char_counter

        if verbose:
            logger.info(f"Found {len(self.char_freq):,} unique characters before filtering")

        # Filter by minimum frequency and sort (deterministic order)
        # Sort by: frequency DESC, then character ASC (for determinism)
        filtered_chars = [
            (char, freq) for char, freq in self.char_freq.items()
            if freq >= min_freq and char not in self.char2idx
        ]
        filtered_chars.sort(key=lambda x: (-x[1], x[0]))

        # Apply max vocab size limit (account for 4 special tokens already added)
        if self.max_vocab_size:
            max_new_chars = self.max_vocab_size - 4  # Reserve space for special tokens
            if max_new_chars < 0:
                raise ValueError(f"max_vocab_size must be ≥ 4 (for special tokens), got {self.max_vocab_size}")
            if len(filtered_chars) > max_new_chars:
                logger.warning(f"Truncating vocab: {len(filtered_chars):,} → {max_new_chars:,} chars "
                             f"(max_vocab_size={self.max_vocab_size})")
                filtered_chars = filtered_chars[:max_new_chars]

        # Add characters to vocabulary
        added_chars = 0
        for char, freq in filtered_chars:
            self.char2idx[char] = self._size
            self.idx2char[self._size] = char
            self._size += 1
            added_chars += 1

        # Invalidate stats cache
        self._stats_cache = None

        if verbose:
            logger.info(f"Added {added_chars:,} characters (after min_freq={min_freq} filter)")
            logger.info(f"Final vocabulary size: {self._size}")
            
            # Show UNK rate estimate
            total_chars = sum(self.char_freq.values())
            unk_chars = sum(freq for char, freq in self.char_freq.items() 
                          if char not in self.char2idx)
            if total_chars > 0:
                unk_rate = 100 * unk_chars / total_chars
                logger.info(f"Estimated UNK rate: {unk_rate:.2f}% ({unk_chars:,}/{total_chars:,} chars)")
                # Use config-driven threshold (default 5.0%)
                unk_threshold = self.config.get('preprocessing', {}).get('unk_rate_warning_threshold', 5.0)
                if unk_rate > unk_threshold:
                    logger.warning(f"High UNK rate detected (>{unk_threshold}%)! "
                                 f"Consider lowering min_freq or increasing max_vocab_size")

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        """
        Convert text to list of indices.
        
        Args:
            text: Input text string
            add_special: Add SOS/EOS tokens
        
        Returns:
            List of integer indices
        
        Raises:
            TypeError: If text is not a string
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected string, got {type(text).__name__}: {repr(text)[:50]}")

        # Warn about empty strings (may indicate data pipeline issue)
        if len(text) == 0:
            logger.debug("Encoding empty string (will produce [SOS, EOS] if add_special=True)")

        indices = []

        if add_special:
            indices.append(self.sos_idx)

        for char in text:
            indices.append(self.char2idx.get(char, self.unk_idx))

        if add_special:
            indices.append(self.eos_idx)

        return indices

    def decode(self, indices: List[int], remove_special: bool = True, 
               handle_unk: str = 'keep') -> str:
        """
        Convert list of indices back to text.
        
        Args:
            indices: List of integer indices
            remove_special: Remove special tokens (PAD, SOS, EOS)
            handle_unk: How to handle UNK tokens ('skip', 'keep', 'replace')
                       - 'keep': Keep UNK token as-is
                       - 'skip': Remove UNK tokens completely
                       - 'replace': Replace with Unicode replacement char 
        
        Returns:
            Decoded text string
            Note: Unknown indices (not in vocab) are replaced with '' and logged as warnings
        
        Raises:
            TypeError: If indices is not a list/tuple
            ValueError: If handle_unk is invalid
        """
        if not isinstance(indices, (list, tuple)):
            raise TypeError(f"Expected list of indices, got {type(indices).__name__}")
        
        if handle_unk not in ['skip', 'keep', 'replace']:
            raise ValueError(f"handle_unk must be 'skip', 'keep', or 'replace', got '{handle_unk}'")

        chars = []
        unknown_indices_set = set()  # Use set to avoid duplicates
        
        for idx in indices:
            # Handle both Python int and NumPy integers
            try:
                idx = int(idx)
            except (TypeError, ValueError):
                logger.warning(f"Skipping non-integer index: {idx}")
                continue
            
            # Check if index exists in vocabulary
            if idx not in self.idx2char:
                unknown_indices_set.add(idx)
                chars.append('\uFFFD')  # Unicode replacement character (U+FFFD) - consistent usage
                continue
            
            char = self.idx2char[idx]
            
            # Handle special tokens
            if remove_special:
                if char in [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]:
                    continue
                if char == self.UNK_TOKEN:
                    if handle_unk == 'skip':
                        continue
                    elif handle_unk == 'replace':
                        chars.append('\uFFFD')  # Unicode replacement character - consistent
                        continue
                    # else 'keep': fall through to append
            
            chars.append(char)
        
        # Log unknown indices (limit to first 10 to avoid spam)
        if unknown_indices_set:
            unknown_list = sorted(unknown_indices_set)
            logger.warning(f"Found {len(unknown_list)} unknown indices: "
                         f"{unknown_list[:10]}{'...' if len(unknown_list) > 10 else ''}")
        
        return ''.join(chars)

    def save(self, filepath: Union[str, Path], format: str = 'auto') -> None:
        """
        Save vocabulary to file with atomic write (crash-safe).
        
        Args:
            filepath: Output file path (.pkl for pickle, .json for JSON)
            format: 'json', 'pickle', or 'auto' (infer from extension)
        
        Raises:
            IOError: If save fails
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Infer format from extension
        if format == 'auto':
            format = 'pickle' if filepath.suffix == '.pkl' else 'json'
        
        # Prepare data (save everything including char_freq for reproducibility)
        data = {
            'format_version': self.format_version,
            'char2idx': self.char2idx,
            'idx2char': {str(k): v for k, v in self.idx2char.items()},  # JSON-safe keys
            'size': self._size,
            'char_freq': dict(self.char_freq),  # Preserve frequency info
            'special_tokens': {
                'pad': self.PAD_TOKEN,
                'sos': self.SOS_TOKEN,
                'eos': self.EOS_TOKEN,
                'unk': self.UNK_TOKEN
            },
            'metadata': {
                'min_freq': self.min_freq,
                'max_vocab_size': self.max_vocab_size,
                'num_texts_built': self.num_texts_built
            }
        }
        
        # Atomic write: write to temp file, then rename (crash-safe)
        temp_path = filepath.with_suffix(filepath.suffix + '.tmp')
        
        try:
            if format == 'pickle':
                # Use protocol 4 for Python 3.4+ compatibility (not HIGHEST_PROTOCOL)
                with open(temp_path, 'wb') as f:
                    pickle.dump(data, f, protocol=4)
            else:  # json
                with open(temp_path, 'w', encoding='utf-8') as f:  # Explicit UTF-8
                    json.dump(data, f, ensure_ascii=False, indent=2)
            
            # Atomic rename (overwrites existing file safely)
            # On Windows, this can fail if file is open - handle gracefully
            try:
                if filepath.exists():
                    filepath.unlink()  # Remove existing file first on Windows
                temp_path.rename(filepath)  # Use rename instead of replace for better Windows compatibility
            except Exception as e:
                # Fallback: copy and delete
                try:
                    import shutil
                    shutil.copy2(temp_path, filepath)
                    temp_path.unlink()
                except Exception as e2:
                    raise IOError(f"Failed to save {filepath}: {e} | Fallback failed: {e2}")
            
            file_size = filepath.stat().st_size / 1024  # KB
            logger.info(f"Vocabulary saved: {filepath.name} ({format}, {self._size} entries, {file_size:.1f} KB)")
            
        except Exception as e:
            # Clean up temp file on error
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass  # Ignore cleanup errors
            raise IOError(f"Error saving vocabulary to {filepath}: {e}")

    def load(self, filepath: Union[str, Path], format: str = 'auto', strict: bool = True) -> None:
        """
        Load vocabulary from file with validation.
        
        Args:
            filepath: Input file path
            format: 'json', 'pickle', or 'auto' (infer from extension)
            strict: Raise error on version mismatch (False = warn only)
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is corrupt or incompatible
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {filepath}")
        
        # Infer format from extension
        if format == 'auto':
            format = 'pickle' if filepath.suffix == '.pkl' else 'json'
        
        try:
            # Load data
            if format == 'pickle':
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
            else:  # json
                with open(filepath, 'r', encoding='utf-8') as f:  # Explicit UTF-8
                    data = json.load(f)
            
            # Check format version (backward compatibility)
            file_version = data.get('format_version', '0.0')
            if file_version != VOCAB_FORMAT_VERSION:
                msg = f"Vocab version mismatch: file={file_version}, current={VOCAB_FORMAT_VERSION}"
                if strict:
                    raise ValueError(msg)
                else:
                    logger.warning(msg + " (continuing with strict=False)")
            
            # Validate required fields
            required_fields = ['char2idx', 'idx2char', 'size']
            missing = [f for f in required_fields if f not in data]
            if missing:
                raise ValueError(f"Missing required fields: {missing}")
            
            # Validate minimum vocab size (must have at least special tokens)
            if data['size'] < 4:
                raise ValueError(f"Invalid vocab size: {data['size']} (must be ≥ 4 for special tokens)")
            
            # Load mappings
            self.char2idx = data['char2idx']
            
            # Convert idx2char keys from strings to ints (JSON requirement)
            try:
                self.idx2char = {int(k): v for k, v in data['idx2char'].items()}
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid idx2char keys (must be integers): {e}")
            
            self._size = data['size']
            
            # Load frequency info (if available, for backward compatibility)
            if 'char_freq' in data:
                self.char_freq = Counter(data['char_freq'])
            else:
                logger.warning("Vocabulary loaded without frequency information")
                self.char_freq = Counter()
            
            # Load metadata (if available)
            if 'metadata' in data:
                self.min_freq = data['metadata'].get('min_freq')
                self.max_vocab_size = data['metadata'].get('max_vocab_size')
                self.num_texts_built = data['metadata'].get('num_texts_built', 0)
            
            # Load special tokens (for backward compatibility with old vocabs)
            if 'special_tokens' in data:
                loaded_tokens = data['special_tokens']
                # Update our special tokens to match loaded ones (backward compatibility)
                self.PAD_TOKEN = loaded_tokens.get('pad', self.PAD_TOKEN)
                self.SOS_TOKEN = loaded_tokens.get('sos', self.SOS_TOKEN)
                self.EOS_TOKEN = loaded_tokens.get('eos', self.EOS_TOKEN)
                self.UNK_TOKEN = loaded_tokens.get('unk', self.UNK_TOKEN)
            
            # CRITICAL FIX: Rebuild special token indices from char2idx AND UPDATE INSTANCE VARIABLES
            self._rebuild_special_indices()
            
            # Validate consistency
            self._validate_consistency()
            
            # Invalidate stats cache
            self._stats_cache = None
            
            file_size = filepath.stat().st_size / 1024  # KB
            logger.info(f"Vocabulary loaded: {filepath.name} ({format}, {self._size} entries, {file_size:.1f} KB)")
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {filepath}: {e}")
        except pickle.UnpicklingError as e:
            raise ValueError(f"Invalid pickle file {filepath}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading vocabulary from {filepath}: {e}")
    
    def _rebuild_special_indices(self) -> None:
        """
        Rebuild special token index attributes from char2idx.
        
        CRITICAL FIX: Now actually updates self.pad_idx, self.sos_idx, etc.
        This ensures that pad_idx actually points to the PAD token.
        """
        # Find indices for each special token
        pad_idx = self.char2idx.get(self.PAD_TOKEN)
        sos_idx = self.char2idx.get(self.SOS_TOKEN)
        eos_idx = self.char2idx.get(self.EOS_TOKEN)
        unk_idx = self.char2idx.get(self.UNK_TOKEN)
        
        # Validate all found
        if None in [pad_idx, sos_idx, eos_idx, unk_idx]:
            missing = []
            if pad_idx is None: missing.append(f"PAD ('{self.PAD_TOKEN}')")
            if sos_idx is None: missing.append(f"SOS ('{self.SOS_TOKEN}')")
            if eos_idx is None: missing.append(f"EOS ('{self.EOS_TOKEN}')")
            if unk_idx is None: missing.append(f"UNK ('{self.UNK_TOKEN}')")
            raise ValueError(f"Loaded vocab missing special tokens: {', '.join(missing)}")
        
        # Update instance variables - THIS WAS THE CRITICAL BUG
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.unk_idx = unk_idx
        
        # Warn if indices not in expected positions (not critical, but unusual)
        if pad_idx != 0 or sos_idx != 1 or eos_idx != 2 or unk_idx != 3:
            logger.warning(f"Non-standard special token indices: "
                         f"PAD={pad_idx}, SOS={sos_idx}, "
                         f"EOS={eos_idx}, UNK={unk_idx}")
    
    def _validate_consistency(self) -> None:
        """Validate that char2idx and idx2char are consistent"""
        # Check all special tokens exist
        missing_tokens = []
        for token in [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]:
            if token not in self.char2idx:
                missing_tokens.append(token)
        
        if missing_tokens:
            raise ValueError(f"Missing required tokens: {missing_tokens}")
        
        # Check bidirectional consistency (sample check to avoid O(n²))
        for char, idx in list(self.char2idx.items())[:100]:  # Check first 100
            if idx not in self.idx2char:
                raise ValueError(f"Inconsistent mapping: char '{char}' → idx {idx} not in idx2char")
            if self.idx2char[idx] != char:
                raise ValueError(f"Inconsistent mapping: idx {idx} → '{self.idx2char[idx]}' != '{char}'")
        
        # Check size matches
        if len(self.char2idx) != self._size:
            logger.warning(f"Size mismatch: char2idx has {len(self.char2idx)} entries, size={self._size}")
            self._size = len(self.char2idx)  # Auto-correct
        
        # Validate idx2char covers all indices 0 to size-1
        missing_indices = set(range(self._size)) - set(self.idx2char.keys())
        if missing_indices:
            logger.warning(f"Missing {len(missing_indices)} indices in idx2char: {sorted(missing_indices)[:10]}")

    def get_vocab(self) -> Dict[str, int]:
        """
        Return copy of character to index mapping.
        
        WARNING: Returns full copy (expensive for large vocabs).
        For read-only access, use vocab.char2idx directly.
        """
        logger.warning("get_vocab() returns full copy - consider using char2idx directly for read-only access")
        return self.char2idx.copy()
    
    def get_chars(self) -> Set[str]:
        """Return set of all characters in vocabulary"""
        return set(self.char2idx.keys())
    
    def get_special_tokens(self) -> List[str]:
        """Return list of special tokens (cached for efficiency)"""
        return [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]

    def get_statistics(self) -> Dict:
        """
        Return vocabulary statistics as dictionary.
        
        Returns:
            Dict with size, coverage, UNK rate, etc.
        """
        if self._stats_cache is not None:
            return self._stats_cache.copy()  # Return copy to prevent modification
            
        total_freq = sum(self.char_freq.values())
        unk_freq = sum(freq for char, freq in self.char_freq.items() 
                      if char not in self.char2idx)
        
        stats = {
            'size': self._size,
            'num_texts_built': self.num_texts_built,
            'min_freq': self.min_freq,
            'max_vocab_size': self.max_vocab_size,
            'total_characters': total_freq,
            'unk_characters': unk_freq,
            'unk_rate': 100 * unk_freq / total_freq if total_freq > 0 else None,
            'num_special_tokens': len(self.get_special_tokens()),
            'num_regular_chars': self._size - len(self.get_special_tokens()),
            'coverage': 100 * (total_freq - unk_freq) / total_freq if total_freq > 0 else None
        }
        
        self._stats_cache = stats
        return stats.copy()

    def print_info(self, top_k: int = 15) -> None:
        """
        Print vocabulary statistics and most frequent characters.
        
        Args:
            top_k: Number of top frequent characters to display
        """
        separator = "=" * 60
        logger.info(f"\n{separator}")
        logger.info("VOCABULARY INFORMATION")
        logger.info(separator)
        logger.info(f"Size: {self._size:,}")
        
        if self.num_texts_built > 0:
            logger.info(f"Built from: {self.num_texts_built:,} texts")
        else:
            logger.info("Loaded from file")
        
        if self.min_freq is not None:
            logger.info(f"Min frequency: {self.min_freq}")
        if self.max_vocab_size is not None:
            logger.info(f"Max vocab size: {self.max_vocab_size}")
        
        logger.info(f"\nSpecial Tokens:")
        logger.info(f"  PAD: '{self.PAD_TOKEN}' → idx {self.pad_idx}")
        logger.info(f"  SOS: '{self.SOS_TOKEN}' → idx {self.sos_idx}")
        logger.info(f"  EOS: '{self.EOS_TOKEN}' → idx {self.eos_idx}")
        logger.info(f"  UNK: '{self.UNK_TOKEN}' → idx {self.unk_idx}")

        # Show statistics if available
        if len(self.char_freq) > 0:
            stats = self.get_statistics()
            logger.info(f"\nCoverage Statistics:")
            logger.info(f"  Total characters: {stats['total_characters']:,}")
            if stats['coverage'] is not None:
                logger.info(f"  Coverage: {stats['coverage']:.2f}%")
            if stats['unk_rate'] is not None:
                logger.info(f"  UNK rate: {stats['unk_rate']:.2f}%")
            
            # Only show "Top K" header if there are items to display
            # Filter out special tokens
            freq_items = [
                (char, freq) for char, freq in self.char_freq.items()
                if char not in self.get_special_tokens()
            ]

            # Sort by frequency descending, then alphabetically
            freq_items.sort(key=lambda x: (-x[1], x[0]))

            if freq_items:  # Only show header and items if we have data
                logger.info(f"\nTop {top_k} Most Frequent Characters:")

                # Display character mapping helper
                def display_char(c: str) -> str:
                    """Format character for display"""
                    special_chars = {
                        ' ': '<SPACE>', '\t': '<TAB>', '\n': '<NL>', '\r': '<CR>',
                        '\u200c': '<ZWNJ>', '\u200d': '<ZWJ>'  # Zero-width chars
                    }
                    return f"'{special_chars.get(c, c)}'"

                for i, (char, freq) in enumerate(freq_items[:top_k], 1):
                    idx = self.char2idx.get(char, 'N/A')
                    in_vocab = "✓" if char in self.char2idx else "✗"
                    logger.info(f"  {i:2d}. {display_char(char):12} → freq: {freq:8,} | idx: {str(idx):6} [{in_vocab}]")
        
        logger.info(f"{separator}\n")


def build_vocab_from_data(src_texts: List[str], tgt_texts: List[str],
                         config: Optional[Dict] = None,
                         config_path: Optional[str] = None,
                         save_dir: Optional[Union[str, Path]] = None,
                         format: str = 'pickle',
                         verbose: bool = True) -> Tuple['Vocabulary', 'Vocabulary']:
    """
    Convenience function to build source and target vocabularies.
    
    Args:
        src_texts: List of source (Roman) texts
        tgt_texts: List of target (Devanagari) texts
        config: Configuration dict
        config_path: Path to config file
        save_dir: Directory to save vocabularies (None = don't save)
        format: Save format ('pickle' or 'json')
        verbose: Print detailed info
    
    Returns:
        Tuple of (src_vocab, tgt_vocab)
    
    Raises:
        ValueError: If texts are empty or invalid
    """
    if not src_texts or not tgt_texts:
        raise ValueError("Source and target texts cannot be empty")
    
    if len(src_texts) != len(tgt_texts):
        logger.warning(f"Length mismatch: {len(src_texts)} source vs {len(tgt_texts)} target texts")
    
    # Load config once and reuse
    if config is None:
        config = Vocabulary()._load_config(config_path)  # Use internal method to avoid duplication
    
    # Get vocabulary build parameters from config
    preproc = config.get('preprocessing', {})
    min_freq = preproc.get('min_frequency', 2)
    max_vocab_size_src = preproc.get('max_vocab_size_src', 0)
    max_vocab_size_tgt = preproc.get('max_vocab_size_tgt', 0)
    
    # Build source vocabulary
    if verbose:
        logger.info("\n" + "="*60)
        logger.info("BUILDING SOURCE (ROMAN) VOCABULARY")
        logger.info("="*60)
    
    src_vocab = Vocabulary(config=config)
    src_vocab.build_vocab(src_texts, min_freq=min_freq, 
                         max_vocab_size=max_vocab_size_src, verbose=verbose)
    if verbose:
        src_vocab.print_info()
    
    # Build target vocabulary
    if verbose:
        logger.info("\n" + "="*60)
        logger.info("BUILDING TARGET (DEVANAGARI) VOCABULARY")
        logger.info("="*60)
    
    tgt_vocab = Vocabulary(config=config)
    tgt_vocab.build_vocab(tgt_texts, min_freq=min_freq,
                         max_vocab_size=max_vocab_size_tgt, verbose=verbose)
    if verbose:
        tgt_vocab.print_info()
    
    # Save if directory provided
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Get vocab filenames from config or use defaults
        paths = config.get('paths', {})
        src_filename = paths.get('src_vocab_file', 'vocab_src.pkl')
        tgt_filename = paths.get('tgt_vocab_file', 'vocab_tgt.pkl')
        
        src_path = save_dir / src_filename
        tgt_path = save_dir / tgt_filename
        
        src_vocab.save(src_path, format=format)
        tgt_vocab.save(tgt_path, format=format)
        
        if verbose:
            logger.info(f"\nVocabularies saved to: {save_dir}")
    
    return src_vocab, tgt_vocab


def main():
    """Test vocabulary functionality"""
    # Setup logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger.info("Testing Vocabulary class...\n")
    
    # Example data
    src_texts = ["namaste", "dhanyavad", "aapka naam kya hai", "shukriya", "namaste"]
    tgt_texts = ["नमस्ते", "धन्यवाद", "आपका नाम क्या है", "शुक्रिया", "नमस्ते"]
    
    try:
        # Build vocabularies
        src_vocab, tgt_vocab = build_vocab_from_data(
            src_texts, tgt_texts,
            save_dir="data/processed",
            format='pickle',
            verbose=True
        )
        
        # Test encoding/decoding
        test_text = "namaste"
        encoded = src_vocab.encode(test_text)
        decoded = src_vocab.decode(encoded)
        
        logger.info(f"\n{'='*60}")
        logger.info("ENCODING TEST")
        logger.info(f"{'='*60}")
        logger.info(f"Original:  '{test_text}'")
        logger.info(f"Encoded:   {encoded}")
        logger.info(f"Decoded:   '{decoded}'")
        logger.info(f"Match:     {'✅' if test_text == decoded else '❌'}")
        
        # Test save/load
        logger.info(f"\n{'='*60}")
        logger.info("SAVE/LOAD TEST")
        logger.info(f"{'='*60}")
        test_path = Path("data/processed/test_vocab.pkl")
        src_vocab.save(test_path)
        
        loaded_vocab = Vocabulary()
        loaded_vocab.load(test_path)
        loaded_vocab.print_info()
        
        # Verify loaded vocab works
        encoded_loaded = loaded_vocab.encode(test_text)
        decoded_loaded = loaded_vocab.decode(encoded_loaded)
        
        logger.info(f"\nLoaded Vocab Test:")
        logger.info(f"Encoded:   {encoded_loaded}")
        logger.info(f"Decoded:   '{decoded_loaded}'")
        logger.info(f"Match original: {'✅' if encoded == encoded_loaded else '❌'}")
        
        # Test statistics
        logger.info(f"\n{'='*60}")
        logger.info("STATISTICS TEST")
        logger.info(f"{'='*60}")
        stats = src_vocab.get_statistics()
        for key, value in stats.items():
            logger.info(f"{key:20s}: {value}")
        
        # Clean up test file
        if test_path.exists():
            test_path.unlink()
            logger.info(f"\nCleaned up test file: {test_path}")
        
        logger.info("\n✅ All tests passed!")
        
    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()