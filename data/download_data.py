#data/download_data.py
"""
Data downloader and preprocessor for Hindi Transliteration (CS772 Assignment 2)
Handles Aksharantar dataset: Roman (English) → Devanagari (Hindi)
Preserves original key names: 'english word' and 'native word'

Optimized for:
- Memory efficiency (streaming file I/O)
- Atomic file writes (crash-safe)
- Config-driven paths and parameters
- Reproducible processing
"""

import os
import json
import zipfile
import requests
import random
import unicodedata
import shutil
import tempfile
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from tqdm import tqdm
import yaml
import hashlib
import sys


class DataDownloader:
    """
    Download and preprocess Aksharantar transliteration data.
    Format: {"english word": "roman", "native word": "देवनागरी", ...}
    PRESERVES ORIGINAL KEY NAMES throughout pipeline.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize data downloader with configuration"""
        # Try to ensure stdout/stderr use UTF-8 to avoid UnicodeEncodeError on Windows consoles
        try:
            if sys.stdout.encoding is None or sys.stdout.encoding.lower() != 'utf-8':
                import io
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        except Exception:
            # If re-wrapping stdout fails, continue without raising — prints may still error on some consoles
            pass
        # Resolve config path relative to project root
        config_path = Path(config_path)
        if not config_path.exists():
            # Try relative to this file
            script_dir = Path(__file__).parent.parent
            config_path = script_dir / "config" / "config.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Extract config values using proper nested structure
        self.language = self.config['data']['language']
        self.base_url = self.config['data']['data_url']
        self.max_train_samples = self.config['data']['max_train_samples']
        self.file_encoding = self.config['data'].get('file_encoding', 'utf-8')
        
        # Get seed from global config
        self.seed = self.config['global']['seed']
        
        # Set random seeds for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Create directories from config paths
        self.raw_dir = Path(self.config['paths']['raw_data_dir'])
        self.processed_dir = Path(self.config['paths']['processed_data_dir'])
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Get raw file names from config (as downloaded from Aksharantar)
        paths = self.config.get('paths', {})
        self.raw_train_file = paths.get('raw_train_file', 'hin_train.json')
        self.raw_valid_file = paths.get('raw_valid_file', 'hin_valid.json')
        self.raw_test_file = paths.get('raw_test_file', 'hin_test.json')
        
        # Get processed file names from config (simplified names)
        self.train_file = paths.get('train_file', 'train.json')
        self.valid_file = paths.get('valid_file', 'valid.json')
        self.test_file = paths.get('test_file', 'test.json')
        
        # Network settings
        self.timeout = self.config.get('llm', {}).get('timeout', 30)
        
        # Data statistics (store summaries, not full lists)
        self.stats = defaultdict(dict)
        
        # Verbose mode
        self.verbose = self.config['data'].get('verbose', False)
    
    def download_file(self, url: str, filepath: Path) -> None:
        """Download file with progress bar, retry logic, and integrity check"""
        max_retries = self.config.get('llm', {}).get('max_retries', 3)
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, stream=True, timeout=self.timeout)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                
                # Atomic write: download to temp file first
                temp_path = filepath.with_suffix('.tmp')
                
                with open(temp_path, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, 
                             desc=f"Downloading {filepath.name}") as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                
                # Verify file size matches
                downloaded_size = temp_path.stat().st_size
                if total_size > 0 and downloaded_size != total_size:
                    raise IOError(f"Size mismatch: expected {total_size}, got {downloaded_size}")
                
                # Atomic rename (crash-safe)
                temp_path.replace(filepath)
                print(f"✅ Downloaded: {filepath.name} ({downloaded_size / 1024 / 1024:.1f} MB)")
                return
                
            except requests.exceptions.RequestException as e:
                print(f"⚠️  Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                # Exponential backoff
                import time
                time.sleep(2 ** attempt)
    
    def download_hindi_data(self) -> None:
        """Download Hindi transliteration data from Aksharantar"""
        print(f"\n{'='*60}")
        print(f"🌐 DOWNLOADING {self.language.upper()} TRANSLITERATION DATA")
        print(f"{'='*60}\n")
        
        # Construct paths
        zip_url = f"{self.base_url.rstrip('/')}/{self.language}.zip"
        zip_path = self.raw_dir / f"{self.language}.zip"
        
        # Check if already downloaded and valid
        if zip_path.exists():
            try:
                # Quick validation: check if it's a valid zip
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    if zf.testzip() is None:
                        print(f"📁 Found valid zip: {zip_path}")
                        self._extract_zip(zip_path)
                        return
            except zipfile.BadZipFile:
                print(f"⚠️  Existing zip corrupted, re-downloading...")
                zip_path.unlink()
        
        # Download
        print(f"📥 Downloading from: {zip_url}")
        self.download_file(zip_url, zip_path)
        
        # Extract
        self._extract_zip(zip_path)
    
    def _extract_zip(self, zip_path: Path) -> None:
        """Extract zip file with validation"""
        print("\n📦 Extracting files...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Validate zip integrity
                corrupt_file = zip_ref.testzip()
                if corrupt_file:
                    raise zipfile.BadZipFile(f"Corrupt file in zip: {corrupt_file}")
                
                # List contents
                file_list = zip_ref.namelist()
                print(f"📋 Found {len(file_list)} files in archive")
                # Safe extraction to avoid zip-slip vulnerabilities
                raw_resolved = self.raw_dir.resolve()
                for member in zip_ref.infolist():
                    member_name = member.filename
                    # Skip directory entries
                    if member_name.endswith('/'):
                        continue
                    target_path = (self.raw_dir / member_name).resolve()
                    if not str(target_path).startswith(str(raw_resolved)):
                        raise RuntimeError(f"Unsafe path in zip file: {member_name}")
                    # Ensure parent directory exists
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    # Extract member safely
                    with zip_ref.open(member) as source, open(target_path, 'wb') as target:
                        shutil.copyfileobj(source, target)

                # Show extracted JSON files
                for file in file_list:
                    if file.endswith('.json'):
                        print(f"   ✓ {file}")
            
            print(f"✅ Extracted to: {self.raw_dir}\n")
            
        except zipfile.BadZipFile as e:
            print(f"❌ Error extracting zip: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text (NFC normalization for Devanagari)"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove zero-width characters
        text = text.replace('\u200b', '').replace('\u200c', '').replace('\u200d', '')
        text = text.replace('\ufeff', '')  # Remove BOM
        
        # Normalize Unicode (critical for Devanagari consistency)
        if self.config['preprocessing'].get('normalize_unicode', True):
            text = unicodedata.normalize('NFC', text)
        
        return text.strip()
    
    def validate_transliteration_pair(self, src: str, tgt: str) -> bool:
        """
        Validate that the pair is valid for Roman→Devanagari transliteration
        """
        if not src or not tgt:
            return False
        
        # Check max sequence length
        max_len = self.config['data']['max_seq_length']
        if len(src) > max_len or len(tgt) > max_len:
            return False
        
        # Check source is primarily ASCII (Roman script)
        allowed_special = set(" -.'")
        non_ascii = sum(1 for c in src if ord(c) > 127 and c not in allowed_special)
        if non_ascii > len(src) * 0.1:  # Allow up to 10% non-ASCII
            return False
        
        # Check target contains Devanagari characters
        devanagari_count = sum(1 for c in tgt if '\u0900' <= c <= '\u097F')
        if devanagari_count == 0:
            return False
        
        # Check reasonable length ratio (avoid corrupted pairs)
        length_ratio = len(tgt) / len(src) if len(src) > 0 else 0
        if length_ratio < 0.3 or length_ratio > 5:
            return False
        
        return True
    
    def process_data_item(self, item: Dict) -> Optional[Dict]:
        """Process and validate a single data item"""
        # Extract fields using original Aksharantar key names
        src_roman = item.get('english word', '').strip()
        tgt_devanagari = item.get('native word', '').strip()
        
        # Clean texts
        src_roman = self.clean_text(src_roman)
        tgt_devanagari = self.clean_text(tgt_devanagari)
        
        # Validate
        if not self.validate_transliteration_pair(src_roman, tgt_devanagari):
            return None
        
        # Provide a stable fallback unique_id when the source data lacks one
        uid = item.get('unique_identifier')
        if not uid:
            # Hash of source + separator + target (stable and deterministic)
            uid = hashlib.sha1(f"{src_roman}||{tgt_devanagari}".encode('utf-8')).hexdigest()

        return {
            'english word': src_roman,
            'native word': tgt_devanagari,
            'src_len': len(src_roman),
            'tgt_len': len(tgt_devanagari),
            'unique_id': uid,
            'source_dataset': item.get('source', 'unknown'),
            'score': item.get('score')
        }
    
    def load_json_data(self, filepath: Path) -> List[Dict]:
        """
        Load JSONL data from file (streaming for memory efficiency)
        """
        data = []
        skipped = 0
        
        if not filepath.exists():
            raise FileNotFoundError(f"❌ File not found: {filepath}")
        
        print(f"📖 Loading: {filepath.name}")

        # Detect whether file is JSON array (single list) or JSONL (one object per line)
        with open(filepath, 'r', encoding=self.file_encoding) as f:
            # Peek first non-whitespace character
            first_chunk = f.read(4096)
            first_char = ''
            for ch in first_chunk:
                if not ch.isspace():
                    first_char = ch
                    break
            f.seek(0)

            if first_char == '[':
                # JSON array
                try:
                    items = json.load(f)
                except json.JSONDecodeError as e:
                    raise
                for idx, item in enumerate(tqdm(items, desc='Reading array items')):
                    try:
                        processed_item = self.process_data_item(item)
                        if processed_item:
                            data.append(processed_item)
                        else:
                            skipped += 1
                    except Exception:
                        skipped += 1
            else:
                # JSONL streaming
                for line_num, line in enumerate(tqdm(f, desc="Reading lines"), 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        item = json.loads(line)
                        processed_item = self.process_data_item(item)

                        if processed_item:
                            data.append(processed_item)
                        else:
                            skipped += 1

                    except json.JSONDecodeError as e:
                        if self.verbose:
                            print(f"   ⚠️  Line {line_num}: Invalid JSON - {e}")
                        skipped += 1
                    except Exception as e:
                        if self.verbose:
                            print(f"   ⚠️  Line {line_num}: Error - {e}")
                        skipped += 1
        
        print(f"   ✅ Loaded: {len(data):,} valid pairs")
        if skipped > 0:
            print(f"   ⚠️  Skipped: {skipped:,} invalid pairs ({100*skipped/(len(data)+skipped):.1f}%)")
        
        return data
    
    def smart_subsample(self, data: List[Dict], max_samples: int) -> List[Dict]:
        """
        Smart subsampling to maintain diversity in:
        1. Length distribution
        2. Source datasets
        3. Character complexity
        
        O(n) with set-based operations
        """
        if len(data) <= max_samples:
            return data
        
        print(f"\n🎯 Smart Subsampling from {len(data):,} to {max_samples:,} examples")
        
        # Analyze data sources
        source_counts = Counter(item['source_dataset'] for item in data)
        print(f"📊 Source distribution:")
        for source, count in source_counts.most_common():
            print(f"   - {source}: {count:,} ({100*count/len(data):.1f}%)")
        
        # Strategy: Stratified sampling with set-based tracking
        sampled = []
        sampled_ids = set()  # O(1) lookup
        
        # Calculate target samples per source (proportional)
        for source, items_count in source_counts.items():
            source_items = [item for item in data if item['source_dataset'] == source]
            target_count = int(max_samples * items_count / len(data))
            target_count = max(1, target_count)  # At least 1 from each source
            
            # Group by length buckets (bins of 5 characters)
            length_buckets = defaultdict(list)
            for item in source_items:
                bucket = item['src_len'] // 5
                length_buckets[bucket].append(item)
            
            # Sample proportionally from each bucket
            samples_per_bucket = max(1, target_count // len(length_buckets))
            
            for bucket_items in length_buckets.values():
                n = min(len(bucket_items), samples_per_bucket)
                selected = random.sample(bucket_items, n)
                
                for item in selected:
                    if item['unique_id'] not in sampled_ids:
                        sampled.append(item)
                        sampled_ids.add(item['unique_id'])
        
        # If we haven't reached target, randomly sample remaining
        if len(sampled) < max_samples:
            remaining = max_samples - len(sampled)
            unsampled = [item for item in data if item['unique_id'] not in sampled_ids]
            
            if unsampled:
                additional = random.sample(unsampled, min(remaining, len(unsampled)))
                sampled.extend(additional)
                for item in additional:
                    sampled_ids.add(item['unique_id'])
        
        # If oversampled, trim
        if len(sampled) > max_samples:
            sampled = random.sample(sampled, max_samples)
        
        # Shuffle for good measure
        random.shuffle(sampled)
        
        print(f"✂️  Subsampled to {len(sampled):,} examples")
        
        # Show final distribution
        final_sources = Counter(item['source_dataset'] for item in sampled)
        print(f"📊 Final distribution:")
        for source, count in final_sources.most_common():
            print(f"   - {source}: {count:,} ({100*count/len(sampled):.1f}%)")
        
        return sampled
    
    def compute_statistics(self, data: List[Dict], split_name: str) -> Dict:
        """
        Compute detailed statistics for a dataset split
        Stores summaries, not full lists (memory efficient)
        """
        src_lengths = [item['src_len'] for item in data]
        tgt_lengths = [item['tgt_len'] for item in data]
        
        stats = {
            'count': len(data),
            'sources': dict(Counter(item['source_dataset'] for item in data)),
            'unique_src': len(set(item['english word'] for item in data)),
            'unique_tgt': len(set(item['native word'] for item in data)),
            
            # Source length stats (summaries only)
            'src_len_mean': float(np.mean(src_lengths)) if src_lengths else 0,
            'src_len_std': float(np.std(src_lengths)) if src_lengths else 0,
            'src_len_min': int(min(src_lengths)) if src_lengths else 0,
            'src_len_max': int(max(src_lengths)) if src_lengths else 0,
            'src_len_median': float(np.median(src_lengths)) if src_lengths else 0,
            
            # Target length stats (summaries only)
            'tgt_len_mean': float(np.mean(tgt_lengths)) if tgt_lengths else 0,
            'tgt_len_std': float(np.std(tgt_lengths)) if tgt_lengths else 0,
            'tgt_len_min': int(min(tgt_lengths)) if tgt_lengths else 0,
            'tgt_len_max': int(max(tgt_lengths)) if tgt_lengths else 0,
            'tgt_len_median': float(np.median(tgt_lengths)) if tgt_lengths else 0,
        }
        
        return stats
    
    def print_statistics(self, data: List[Dict], split_name: str) -> None:
        """Print detailed statistics about the dataset"""
        stats = self.compute_statistics(data, split_name)
        self.stats[split_name] = stats
        
        print(f"\n{'='*60}")
        print(f"📊 {split_name.upper()} SET STATISTICS")
        print(f"{'='*60}")
        print(f"Total examples: {stats['count']:,}")
        print(f"Unique source words: {stats['unique_src']:,} ({100*stats['unique_src']/stats['count']:.1f}% unique)")
        print(f"Unique target words: {stats['unique_tgt']:,} ({100*stats['unique_tgt']/stats['count']:.1f}% unique)")
        
        print(f"\n📏 Length Statistics:")
        print(f"Source (Roman):")
        print(f"  Mean:   {stats['src_len_mean']:.1f} ± {stats['src_len_std']:.1f}")
        print(f"  Median: {stats['src_len_median']:.1f}")
        print(f"  Range:  [{stats['src_len_min']}, {stats['src_len_max']}]")
        
        print(f"Target (Devanagari):")
        print(f"  Mean:   {stats['tgt_len_mean']:.1f} ± {stats['tgt_len_std']:.1f}")
        print(f"  Median: {stats['tgt_len_median']:.1f}")
        print(f"  Range:  [{stats['tgt_len_min']}, {stats['tgt_len_max']}]")
        
        print(f"\n📚 Data Sources:")
        for source, count in sorted(stats['sources'].items(), key=lambda x: x[1], reverse=True):
            percentage = 100 * count / stats['count']
            print(f"  {source:20s}: {count:6,} ({percentage:5.1f}%)")
        
        if len(data) > 0:
            print(f"\n🔤 Sample Pairs:")
            samples = random.sample(data, min(5, len(data)))
            for item in samples:
                print(f"  '{item['english word']}' → '{item['native word']}'")
    
    def save_processed_data(self, train: List[Dict], valid: List[Dict], test: List[Dict]) -> None:
        """
        Save processed data in JSONL format - PRESERVING ORIGINAL KEY NAMES
        Uses atomic writes (crash-safe)
        """
        datasets = {
            'train': train,
            'valid': valid,
            'test': test
        }
        
        print(f"\n{'='*60}")
        print(f"💾 SAVING PROCESSED DATA")
        print(f"{'='*60}")
        
        for split_name, data in datasets.items():
            filepath = self.processed_dir / f"{split_name}.json"
            temp_path = filepath.with_suffix('.tmp')
            
            # Write to temp file first (atomic write)
            with open(temp_path, 'w', encoding=self.file_encoding) as f:
                for item in data:
                    save_item = {
                        'english word': item['english word'],
                        'native word': item['native word'],
                        'source_dataset': item.get('source_dataset', 'unknown')
                    }
                    f.write(json.dumps(save_item, ensure_ascii=False) + '\n')
            
            # Atomic rename (crash-safe)
            temp_path.replace(filepath)
            
            file_size = filepath.stat().st_size / (1024 * 1024)
            print(f"✅ {split_name:8s}: {len(data):6,} examples → {filepath.name} ({file_size:.1f} MB)")
    
    def validate_processed_file(self, filepath: Path) -> bool:
        """Validate that a processed JSON file is well-formed"""
        try:
            with open(filepath, 'r', encoding=self.file_encoding) as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    # Check required keys
                    if 'english word' not in item or 'native word' not in item:
                        print(f"⚠️  {filepath.name} line {line_num}: Missing required keys")
                        return False
            return True
        except Exception as e:
            print(f"⚠️  {filepath.name} validation failed: {e}")
            return False
    
    def process_data(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Main processing pipeline"""
        print(f"\n{'='*60}")
        print(f"📂 LOADING RAW DATA")
        print(f"{'='*60}\n")
        
        # FIXED: Use raw file names from config for loading
        train_path = self.raw_dir / self.raw_train_file
        valid_path = self.raw_dir / self.raw_valid_file
        test_path = self.raw_dir / self.raw_test_file
        
        # Load data
        train_data = self.load_json_data(train_path)
        valid_data = self.load_json_data(valid_path)
        test_data = self.load_json_data(test_path)
        
        print(f"\n📊 Raw Data Summary:")
        print(f"  Train: {len(train_data):,} examples")
        print(f"  Valid: {len(valid_data):,} examples")
        print(f"  Test:  {len(test_data):,} examples")
        print(f"  Total: {len(train_data) + len(valid_data) + len(test_data):,} examples")
        
        # Smart subsampling for training data (assignment requirement ≤100k)
        if len(train_data) > self.max_train_samples:
            print(f"\n⚠️  Training set exceeds limit ({len(train_data):,} > {self.max_train_samples:,})")
            train_data = self.smart_subsample(train_data, self.max_train_samples)
        
        # Compute and print statistics
        self.print_statistics(train_data, "Training")
        self.print_statistics(valid_data, "Validation")
        self.print_statistics(test_data, "Test")
        
        # Save processed data (uses simplified names: train.json, valid.json, test.json)
        self.save_processed_data(train_data, valid_data, test_data)
        
        # Validate saved files
        print(f"\n🔍 Validating saved files...")
        for split in ['train', 'valid', 'test']:
            filepath = self.processed_dir / f"{split}.json"
            if self.validate_processed_file(filepath):
                print(f"   ✅ {split}.json is valid")
            else:
                print(f"   ❌ {split}.json validation failed!")
        
        return train_data, valid_data, test_data
    
    def check_data_ready(self) -> bool:
        """Check if processed data already exists and is valid"""
        required_files = ['train.json', 'valid.json', 'test.json']
        
        for filename in required_files:
            filepath = self.processed_dir / filename
            if not filepath.exists():
                return False
            # Quick validation: check it's valid JSON
            if not self.validate_processed_file(filepath):
                return False
        
        return True
    
    def run(self, force_reprocess: bool = False) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Main entry point for data processing
        
        Args:
            force_reprocess: If True, reprocess even if data exists
        """
        print(f"\n{'='*70}")
        print(f"🚀 AKSHARANTAR DATA PROCESSOR - CS772 Assignment 2")
        print(f"   Language: Hindi | Direction: Roman → Devanagari")
        print(f"   Seed: {self.seed} | Max Train: {self.max_train_samples:,}")
        print(f"{'='*70}")
        
        # Check if data already processed (unless forced)
        if not force_reprocess and self.check_data_ready():
            print("\n✅ Processed data found and validated. Skipping download/processing.")
            print("   Set force_reprocess=True to re-download and reprocess.")
            
            # Load existing data
            train_data = self.load_json_data(self.processed_dir / 'train.json')
            valid_data = self.load_json_data(self.processed_dir / 'valid.json')
            test_data = self.load_json_data(self.processed_dir / 'test.json')
        else:
            if force_reprocess:
                print("♻️  Force reprocessing enabled...")
            
            # Download if needed
            self.download_hindi_data()
            
            # Process data
            train_data, valid_data, test_data = self.process_data()
        
        # Final summary
        print(f"\n{'='*60}")
        print(f"✨ DATA PROCESSING COMPLETE!")
        print(f"{'='*60}")
        print(f"📈 Final Dataset Sizes:")
        print(f"   Training:   {len(train_data):,} examples")
        print(f"   Validation: {len(valid_data):,} examples")
        print(f"   Test:       {len(test_data):,} examples")
        print(f"   Total:      {len(train_data) + len(valid_data) + len(test_data):,} examples")
        print(f"\n📁 Processed data saved to: {self.processed_dir}")
        print(f"{'='*60}\n")
        
        return train_data, valid_data, test_data


def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and preprocess Hindi transliteration data")
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--force', action='store_true',
                       help='Force reprocessing even if data exists')
    args = parser.parse_args()
    
    try:
        downloader = DataDownloader(config_path=args.config)
        train_data, valid_data, test_data = downloader.run(force_reprocess=args.force)
        
        # Save statistics to file
        stats_file = downloader.processed_dir / "statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(downloader.stats, f, indent=2, ensure_ascii=False)
        print(f"📊 Statistics saved to: {stats_file}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())