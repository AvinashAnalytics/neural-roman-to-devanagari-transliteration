"""
Quick vocabulary inspection script
Usage: python check_vocab.py
"""

from utils.vocab import Vocabulary
from pathlib import Path
import pickle

def inspect_vocabulary(vocab_path, name="Vocabulary"):
    """Load and print detailed vocab statistics"""
    print(f"\n{'='*70}")
    print(f"📊 {name.upper()}")
    print(f"{'='*70}")
    
    # Method 1: Load with Vocabulary class (recommended)
    vocab = Vocabulary()
    vocab.load(vocab_path)
    
    # Print full info
    vocab.print_info(top_k=20)
    
    # Get statistics
    stats = vocab.get_statistics()
    
    # Additional analysis
    print(f"\n📈 DETAILED STATISTICS:")
    print(f"   Total entries:        {stats['size']:,}")
    print(f"   Special tokens:       {stats['num_special_tokens']}")
    print(f"   Regular characters:   {stats['num_regular_chars']:,}")
    print(f"   Texts used for build: {stats['num_texts_built']:,}")
    print(f"   Min frequency:        {stats['min_freq']}")
    
    if stats['total_characters'] > 0:
        print(f"\n   Character coverage:   {stats['coverage']:.2f}%")
        print(f"   UNK rate:             {stats['unk_rate']:.2f}%")
        print(f"   Total chars seen:     {stats['total_characters']:,}")
        print(f"   Unknown chars:        {stats['unk_characters']:,}")
    
    # Character breakdown
    special = vocab.get_special_tokens()
    regular = [c for c in vocab.get_chars() if c not in special]
    
    print(f"\n🔤 CHARACTER BREAKDOWN:")
    print(f"   Alphabetic:   {sum(1 for c in regular if c.isalpha())}")
    print(f"   Digits:       {sum(1 for c in regular if c.isdigit())}")
    print(f"   Whitespace:   {sum(1 for c in regular if c.isspace())}")
    print(f"   Punctuation:  {sum(1 for c in regular if not c.isalnum() and not c.isspace())}")
    
    return vocab, stats


def compare_vocabularies():
    """Compare source and target vocabularies"""
    print(f"\n{'='*70}")
    print(f"🔄 VOCABULARY COMPARISON")
    print(f"{'='*70}")
    
    src_vocab, src_stats = inspect_vocabulary('data/processed/vocab_src.pkl', 'Source (Roman)')
    tgt_vocab, tgt_stats = inspect_vocabulary('data/processed/vocab_tgt.pkl', 'Target (Devanagari)')
    
    print(f"\n{'='*70}")
    print(f"📊 COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n{'Metric':<30} {'Source':>15} {'Target':>15}")
    print(f"{'-'*30} {'-'*15} {'-'*15}")
    print(f"{'Total size':<30} {src_stats['size']:>15,} {tgt_stats['size']:>15,}")
    print(f"{'Regular characters':<30} {src_stats['num_regular_chars']:>15,} {tgt_stats['num_regular_chars']:>15,}")
    print(f"{'Training texts':<30} {src_stats['num_texts_built']:>15,} {tgt_stats['num_texts_built']:>15,}")
    
    if src_stats['coverage']:
        print(f"{'Coverage %':<30} {src_stats['coverage']:>14.2f}% {tgt_stats['coverage']:>14.2f}%")
        print(f"{'UNK rate %':<30} {src_stats['unk_rate']:>14.2f}% {tgt_stats['unk_rate']:>14.2f}%")
    
    # Size ratio
    ratio = tgt_stats['size'] / src_stats['size'] if src_stats['size'] > 0 else 0
    print(f"\n💡 Target vocab is {ratio:.2f}x larger than source")
    print(f"   (Devanagari has more characters than Roman script)")


def show_sample_encodings():
    """Show how sample words are encoded"""
    print(f"\n{'='*70}")
    print(f"🧪 SAMPLE ENCODINGS")
    print(f"{'='*70}")
    
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    src_vocab.load('data/processed/vocab_src.pkl')
    tgt_vocab.load('data/processed/vocab_tgt.pkl')
    
    # Sample Roman words
    test_words = [
        ("namaste", "नमस्ते"),
        ("bharat", "भारत"),
        ("hindi", "हिंदी"),
        ("delhi", "दिल्ली"),
        ("mumbai", "मुंबई")
    ]
    
    print(f"\n{'Roman':<15} {'Encoded':<30} {'Devanagari':<15} {'Encoded':<30}")
    print(f"{'-'*15} {'-'*30} {'-'*15} {'-'*30}")
    
    for roman, devanagari in test_words:
        try:
            src_enc = src_vocab.encode(roman)
            tgt_enc = tgt_vocab.encode(devanagari)
            
            # Truncate encoding display if too long
            src_str = str(src_enc)[:28] + ".." if len(str(src_enc)) > 30 else str(src_enc)
            tgt_str = str(tgt_enc)[:28] + ".." if len(str(tgt_enc)) > 30 else str(tgt_enc)
            
            print(f"{roman:<15} {src_str:<30} {devanagari:<15} {tgt_str:<30}")
        except Exception as e:
            print(f"{roman:<15} ERROR: {e}")


def check_vocab_files_exist():
    """Verify vocab files exist"""
    print(f"\n{'='*70}")
    print(f"📁 FILE CHECK")
    print(f"{'='*70}\n")
    
    files = {
        'Source vocab': 'data/processed/vocab_src.pkl',
        'Target vocab': 'data/processed/vocab_tgt.pkl',
        'Train data': 'data/processed/train.json',
        'Valid data': 'data/processed/valid.json',
        'Test data': 'data/processed/test.json'
    }
    
    all_exist = True
    for name, path in files.items():
        p = Path(path)
        if p.exists():
            size = p.stat().st_size / 1024  # KB
            if size > 1024:
                size_str = f"{size/1024:.1f} MB"
            else:
                size_str = f"{size:.1f} KB"
            print(f"✅ {name:<20} {path:<35} ({size_str})")
        else:
            print(f"❌ {name:<20} {path:<35} (MISSING)")
            all_exist = False
    
    if not all_exist:
        print(f"\n⚠️  Some files missing! Run: python -m data.download_data --force")
        return False
    
    print(f"\n✅ All required files found!")
    return True


if __name__ == "__main__":
    import sys
    
    # Check files exist first
    if not check_vocab_files_exist():
        sys.exit(1)
    
    # Run full comparison
    try:
        compare_vocabularies()
        show_sample_encodings()
        
        print(f"\n{'='*70}")
        print(f"✅ VOCABULARY INSPECTION COMPLETE")
        print(f"{'='*70}\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print(f"   Run: python -m data.download_data --force")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)