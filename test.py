# check_files.py
import os
import glob

print("Checking for model files...")
print("\n📁 outputs/checkpoints/:")
if os.path.exists('outputs/checkpoints'):
    files = os.listdir('outputs/checkpoints')
    for f in files:
        size = os.path.getsize(f'outputs/checkpoints/{f}') / (1024*1024)  # MB
        print(f"  - {f} ({size:.2f} MB)")
else:
    print("  Directory doesn't exist!")

print("\n📁 data/processed/:")
if os.path.exists('data/processed'):
    files = os.listdir('data/processed')
    for f in files:
        print(f"  - {f}")