"""
Simple setup script for N-Gram Language Model Homework
Downloads PTB dataset using kagglehub
"""

import os
import shutil

def main():
    print("="*60)
    print("Downloading Penn Treebank Dataset")
    print("="*60)
    
    # Install kagglehub if needed
    try:
        import kagglehub
    except ImportError:
        print("\nInstalling kagglehub...")
        import subprocess
        subprocess.check_call(["pip", "install", "kagglehub"])
        import kagglehub
        print("✓ Installed kagglehub")
    
    # Download dataset
    print("\nDownloading dataset from Kaggle...")
    print("(This may take a minute...)")
    
    try:
        path = kagglehub.dataset_download("aliakay8/penn-treebank-dataset")
        print(f"\n✓ Downloaded to: {path}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure you:")
        print("1. Have a Kaggle account")
        print("2. Accepted dataset terms: https://www.kaggle.com/datasets/aliakay8/penn-treebank-dataset")
        print("3. Set up Kaggle API: https://www.kaggle.com/docs/api")
        return
    
    # Copy files to current directory
    print("\nCopying files to current directory...")
    files = ['ptb.train.txt', 'ptb.valid.txt', 'ptb.test.txt']
    
    # The files are in a ptbdataset subdirectory
    dataset_path = os.path.join(path, 'ptbdataset')
    
    for filename in files:
        src = os.path.join(dataset_path, filename)
        if os.path.exists(src):
            shutil.copy2(src, filename)
            print(f"✓ {filename}")
        else:
            print(f"✗ {filename} not found at {src}")
    
    print("\n" + "="*60)
    print("✓ Setup complete! Run: python main.py")
    print("="*60)

if __name__ == "__main__":
    main()