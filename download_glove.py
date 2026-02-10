import os
import urllib.request
import zipfile
import sys
from pathlib import Path

def download_glove_embeddings():
    """Download and extract GloVe embeddings from Stanford NLP"""
    
    # URLs for different GloVe embeddings
    glove_urls = {
        "glove.6B.50d": {
            "url": "https://nlp.stanford.edu/data/glove.6B.zip",
            "file": "glove.6B.50d.txt",
            "size": "50d"
        },
        "glove.6B.100d": {
            "url": "https://nlp.stanford.edu/data/glove.6B.zip", 
            "file": "glove.6B.100d.txt",
            "size": "100d"
        },
        "glove.6B.200d": {
            "url": "https://nlp.stanford.edu/data/glove.6B.zip",
            "file": "glove.6B.200d.txt", 
            "size": "200d"
        },
        "glove.6B.300d": {
            "url": "https://nlp.stanford.edu/data/glove.6B.zip",
            "file": "glove.6B.300d.txt",
            "size": "300d"
        }
    }
    
    print("🔍 Available GloVe embeddings:")
    print("1. glove.6B.50d (50 dimensions) - Fastest")
    print("2. glove.6B.100d (100 dimensions)")  
    print("3. glove.6B.200d (200 dimensions)")
    print("4. glove.6B.300d (300 dimensions) - Most detailed")
    
    choice = input("\nEnter choice (1-4, default=1): ").strip() or "1"
    
    size_map = {"1": "50d", "2": "100d", "3": "200d", "4": "300d"}
    selected_size = size_map.get(choice, "50d")
    selected_key = f"glove.6B.{selected_size}"
    
    if selected_key not in glove_urls:
        print("Invalid choice, using 50d")
        selected_key = "glove.6B.50d"
    
    config = glove_urls[selected_key]
    zip_file = "glove.6B.zip"
    target_file = config["file"]
    
    # Check if file already exists
    if os.path.exists(target_file):
        print(f"✅ {target_file} already exists!")
        return target_file
    
    # Download if zip doesn't exist
    if not os.path.exists(zip_file):
        print(f"📥 Downloading GloVe embeddings...")
        print(f"URL: {config['url']}")
        print(f"File size: ~860MB")
        print("This may take several minutes depending on your internet connection...")
        
        def download_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, (downloaded * 100) / total_size)
            bar_length = 40
            filled_length = int(bar_length * downloaded // total_size)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            print(f'\r[{bar}] {percent:.1f}% ({downloaded//1024//1024}MB/{total_size//1024//1024}MB)', end='')
        
        try:
            urllib.request.urlretrieve(config["url"], zip_file, download_progress)
            print("\n✅ Download completed!")
        except Exception as e:
            print(f"\n❌ Download failed: {e}")
            return None
    
    # Extract the specific file
    if os.path.exists(zip_file):
        print(f"📦 Extracting {target_file}...")
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                # Extract only the target file
                zip_ref.extract(target_file)
            print(f"✅ Extracted {target_file}")
            
            # Clean up zip file to save space
            cleanup = input("Delete zip file to save space? (y/N): ").strip().lower()
            if cleanup == 'y':
                os.remove(zip_file)
                print("🗑️ Zip file deleted")
                
            return target_file
            
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            return None
    
    return None

def create_sample_glove_file(filename="sample_glove.txt"):
    """Create a small sample GloVe file for testing"""
    print(f"📄 Creating sample GloVe file: {filename}")
    
    # Extended sample data with more realistic embeddings
    sample_data = """the -0.038194 -0.24487 0.72812 -0.39961 0.083172 0.043953 -0.39141 0.3344 -0.57545 0.087459
of -0.10767 -0.11053 0.59812 -0.54361 0.67396 0.10663 -0.038867 0.35481 -0.31024 -0.1678
to -0.21322 -0.47368 0.057986 -0.65597 0.21924 0.46303 -0.094386 0.41114 -0.23817 0.089428
and -0.33979 0.20941 0.46348 -0.64792 -0.38377 0.038034 -0.17920 0.55063 -0.14030 -0.25209
in 0.26818 0.14346 -0.27877 0.016257 0.11384 0.69923 -0.51332 -0.47368 -0.33075 -0.13834
a 0.21705 0.46515 -0.46757 0.10082 1.0135 0.74845 -0.53104 -0.26256 0.16812 0.13182
is -0.27158 0.10729 -0.068723 0.14127 0.51465 0.63513 -0.25757 -0.52262 0.16154 0.094967
that -0.21419 0.45521 -0.28864 -0.63256 -0.15284 0.59999 -0.034379 -0.21298 0.38447 -0.14801
for 0.20741 0.14441 -0.46221 0.38561 0.75939 0.67199 -0.93118 0.038321 -0.34434 -0.18044
it -0.15872 0.36161 0.21648 -0.33012 0.25264 1.2644 -0.78878 -0.22994 0.023324 -0.48705
king -0.50451 0.68607 0.59517 -0.022801 0.60046 -0.13498 -0.08813 0.47377 -0.61798 -0.31012
queen -0.30884 0.50195 0.61688 -0.14618 0.58413 -0.06016 0.0070928 0.49671 -0.55128 -0.38062
man -0.11285 0.23434 0.23102 -0.47434 0.15524 0.14216 -0.34809 0.32474 -0.41725 -0.44835
woman -0.27168 0.14978 0.33157 -0.20251 0.12138 0.026392 -0.29374 0.26466 -0.33699 -0.32633
prince -0.48049 0.62237 0.51220 -0.078379 0.47474 -0.15186 -0.12413 0.54482 -0.69124 -0.23590
princess -0.35847 0.40543 0.59924 -0.22585 0.47549 -0.089821 -0.09671 0.38922 -0.62024 -0.34167
doctor -0.31731 0.30643 0.39148 -0.25006 0.33308 0.091046 -0.21921 0.37685 -0.45642 -0.27814
nurse -0.42123 0.28945 0.41857 -0.31273 0.30142 0.12083 -0.19874 0.40321 -0.51938 -0.22657
cat -0.45441 0.32295 0.42174 -0.20981 0.34637 0.077293 -0.16547 0.41163 -0.47385 -0.26994
dog -0.44213 0.35689 0.38947 -0.28514 0.37562 0.10324 -0.22815 0.39471 -0.44382 -0.31085
car -0.31586 0.45321 0.23479 -0.14257 0.42891 0.21456 -0.31247 0.32864 -0.38295 -0.24731
bike -0.38429 0.41627 0.28341 -0.22458 0.39174 0.18927 -0.28439 0.35782 -0.41683 -0.27952
house -0.29542 0.38164 0.35827 -0.26734 0.31987 0.13546 -0.24381 0.37291 -0.42851 -0.25493
home -0.25694 0.34782 0.41523 -0.21847 0.35768 0.16283 -0.20574 0.39164 -0.38627 -0.23875
happy 0.12847 0.64235 -0.23571 0.028394 0.79418 0.53162 -0.68342 0.14729 -0.12648 -0.41357
sad -0.35482 0.41927 0.25863 -0.38247 0.23594 0.18472 -0.31685 0.42758 -0.49372 -0.28194"""
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(sample_data)
    
    print(f"✅ Created sample GloVe file with {len(sample_data.split(chr(10)))} words")
    return filename

if __name__ == "__main__":
    print("🌟 GloVe Embeddings Downloader")
    print("=" * 50)
    
    print("\nOptions:")
    print("1. Download FULL GloVe embeddings (6B tokens, ~860MB)")
    print("2. Use sample GloVe file for testing (26 words)")
    
    choice = input("\nEnter choice (1-2, default=1): ").strip() or "1"
    
    if choice == "1":
        print("\n🚀 Starting full GloVe download...")
        result = download_glove_embeddings()
        if result:
            print(f"\n🎉 Success! GloVe embeddings saved as: {result}")
            print(f"You can now use this file in the Word Embedding Explorer")
        else:
            print(f"\n❌ Download failed.")
            print("Creating sample file as fallback...")
            create_sample_glove_file("glove_sample.txt")
    else:
        print("\n📄 Creating sample GloVe file for demo...")
        sample_file = create_sample_glove_file("glove_sample.txt")
        print(f"🎉 Success! Sample GloVe embeddings saved as: {sample_file}")
    
    print("\n🚀 To use with Streamlit app, run:")
    print("   streamlit run app.py")