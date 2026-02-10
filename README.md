# Word Embedding Explorer Setup Guide

## Quick Start

1. **Run the Application**:
   ```bash
   # Option 1: Use the batch file (Windows)
   run_app.bat
   
   # Option 2: Manual command
   streamlit run app.py
   ```

2. **Open in Browser**: Go to `http://localhost:8501`

3. **Load Embeddings**: Click "Load Demo GloVe Data" in the sidebar to start exploring

## Full GloVe Embeddings Download

### Automatic Download (Recommended)
Run the download script:
```bash
python download_glove.py
```

### Manual Download
1. Visit: https://nlp.stanford.edu/projects/glove/
2. Download: `glove.6B.zip` (~860MB)
3. Extract files:
   - `glove.6B.50d.txt` (50 dimensions) - Fastest
   - `glove.6B.100d.txt` (100 dimensions)
   - `glove.6B.200d.txt` (200 dimensions) 
   - `glove.6B.300d.txt` (300 dimensions) - Most detailed

### Alternative Sources
- **Hugging Face**: https://huggingface.co/stanfordnlp/glove
- **PyTorch**: Available in torchtext.datasets
- **Direct**: http://nlp.stanford.edu/data/glove.6B.zip

## Features Overview

### 📊 Word Info
- Explore loaded vocabulary
- View word vector representations
- Interactive vector visualization

### 🔍 Similarity 
- Calculate cosine similarity between word pairs
- Visual similarity gauge
- Real-time results

### 🧠 Analogies
- Solve word analogies: "king - man + woman ≈ queen"
- Configurable number of results
- Interactive word selection

### 📈 Visualization
- **t-SNE**: Non-linear dimensionality reduction
- **PCA**: Linear dimensionality reduction  
- Custom word selection
- Interactive scatter plots

### ⚖️ Model Comparison
- Compare GloVe vs Word2Vec vs FastText
- Out-of-vocabulary (OOV) handling demo
- Performance metrics

## Troubleshooting

### Streamlit Connection Error
If you see "Connection error" in the browser:
1. Check if Streamlit is running in terminal
2. Restart with: `streamlit run app.py`
3. Try different port: `streamlit run app.py --server.port 8502`

### Missing Dependencies
Install required packages:
```bash
pip install streamlit numpy matplotlib scikit-learn gensim
```

### Memory Issues
- Use smaller embeddings (50d instead of 300d)
- Limit number of words loaded
- Close other applications

### File Not Found
- Ensure you're in the correct directory
- Check file paths are correct
- Use absolute paths if needed

## Advanced Usage

### Custom GloVe Files
1. Upload your own GloVe file via the sidebar
2. Supported format: `word value1 value2 ... valueN`
3. Any number of dimensions supported

### Batch Processing
```python
from glove_loader import GloveLoader

# Load large embeddings
loader = GloveLoader('glove.6B.300d.txt')
embeddings = loader.load(limit=50000)

# Batch similarity calculations
similarities = []
word_pairs = [('king', 'queen'), ('man', 'woman')]
for w1, w2 in word_pairs:
    sim = loader.similarity(w1, w2)
    similarities.append((w1, w2, sim))
```

### API Usage
The GloveLoader class provides:
- `load(limit=None)`: Load embeddings
- `get_vector(word)`: Get word vector  
- `similarity(word1, word2)`: Cosine similarity
- `analogy(a, b, c, top_n=1)`: Word analogies

## Project Structure
```
Word Embedding Explorer/
├── app.py              # Main Streamlit application
├── glove_loader.py     # GloVe embeddings loader
├── download_glove.py   # Download script
├── run_app.bat         # Windows launcher
├── requirements.txt    # Python dependencies
├── glove_sample.txt    # Sample embeddings
└── README.md          # This file
```

## Performance Tips
- **Fast loading**: Use 50d embeddings for development
- **Quality**: Use 300d embeddings for production
- **Memory**: Limit vocabulary size for large files
- **Speed**: Pre-filter words by frequency

## Citation
If using GloVe embeddings:
```
@inproceedings{pennington2014glove,
  title={Glove: Global vectors for word representation},
  author={Pennington, Jeffrey and Socher, Richard and Manning, Christopher D},
  booktitle={Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP)},
  pages={1532--1543},
  year={2014}
}
```