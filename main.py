# Word Embedding Explorer
# Main script to load GloVe embeddings and test basic functionality

from glove_loader import GloveLoader
import os
import numpy as np

# Path to GloVe file (update this to your local path)
glove_path = 'glove.6B.50d.txt'  # Example filename

# Create a mock GloVe file for demo if the real one doesn't exist
if not os.path.exists(glove_path):
    print(f"GloVe file {glove_path} not found. Creating demo file...")
    with open(glove_path, 'w') as f:
        f.write("king 0.1 0.2 0.3 0.4 0.5\n")
        f.write("queen 0.15 0.25 0.35 0.45 0.55\n")
        f.write("man 0.2 0.1 0.3 0.4 0.5\n")
        f.write("woman 0.25 0.15 0.35 0.45 0.55\n")
        f.write("prince 0.12 0.22 0.32 0.42 0.52\n")
        f.write("princess 0.17 0.27 0.37 0.47 0.57\n")
        f.write("doctor 0.3 0.4 0.5 0.6 0.7\n")
        f.write("nurse 0.35 0.45 0.55 0.65 0.75\n")
        f.write("cat 0.5 0.6 0.7 0.8 0.9\n")
        f.write("dog 0.55 0.65 0.75 0.85 0.95\n")

loader = GloveLoader(glove_path)
print('Loading GloVe embeddings...')
embeddings = loader.load(limit=10000)  # Load first 10k words for demo
print(f'Loaded {len(embeddings)} words.')

# Test: Get vector for a word
word = 'king'
vector = loader.get_vector(word)
print(f'Vector for "{word}":', vector)

# Word similarity demo
sim = loader.similarity('king', 'queen')
print(f'Similarity between "king" and "queen": {sim:.4f}')

# Analogy demo: king - man + woman ≈ ?
analogy_result = loader.analogy('king', 'man', 'woman', top_n=3)
print('Analogy (king - man + woman):')
for word, score in analogy_result:
	print(f'  {word}: {score:.4f}')

# Visualization: t-SNE and PCA
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for headless environment
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def plot_embeddings(embeddings, words, method='tsne'):
    X = [embeddings[w] for w in words if w in embeddings]
    labels = [w for w in words if w in embeddings]
    
    if len(X) < 2:
        print(f"Not enough words found for visualization (need at least 2, got {len(X)})")
        return
        
    X = np.array(X)  # Convert to numpy array
    
    if method == 'tsne':
        if X.shape[0] < 4:  # t-SNE needs at least 4 samples, use PCA instead
            print("Not enough samples for t-SNE, using PCA instead")
            method = 'pca'
        else:
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, X.shape[0]-1))
    
    if method == 'pca':
        reducer = PCA(n_components=2)
        
    X_2d = reducer.fit_transform(X)
    plt.figure(figsize=(8,6))
    plt.scatter(X_2d[:,0], X_2d[:,1])
    for i, label in enumerate(labels):
        plt.annotate(label, (X_2d[i,0], X_2d[i,1]))
    plt.title(f'{method.upper()} visualization of embeddings')
    
    # Save plot instead of showing
    filename = f'embeddings_{method}_plot.png'
    plt.savefig(filename)
    plt.close()
    print(f"Saved {method.upper()} plot to {filename}")

# Example cluster visualization
cluster_words = ['king', 'queen', 'man', 'woman', 'prince', 'princess', 'doctor', 'nurse', 'cat', 'dog']
plot_embeddings(embeddings, cluster_words, method='tsne')
plot_embeddings(embeddings, cluster_words, method='pca')

# --- Word2Vec and FastText Comparison ---
try:
    from gensim.models import Word2Vec, FastText
    
    # Sample sentences for training demo models
    sentences = [
        ['king', 'queen', 'man', 'woman', 'prince', 'princess', 'doctor', 'nurse', 'cat', 'dog'],
        ['the', 'king', 'is', 'a', 'man'],
        ['the', 'queen', 'is', 'a', 'woman'],
        ['the', 'prince', 'is', 'the', 'son', 'of', 'the', 'king'],
        ['the', 'princess', 'is', 'the', 'daughter', 'of', 'the', 'queen'],
        ['the', 'doctor', 'helps', 'the', 'nurse'],
        ['the', 'cat', 'chases', 'the', 'dog'],
    ]
    
    # Train Word2Vec
    wv_model = Word2Vec(sentences, vector_size=50, min_count=1, window=3, sg=1)
    # Train FastText
    ft_model = FastText(sentences, vector_size=50, min_count=1, window=3, sg=1)
    
    # Compare similarity
    print('\nWord2Vec similarity (king, queen):', wv_model.wv.similarity('king', 'queen'))
    print('FastText similarity (king, queen):', ft_model.wv.similarity('king', 'queen'))
    print('GloVe similarity (king, queen):', loader.similarity('king', 'queen'))
    
    # Analogy: king - man + woman
    print('\nWord2Vec analogy (king - man + woman):', wv_model.wv.most_similar(positive=['king', 'woman'], negative=['man'], topn=3))
    print('FastText analogy (king - man + woman):', ft_model.wv.most_similar(positive=['king', 'woman'], negative=['man'], topn=3))
    print('GloVe analogy (king - man + woman):', loader.analogy('king', 'man', 'woman', top_n=3))

    # --- FastText OOV Handling Demo ---
    oov_word = 'unicornify'
    try:
        ft_vec = ft_model.wv[oov_word]
        print(f'\nFastText vector for OOV word "{oov_word}":', ft_vec[:5], '...')
    except KeyError:
        print(f'\nFastText could not handle OOV word "{oov_word}".')
    
    try:
        wv_vec = wv_model.wv[oov_word]
        print(f'Word2Vec vector for OOV word "{oov_word}":', wv_vec[:5], '...')
    except KeyError:
        print(f'Word2Vec could not handle OOV word "{oov_word}".')
    
    glove_vec = loader.get_vector(oov_word)
    if glove_vec is not None:
        print(f'GloVe vector for OOV word "{oov_word}":', glove_vec[:5], '...')
    else:
        print(f'GloVe could not handle OOV word "{oov_word}".')

except ImportError:
    print("\n--- Gensim not available ---")
    print("To run Word2Vec and FastText comparisons, install gensim:")
    print("pip install gensim")
    print("\nDemo completed with GloVe functionality only.")

print("\n=== Word Embedding Explorer Demo Complete ===")
print("✓ GloVe embeddings loaded and tested")
print("✓ Similarity calculations working")  
print("✓ Analogy tasks completed")
print("✓ Visualization plots saved (t-SNE and PCA)")
if 'Word2Vec' in locals():
    print("✓ Word2Vec vs FastText vs GloVe comparison completed")
    print("✓ FastText OOV handling demonstrated")
else:
    print("- Word2Vec/FastText comparison skipped (gensim not available)")
