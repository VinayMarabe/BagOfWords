import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
from glove_loader import GloveLoader

# Page config
st.set_page_config(
    page_title="Word Embedding Explorer",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("🔍 Word Embedding Explorer")
st.markdown("Explore word embeddings with GloVe, Word2Vec, and FastText models")

# Sidebar for model configuration
st.sidebar.header("Configuration")

# Initialize session state
if 'glove_loaded' not in st.session_state:
    st.session_state.glove_loaded = False
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = {}
if 'loader' not in st.session_state:
    st.session_state.loader = None

# GloVe file upload or demo data
st.sidebar.subheader("GloVe Embeddings")

# Check for available GloVe files
full_glove_files = []
for filename in ['glove.6B.50d.txt', 'glove.6B.100d.txt', 'glove.6B.200d.txt', 'glove.6B.300d.txt']:
    if os.path.exists(filename):
        file_size = os.path.getsize(filename) / (1024 * 1024)  # Size in MB
        full_glove_files.append((filename, file_size))

if full_glove_files:
    st.sidebar.info(f"✅ Found {len(full_glove_files)} full GloVe file(s)")
    use_full_glove = st.sidebar.checkbox("Use full GloVe embeddings", value=True)
    
    if use_full_glove:
        # Select which GloVe file to use
        if len(full_glove_files) > 1:
            file_options = [f"{fname} ({size:.0f}MB)" for fname, size in full_glove_files]
            selected_idx = st.sidebar.selectbox("Select GloVe file:", range(len(file_options)), 
                                              format_func=lambda x: file_options[x])
            selected_file = full_glove_files[selected_idx][0]
        else:
            selected_file = full_glove_files[0][0]
            st.sidebar.write(f"Using: {selected_file} ({full_glove_files[0][1]:.0f}MB)")
        
        limit = st.sidebar.number_input("Limit words (0 = all, max 50k for performance)", 
                                       min_value=0, max_value=400000, value=50000)
        
        if st.sidebar.button("Load Full GloVe Data"):
            with st.spinner(f"Loading {selected_file}..."):
                st.session_state.loader = GloveLoader(selected_file)
                st.session_state.embeddings = st.session_state.loader.load(limit=limit if limit > 0 else None)
                st.session_state.glove_loaded = True
                
            st.success(f"✅ Loaded {len(st.session_state.embeddings):,} words from {selected_file}!")
            st.info("🚀 Full GloVe embeddings ready for exploration!")
    else:
        use_demo_data = True
else:
    st.sidebar.warning("⚠️ No full GloVe files found")
    use_demo_data = st.sidebar.checkbox("Use demo data", value=True)

if not full_glove_files or (full_glove_files and not use_full_glove):
    if use_demo_data:
        if st.sidebar.button("Load Demo GloVe Data"):
            # Use the existing sample file or create demo data
            glove_path = 'glove_sample.txt'
            if not os.path.exists(glove_path):
                glove_path = 'demo_glove.txt'
                with open(glove_path, 'w') as f:
                    demo_words = {
                        'king': [0.1, 0.2, 0.3, 0.4, 0.5],
                        'queen': [0.15, 0.25, 0.35, 0.45, 0.55],
                        'man': [0.2, 0.1, 0.3, 0.4, 0.5],
                        'woman': [0.25, 0.15, 0.35, 0.45, 0.55],
                        'prince': [0.12, 0.22, 0.32, 0.42, 0.52],
                        'princess': [0.17, 0.27, 0.37, 0.47, 0.57],
                        'doctor': [0.3, 0.4, 0.5, 0.6, 0.7],
                        'nurse': [0.35, 0.45, 0.55, 0.65, 0.75],
                        'cat': [0.5, 0.6, 0.7, 0.8, 0.9],
                        'dog': [0.55, 0.65, 0.75, 0.85, 0.95],
                        'car': [0.4, 0.3, 0.8, 0.2, 0.6],
                        'bike': [0.45, 0.35, 0.85, 0.25, 0.65],
                        'house': [0.7, 0.8, 0.2, 0.9, 0.1],
                        'home': [0.75, 0.85, 0.25, 0.95, 0.15],
                        'happy': [0.9, 0.1, 0.8, 0.3, 0.7],
                        'sad': [0.1, 0.9, 0.2, 0.7, 0.3]
                    }
                    for word, vec in demo_words.items():
                        f.write(f"{word} {' '.join(map(str, vec))}\n")
            
            st.session_state.loader = GloveLoader(glove_path)
            st.session_state.embeddings = st.session_state.loader.load()
            st.session_state.glove_loaded = True
            
            # Check if we loaded from the sample file
            file_info = "sample" if glove_path == 'glove_sample.txt' else "demo"
            st.success(f"✅ Loaded {len(st.session_state.embeddings)} {file_info} words!")
            if glove_path == 'glove_sample.txt':
                st.info("📚 Using realistic word vectors from sample file")
            else:
                st.info("🎯 Using simple demo vectors for testing")

else:
    uploaded_file = st.sidebar.file_uploader("Upload GloVe file", type=['txt'])
    if uploaded_file is not None:
        with open('uploaded_glove.txt', 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        limit = st.sidebar.number_input("Limit words (0 = all)", min_value=0, value=10000)
        if st.sidebar.button("Load GloVe File"):
            st.session_state.loader = GloveLoader('uploaded_glove.txt')
            st.session_state.embeddings = st.session_state.loader.load(limit=limit if limit > 0 else None)
            st.session_state.glove_loaded = True
            st.success(f"✅ Loaded {len(st.session_state.embeddings)} words!")

# Main content tabs
if st.session_state.glove_loaded:
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Word Info", "🔍 Similarity", "🧠 Analogies", "📈 Visualization", "⚖️ Model Comparison"])
    
    with tab1:
        st.header("Word Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Available Words")
            word_list = list(st.session_state.embeddings.keys())
            st.write(f"Total words loaded: {len(word_list)}")
            st.write("Sample words:", word_list[:20])
        
        with col2:
            st.subheader("Word Vector")
            selected_word = st.selectbox("Select a word:", word_list)
            if selected_word:
                vector = st.session_state.loader.get_vector(selected_word)
                if vector is not None:
                    st.write(f"Vector for '{selected_word}':")
                    st.write(vector)
                    st.bar_chart(pd.DataFrame({'dimensions': range(len(vector)), 'values': vector}).set_index('dimensions'))
    
    with tab2:
        st.header("Word Similarity")
        col1, col2 = st.columns(2)
        
        with col1:
            word1 = st.selectbox("First word:", word_list, key="sim_word1")
        with col2:
            word2 = st.selectbox("Second word:", word_list, key="sim_word2", index=1 if len(word_list) > 1 else 0)
        
        if st.button("Calculate Similarity"):
            similarity = st.session_state.loader.similarity(word1, word2)
            if similarity is not None:
                st.metric("Cosine Similarity", f"{similarity:.4f}")
                
                # Create similarity gauge
                fig, ax = plt.subplots(figsize=(8, 4))
                colors = ['red' if similarity < 0.3 else 'orange' if similarity < 0.7 else 'green']
                bars = ax.barh(['Similarity'], [similarity], color=colors[0])
                ax.set_xlim(0, 1)
                ax.set_xlabel('Similarity Score')
                ax.set_title(f'Similarity: "{word1}" ↔ "{word2}"')
                st.pyplot(fig)
            else:
                st.error("Could not calculate similarity")
    
    with tab3:
        st.header("Word Analogies")
        st.markdown("Find words that complete the analogy: **A - B + C ≈ ?**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            word_a = st.selectbox("Word A:", word_list, key="analogy_a")
        with col2:
            word_b = st.selectbox("Word B:", word_list, key="analogy_b", index=2 if len(word_list) > 2 else 0)
        with col3:
            word_c = st.selectbox("Word C:", word_list, key="analogy_c", index=1 if len(word_list) > 1 else 0)
        
        top_n = st.slider("Number of results:", 1, 10, 5)
        
        if st.button("Find Analogies"):
            analogy_results = st.session_state.loader.analogy(word_a, word_b, word_c, top_n=top_n)
            if analogy_results:
                st.subheader(f"Results for: {word_a} - {word_b} + {word_c} ≈ ?")
                
                results_df = pd.DataFrame(analogy_results, columns=['Word', 'Score'])
                
                # Display as table
                st.dataframe(results_df, use_container_width=True)
                
                # Display as bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(results_df['Word'], results_df['Score'])
                ax.set_xlabel('Similarity Score')
                ax.set_title('Analogy Results')
                ax.invert_yaxis()
                st.pyplot(fig)
            else:
                st.error("Could not compute analogies")
    
    with tab4:
        st.header("Embedding Visualization")
        
        # Word selection for visualization
        st.subheader("Select Words to Visualize")
        col1, col2 = st.columns(2)
        
        with col1:
            viz_method = st.radio("Visualization Method:", ["PCA", "t-SNE"])
        
        with col2:
            word_selection_method = st.radio("Word Selection:", ["Predefined", "Custom"])
        
        if word_selection_method == "Predefined":
            selected_words = st.multiselect(
                "Choose words:", 
                word_list,
                default=word_list[:min(10, len(word_list))]
            )
        else:
            custom_words = st.text_area(
                "Enter words (one per line):",
                value="\n".join(word_list[:5])
            ).split('\n')
            selected_words = [w.strip() for w in custom_words if w.strip()]
        
        if st.button("Generate Visualization") and len(selected_words) >= 2:
            # Prepare data
            X = []
            labels = []
            
            for word in selected_words:
                if word in st.session_state.embeddings:
                    X.append(st.session_state.embeddings[word])
                    labels.append(word)
            
            if len(X) >= 2:
                X = np.array(X)
                
                # Apply dimensionality reduction
                if viz_method == "t-SNE":
                    if X.shape[0] < 4:
                        st.warning("Not enough samples for t-SNE, using PCA instead")
                        reducer = PCA(n_components=2)
                    else:
                        perplexity = min(30, X.shape[0] - 1)
                        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                else:
                    reducer = PCA(n_components=2)
                
                X_2d = reducer.fit_transform(X)
                
                # Create plot
                fig, ax = plt.subplots(figsize=(12, 8))
                scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.7, s=100)
                
                # Add labels
                for i, label in enumerate(labels):
                    ax.annotate(label, (X_2d[i, 0], X_2d[i, 1]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=10)
                
                ax.set_title(f'{viz_method} Visualization of Word Embeddings')
                ax.set_xlabel(f'{viz_method} Component 1')
                ax.set_ylabel(f'{viz_method} Component 2')
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
                # Show explained variance for PCA
                if viz_method == "PCA":
                    explained_variance = reducer.explained_variance_ratio_
                    st.info(f"Explained variance: PC1: {explained_variance[0]:.3f}, PC2: {explained_variance[1]:.3f}, Total: {sum(explained_variance):.3f}")
            
            else:
                st.error("Not enough valid words found in embeddings")
    
    with tab5:
        st.header("Model Comparison")
        
        try:
            from gensim.models import Word2Vec, FastText
            
            if st.button("Train and Compare Models"):
                # Create training sentences
                sentences = [
                    ['king', 'queen', 'man', 'woman', 'prince', 'princess'],
                    ['the', 'king', 'is', 'a', 'man'],
                    ['the', 'queen', 'is', 'a', 'woman'],
                    ['the', 'prince', 'is', 'the', 'son', 'of', 'the', 'king'],
                    ['the', 'princess', 'is', 'the', 'daughter', 'of', 'the', 'queen'],
                    ['doctor', 'nurse', 'hospital', 'medicine'],
                    ['cat', 'dog', 'animal', 'pet'],
                ]
                
                with st.spinner("Training models..."):
                    # Train models
                    wv_model = Word2Vec(sentences, vector_size=50, min_count=1, window=3, sg=1)
                    ft_model = FastText(sentences, vector_size=50, min_count=1, window=3, sg=1)
                
                st.success("Models trained successfully!")
                
                # Comparison table
                comparison_data = []
                
                test_words = [('king', 'queen'), ('man', 'woman'), ('doctor', 'nurse')]
                
                for word1, word2 in test_words:
                    row = {'Word Pair': f"{word1} - {word2}"}
                    
                    # GloVe similarity
                    glove_sim = st.session_state.loader.similarity(word1, word2)
                    row['GloVe'] = f"{glove_sim:.4f}" if glove_sim else "N/A"
                    
                    # Word2Vec similarity
                    try:
                        w2v_sim = wv_model.wv.similarity(word1, word2)
                        row['Word2Vec'] = f"{w2v_sim:.4f}"
                    except:
                        row['Word2Vec'] = "N/A"
                    
                    # FastText similarity
                    try:
                        ft_sim = ft_model.wv.similarity(word1, word2)
                        row['FastText'] = f"{ft_sim:.4f}"
                    except:
                        row['FastText'] = "N/A"
                    
                    comparison_data.append(row)
                
                st.subheader("Similarity Comparison")
                st.dataframe(pd.DataFrame(comparison_data))
                
                # OOV handling demo
                st.subheader("Out-of-Vocabulary (OOV) Handling")
                
                oov_word = st.text_input("Test OOV word:", value="unknownword123")
                
                if st.button("Test OOV Handling"):
                    oov_results = {}
                    
                    # GloVe
                    glove_vec = st.session_state.loader.get_vector(oov_word)
                    oov_results['GloVe'] = "✅ Found" if glove_vec is not None else "❌ Not found"
                    
                    # Word2Vec
                    try:
                        wv_vec = wv_model.wv[oov_word]
                        oov_results['Word2Vec'] = "✅ Found"
                    except:
                        oov_results['Word2Vec'] = "❌ Not found"
                    
                    # FastText
                    try:
                        ft_vec = ft_model.wv[oov_word]
                        oov_results['FastText'] = "✅ Can generate vector"
                    except:
                        oov_results['FastText'] = "❌ Cannot generate"
                    
                    oov_df = pd.DataFrame([oov_results])
                    st.dataframe(oov_df)
                    
                    st.info("💡 FastText can handle OOV words by using subword information (character n-grams)")
        
        except ImportError:
            st.warning("⚠️ Gensim not available. Install gensim to compare Word2Vec and FastText models.")
            st.code("pip install gensim")

else:
    st.info("👈 Please load GloVe embeddings from the sidebar to start exploring!")
    
    # Show sample usage
    st.markdown("""
    ## Features
    
    - **📊 Word Info**: Explore loaded words and their vector representations
    - **🔍 Similarity**: Calculate cosine similarity between word pairs
    - **🧠 Analogies**: Solve word analogies (king - man + woman ≈ queen)
    - **📈 Visualization**: Create 2D visualizations using t-SNE or PCA
    - **⚖️ Model Comparison**: Compare GloVe, Word2Vec, and FastText models
    
    ## Getting Started
    
    1. Use the demo data or upload your own GloVe file
    2. Explore different tabs to analyze word embeddings
    3. Try different word combinations and visualization methods
    """)