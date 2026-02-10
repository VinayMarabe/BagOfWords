import numpy as np

class GloveLoader:
    def __init__(self, glove_path):
        self.glove_path = glove_path
        self.embeddings = {}
        self.dim = None

    def load(self, limit=None):
        with open(self.glove_path, 'r', encoding='utf8') as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                parts = line.strip().split()
                word = parts[0]
                vector = np.array(parts[1:], dtype=np.float32)
                self.embeddings[word] = vector
                if self.dim is None:
                    self.dim = len(vector)
        return self.embeddings

    def get_vector(self, word):
        return self.embeddings.get(word)

    def similarity(self, word1, word2):
        v1 = self.get_vector(word1)
        v2 = self.get_vector(word2)
        if v1 is None or v2 is None:
            return None
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def analogy(self, word_a, word_b, word_c, top_n=1):
        # king - man + woman ≈ queen
        v_a = self.get_vector(word_a)
        v_b = self.get_vector(word_b)
        v_c = self.get_vector(word_c)
        if v_a is None or v_b is None or v_c is None:
            return None
        target = v_a - v_b + v_c
        # Find closest words
        sims = {}
        for word, vec in self.embeddings.items():
            if word in [word_a, word_b, word_c]:
                continue
            sims[word] = np.dot(target, vec) / (np.linalg.norm(target) * np.linalg.norm(vec))
        sorted_words = sorted(sims.items(), key=lambda x: x[1], reverse=True)
        return sorted_words[:top_n]
