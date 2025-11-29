import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SimpleRetriever:
    def __init__(self, docs_dir=r"E:\Projects\copilot_DSPy\docs", top_k=3):
        self.docs = []
        self.doc_ids = []
        for i, filename in enumerate(os.listdir(docs_dir)):
            path = os.path.join(docs_dir, filename)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
                self.docs.append(text)
                self.doc_ids.append(f"{filename}::chunk0")
        self.vectorizer = TfidfVectorizer()
        self.doc_vectors = self.vectorizer.fit_transform(self.docs)
        self.top_k = top_k

    def retrieve(self, query: str):
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.doc_vectors)[0]
        top_idx = np.argsort(scores)[::-1][:self.top_k]
        results = [{"doc_id": self.doc_ids[i], "content": self.docs[i], "score": float(scores[i])} for i in top_idx]
        return results