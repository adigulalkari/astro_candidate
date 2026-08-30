import os
import json
import joblib
import numpy as np

class LiteratureRetriever:
    def __init__(self):
        models_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "models"
        )
        self.vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")
        self.matrix_path = os.path.join(models_dir, "tfidf_matrix.joblib")
        self.metadata_path = os.path.join(models_dir, "papers_metadata.json")

        self.vectorizer = None
        self.tfidf_matrix = None
        self.metadata = []

        if os.path.exists(self.vectorizer_path) and os.path.exists(self.matrix_path) and os.path.exists(self.metadata_path):
            try:
                self.vectorizer = joblib.load(self.vectorizer_path)
                self.tfidf_matrix = joblib.load(self.matrix_path)
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                print(f"[RAG Retriever] Successfully loaded index with {len(self.metadata)} documents.")
            except Exception as e:
                print(f"[RAG Retriever] Error loading index files: {e}")
        else:
            print("[RAG Retriever] Warning: Index files not found. RAG search will be inactive.")

    def search_literature(self, query: str, top_k: int = 5, filter_planet: str = None, filter_star: str = None) -> list:
        if self.vectorizer is None or self.tfidf_matrix is None:
            return []

        # 1. Embed query
        query_vec = self.vectorizer.transform([query])

        # 2. Compute Cosine Similarity (dot product for normalized TF-IDF vectors)
        # tfidf_matrix shape: (n_docs, n_vocab)
        # query_vec shape: (1, n_vocab)
        similarities = (self.tfidf_matrix * query_vec.T).toarray().flatten()

        # 3. Sort indices by similarity descending
        ranked_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in ranked_indices:
            score = float(similarities[idx])
            paper = self.metadata[idx]

            # Apply hard filters if requested
            if filter_planet:
                # Check if planet is in mentioned_planets (case insensitive comparison)
                mentioned_lower = [p.lower() for p in paper.get("mentioned_planets", [])]
                if filter_planet.lower() not in mentioned_lower:
                    continue

            if filter_star:
                # Check if star is in mentioned_stars (case insensitive comparison)
                mentioned_lower = [s.lower() for s in paper.get("mentioned_stars", [])]
                if filter_star.lower() not in mentioned_lower:
                    continue

            # Skip results with zero similarity unless we are just filtering by planet/star and want any mention
            if score <= 0.0 and not (filter_planet or filter_star):
                continue

            results.append({
                "paper_id": paper["paper_id"],
                "title": paper["title"],
                "authors": paper["authors"],
                "year": paper["year"],
                "url": paper["url"],
                "score": score,
                "mentioned_planets": paper.get("mentioned_planets", []),
                "mentioned_stars": paper.get("mentioned_stars", [])
            })

            if len(results) >= top_k:
                break

        return results
