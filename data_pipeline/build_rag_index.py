import os
import json
import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def build_rag_index():
    papers_path = "data/raw/papers.jsonl"
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")
    matrix_path = os.path.join(models_dir, "tfidf_matrix.joblib")
    papers_metadata_path = os.path.join(models_dir, "papers_metadata.json")

    if not os.path.exists(papers_path):
        print(f"[RAG Indexer] Error: Raw papers file does not exist at {papers_path}. Run download_papers.py first.")
        return

    print(f"[RAG Indexer] Loading papers from {papers_path}...")
    papers = []
    with open(papers_path, "r", encoding="utf-8") as f:
        for line in f:
            papers.append(json.loads(line))

    if not papers:
        print("[RAG Indexer] Error: No papers found to index.")
        return

    print(f"[RAG Indexer] Found {len(papers)} papers. Building TF-IDF index...")
    
    # Extract abstracts and metadata
    corpus = []
    metadata = []
    
    for paper in papers:
        # Combine title and abstract for richer search context
        combined_text = f"{paper['title']}. {paper['abstract']}"
        corpus.append(combined_text)
        
        metadata.append({
            "paper_id": paper["paper_id"],
            "title": paper["title"],
            "authors": paper["authors"],
            "year": paper["year"],
            "url": paper["url"],
            "mentioned_planets": paper.get("mentioned_planets", []),
            "mentioned_stars": paper.get("mentioned_stars", [])
        })

    # Train TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Save to disk
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(tfidf_matrix, matrix_path)
    with open(papers_metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[RAG Indexer] TF-IDF Index successfully built and saved to {models_dir}/")
    print(f"[RAG Indexer] Vocabulary size: {len(vectorizer.vocabulary_)} terms.")

if __name__ == "__main__":
    build_rag_index()
