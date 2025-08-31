import json
import re
from pathlib import Path
from typing import List, Dict

import numpy as np
from termcolor import colored
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


KEYWORDS_DATA_DIR = Path("data/keywords")
SEARCH_EXPORT_PATH = Path("data/search_results.json")


# -----------------------------
# Data loading / indexing
# -----------------------------
def load_papers(file_path: Path) -> List[Dict]:
    papers = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            papers.append(json.loads(line))
    return papers


def build_tfidf_matrix(papers: List[Dict]):
    abstracts = [p.get("abstract", "") for p in papers]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(abstracts)
    return vectorizer, tfidf_matrix


def search_papers(papers, vectorizer, tfidf_matrix, query: str, top_n: int = 5):
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_idx = sims.argsort()[-top_n:][::-1]

    results = []
    for i in top_idx:
        p = papers[i]
        results.append(
            {
                "title": p.get("title", "N/A"),
                "abstract": p.get("abstract", "N/A"),
                "keywords": p.get("keywords", []),
                "score": float(sims[i]),
            }
        )
    return results


# -----------------------------
# Pretty printing & summary
# -----------------------------
def highlight(text: str, query_terms: List[str]) -> str:
    """Very simple highlighter: bold red for exact term matches."""
    out = text
    for term in query_terms:
        if not term:
            continue
        out = re.sub(
            rf"(?i)\b({re.escape(term)})\b",
            lambda m: colored(m.group(1), "red", attrs=["bold"]),
            out,
        )
    return out


def summarize_results(results: List[Dict], query: str, top_n: int = 3) -> str:
    """Extractive summary: pick sentences from abstracts that contain most query terms."""
    if not results:
        return "No results to summarize."

    q_terms = set(w for w in re.findall(r"\w+", query.lower()) if len(w) > 2)
    if not q_terms:
        return "Query too short to summarize."

    sentences = []
    for r in results:
        # split on sentence boundaries
        sents = re.split(r"(?<=[.!?])\s+", r.get("abstract", ""))
        sentences.extend(sents)

    scored = []
    for s in sentences:
        words = set(re.findall(r"\w+", s.lower()))
        score = len(words & q_terms)
        if score > 0:
            scored.append((score, s.strip()))

    if not scored:
        return "No good sentences matched the query terms."

    scored.sort(key=lambda x: x[0], reverse=True)
    best = [s for _, s in scored[:top_n]]
    return " ".join(best)


# -----------------------------
# CLI app
# -----------------------------
def main():
    # 1) load data
    kw_files = list(KEYWORDS_DATA_DIR.glob("keywords_*.jsonl"))
    if not kw_files:
        print("❌ No keyword files found. Run extract_keywords.py first.")
        return

    file_path = kw_files[0]
    print(f"📂 Loading {file_path} ...")
    papers = load_papers(file_path)

    # 2) build index
    vectorizer, tfidf_matrix = build_tfidf_matrix(papers)

    # 3) interactive loop
    while True:
        query = input("\n🔎 Enter your search query (or type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue

        results = search_papers(papers, vectorizer, tfidf_matrix, query, top_n=5)

        # pretty print
        q_terms = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2]
        print("\n=== Top Results ===")
        for i, r in enumerate(results, 1):
            title = highlight(r["title"], q_terms)
            abstract_snippet = highlight(r["abstract"][:300], q_terms)

            print(f"\n[{i}] 📖 {title}")
            print(f"    ⭐ Score: {r['score']:.4f}")
            print(f"    📝 Keywords: {', '.join(r.get('keywords', []))}")
            print(f"    📄 Abstract: {abstract_snippet}...")

        # summary
        summary = summarize_results(results, query, top_n=3)
        print("\n📝 Summary of top results:")
        print(summary)

        # export
        SEARCH_EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SEARCH_EXPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to {SEARCH_EXPORT_PATH}")


if __name__ == "__main__":
    main()
