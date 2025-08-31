import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

PROCESSED_DATA_DIR = Path("data/processed")
OUTPUT_DATA_DIR = Path("data/keywords")
OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_papers(file_path: Path):
    papers = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            papers.append(json.loads(line))
    return papers


def extract_keywords_tfidf(papers, top_n=10):
    abstracts = [p.get("abstract", "") for p in papers]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(abstracts)
    feature_names = np.array(vectorizer.get_feature_names_out())

    results = []
    for idx, paper in enumerate(papers):
        row = tfidf_matrix[idx].toarray().flatten()
        top_indices = row.argsort()[-top_n:][::-1]
        keywords = feature_names[top_indices]
        paper["keywords"] = keywords.tolist()
        results.append(paper)
    return results


def process_file(input_file: Path, output_file: Path, top_n=10):
    papers = load_papers(input_file)
    enriched = extract_keywords_tfidf(papers, top_n=top_n)
    with open(output_file, "w", encoding="utf-8") as f:
        for record in enriched:
            f.write(json.dumps(record) + "\n")
    print(f"✅ Keywords extracted for {len(enriched)} papers → {output_file}")


def main():
    for processed_file in PROCESSED_DATA_DIR.glob("processed_*.jsonl"):
        output_file = OUTPUT_DATA_DIR / processed_file.name.replace(
            "processed_", "keywords_"
        )
        process_file(processed_file, output_file, top_n=10)


if __name__ == "__main__":
    main()
