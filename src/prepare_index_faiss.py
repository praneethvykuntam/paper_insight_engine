import json
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

PASSAGES_DIR = Path("data/processed/passages")
PAPERS_DIR = Path("data/processed")
SEMANTIC_DIR = Path("data/semantic")
SEMANTIC_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 256


def load_passages() -> List[Dict]:
    files = sorted(PASSAGES_DIR.glob("processed_passages_*.jsonl"))
    if not files:
        raise SystemExit("âŒ No passages found. Run process_papers_advanced.py first.")
    passages = []
    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                passages.append(json.loads(line))
    return passages


def load_paper_meta() -> Dict[str, Dict]:
    """Map pmid -> {title, pub_date, journal} from cleaned files."""
    meta: Dict[str, Dict] = {}
    files = sorted(PAPERS_DIR.glob("processed_advanced_*.jsonl"))
    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                pmid = str(rec.get("pmid"))
                if pmid and pmid not in meta:
                    meta[pmid] = {
                        "title": rec.get("title", ""),
                        "pub_date": rec.get("pub_date", "NA"),
                        "journal": rec.get("journal", ""),
                    }
    return meta


def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    vecs = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding"):
        batch = texts[i : i + BATCH_SIZE]
        emb = model.encode(
            batch,
            batch_size=min(BATCH_SIZE, 64),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        vecs.append(emb.astype(np.float32))
    return np.vstack(vecs)


def build_faiss(embeddings: np.ndarray) -> faiss.Index:
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # inner product on normalized vectors == cosine sim
    index.add(embeddings)
    return index


def main():
    print("ðŸ“¥ Loading passages & metadata ...")
    passages = load_passages()  # [{pmid, sent_id, passage}]
    paper_meta = load_paper_meta()  # pmid -> meta

    texts = [p["passage"] for p in passages]

    print(f"ðŸ§  Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"ðŸ”¢ Embedding {len(texts)} passages ...")
    X = embed_texts(model, texts)  # (N, d) float32 normalized

    print("ðŸ“¦ Building FAISS index ...")
    index = build_faiss(X)

    # Save FAISS
    faiss_path = SEMANTIC_DIR / "faiss.index"
    faiss.write_index(index, str(faiss_path))

    # Save metadata row-aligned with FAISS ids
    meta_path = SEMANTIC_DIR / "meta.jsonl"
    with meta_path.open("w", encoding="utf-8") as f:
        for i, p in enumerate(passages):
            pmid = str(p.get("pmid"))
            meta = paper_meta.get(pmid, {"title": "", "pub_date": "NA", "journal": ""})
            f.write(
                json.dumps(
                    {
                        "id": i,
                        "pmid": pmid,
                        "sent_id": p.get("sent_id"),
                        "passage": p.get("passage", ""),
                        "title": meta.get("title", ""),
                        "pub_date": meta.get("pub_date", "NA"),
                        "journal": meta.get("journal", ""),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # Save config
    cfg_path = SEMANTIC_DIR / "config.json"
    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"model": MODEL_NAME, "dim": int(X.shape[1]), "count": int(X.shape[0])},
            f,
            indent=2,
        )

    print(f"âœ… Saved index â†’ {faiss_path}")
    print(f"âœ… Saved metadata â†’ {meta_path}")
    print(f"âœ… Saved config â†’ {cfg_path}")


if __name__ == "__main__":
    main()
