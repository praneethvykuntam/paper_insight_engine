import json
import re
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz import utils as rf_utils
from sentence_transformers import SentenceTransformer

SEMANTIC_DIR = Path("data/semantic")


def load_config():
    return json.loads((SEMANTIC_DIR / "config.json").read_text(encoding="utf-8"))


def load_meta() -> List[Dict]:
    meta_fp = SEMANTIC_DIR / "meta.jsonl"
    meta = []
    with meta_fp.open("r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    return meta


def load_index():
    return faiss.read_index(str(SEMANTIC_DIR / "faiss.index"))


def embed_query(model, text: str) -> np.ndarray:
    x = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)
    return x.astype(np.float32)


def normalize_terms(query: str) -> List[str]:
    return [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2]


def fuzzy_overlap_score(text: str, query: str, threshold: int = 85) -> float:
    terms = normalize_terms(query)
    if not terms:
        return 0.0
    t = rf_utils.default_process(text or "").lower()
    hits = 0
    for term in terms:
        if term in t:
            hits += 1
            continue
        if fuzz.partial_ratio(term, t) >= threshold:
            hits += 1
    return hits / len(terms)


def hybrid_rerank(
    cands: List[Dict], query: str, alpha: float = 0.7, fuzzy_threshold: int = 85
) -> List[Dict]:
    out = []
    for c in cands:
        kw = fuzzy_overlap_score(c.get("passage", ""), query, threshold=fuzzy_threshold)
        c["kw_overlap"] = kw
        c["hybrid_score"] = alpha * float(c.get("score", 0.0)) + (1 - alpha) * kw
        out.append(c)
    out.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return out


def dedup_by_paper(items: List[Dict]) -> List[Dict]:
    groups: Dict[str, List[Dict]] = {}
    for it in items:
        pmid = it.get("pmid")
        title = (it.get("title") or "").strip().lower()
        key = pmid if pmid else f"title::{re.sub(r'\\s+', ' ', title)}"
        groups.setdefault(key, []).append(it)

    deduped = []
    for _, arr in groups.items():
        best = max(
            arr, key=lambda x: (x.get("hybrid_score", -1e9), x.get("score", -1e9))
        )
        merged_count = len(arr) - 1
        best = dict(best)
        best["merged_passages"] = merged_count
        deduped.append(best)

    deduped.sort(
        key=lambda x: (x.get("hybrid_score", -1e9), x.get("score", -1e9)), reverse=True
    )
    return deduped


def main():
    cfg = load_config()
    index = load_index()
    meta = load_meta()
    model = SentenceTransformer(
        cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2")
    )
    print(f"📦 Index dim={cfg['dim']} vectors={cfg['count']} | model={cfg['model']}")

    while True:
        q = input("\n🔎 Enter query (or 'exit'): ").strip()
        if q.lower() == "exit":
            break
        if not q:
            continue

        qv = embed_query(model, q)
        k_stage1 = 100
        sims, ids = index.search(qv, k_stage1)
        sims, ids = sims[0], ids[0]

        cands = []
        for score, idx in zip(sims, ids):
            if idx == -1:
                continue
            m = meta[idx]
            cands.append(
                {
                    "score": float(score),
                    "pmid": m["pmid"],
                    "sent_id": m["sent_id"],
                    "title": m.get("title", ""),
                    "journal": m.get("journal", ""),
                    "pub_date": m.get("pub_date", ""),
                    "passage": m.get("passage", ""),
                }
            )

        ranked = hybrid_rerank(cands, q, alpha=0.7, fuzzy_threshold=85)
        deduped = dedup_by_paper(ranked)
        results = deduped[:5]

        print("\n=== Top results (semantic • hybrid • dedup • fuzzy) ===")
        for rank, r in enumerate(results, 1):
            print(f"\n[{rank}] 📖 {r['title'] or '(title unavailable)'}")
            print(
                f"    ⭐ cos={r.get('score',0):.4f} • kw={r.get('kw_overlap',0):.2f} • hybrid={r.get('hybrid_score',0):.4f} • merged_passages={r.get('merged_passages',0)}"
            )
            print(f"    🏷️  PMID: {r['pmid']}  •  {r['journal']}  •  {r['pub_date']}")
            snippet = r["passage"][:280] + ("..." if len(r["passage"]) > 280 else "")
            print(f"    📄 {snippet}")


if __name__ == "__main__":
    main()
