import json
import re
from collections import defaultdict
from datetime import datetime

import faiss
import numpy as np
import streamlit as st
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Helper functions
# -----------------------------
def keyword_overlap_score(text: str, query: str) -> float:
    q_terms = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2]
    if not q_terms:
        return 0.0
    t = text.lower()
    hits = 0
    for w in q_terms:
        if w in t:
            hits += 1
        else:
            for token in re.findall(r"\w+", t):
                if fuzz.ratio(w, token) >= 85:  # fuzzy match
                    hits += 1
                    break
    return hits / len(q_terms)


def hybrid_rerank(candidates, query, alpha=0.7):
    out = []
    for c in candidates:
        kw = keyword_overlap_score(c["passage"], query)
        c["hybrid_score"] = alpha * float(c["score"]) + (1 - alpha) * kw
        out.append(c)
    out.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return out


def filter_results(
    results, year_min, year_max, include_terms, exclude_terms, journal_filter
):
    filtered = []
    for r in results:
        # Year filter
        year = None
        if r.get("pub_date"):
            try:
                year = int(r["pub_date"].split("-")[0])
            except (KeyError, ValueError, AttributeError):
                year = None
        if year and (year < year_min or year > year_max):
            continue

        # Journal filter
        if journal_filter:
            keep = False
            for kw in journal_filter:
                if kw.lower() in r.get("journal", "").lower():
                    keep = True
            if not keep:
                continue

        # Include terms filter
        if include_terms:
            if not any(kw.lower() in r["passage"].lower() for kw in include_terms):
                continue

        # Exclude terms filter
        if exclude_terms:
            if any(kw.lower() in r["passage"].lower() for kw in exclude_terms):
                continue

        filtered.append(r)
    return filtered


def highlight_text(text: str, query: str) -> str:
    """Highlight query terms (and fuzzy variants) in passage or summary text."""
    terms = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2]
    if not terms:
        return text

    def repl(match):
        word = match.group(0)
        lw = word.lower()
        for t in terms:
            if t == lw or fuzz.ratio(t, lw) >= 85:
                return f"<mark>{word}</mark>"
        return word

    return re.sub(r"\w+", repl, text, flags=re.IGNORECASE)


def dedup_by_paper(results):
    """Collapse multiple passages from the same paper into one entry with snippets."""
    grouped = defaultdict(list)
    meta = {}
    for r in results:
        key = (r["pmid"], r["title"])
        grouped[key].append(r["passage"])
        meta[key] = r
    deduped = []
    for key, passages in grouped.items():
        r = meta[key]
        r["snippets"] = passages[:3]  # keep first 3 passages
        deduped.append(r)
    return deduped


def summarize_results(results):
    """Naive extractive summary from top 3 snippets."""
    snippets = []
    for r in results[:3]:
        snippets.extend(r.get("snippets", [r["passage"]]))
    text = " ".join(snippets[:5])
    return text if text else "No summary available."


# -----------------------------
# Streamlit UI
# -----------------------------
st.title(" Paper Insight Engine")

mode = st.radio("Search mode", ["Semantic (FAISS)", "TF-IDF (keywords)"])
top_k = st.slider("Top results", 5, 20, 10)
alpha = st.slider("Hybrid  (semantic weight)", 0.0, 1.0, 0.7)

query = st.text_input(" Enter your search query")

# Sidebar filters
st.sidebar.header("Filters")
year_min, year_max = st.sidebar.slider(
    "Publication Year Range", 1990, datetime.now().year, (2015, datetime.now().year)
)
journal_kw = st.sidebar.text_input("Journal contains (comma separated)")
journal_filter = [j.strip() for j in journal_kw.split(",") if j.strip()]
include_kw = st.sidebar.text_input("Must include terms (comma separated)")
include_terms = [t.strip() for t in include_kw.split(",") if t.strip()]
exclude_kw = st.sidebar.text_input("Exclude terms (comma separated)")
exclude_terms = [t.strip() for t in exclude_kw.split(",") if t.strip()]

if query:
    # Load FAISS + metadata
    index = faiss.read_index("data/semantic/faiss.index")
    with open("data/semantic/meta.jsonl", "r", encoding="utf-8") as f:
        meta = [json.loads(line) for line in f]

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    if mode == "Semantic (FAISS)":
        qv = model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)
        sims, ids = index.search(qv, top_k * 20)
        sims, ids = sims[0], ids[0]
        candidates = []
        for score, idx in zip(sims, ids):
            if idx == -1:
                continue
            m = meta[idx]
            candidates.append(
                {
                    "score": float(score),
                    "pmid": m["pmid"],
                    "title": m.get("title", ""),
                    "journal": m.get("journal", ""),
                    "pub_date": m.get("pub_date", ""),
                    "passage": m.get("passage", ""),
                }
            )
        ranked = hybrid_rerank(candidates, query, alpha)
        results = filter_results(
            ranked, year_min, year_max, include_terms, exclude_terms, journal_filter
        )

    else:  # TF-IDF
        corpus = [r["passage"] for r in meta]
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)
        qv = vectorizer.transform([query])
        sims = cosine_similarity(qv, X).flatten()
        ranked_ids = sims.argsort()[::-1][: top_k * 20]
        results = []
        for idx in ranked_ids:
            m = meta[idx]
            results.append(
                {
                    "score": float(sims[idx]),
                    "pmid": m["pmid"],
                    "title": m.get("title", ""),
                    "journal": m.get("journal", ""),
                    "pub_date": m.get("pub_date", ""),
                    "passage": m.get("passage", ""),
                }
            )
        results = filter_results(
            results, year_min, year_max, include_terms, exclude_terms, journal_filter
        )

    # Deduplicate
    deduped = dedup_by_paper(results)

    # Summary
    summary = summarize_results(deduped)
    summary_hl = highlight_text(summary, query)
    st.subheader(" Summary")
    st.markdown(summary_hl, unsafe_allow_html=True)

    # Display results
    st.subheader("Top results")
    for i, r in enumerate(deduped[:top_k], 1):
        snippets_hl = " ".join(
            [highlight_text(s, query) for s in r.get("snippets", [])]
        )
        st.markdown(
            f"""
        **[{i}] {r['title']}**  
         PMID: {r['pmid']}  {r['journal']} {r['pub_date']}  
         {snippets_hl}  
        """,
            unsafe_allow_html=True,
        )
