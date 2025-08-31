import json
import re
import unicodedata
from html import unescape
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from bs4 import BeautifulSoup
from tqdm import tqdm

# --------- Paths ---------
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PASSAGES_DIR = PROCESSED_DIR / "passages"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
PASSAGES_DIR.mkdir(parents=True, exist_ok=True)

# --------- Optional spaCy (lemmatization, sentencizer) ---------
_SPACY_OK = False
try:
    import spacy

    try:
        _nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
        _nlp.enable_pipe("senter") if "senter" in _nlp.pipe_names else None
        _SPACY_OK = True
    except Exception:
        # Model not present; weâ€™ll fallback later
        _SPACY_OK = False
except Exception:
    _SPACY_OK = False


def _strip_html(text: str) -> str:
    """Remove HTML/XML tags safely with BeautifulSoup (keeps text)."""
    if not text:
        return ""
    try:
        return BeautifulSoup(text, "html.parser").get_text(" ", strip=True)
    except Exception:
        # fallback: crude tag removal
        return re.sub(r"<.*?>", " ", text)


def _normalize_unicode(text: str) -> str:
    """Normalize different unicode forms & entities (â€“ â†’ -, smart quotes, etc.)."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = unescape(text)  # &nbsp; â†’ space, &amp; â†’ &
    # Replace fancy dashes/quotes with ASCII-ish equivalents
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    return text


def _remove_citations_boilerplate(text: str) -> str:
    """Remove patterns like [1], (2), reference markers, figure/table hints."""
    if not text:
        return ""
    # Common numeric citations
    text = re.sub(r"\[\s*\d+\s*\]", " ", text)  # [1]
    text = re.sub(r"\(\s*\d+\s*\)", " ", text)  # (2)
    text = re.sub(r"\[\s*\d+\s*-\s*\d+\s*\]", " ", text)  # [3-5]
    # Author-year-ish (light touch to avoid over-deleting)
    text = re.sub(r"\([A-Z][A-Za-z-]+,?\s*\d{4}\)", " ", text)
    # Figure/table artifacts
    text = re.sub(r"\b(Fig(ure)?|Table)\s*\d+[A-Za-z]?\b", " ", text, flags=re.I)
    return text


def _basic_token_cleanup(text: str) -> str:
    """Keep letters, numbers, and sentence punctuation; collapse spaces."""
    if not text:
        return ""
    # Allow .,!?;:()- for readability; drop other symbols
    text = re.sub(r"[^A-Za-z0-9\.\,\!\?\;\:\-\(\)\'\"\s]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text(text: str, lemmatize: bool = True) -> str:
    """
    Advanced cleaner:
      - HTML removal, unicode normalization, boilerplate removal,
        punctuation/whitespace normalization
      - Optional lemmatization via spaCy (if available)
    """
    if not text:
        return ""

    text = _strip_html(text)
    text = _normalize_unicode(text)
    text = _remove_citations_boilerplate(text)
    text = _basic_token_cleanup(text)

    if lemmatize and _SPACY_OK:
        # Lemmatize and drop stopwords if desired (keep stopwords to maintain phrase shape)
        doc = _nlp(text)
        # Conservative: lemmatize but keep stopwords
        text = " ".join(
            tok.lemma_ if tok.lemma_ != "-PRON-" else tok.text for tok in doc
        )
        text = re.sub(r"\s+", " ", text).strip().lower()
    else:
        # If no spaCy, just lowercase
        text = text.lower()

    return text


def split_sentences(text: str) -> List[str]:
    """
    Sentence segmentation:
      - use spaCy if available; else fallback to regex splitter.
    """
    if not text:
        return []
    if _SPACY_OK:
        doc = _nlp(text)
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        return sents
    # Simple regex fallback
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def dedup_sentences(sents: List[str]) -> List[str]:
    """Remove exact & trivial near-duplicates by normalized form."""
    seen = set()
    out = []
    for s in sents:
        key = re.sub(r"\s+", " ", s.lower()).strip()
        if key and key not in seen:
            seen.add(key)
            out.append(s)
    return out


def process_record(rec: Dict) -> Optional[Dict]:
    """Clean a single paper record. Returns None if unusable."""
    title = clean_text(rec.get("title", ""))
    abstract = clean_text(rec.get("abstract", ""))

    if not abstract and not title:
        return None

    out = {
        "pmid": rec.get("pmid"),
        "title": title,
        "abstract": abstract,
        "journal": clean_text(rec.get("journal", ""), lemmatize=False),
        "pub_date": rec.get("pub_date", "NA"),
    }
    return out


def to_passages(
    rec: Dict, min_len: int = 25, max_len: int = 1200
) -> List[Tuple[str, int, str]]:
    """
    Split a cleaned record into sentence-level passages.
    Returns list of (pmid, idx, passage_text).
    """
    sents = split_sentences(rec.get("abstract", ""))
    sents = [s for s in sents if min_len <= len(s) <= max_len]
    sents = dedup_sentences(sents)
    passages = [(rec.get("pmid"), i, s) for i, s in enumerate(sents)]
    return passages


def process_file(raw_fp: Path):
    """
    Read a raw JSONL file and produce:
      - processed_*.jsonl (cleaned title/abstract/metadata)
      - passages/processed_passages_*.jsonl (sentence-level passages)
    """
    cleaned_records: List[Dict] = []
    all_passages: List[Dict] = []

    with raw_fp.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            cleaned = process_record(rec)
            if not cleaned:
                continue
            cleaned_records.append(cleaned)
            # passages
            for pmid, idx, sent in to_passages(cleaned):
                all_passages.append({"pmid": pmid, "sent_id": idx, "passage": sent})

    # Write cleaned records
    out_clean = PROCESSED_DIR / raw_fp.name.replace("pubmed_", "processed_advanced_")
    with out_clean.open("w", encoding="utf-8") as f:
        for r in cleaned_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Write sentence-level passages
    out_pass = PASSAGES_DIR / raw_fp.name.replace("pubmed_", "processed_passages_")
    with out_pass.open("w", encoding="utf-8") as f:
        for p in all_passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"âœ… {raw_fp.name}: {len(cleaned_records)} papers â†’ {out_clean.name}")
    print(
        f"âœ… {raw_fp.name}: {len(all_passages)} passages â†’ passages/{out_pass.name}"
    )


def main():
    raw_files = sorted(RAW_DATA_DIR.glob("pubmed_*.jsonl"))
    if not raw_files:
        print("âŒ No raw files found in data/raw/. Run ingestion first.")
        return

    print(
        f"Processing {len(raw_files)} raw file(s)... (spaCy={'ON' if _SPACY_OK else 'OFF'})"
    )
    for fp in tqdm(raw_files, desc="Files"):
        process_file(fp)


if __name__ == "__main__":
    main()
