import json
import os
from pathlib import Path
from time import sleep
from typing import List

from Bio import Entrez

# Load credentials from env
Entrez.email = os.getenv("NCBI_EMAIL")
Entrez.api_key = os.getenv("NCBI_API_KEY")

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_papers(
    query: str, max_results: int = 2000, batch_size: int = 500
) -> List[dict]:
    """Fetch PubMed papers for a given query in batches."""
    results = []
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, usehistory="y")
    search_results = Entrez.read(handle)
    handle.close()

    webenv = search_results["WebEnv"]
    query_key = search_results["QueryKey"]
    total_count = int(search_results["Count"])
    print(
        f"🔍 Query: {query} → Found {total_count} results (fetching up to {max_results})"
    )

    fetched = 0
    while fetched < min(total_count, max_results):
        batch_size = min(batch_size, max_results - fetched)
        handle = Entrez.efetch(
            db="pubmed",
            rettype="abstract",
            retmode="xml",
            retstart=fetched,
            retmax=batch_size,
            webenv=webenv,
            query_key=query_key,
        )
        records = Entrez.read(handle)
        handle.close()

        for article in records["PubmedArticle"]:
            pmid = article["MedlineCitation"]["PMID"]
            art = article["MedlineCitation"]["Article"]

            title = art.get("ArticleTitle", "")
            abstract = ""
            if "Abstract" in art and "AbstractText" in art["Abstract"]:
                abstract = " ".join(str(t) for t in art["Abstract"]["AbstractText"])

            journal = art["Journal"]["Title"] if "Journal" in art else ""
            pub_date = art["Journal"]["JournalIssue"].get("PubDate", {}).get("Year", "")

            results.append(
                {
                    "pmid": str(pmid),
                    "title": str(title),
                    "abstract": str(abstract),
                    "journal": str(journal),
                    "pub_date": str(pub_date),
                }
            )

        fetched += batch_size
        print(f"   ✅ Retrieved {fetched}/{min(total_count, max_results)}")
        sleep(0.34)  # API rate limit (NCBI allows ~3 requests/sec)

    return results


def save_results(query: str, papers: List[dict]):
    safe_q = query.lower().replace(" ", "_")
    out_fp = DATA_DIR / f"pubmed_{safe_q}.jsonl"
    with out_fp.open("w", encoding="utf-8") as f:
        for rec in papers:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"💾 Saved {len(papers)} papers → {out_fp}")


def main():
    queries = [
        "hallucination mitigation LLM",
        "retrieval augmented generation",
        "sentiment analysis",
    ]
    for q in queries:
        papers = fetch_papers(q, max_results=1000)  # adjust up if needed
        save_results(q, papers)


if __name__ == "__main__":
    main()
