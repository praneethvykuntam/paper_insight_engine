import json
from pathlib import Path
import re

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

def clean_text(text: str) -> str:
    """Basic text cleaning: remove newlines, excessive spaces, special chars"""
    if not text:
        return ""
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def process_file(input_file: Path, output_file: Path):
    """Read a JSONL file, clean the abstracts, and save processed output"""
    processed = []
    with open(input_file, "r", encoding="utf-8") as infile:
        for line in infile:
            record = json.loads(line)
            record["title"] = clean_text(record.get("title", ""))
            record["abstract"] = clean_text(record.get("abstract", ""))
            processed.append(record)

    with open(output_file, "w", encoding="utf-8") as outfile:
        for record in processed:
            outfile.write(json.dumps(record) + "\n")

    print(f"✅ Processed {len(processed)} papers → {output_file}")

def main():
    # Process all raw pubmed files
    for raw_file in RAW_DATA_DIR.glob("pubmed_*.jsonl"):
        output_file = PROCESSED_DATA_DIR / raw_file.name.replace("pubmed_", "processed_")
        process_file(raw_file, output_file)

if __name__ == "__main__":
    main()
