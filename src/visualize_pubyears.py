import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Path to your processed PubMed JSONL file
DATA_PATH = Path("data/processed")
OUT_PATH = Path("docs/images")
OUT_PATH.mkdir(parents=True, exist_ok=True)

def load_data():
    files = list(DATA_PATH.glob("processed_advanced_*.jsonl"))
    if not files:
        raise FileNotFoundError("No processed_advanced_*.jsonl file found in data/processed/")
    records = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fin:
            for line in fin:
                records.append(json.loads(line))
    return pd.DataFrame(records)

def plot_pubyears(df):
    # Extract year from pub_date
    df["year"] = pd.to_datetime(df["pub_date"], errors="coerce").dt.year
    year_counts = df["year"].value_counts().sort_index()

    # Plot
    plt.figure(figsize=(8, 5))
    year_counts.plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title("Publication Year Distribution", fontsize=14)
    plt.xlabel("Year")
    plt.ylabel("Number of Papers")
    plt.xticks(rotation=45)
    plt.tight_layout()

    out_file = OUT_PATH / "figure1_overview.png"
    plt.savefig(out_file, dpi=150)
    print(f"✅ Saved: {out_file}")

if __name__ == "__main__":
    df = load_data()
    plot_pubyears(df)
