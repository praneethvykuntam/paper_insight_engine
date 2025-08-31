import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

def plot_workflow():
    steps = [
        "Data Ingestion\n(PubMed)",
        "Cleaning /\nPreprocessing",
        "Passage\nSegmentation",
        "Embedding\n(SBERT)",
        "Vector Indexing\n(FAISS)",
        "Hybrid Reranking\n(Semantic + TF-IDF)",
        "Filters & UI\n(Streamlit + Summary)"
    ]

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.axis("off")

    # coordinates for boxes
    x_positions = range(len(steps))
    y = 0.5

    for i, step in enumerate(steps):
        # box
        ax.add_patch(plt.Rectangle((i, y), 0.9, 0.5, fill=False, edgecolor="black", linewidth=1.5))
        # text (smaller font size)
        ax.text(i + 0.45, y + 0.25, step, ha="center", va="center", fontsize=9, wrap=True)

        # arrow to next
        if i < len(steps) - 1:
            arrow = FancyArrowPatch(
                (i + 0.9, y + 0.25), (i + 1, y + 0.25),
                arrowstyle="->", mutation_scale=15, linewidth=1.2, color="tab:blue"
            )
            ax.add_patch(arrow)

    plt.tight_layout()
    plt.savefig("docs/images/figure4_workflow.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    plot_workflow()
