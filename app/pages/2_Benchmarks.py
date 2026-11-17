from pathlib import Path

import pandas as pd
import streamlit as st

from page_common import OUTPUTS_DIR, apply_cineverse_style, load_metrics


apply_cineverse_style(
    "Model benchmarks",
    "Compare ranking quality, coverage, diversity, and neural training behavior across CineVerse models.",
)

metrics = load_metrics()
if metrics.empty:
    st.warning("No evaluation metrics are available yet. Run the evaluation pipeline first.")
else:
    labels = {
        "popularity": "Popularity baseline",
        "random": "Random baseline",
        "user_cf": "User-based collaborative filtering",
        "item_cf": "Item-based collaborative filtering",
        "svd": "SVD matrix factorization",
        "content": "Content-based filtering",
        "hybrid": "Weighted hybrid",
        "ncf": "Neural collaborative filtering",
    }
    display = metrics.copy()
    display["Model"] = display["model"].map(labels).fillna(display["model"])
    columns = [column for column in ["Model", "precision_at_10", "recall_at_10", "ndcg_at_10", "rmse"] if column in display]
    st.dataframe(display[columns], hide_index=True, width="stretch")

for filename, caption in [
    ("metrics_comparison.png", "Ranking metrics across models"),
    ("coverage_diversity.png", "Catalog coverage and popularity concentration"),
    ("ncf_training_curves.png", "NCF training loss and validation Recall@10"),
]:
    path = OUTPUTS_DIR / filename
    if path.is_file():
        st.image(str(path), caption=caption, width="stretch")

qualitative = OUTPUTS_DIR / "qualitative_ratings.csv"
if qualitative.is_file():
    with st.expander("Qualitative evaluation of recommendations"):
        st.dataframe(pd.read_csv(qualitative), hide_index=True, width="stretch")
