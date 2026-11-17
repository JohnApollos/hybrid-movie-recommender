import streamlit as st

from page_common import apply_cineverse_style


apply_cineverse_style(
    "How CineVerse works",
    "A visual overview of the data, recommendation models, evaluation, and product layer.",
)

st.markdown(
    """
    ```text
    MovieLens metadata and ratings
                    │
                    ▼
       preprocessing and train/test splits
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
     User-CF     Item-CF       SVD
        └───────────┬───────────┘
                    ▼
      Content features + weighted hybrid
                    │
                    ▼
          Published top-10 CSV outputs
                    │
                    ▼
      CineVerse cards, posters, feedback,
      cold-start discovery, and benchmarks
    ```

    ### Recommendation layers

    - **Collaborative filtering:** learns relationships between users and movies.
    - **Content-based filtering:** uses genres and tags for cold-start discovery.
    - **Hybrid ranking:** combines complementary model signals.
    - **NCF:** provides a neural collaborative-filtering comparison.
    - **Evaluation:** reports Precision@10, Recall@10, NDCG@10, RMSE, coverage, and diversity.
    """
)
