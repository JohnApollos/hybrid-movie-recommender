# 🎬 Hybrid Movie Recommender

A hybrid movie recommendation system combining **user-based & item-based collaborative
filtering, SVD matrix factorization, content-based filtering (genres + tags), and Neural
Collaborative Filtering** — trained on MovieLens, evaluated against baselines, and deployed
as an interactive Streamlit web app.

> Recommender Systems Practical Study — group project, BSc Data Science & Analytics, JKUAT.

## ✨ Features
- **User-based CF** — recommendations from similar users' tastes
- **Item-based CF** — recommendations from similar items
- **SVD** — latent-factor rating prediction
- **Content-based** — TF-IDF over genres & user tags
- **Weighted hybrid** — tuned blend of the above (core deliverable)
- **NCF** — neural collaborative filtering (PyTorch)
- **Cold-start fallback** — new users get content-based recommendations from a stated profile
- **Evaluation suite** — Precision@10, Recall@10, NDCG@10 (+ RMSE for rating predictors) vs. popularity/random baselines
- **Web app** — Streamlit UI with 👍/👎 feedback logging

## 🏗 Architecture
```
MovieLens (ratings, genres, tags)
        │
        ▼
Preprocessing + leave-one-out split
        │
        ├──► User-CF ──┐
        ├──► Item-CF ──┤
        ├──► SVD ──────┼──► Weighted Hybrid ──► Top-10 lists ──► Streamlit app
        └──► Content ──┘         ▲                    │
                │                │                    ▼
                └── cold-start fallback        Evaluation (P@10, R@10, NDCG@10)
        NCF ──────────────────────────────────► evaluated alongside
```

## 📊 Dataset
**MovieLens ml-latest-small** — 100,836 ratings, 9,742 movies, 610 users, plus genres & tags.

Per GroupLens terms, the data is **not redistributed in this repo**. Fetch it with:

**Recommended (works on Windows / macOS / Linux / Colab):**
```bash
python data/download.py
```

**Manual fallback:** download
https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
and extract into `data/ml-latest-small/`.
If GroupLens is unreachable, use the official Kaggle mirror: `grouplens/movielens-latest-small`.

**Verify after extraction:** `ratings.csv` 100,836 rows · `movies.csv` 9,742 · `tags.csv` 3,683 · `links.csv` 9,742.

Citation: Harper, F. M., & Konstan, J. A. (2015). *The MovieLens Datasets: History and Context.*
ACM Transactions on Interactive Intelligent Systems, 5(4).

## 📁 Repository Structure
```
├── data/                  # MovieLens files (gitignored — download locally)
├── notebooks/             # EDA + experiments
├── src/
│   ├── preprocessing.py   # cleaning + train/test split
│   ├── models/
│   │   ├── baselines.py   # popularity + random
│   │   ├── user_cf.py
│   │   ├── item_cf.py
│   │   ├── svd_mf.py
│   │   ├── content.py     # TF-IDF genres + tags
│   │   ├── hybrid.py      # weighted blend + cold-start fallback
│   │   └── ncf.py         # neural collaborative filtering
│   └── evaluate.py
├── outputs/               # top-10 CSVs + metrics (committed — powers the app)
├── app/app.py             # Streamlit UI + feedback logging
├── docs/                  # report, slides, contributions
└── requirements.txt
```

## 🔌 Integration Contract (read before coding!)
Squads work independently; files are the interface. **Every model script must write:**

`outputs/{model}_top10.csv` with columns:
```
userId, movieId, rank, score
```
where `{model}` ∈ `popularity, random, user_cf, item_cf, svd, content, hybrid, ncf`.

`src/evaluate.py` writes `outputs/metrics.csv`:
```
model, precision_at_10, recall_at_10, ndcg_at_10, rmse
```
The app and evaluation read **only these files** — never import another squad's code.

**Evaluation protocol:** leave-one-out (each user's latest interaction held out);
a held-out item is *relevant* if its rating ≥ 4.0. RMSE uses an 80/20 random rating split.

## 🚀 Getting Started
**Option A — Google Colab (recommended, zero setup):** open a notebook from `notebooks/`,
`pip install scikit-surprise`, mount Drive or clone this repo.

**Option B — local:**
```bash
git clone https://github.com/JohnApollos/hybrid-movie-recommender.git
cd hybrid-movie-recommender
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# download the dataset (see above)
```
> ⚠️ If `scikit-surprise` fails to install: `pip install "numpy<2"` first, then retry
> (on Colab, restart the session afterwards).

**Run the pipeline:**
```bash
python src/preprocessing.py
python src/models/hybrid.py        # or any model script
python src/evaluate.py
streamlit run app/app.py
```

## 📈 Results
| Model | P@10 | R@10 | NDCG@10 | RMSE |
|---|---|---|---|---|
| Popularity (baseline) | – | – | – | – |
| User-based CF | – | – | – | – |
| Item-based CF | – | – | – | – |
| SVD | – | – | – | – |
| Content-based | – | – | – | – |
| **Hybrid** | – | – | – | – |
| NCF | – | – | – | – |

*Table populated from `outputs/metrics.csv` after the evaluation run.*

## ⚖️ Ethics & Limitations
- **Popularity bias:** CF tends to over-recommend blockbusters; we report catalog coverage
  alongside accuracy.
- **Filter bubbles:** the hybrid's content component can narrow exposure; diversity is
  measured, not assumed.
- **Privacy:** MovieLens uses anonymized IDs; the app collects only optional 👍/👎 feedback
  (logged to `feedback.csv` for future re-ranking) — no personal data.
- **Data license:** MovieLens is used under GroupLens' research terms and not redistributed.

## 👥 Team
| Squad | Members |
|---|---|
| Data | |
| Collaborative Filtering | |
| Content + Hybrid | |
| Deep Learning | |
| Evaluation | |
| App | |
| Report + Slides | |

Full contribution log: `docs/contributions.md`.

## 📄 License
Code: MIT. Data: GroupLens research terms (see Dataset section).
