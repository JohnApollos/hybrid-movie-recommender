# 👥 Team Contributions Log

**Project:** Hybrid Movie Recommender System  
**Course:** Recommender Systems Practical Study  
**Institution:** Jomo Kenyatta University of Agriculture and Technology (JKUAT)  
**Program:** BSc Data Science & Analytics  

---

## Task Allocation & Deliverables Matrix

The table below documents the task allocations, assigned seats, specific deliverables, evidence artifacts, and completion status for all 15 members and coordination roles.

| # | Name | Adm. No. | Seat / Task | Deliverable Description | Evidence Artifact (Repo / Doc Link) | Status |
|---|---|---|---|---|---|---|
| **L** | Group Leader | SCT211-0001/2022 | Project Lead & Coordination | Scope definition, task board, repo scaffolding, dependency setup, submission packaging | `README.md`, `LICENSE`, `requirements.txt`, repo root | ✅ Done |
| **1** | Team Member 1 | SCT211-0002/2022 | Seat 1: Data Acquisition | MovieLens download script, row count verification, directory skeleton, data dictionary | `data/download.py`, `docs/data_dictionary.md` | ✅ Done |
| **2** | Team Member 2 | SCT211-0003/2022 | Seat 2: Data Preparation & EDA | Exploratory data analysis, sparsity/rating distribution plots, cleaning, LOO + 80/20 splits | `notebooks/01_eda.ipynb`, `src/preprocessing.py`, `data/splits/` | ✅ Done |
| **3** | Team Member 3 | SCT211-0004/2022 | Seat 3: Baseline Recommenders | Popularity ranker, uniform random baseline ranker, top-10 contract outputs | `src/models/baselines.py`, `outputs/popularity_top10.csv`, `outputs/random_top10.csv` | ✅ Done |
| **4** | Team Member 4 | SCT211-0005/2022 | Seat 4: User-Based CF | User-user cosine similarity with mean centering, k-nearest neighbors tuning, top-10 contract | `src/models/user_cf.py`, `outputs/user_cf_top10.csv` | ✅ Done |
| **5** | Team Member 5 | SCT211-0006/2022 | Seat 5: Item-Based CF & SVD | Item-item adjusted cosine similarity, SVD latent factor matrix factorization, RMSE calculation | `src/models/item_cf.py`, `src/models/svd_mf.py`, `outputs/item_cf_top10.csv`, `outputs/svd_top10.csv`, `outputs/rmse_svd.csv`, `outputs/rmse_item_cf.csv` | ✅ Done |
| **6** | Team Member 6 | SCT211-0007/2022 | Seat 6: Content-Based Filtering | TF-IDF feature extraction over genres & user tags, cold-start seed matching functions | `src/models/content.py`, `outputs/content_top10.csv` | ✅ Done |
| **7** | Team Member 7 | SCT211-0008/2022 | Seat 7: Weighted Hybrid Engine | Normalization (z-score), validation grid search for model weights, cold-start fallback | `src/models/hybrid.py`, `outputs/hybrid_top10.csv`, `outputs/hybrid_weights.json` | ✅ Done |
| **8** | Team Member 8 | SCT211-0009/2022 | Seat 8: Deep Learning (NCF) | PyTorch Neural Collaborative Filtering (NeuMF: GMF + MLP layers), BCE implicit loss | `src/models/ncf.py`, `outputs/ncf_top10.csv`, `outputs/ncf_training_log.csv` | ✅ Done |
| **9** | Team Member 9 | SCT211-0010/2022 | Seat 9: NCF Tuning & Curves | Hyperparameter sweep (dimensions, learning rates, negative sampling), training curve plotting | `src/models/ncf_tuning.py`, `outputs/ncf_tuning_results.csv`, `outputs/ncf_training_curves.png` | ✅ Done |
| **10** | Team Member 10 | SCT211-0011/2022 | Seat 10: Evaluation Module | Core metrics evaluation suite: P@10, R@10, NDCG@10, RMSE computation across all models | `src/evaluate.py`, `outputs/metrics.csv` | ✅ Done |
| **11** | Team Member 11 | SCT211-0012/2022 | Seat 11: Benchmark Analysis | Comparison plots vs baselines, catalog coverage, popularity bias analysis, README update | `outputs/metrics_comparison.png`, `outputs/coverage_diversity.png`, `outputs/error_analysis.csv`, `README.md` | ✅ Done |
| **12** | Team Member 12 | SCT211-0013/2022 | Seat 12: Streamlit Deployment | Interactive Streamlit application, user picker, model switcher, live 👍/👎 feedback logger | `app/app.py`, `feedback.csv`, `.streamlit/config.toml` | ✅ Done |
| **13** | Team Member 13 | SCT211-0014/2022 | Seat 13: Cloud & Demo Video | Streamlit cloud configuration, mobile phone testing log, 3-minute backup demo video script | `docs/demo_video_storyboard.md`, `docs/mobile_testing_bug_log.md` | ✅ Done |
| **14** | Team Member 14 | SCT211-0015/2022 | Seat 14: Comprehensive Report | Academic report skeleton & compilation (12–15 pages, IEEE format, figures & tables) | `docs/report.md`, `docs/Group##_MovieRecommender_Report.pdf` | ✅ Done |
| **15** | Team Member 15 | SCT211-0016/2022 | Seat 15: Presentation & Slides | 12-slide presentation deck (16:9), presentation script, team Q&A defense document | `docs/slides.md`, `docs/Group##_MovieRecommender_Slides.pdf`, `docs/presentation_script.md`, `docs/qa_document.md` | ✅ Done |

---

## 10-Minute Menu Task Deliverables

In addition to seat assignments, the following focused agile menu deliverables were successfully completed:

1. **Menu 1 (Phone Testing & Bug Log):** Cross-device testing of Streamlit UI on iOS/Android; layout responsive validation (`docs/mobile_testing_bug_log.md`).
2. **Menu 2 (Qualitative Evaluation):** Human scoring of 20 model recommendations across diverse user archetypes (`outputs/qualitative_ratings.csv`).
3. **Menu 3 (Cold-Start Persona Profiles):** Creation of 5 realistic, multifaceted test user profiles with seed genres and movies (`app/demo_profiles.json`).
4. **Menu 4 (Academic References):** Compilation of 5 canonical IEEE references covering MovieLens, NCF, Matrix Factorization, Item-CF, and Recommender Systems Handbook (`docs/report.md`).
5. **Menu 5 (Report Quality Assurance):** Proofreading, citation cross-checking, table/figure captioning, and formatting compliance.
6. **Menu 6 (Architecture Visualization):** Creation of system architecture block diagram and data pipeline schematics (`app/app.py`, `docs/report.md`, `docs/slides.md`).
7. **Menu 7 (Ethics & Bias Deep Dive):** Empirical writeup on catalog coverage, popularity concentration, and filter bubble mitigation (`docs/report.md`).
8. **Menu 8 (Q&A Defense Coordination):** 15 technical questions and answers mapped to team owners (`docs/qa_document.md`).
9. **Menu 9 (Submission Protocol Confirmation):** Class representative liaison and final packaging checklist (`docs/submission_checklist.md`).
