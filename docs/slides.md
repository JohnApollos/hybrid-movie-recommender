# 🎬 Presentation Slides: Hybrid Movie Recommender System

**Format:** 16:9 Landscape Widescreen | Light Background / Dark Text (Projector-Safe)  
**Target Duration:** 12 Minutes (9 Minutes Presentation + 3 Minutes Live App Demo)  
**Course:** Recommender Systems Practical Study, BSc Data Science & Analytics, JKUAT  
**Group:** Group 04  

---

### Slide 1: Title Slide
- **Title:** Hybrid Movie Recommendation System
- **Subtitle:** Practical Study on Collaborative Filtering, Latent Factors, Content Metadata, and Deep Learning
- **Institution:** Jomo Kenyatta University of Agriculture and Technology (JKUAT)
- **Program:** BSc Data Science & Analytics | Department of Computing & Data Science
- **Team:** Group 04 (15 Project Members)
- **Date:** November 2026

---

### Slide 2: Problem Statement & Objectives
- **The Challenge:** Digital streaming platforms face extreme catalog size and choice paralysis.
- **Sparsity Barrier:** Real-world interaction matrices are over 98% sparse; users see <2% of catalog.
- **Cold-Start Failure:** Pure collaborative filtering cannot recommend for newly registered users.
- **Popularity Trap:** Basic algorithms over-recommend blockbusters, creating cultural echo chambers.
- **Project Objective:** Build, tune, and evaluate an integrated hybrid recommender (CF + SVD + Content + NCF) with live web deployment and feedback logging.

---

### Slide 3: Dataset Architecture & Exploratory Insights
- **Dataset:** MovieLens 100k (`ml-latest-small`) — 100,836 ratings, 9,742 movies, 610 users.
- **Matrix Sparsity:** 98.30% sparse (only 1.70% of potential interaction matrix observed).
- **Rating Distribution:** Median rating is 3.5 ★; **48.2% of ratings are $\ge 4.0$** (ground-truth relevance threshold).
- **Extreme Long-Tail:** Top 100 movies command 16.1% of ratings; 62.5% of movies have fewer than 5 ratings.
- **Metadata Availability:** Genres present for 99.7% of movies; tags are sparse (16.1% of movies).

---

### Slide 4: System Architecture & Workflow Pipeline
- **Decoupled Architecture:** Autonomous squad modules reporting through standard top-10 contracts.
- **Data Preprocessing:** Strict temporal Leave-One-Out split (ranking) + 80/20 random split (RMSE).
- **Constituent Engines:**
  - *Memory CF:* User-Based CF (Cosine) + Item-Based CF (Adjusted Cosine).
  - *Latent Factor:* SVD Matrix Factorization (100 factors, user/item biases).
  - *Content Engine:* TF-IDF feature space over combined genres and user tags.
  - *Deep Learning:* PyTorch NeuMF (GMF + MLP branches).
- **Weighted Ensemble:** Per-user z-score normalization + validation-tuned linear blend.
- **Serving Layer:** Interactive Streamlit web interface with persistent feedback capture.

---

### Slide 5: Classical Collaborative Filtering & SVD
- **User-Based CF:**
  - Cosine similarity on mean-centered ratings ($R - \bar{r}_u$).
  - $k=50$ nearest neighbors with similarity shrinkage $\beta=5.0$ to damp small overlaps.
- **Item-Based CF:**
  - Adjusted cosine similarity accounting for user rating scales.
  - Co-rater shrinkage $\lambda_{\text{sim}}=10.0$ and score shrinkage $\gamma=1.0$ for stable neighborhoods.
- **SVD Matrix Factorization:**
  - Decomposes matrix into 100 latent factors + global, user, and item biases.
  - Minimizes regularized squared error via SGD over 20 epochs.
  - **Achieved best rating reconstruction accuracy: RMSE = 0.8831.**

---

### Slide 6: Content-Based Filtering & Weighted Hybrid (Core Deliverable)
- **Content-Based Engine (TF-IDF):**
  - Sublinear TF-IDF vectors combining pipe-delimited genres and user-applied tags.
  - User profiles formed as rating-weighted centroids of watched movies.
  - Achieves highest catalog exploration (**21.68% catalog coverage**).
- **Weighted Hybrid Strategy:**
  - Standardizes disparate raw model scores into per-user z-scores.
  - Grid search on validation split ($N_{\text{val}}=5$) avoids test data leakage.
  - Optimal weights: **$w_{\text{user}} = 0.60, \; w_{\text{item}} = 0.40$** (Validation Recall = 0.0913).
  - Synergistic blend balances peer taste patterns with item-item coherence.

---

### Slide 7: Deep Learning: Neural Collaborative Filtering (NCF)
- **Architecture:** PyTorch NeuMF combining Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP).
- **Non-Linear Interactions:** MLP learns non-linear latent feature combinations via 3 hidden dense layers.
- **Implicit Training:** BCE loss with 4 negative samples per positive interaction.
- **Hyperparameter Tuning:** Swept 7 configurations across dimensions, dropout, learning rate, and negatives.
- **Key Finding:** Compact architecture (dim=16, layers 32-16-8, 10% dropout) prevented overfitting, achieving **Validation Recall = 0.0824** at epoch 16.

---

### Slide 8: Evaluation Protocol & Rigorous Metrics
- **Protocol:** Chronological Leave-One-Out (LOO) holding out each user's latest interaction.
- **Relevance Criteria:** Held-out movie must have true rating **$\ge 4.0$** (363 relevant evaluation users).
- **Evaluation Dimensions:**
  - *Recall@10:* Proportion of relevant held-out items recovered in top-10.
  - *Precision@10:* Proportion of top-10 recommendations that are relevant hits.
  - *NDCG@10:* Normalized Discounted Cumulative Gain with logarithmic rank discount.
  - *RMSE:* Rating prediction error on independent 80/20 test split.
  - *Catalog Coverage:* % of unique catalog items recommended across all users.

---

### Slide 9: Experimental Results vs. Baselines
- **Results Summary Table:**
  - *Popularity:* Recall@10 = 0.0634, NDCG@10 = 0.0299, Coverage = 1.25%
  - *Random:* Recall@10 = 0.0000, NDCG@10 = 0.0000, Coverage = 46.19%
  - *User-CF:* Recall@10 = 0.0909, NDCG@10 = 0.0459, RMSE = 0.9227
  - *Item-CF:* Recall@10 = 0.0661, NDCG@10 = 0.0411, RMSE = 0.8922
  - *SVD (Latent):* Recall@10 = 0.0386, NDCG@10 = 0.0209, **RMSE = 0.8831**
  - *Content (TF-IDF):* Recall@10 = 0.0165, NDCG@10 = 0.0081, **Coverage = 21.68%**
  - **Weighted Hybrid:** **Recall@10 = 0.0964**, **Precision@10 = 0.0096**, **NDCG@10 = 0.0521**
  - *NCF (PyTorch):* Recall@10 = 0.0744, NDCG@10 = 0.0377, Coverage = 8.49%
- **Outcome:** The Weighted Hybrid decisively beats all individual models in ranking accuracy.

---

### Slide 10: Live Demonstration (Web App & Cold-Start)
- **Live Streamlit Interface:** Interactive deployment at `localhost:8501`.
- **Existing User Recommender:**
  - Select User #10 $\rightarrow$ Inspect user taste profile (140 ratings, Action/Crime).
  - Compare model outputs: Popularity vs SVD vs Weighted Hybrid.
  - Live 👍/👎 feedback logged to persistent `feedback.csv`.
- **Cold-Start Onboarding Engine:**
  - New user with zero history $\rightarrow$ Selects "Sci-Fi Fan" persona or custom genres.
  - Instant Content-based TF-IDF generates genre-aligned recommendations in <40ms.
- **Model Benchmarks Dashboard:** Real-time metrics tables, comparison charts, and tuning curves.

---

### Slide 11: Ethical Considerations, Bias & Limitations
- **Popularity Bias Mitigation:**
  - Popularity baseline allocates 56.8% of recommendation slots to top-10 blockbusters.
  - Hybrid expands catalog coverage to 401 unique movies (4.12%), tripling catalog discovery.
- **Filter Bubble Prevention:** Blending collaborative signals prevents narrow metadata trapping.
- **Privacy by Design:** Anonymized user IDs (no PII); local feedback logging prevents data tracking.
- **Limitations & Future Work:**
  - Current models evaluate static interactions $\rightarrow$ Future work: Sequential models (SASRec/GRU4Rec).
  - Graph Neural Networks (LightGCN) for high-order bipartite graph connectivity.

---

### Slide 12: Conclusion & Q&A
- **Summary:** Delivered a fully functional, validated end-to-end Hybrid Recommender System.
- **Key Takeaway:** Combining memory-based CF with item similarity achieves state-of-the-art ranking (Recall@10 = 0.0964), while content fallback provides instant cold-start resilience.
- **Open-Source Codebase:** Full source code, test splits, notebooks, and models on GitHub.
- **Thank You!** We now invite questions and feedback from the panel.
