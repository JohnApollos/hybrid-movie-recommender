# Hybrid Movie Recommendation System: An End-to-End Practical Study on Collaborative Filtering, Latent Factor Modeling, Content-Based Metadata, and Deep Neural Networks

**Course:** Recommender Systems Practical Study (BSc Data Science & Analytics)  
**Institution:** Jomo Kenyatta University of Agriculture and Technology (JKUAT)  
**Department:** Department of Computing & Data Science  
**Date:** November 2026  
**Group Number:** Group 04  

**Authors & Project Contributors:**
1. Group Leader (Lead Coordinator, Architecture & Deployment) – SCT211-0001/2022
2. Team Member 1 (Data Acquisition, Validation & Dictionary) – SCT211-0002/2022
3. Team Member 2 (Data Preprocessing, Cleaning & EDA) – SCT211-0003/2022
4. Team Member 3 (Baseline Modeling: Popularity & Random) – SCT211-0004/2022
5. Team Member 4 (User-Based Collaborative Filtering) – SCT211-0005/2022
6. Team Member 5 (Item-Based CF & SVD Matrix Factorization) – SCT211-0006/2022
7. Team Member 6 (Content-Based TF-IDF & Metadata Filtering) – SCT211-0007/2022
8. Team Member 7 (Weighted Hybrid Engine & Cold-Start Fallback) – SCT211-0008/2022
9. Team Member 8 (Neural Collaborative Filtering in PyTorch) – SCT211-0009/2022
10. Team Member 9 (NCF Hyperparameter Tuning & Training Dynamics) – SCT211-0010/2022
11. Team Member 10 (Evaluation Protocol & Metrics Module) – SCT211-0011/2022
12. Team Member 11 (Empirical Benchmark Analysis & Catalog Coverage) – SCT211-0012/2022
13. Team Member 12 (Interactive Streamlit Web Application & UI) – SCT211-0013/2022
14. Team Member 13 (Deployment Engineering & Video Demonstration) – SCT211-0014/2022
15. Team Member 14 (Technical Documentation & Academic Report Compilation) – SCT211-0015/2022
16. Team Member 15 (Presentation Slide Deck, Script & Defense Curation) – SCT211-0016/2022

**Course Lecturer:** Department of Computing & Data Science, JKUAT  

---

## Abstract
Modern digital streaming platforms require recommendation algorithms capable of accurately estimating user preference while navigating extreme matrix sparsity, cold-start limitations, and algorithmic popularity bias. This study presents an end-to-end implementation and comparative evaluation of an adaptive hybrid movie recommender system trained on the benchmark MovieLens 100k dataset (100,836 ratings, 9,742 movies, 610 users). We systematically formulate, implement, and benchmark eight distinct algorithms: Popularity and Uniform Random baselines, User-based Collaborative Filtering (User-CF), Item-based Collaborative Filtering (Item-CF), Singular Value Decomposition (SVD) matrix factorization, Content-based Filtering via TF-IDF over genres and user-generated tags, a validation-tuned Weighted Hybrid, and a deep Neural Collaborative Filtering (NCF/NeuMF) architecture implemented in PyTorch. 

Under a rigorous chronological Leave-One-Out (LOO) ranking protocol with a strict relevance threshold ($\text{rating} \ge 4.0$), our Weighted Hybrid achieves superior performance across all ranking dimensions with a Recall@10 of **0.0964**, Precision@10 of **0.0096**, and NDCG@10 of **0.0521**, outperforming the best individual collaborative model (User-CF, Recall@10 = 0.0909) and deep learning baseline (NCF, Recall@10 = 0.0744). For explicit rating reconstruction, SVD achieves the lowest Root Mean Squared Error (RMSE = **0.8831**) on an independent 80/20 test split. Furthermore, our empirical catalog coverage audit demonstrates that the Hybrid expands unique catalog exploration to 401 movies (4.12%), effectively countering the severe popularity concentration of pure popularity (56.8% top-10 slot hogging). The entire pipeline is operationalized within an interactive Streamlit web application featuring instant cold-start fallback onboarding and real-time 👍/👎 feedback logging.

---

## Table of Contents
1. **Introduction**
   - 1.1 Problem Statement & Background
   - 1.2 Study Objectives
   - 1.3 The Rationale for a Hybrid Architecture
2. **Data Acquisition, Preprocessing & Exploratory Analysis**
   - 2.1 Dataset Description & Academic Licensing
   - 2.2 Exploratory Data Analysis & Empirical Observations
   - 2.3 Data Preprocessing & Cleaning Pipeline
   - 2.4 Splitting Protocols: Leave-One-Out & 80/20 RMSE Splits
3. **Methodology & Algorithmic Formulation**
   - 3.1 End-to-End System Architecture
   - 3.2 Baseline Models (Popularity & Random)
   - 3.3 User-Based Collaborative Filtering (User-CF)
   - 3.4 Item-Based Collaborative Filtering (Item-CF)
   - 3.5 Latent Factor Matrix Factorization (SVD)
   - 3.6 Content-Based Filtering (TF-IDF Metadata Engine)
   - 3.7 The Weighted Hybrid Recommender
   - 3.8 Deep Learning: Neural Collaborative Filtering (NCF / NeuMF)
   - 3.9 Algorithmic Parameter Synthesis
4. **Experimental Evaluation & Results**
   - 4.1 Evaluation Protocol & Metrics Formulation
   - 4.2 Benchmark Results vs. Baselines
   - 4.3 Deep Learning Ablation & NCF Hyperparameter Tuning
   - 4.4 Cold-Start Fallback Empirical Evaluation
   - 4.5 Error Analysis Across User Activity & Item Tiers
   - 4.6 Qualitative Evaluation of Recommendations
5. **Deployment & Interactive Web Application**
   - 5.1 Streamlit Architectural Interface
   - 5.2 Real-Time User Feedback Logging Mechanism
   - 5.3 Cold-Start Onboarding Demonstration
6. **Ethical Implications, Bias Mitigation & Privacy**
   - 6.1 Popularity Bias & The Long-Tail Problem
   - 6.2 Filter Bubbles & Serendipity Management
   - 6.3 Data Privacy, Anonymization & Governance
7. **Limitations & Future Work**
8. **Conclusion**
- **References**
- **Appendix A: Detailed Contributions Matrix**
- **Appendix B: Project Data Dictionary**

---

## 1. Introduction

### 1.1 Problem Statement & Background
The exponential growth of digital media catalogs has fundamentally altered how consumers discover entertainment. Commercial video streaming platforms such as Netflix, Amazon Prime Video, and Disney+ host tens of thousands of media titles, creating severe choice overload for users. Recommender systems have emerged as the foundational infrastructure of the digital economy, filtering vast information spaces to provide personalized recommendations aligned with individual taste profiles [1].

In academic and industrial practice, recommender algorithms are traditionally bifurcated into collaborative filtering (CF) and content-based filtering (CBF). Collaborative filtering leverages the collective wisdom of crowds by analyzing interaction matrices, assuming that users who agreed on items in the past will agree in the future [3], [4]. Content-based filtering, conversely, recommends items possessing similar auxiliary attributes (such as genres, directors, and keywords) to those previously preferred by the target user [5]. Both approaches, however, suffer from intrinsic structural limitations:
- **Matrix Sparsity:** In typical commercial catalogs, users interact with fewer than 1% to 2% of available inventory, resulting in extreme matrix sparsity that impairs neighbor discovery in memory-based CF.
- **The Cold-Start Problem:** Pure collaborative filtering is structurally incapable of generating recommendations for newly registered users with zero interaction history, as no neighborhood similarity or latent factor projection can be computed.
- **Popularity Bias & Monoculture:** Collaborative models often over-index on blockbusters, concentrating recommendations on a narrow band of hit titles and starving niche content in the long tail.

### 1.2 Study Objectives
This study addresses these fundamental challenges by implementing an integrated, production-grade hybrid movie recommendation engine. Specifically, our objectives are:
1. **Data Engineering:** Ingest, audit, clean, and split the MovieLens 100k benchmark dataset using strict temporal leave-one-out and random 80/20 splits without data leakage.
2. **Multi-Paradigm Algorithmic Implementation:** Develop, tune, and evaluate eight distinct recommender paradigms spanning heuristic baselines, neighborhood collaborative filtering (User-CF, Item-CF), latent factor matrix factorization (SVD), metadata content filtering (TF-IDF on genres and tags), and deep learning (Neural Collaborative Filtering in PyTorch).
3. **Hybrid Optimization:** Engineer a z-score normalized Weighted Hybrid that programmatically optimizes model weights on validation data to surpass individual constituent models.
4. **Cold-Start Engineering:** Design and benchmark a seamless cold-start fallback mechanism that captures user taste via stated genres or liked seed movies.
5. **Empirical Evaluation & Ethical Audit:** Benchmark all models across Precision@10, Recall@10, NDCG@10, RMSE, catalog coverage, and popularity concentration metrics.
6. **Web Deployment:** Deploy an interactive web application with user taste inspection, live model switching, cold-start onboarding, and persistent 👍/👎 feedback logging.

### 1.3 The Rationale for a Hybrid Architecture
Hybrid recommendation systems combine the complementary strengths of collaborative and content-based approaches while mutually neutralizing their individual deficiencies [5]. Collaborative models excel at discovering cross-genre, serendipitous affinities that cannot be deduced from surface-level metadata alone. Conversely, content-based models provide robust item representations independent of user interaction counts, providing an immediate fallback for unrated items and cold-start users. By normalizing model scores to a common distribution and blending them through validation-tuned weights, our hybrid system achieves both high predictive accuracy and broader catalog exploration.

---

## 2. Data Acquisition, Preprocessing & Exploratory Analysis

### 2.1 Dataset Description & Academic Licensing
Our study utilizes the canonical **MovieLens ml-latest-small** dataset collected and maintained by GroupLens Research at the University of Minnesota [1]. The dataset contains **100,836 ratings** and **3,683 tag applications** across **9,742 movies** generated by **610 users** between March 29, 1996 and September 24, 2018. All users in the dataset have rated at least 20 movies, ensuring a minimum baseline of interaction history.

Per the GroupLens research license agreement, the raw data is utilized exclusively for non-commercial research purposes and is excluded from repository redistribution via `.gitignore`. The dataset is ingested dynamically via a verified download and extraction script (`data/download.py`).

### 2.2 Exploratory Data Analysis & Empirical Observations
Exploratory data analysis (documented in `notebooks/01_eda.ipynb`) revealed several critical structural characteristics:
- **Matrix Sparsity:** With 610 users and 9,724 unique movies represented in the rating matrix, the total possible interaction space is $610 \times 9,724 = 5,931,640$ entries. The actual observed rating count of 100,836 yields an interaction density of **1.70%** and a matrix sparsity of **98.30%**.
- **Rating Value Distribution:** Ratings span from 0.5 to 5.0 stars in half-star increments. The overall rating distribution is left-skewed with a mean of **3.50** and a median of **3.50**. Critically, **48.2%** of all ratings are $\ge 4.0$, while ratings below 3.0 represent fewer than 22% of entries.
- **Power-Law Item Distribution:** Item consumption exhibits extreme long-tail behavior. The top 100 most rated movies account for **16.1%** of all ratings in the catalog. The single most rated movie (*Forrest Gump*, 1994) received 329 ratings. In stark contrast, 3,446 movies (35.4%) received only 1 rating, and 6,074 movies (**62.5%**) received fewer than 5 ratings.
- **User Activity Skewness:** User participation ranges from the minimum threshold of 20 ratings up to 2,698 ratings, with a median of 70.5 ratings per user.
- **Metadata Availability:** All 9,742 movies feature genre classifications, with only 34 movies (0.35%) labeled as `(no genres listed)`. On average, movies possess 2.27 genres across 20 distinct categories (dominated by Drama, Comedy, Action, and Thriller). Free-form tags, however, are highly sparse: only 1,572 movies (16.1%) possess at least one tag, applied by just 58 users (9.5%).

### 2.3 Data Preprocessing & Cleaning Pipeline
Data cleaning and standardization steps implemented in `src/preprocessing.py` and `src/models/content.py` include:
1. **Null & Duplicate Handling:** Verification confirmed zero missing values across `ratings.csv` and zero duplicate `(userId, movieId)` interaction pairs. In `tags.csv`, entries missing text tags were dropped.
2. **Genre Normalization:** Genres are parsed from pipe-delimited strings and prefixed with `g_` (e.g. `g_scifi`, `g_action`) to prevent token collisions between genre classes and common tag words during TF-IDF tokenization.
3. **Tag Text Aggregation:** All tags assigned to a movie across different users are normalized to lowercase and aggregated into a unified space-delimited text string. Movies lacking tags receive an empty string, ensuring graceful fallback to genre-only representations.

### 2.4 Splitting Protocols: Leave-One-Out & 80/20 Splits
To ensure complete separation between evaluation modes and prevent data leakage, we prepared two independent data splits:
1. **Leave-One-Out (LOO) Ranking Split (`data/splits/train_loo.csv`, `test_loo.csv`):**
   For each of the 610 users, ratings are sorted chronologically by timestamp (with `movieId` as tie-breaker). The single latest interaction is extracted into the test set (610 rows), while the remaining 100,226 interactions constitute the training set. A test interaction is classified as *relevant* if and only if its rating is $\ge 4.0$ (resulting in **363 relevant evaluation users**).
2. **Random 80/20 Rating Split (`data/splits/train_rmse.csv`, `test_rmse.csv`):**
   To evaluate explicit rating prediction error (RMSE) without temporal ordering constraints, ratings are partitioned randomly into an 80% training set (80,668 rows) and a 20% test set (20,168 rows) using a fixed random seed (`seed=42`).

---

## 3. Methodology & Algorithmic Formulation

### 3.1 End-to-End System Architecture
The hybrid recommender follows a strictly decoupled architectural contract. Squad models execute independently against the training data and export standard top-10 candidate CSV files formatted as `userId, movieId, rank, score`. The Weighted Hybrid ingests these candidate outputs, performs user-level normalization, and synthesizes final blended recommendation lists.

```
┌────────────────────────────────────────────────────────────────────────┐
│                       MovieLens ml-latest-small                        │
│              (100,836 ratings · 9,742 movies · 610 users)              │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 Data Preprocessing & Train/Test Split                  │
│         Leave-One-Out (Latest Rating) + 80/20 Random RMSE Split        │
└───────┬───────────────────────────┬────────────────────────────┬───────┘
        │                           │                            │
        ▼                           ▼                            ▼
┌──────────────┐            ┌──────────────┐             ┌──────────────┐
│   User-CF    │            │   Item-CF    │             │    SVD MF    │
│  Cosine Sim  │            │  Adj. Cosine │             │Latent Factors│
└───────┬──────┘            └───────┬──────┘             └──────┬───────┘
        │                           │                           │
        └─────────────┬─────────────┴───────────────────────────┘
                      │
                      ▼
        ┌───────────────────────────┐         ┌───────────────────────────┐
        │   Content-Based Engine    │         │ Neural Collaborative Filt.│
        │  TF-IDF (Genres + Tags)   │         │    NeuMF (GMF + MLP)      │
        └─────────────┬─────────────┘         └─────────────┬─────────────┘
                      │                                     │
                      ▼                                     ▼
        ┌───────────────────────────┐         ┌───────────────────────────┐
        │      Weighted Hybrid      │         │     NCF Ranker Top-10     │
        │   Normalized Z-Score Blend│         │   BCE Implicit Loss     │
        └─────────────┬─────────────┘         └─────────────┬─────────────┘
                      │                                     │
                      └──────────────────┬──────────────────┘
                                         │
                                         ▼
                      ┌─────────────────────────────────────┐
                      │   Evaluation & Deployment Engine    │
                      │   Streamlit Web App + Feedback Log  │
                      └─────────────────────────────────────┘
```
*Figure 1: End-to-end architecture of the Hybrid Movie Recommendation System.*

### 3.2 Baseline Models (Popularity & Random)
To establish rigorous empirical benchmarks, two baseline rankers were implemented in `src/models/baselines.py`:
- **Popularity Baseline:** Ranks movies globally based on the total number of ratings received in the training split, breaking ties by `movieId`. For each user, already-rated movies are filtered out, and the top-10 most popular unconsumed movies are recommended.
- **Uniform Random Baseline:** For each user, candidate movies from the unrated catalog are sampled uniformly at random using pseudorandom generator seed 42.

### 3.3 User-Based Collaborative Filtering (User-CF)
Implemented in `src/models/user_cf.py`, User-CF identifies peers with historical rating patterns similar to the active user. Given rating matrix $R \in \mathbb{R}^{U \times I}$ and binary observation mask $B \in \{0, 1\}^{U \times I}$, user mean ratings are computed as $\bar{r}_u = \sum_i R_{u,i} / \sum_i B_{u,i}$. Mean-centered ratings are defined as:
$$R^c_{u,i} = (R_{u,i} - \bar{r}_u) \cdot B_{u,i}$$

Cosine similarity between users $u$ and $v$ is computed over mean-centered vectors:
$$\text{sim}(u, v) = \frac{\sum_{i} R^c_{u,i} R^c_{v,i}}{\sqrt{\sum_{i} (R^c_{u,i})^2} \sqrt{\sum_{i} (R^c_{v,i})^2}}$$

Negative similarities are clamped to zero. For each user, similarity is restricted to their top $k=50$ positive neighbors ($N_k(u)$). Predicted scores for unrated movies are computed with similarity shrinkage $\beta=5.0$:
$$\text{score}(u, i) = \frac{\sum_{v \in N_k(u)} \text{sim}(u, v) \cdot R^c_{v,i}}{\sum_{v \in N_k(u)} \text{sim}(u, v) \cdot B_{v,i} + \beta}$$
Movies previously rated by the user or unsupported by neighbors are assigned $-\infty$.

### 3.4 Item-Based Collaborative Filtering (Item-CF)
Implemented in `src/models/item_cf.py`, Item-CF computes similarities between items along user rating dimensions [4]. Adjusted cosine similarity between items $i$ and $j$ accounts for user rating scale discrepancies:
$$s_{i,j} = \frac{\sum_{u \in U_{i,j}} (R_{u,i} - \bar{r}_u)(R_{u,j} - \bar{r}_u)}{\sqrt{\sum_{u \in U_{i,j}} (R_{u,i} - \bar{r}_u)^2} \sqrt{\sum_{u \in U_{i,j}} (R_{u,j} - \bar{r}_u)^2}}$$

To penalize item pairs co-rated by very few users, similarity shrinkage is applied with $\lambda_{\text{sim}} = 10.0$:
$$s'_{i,j} = s_{i,j} \cdot \frac{|U_{i,j}|}{|U_{i,j}| + \lambda_{\text{sim}}}$$

For each movie, the top $k=50$ neighbors are retained in a sparse matrix $S$. Scores for unrated items are aggregated across neighbors the user has rated with score shrinkage $\gamma=1.0$:
$$\text{score}(u, i) = \frac{\sum_{j \in N_k(i)} s'_{i,j} \cdot R^c_{u,j}}{\sum_{j \in N_k(i)} s'_{i,j} \cdot B_{u,j} + \gamma}$$

For explicit rating prediction on the 80/20 test split, the predicted rating is reconstructed as:
$$\hat{r}_{u,i} = \text{clip}\left(\bar{r}_u + \frac{\sum_{j \in N_k(i)} s'_{i,j} (R_{u,j} - \bar{r}_u)}{\sum_{j \in N_k(i)} s'_{i,j} + \epsilon}, 0.5, 5.0\right)$$

### 3.5 Latent Factor Matrix Factorization (SVD)
Implemented in `src/models/svd_mf.py` using `scikit-surprise`, regularized SVD decomposes the interaction matrix into user latent factors $p_u \in \mathbb{R}^k$ and item latent factors $q_i \in \mathbb{R}^k$ ($k=100$) [3]. The predicted rating incorporates global mean $\mu$, user bias $b_u$, and item bias $b_i$:
$$\hat{r}_{u,i} = \mu + b_u + b_i + p_u^T q_i$$

The parameters are learned by minimizing regularized squared error via Stochastic Gradient Descent (SGD) over 20 epochs:
$$\min_{p, q, b} \sum_{(u,i) \in \mathcal{K}} (r_{u,i} - \hat{r}_{u,i})^2 + \lambda (\|p_u\|_2^2 + \|q_i\|_2^2 + b_u^2 + b_i^2)$$

For top-10 ranking, $\hat{r}_{u,i}$ is computed across all unrated movies and sorted in descending order.

### 3.6 Content-Based Filtering (TF-IDF Metadata Engine)
Implemented in `src/models/content.py`, the content engine constructs an $L_2$-normalized TF-IDF vector $\vec{x}_i$ for each movie from its combined genres and user tags with sublinear term frequency scaling:
$$\text{tf-idf}(t, d) = (1 + \log(\text{tf}(t, d))) \cdot \log\left(\frac{1 + |D|}{1 + \text{df}(t)}\right) + 1$$

User profile vectors $\vec{p}_u$ are formed as the rating-weighted centroid of their rated movie representations:
$$\vec{p}_u = \text{normalize}\left(\sum_{i \in I_u} (r_{u,i} - \bar{r}_u) \vec{x}_i\right)$$

The cosine affinity score between user profile $\vec{p}_u$ and candidate movie $\vec{x}_i$ is:
$$\text{score}(u, i) = \vec{p}_u \cdot \vec{x}_i^T + 10^{-6} \cdot \frac{\text{pop}_i}{\max(\text{pop})}$$
where a minuscule popularity tie-breaker resolves identical metadata vectors without distorting ranking.

### 3.7 The Weighted Hybrid Recommender
Implemented in `src/models/hybrid.py`, the Weighted Hybrid unifies User-CF, Item-CF, SVD, and Content models into an ensemble ranker. Because individual algorithms output scores on fundamentally disparate scales (e.g. SVD outputs stars [0.5, 5.0], User-CF outputs centered deviations [-4, +4], and Content outputs cosine similarities [0, 1]), raw score addition would cause arbitrary model dominance.

To enforce fair contribution, candidate scores for each model $m$ are z-score standardized per user across all unrated candidate movies:
$$Z_m(u, i) = \frac{S_m(u, i) - \mu_m(u)}{\sigma_m(u) + \epsilon}$$

The hybrid score is a linear combination of standardized scores:
$$S_{\text{hybrid}}(u, i) = \sum_{m \in M} w_m \cdot Z_m(u, i), \quad \text{where } \sum_{m} w_m = 1.0, \; w_m \ge 0$$

**Leakage-Free Weight Tuning:**  
Weights were optimized via a grid search ($0.0 \le w_m \le 1.0$, step 0.1) on an independent validation split comprising each user's last 5 training ratings ($N_{\text{val}}=5$). The validation objective maximized Recall@10 on relevant held-out items ($\text{rating} \ge 4.0$). The grid search converged to:
$$w_{\text{user}} = 0.60, \quad w_{\text{item}} = 0.40, \quad w_{\text{svd}} = 0.00, \quad w_{\text{content}} = 0.00$$
achieving a peak validation Recall@10 of **0.0913**. The sequestered test set was evaluated exactly once with these weights locked.

### 3.8 Deep Learning: Neural Collaborative Filtering (NCF / NeuMF)
Implemented in `src/models/ncf.py` in PyTorch, Neural Collaborative Filtering models non-linear user-item latent interactions [2]. NeuMF fuses a Generalized Matrix Factorization (GMF) branch with a Multi-Layer Perceptron (MLP) branch:

$$\mathbf{\phi}^{\text{GMF}} = \mathbf{p}_u^G \odot \mathbf{q}_i^G$$
$$\mathbf{\phi}^{\text{MLP}} = a_L\left(\mathbf{W}_L^T \dots a_1(\mathbf{W}_1^T [\mathbf{p}_u^M; \mathbf{q}_i^M] + \mathbf{b}_1) \dots + \mathbf{b}_L\right)$$
$$\hat{y}_{u,i} = \sigma\left(\mathbf{h}^T [\mathbf{\phi}^{\text{GMF}}; \mathbf{\phi}^{\text{MLP}}]\right)$$

The model is trained on implicit feedback with Binary Cross-Entropy (BCE) loss:
$$\mathcal{L} = -\sum_{(u,i) \in \mathcal{Y} \cup \mathcal{Y}^-} \left( y_{u,i} \log \hat{y}_{u,i} + (1 - y_{u,i}) \log(1 - \hat{y}_{u,i}) \right)$$
where each rated movie serves as a positive ($y=1$), and 4 unrated movies are sampled uniformly at random as negative instances ($y=0$) each epoch.

### 3.9 Algorithmic Parameter Synthesis
*Table 1: Algorithmic parameters and hyperparameter configurations across all models.*

| Model | Hyperparameter | Configured Value | Selection Rationale |
|---|---|---|---|
| **Popularity** | Cutoff Rank $K$ | 10 | Assignment standard evaluation depth |
| **User-CF** | Neighborhood Size $k$ | 50 | Eliminates distant noise while preserving peer diversity |
| **User-CF** | Shrinkage $\beta$ | 5.0 | Dampens similarity weights backed by few co-rated movies |
| **Item-CF** | Neighborhood Size $k$ | 50 | Optimized item neighborhood for Adjusted Cosine |
| **Item-CF** | Sim Shrinkage $\lambda_{\text{sim}}$ | 10.0 | Penalizes similarity for item pairs with small co-rater counts |
| **Item-CF** | Score Shrinkage $\gamma$ | 1.0 | Stabilizes candidate scores backed by sparse neighbor evidence |
| **SVD** | Latent Factors $k$ | 100 | Standard factor dimensionality for MovieLens benchmarks |
| **SVD** | Epochs & Learning Rate | 20 epochs, $\gamma=0.005$ | SGD convergence on training split |
| **Content** | Vectorizer | Sublinear TF-IDF | Prevents high-frequency tag dominance |
| **Hybrid** | Weights ($w_u, w_i, w_s, w_c$) | $(0.60, 0.40, 0.00, 0.00)$ | Optimal grid search on validation split ($N_{\text{val}}=5$) |
| **NCF** | GMF Embedding Dim | 16 | Tuned to prevent overfitting on 100k samples |
| **NCF** | MLP Architecture | (32, 16, 8), Dropout=0.1 | Dense feedforward projection with ReLU activations |
| **NCF** | Negatives per Positive | 4 | Balances gradient stability and memory footprint |
| **NCF** | Optimizer & Batch Size | Adam ($\text{lr}=10^{-3}$), Batch 1024 | Stable mini-batch gradient descent |

---

## 4. Experimental Evaluation & Results

### 4.1 Evaluation Protocol & Metrics Formulation
All ranking models were evaluated under a rigorous Leave-One-Out (LOO) protocol on the held-out test split (`data/splits/test_loo.csv`). Let $U_{\text{rel}}$ denote the set of users whose held-out test item $i_u^*$ received a true rating $r_{u, i_u^*} \ge 4.0$ ($|U_{\text{rel}}| = 363$). Let $\text{rank}(u, i_u^*)$ denote the 1-based rank of the held-out item in the model's top-10 list:
- **Hit@10:**
  $$\text{Hit@10}(u) = \mathbb{I}\left(\text{rank}(u, i_u^*) \le 10\right)$$
- **Recall@10:**
  $$\text{Recall@10} = \frac{1}{|U_{\text{rel}}|} \sum_{u \in U_{\text{rel}}} \text{Hit@10}(u)$$
- **Precision@10:**
  $$\text{Precision@10} = \frac{1}{|U_{\text{rel}}|} \sum_{u \in U_{\text{rel}}} \frac{\text{Hit@10}(u)}{10} = \frac{\text{Recall@10}}{10}$$
- **NDCG@10 (Normalized Discounted Cumulative Gain):**
  Since there is exactly one relevant test item per user, $\text{IDCG}@10 = \frac{1}{\log_2(1 + 1)} = 1.0$.
  $$\text{NDCG@10} = \frac{1}{|U_{\text{rel}}|} \sum_{u \in U_{\text{rel}}} \frac{\text{Hit@10}(u)}{\log_2(\text{rank}(u, i_u^*) + 1)}$$
- **RMSE (Root Mean Squared Error):**
  Evaluated on the independent 80/20 test split ($N_{\text{test}} = 20,168$):
  $$\text{RMSE} = \sqrt{\frac{1}{N_{\text{test}}} \sum_{(u,i) \in \text{Test}} (r_{u,i} - \hat{r}_{u,i})^2}$$

### 4.2 Benchmark Results vs. Baselines
*Table 2: Comprehensive benchmark evaluation results across all models.*

| Model | Precision@10 | Recall@10 | NDCG@10 | RMSE (80/20) | Catalog Coverage | Top-10 Concentration |
|---|---|---|---|---|---|---|
| **Popularity (Baseline)** | 0.0063 | 0.0634 | 0.0299 | – | 1.25% (122) | 56.77% |
| **Random (Baseline)** | 0.0000 | 0.0000 | 0.0000 | – | 46.19% (4,500) | 0.69% |
| **User-Based CF** | 0.0091 | 0.0909 | 0.0459 | 0.9227 | 3.12% (304) | 33.79% |
| **Item-Based CF** | 0.0066 | 0.0661 | 0.0411 | 0.8922 | 10.42% (1,015) | 12.07% |
| **SVD Matrix Factorization** | 0.0039 | 0.0386 | 0.0209 | **0.8831** | 4.10% (399) | 23.02% |
| **Content-Based (TF-IDF)** | 0.0017 | 0.0165 | 0.0081 | – | **21.68% (2,112)** | 7.46% |
| **Weighted Hybrid** | **0.0096** | **0.0964** | **0.0521** | – | 4.12% (401) | 30.41% |
| **Neural CF (NCF)** | 0.0074 | 0.0744 | 0.0377 | – | 8.49% (827) | 20.80% |

**Key Analytical Findings:**
1. **The Weighted Hybrid Dominates Ranking:** The Weighted Hybrid achieved the highest score across all three ranking metrics (Recall@10 = **0.0964**, NDCG@10 = **0.0521**), successfully recovering 35 held-out test movies. Blending user-peer affinities with item-item co-occurrence signals produces a synergistic effect that exceeds the capability of any single constituent algorithm.
2. **SVD Excels at Rating Prediction but Lags in Ranking:** While SVD achieved the lowest rating prediction error (RMSE = **0.8831**), its ranking performance was modest (Recall@10 = 0.0386). This highlights the fundamental discrepancy in recommender systems between point-wise rating estimation and top-$N$ candidate ranking.
3. **Deep Learning Performance:** PyTorch NCF performed strongly (Recall@10 = **0.0744**, NDCG@10 = **0.0377**), outperforming pure Item-CF and SVD, proving that neural embedding projections effectively learn non-linear collaborative interactions from implicit signals.

### 4.3 Deep Learning Ablation & NCF Hyperparameter Tuning
As documented in `outputs/ncf_tuning_results.csv` and `outputs/ncf_training_curves.png`, we conducted a systematic ablation study over 7 NCF configurations on validation recall:
- **Baseline (dim=32, layers 64-32-16):** Validation Recall = 0.0771 at epoch 9.
- **Tuned Compact (dim=16, layers 32-16-8):** Validation Recall = **0.0824** at epoch 16 (best performing).
- **High Capacity (dim=64, layers 128-64-32):** Suffered rapid overfitting, peaking at epoch 4 with Recall = 0.0754.
- **High Regularization (dim=32, dropout=0.3):** Validation Recall = 0.0807 at epoch 14.
- **Negative Ratio Sweep (neg=8):** Validation Recall = 0.0824 at epoch 10.

Due to the moderate size of MovieLens 100k, compact embedding dimensions (dim=16) with 10% dropout act as a strong regularizer, preventing memorization of sparse user interactions.

### 4.4 Cold-Start Fallback Empirical Evaluation
To test the cold-start fallback function (`outputs/cold_start_eval.csv`), we simulated five distinct user archetypes (Sci-Fi, Romance/Comedy, Animation/Family, Crime/Thriller, and Drama/War).
- When prompted with **stated genres**, the TF-IDF engine achieved an average **Genre Hit Ratio@10 of 100.0%**, populating top-10 lists entirely with titles containing the requested genre classifications.
- When prompted with **liked seed movies**, the engine matched plot keywords and thematic tags, achieving an average Genre Hit Ratio@10 of **94.0%** while introducing novel titles that users had not explicitly specified.

### 4.5 Error Analysis Across User Activity & Item Tiers
Error analysis (`outputs/error_analysis.csv`) broke down test hit rates by user activity quartiles:
- **Quartile 1 (Low Activity, 20–35 ratings):** Hybrid Hit Rate = **3.9%**.
- **Quartile 2 (Mid-Low, 36–70 ratings):** Hybrid Hit Rate = **5.2%**.
- **Quartile 3 (Mid-High, 71–168 ratings):** Hybrid Hit Rate = **6.5%**.
- **Quartile 4 (High Activity, 169–2,698 ratings):** Hybrid Hit Rate = **9.8%**.

As expected, collaborative filtering accuracy correlates directly with user interaction density. Users in Q1 suffer from sparse co-occurrence overlaps, confirming why a hybrid architecture with content fallback is essential in production.

### 4.6 Qualitative Evaluation of Recommendations
As required by Menu 2, qualitative human evaluation was performed across 20 recommendations generated for diverse user profiles (`outputs/qualitative_ratings.csv`). Independent human assessors scored recommendations on a 1-to-5 scale across **Relevance** (mean = **4.40 / 5.0**) and **Serendipity** (mean = **3.65 / 5.0**), confirming that top recommendations were topical, high-quality, and plausible.

---

## 5. Deployment & Interactive Web Application

### 5.1 Streamlit Architectural Interface
The recommendation system is fully deployed as an interactive web application built with Streamlit (`app/app.py`). Designed with a responsive, modern dark aesthetic, the application provides five primary modules:
1. **Existing User Recommender:** Enables selection of any user ID (1–610), displays their historical rating statistics and preferred genres, and renders top-10 recommendation cards complete with rank badges, metadata tags, and similarity scores.
2. **Side-by-Side Model Comparator:** Renders an interactive matrix comparing recommendations across all eight algorithms simultaneously for any selected user.
3. **Cold-Start Onboarding Engine:** Provides five pre-configured persona profiles and an interactive custom preference builder (genre multiselect and seed movie picker) generating real-time recommendations.
4. **Benchmarks & Analytics Suite:** Embeds the official metrics table, comparison bar charts, and NCF training dynamics curves.
5. **Ethics & Architecture Explorer:** Features system architecture diagrams, mathematical formulations, and ethical impact assessments.

### 5.2 Real-Time User Feedback Logging Mechanism
Addressing **Additional Consideration 2**, the web application incorporates a real-time feedback loop. Every recommendation card includes **👍 Yes** and **👎 No** reaction buttons. When clicked:
- The interaction is appended asynchronously to `feedback.csv` recording `timestamp`, `userId`, `movieId`, `title`, `model`, `reaction`, and context.
- A visual confirmation toast acknowledges user input without triggering a disruptive full-page reload.
- These interaction signals can be ingested by future training cycles to incrementally adjust user profiles and penalize disliked candidate items.

### 5.3 Cold-Start Onboarding Demonstration
The cold-start module demonstrates graceful degradation for unindexed users. By routing queries through `recommend_from_genres()` and `recommend_from_movies()`, the system computes cosine similarities across $L_2$-normalized TF-IDF movie vectors in under 40 milliseconds, providing instantaneous recommendations with zero dependency on collaborative history.

---

## 6. Ethical Implications, Bias Mitigation & Privacy

### 6.1 Popularity Bias & The Long-Tail Problem
Recommendation systems inherently risk reinforcing Matthew effects ("the rich get richer") [5]. Our catalog coverage analysis revealed stark empirical evidence:
- The **Popularity Baseline** covered only **1.25%** of the catalog (122 / 9,742 movies), with its top 10 movies claiming **56.77%** of all generated recommendation positions.
- In contrast, **Content-Based Filtering** achieved **21.68% catalog coverage** (2,112 movies) with a top-10 concentration of only 7.46%.
- Our **Weighted Hybrid** expanded catalog coverage to **4.12%** (401 movies), successfully breaking blockbuster concentration while preserving top-tier ranking accuracy.

### 6.2 Filter Bubbles & Serendipity Management
Over-reliance on content matching leads to narrow filter bubbles where users are repeatedly served variations of content they have already consumed. By blending collaborative filtering signals ($w_{\text{user}}=0.60, w_{\text{item}}=0.40$), our hybrid introduces serendipitous cross-genre recommendations discovered through peer cohort patterns.

### 6.3 Data Privacy, Anonymization & Governance
- **Zero Personally Identifiable Information:** The MovieLens dataset contains strictly anonymized numeric identifiers. No names, IP addresses, emails, or device fingerprints are stored.
- **Local Data Governance:** Live feedback collected via Streamlit is written strictly to local storage (`feedback.csv`) and is never transmitted to external ad brokers or third-party tracking services.

---

## 7. Limitations & Future Work
While our hybrid system achieves strong empirical performance, several limitations provide avenues for future research:
1. **Static vs. Sequential Modeling:** Our models treat ratings as static sets rather than sequential trajectories. Future extensions could implement recurrent architectures (GRU4Rec) or self-attention sequential models (SASRec) to capture evolving user taste over time.
2. **Graph Neural Networks (GNNs):** Exploring LightGCN or PinSage could further capture high-order collaborative connectivity in bipartite user-item graphs.
3. **Multimodal Rich Features:** MovieLens small lacks rich multimodal assets (movie poster images, trailers, audio). Ingesting visual poster embeddings via pre-trained vision transformers (ViT) represents a promising direction for enhanced content representation.

---

## 8. Conclusion
This practical study successfully designed, implemented, and validated a comprehensive Hybrid Movie Recommendation System on the MovieLens 100k benchmark. By uniting memory-based collaborative filtering, latent factor matrix factorization, TF-IDF metadata modeling, and PyTorch deep learning, our validation-tuned Weighted Hybrid achieved state-of-the-art performance across all ranking dimensions (Recall@10 = **0.0964**, NDCG@10 = **0.0521**), outperforming all individual baselines. The system demonstrates robust cold-start resilience through content fallback, mitigates popularity bias, and is fully deployed as an interactive Streamlit application with live feedback capture. All code, data pipelines, and documentation are openly available and reproducible.

---

## References

- [1] F. M. Harper and J. A. Konstan, "The MovieLens datasets: History and context," *ACM Trans. Interact. Intell. Syst.*, vol. 5, no. 4, pp. 1–19, 2015.
- [2] X. He, L. Liao, H. Zhang, L. Nie, X. Hu, and T.-S. Chua, "Neural collaborative filtering," in *Proc. 26th Int. Conf. World Wide Web (WWW)*, 2017, pp. 173–182.
- [3] Y. Koren, R. Bell, and C. Volinsky, "Matrix factorization techniques for recommender systems," *Computer*, vol. 42, no. 8, pp. 30–37, 2009.
- [4] B. Sarwar, G. Karypis, J. Konstan, and J. Riedl, "Item-based collaborative filtering recommendation algorithms," in *Proc. 10th Int. Conf. World Wide Web (WWW)*, 2001, pp. 285–295.
- [5] F. Ricci, L. Rokach, and B. Shapira, Eds., *Recommender Systems Handbook*, 3rd ed. New York, NY, USA: Springer, 2022.

---

## Appendix A: Contributions Table
*(Refer to `docs/contributions.md` for complete seat and menu mapping).*

| Seat | Name | Task / Deliverable | Status |
|---|---|---|---|
| **L** | Group Leader | Project management, repo scaffolding, submission packaging | Completed |
| **Seat 1** | Team Member 1 | Data acquisition, row count verification, data dictionary | Completed |
| **Seat 2** | Team Member 2 | Preprocessing, cleaning, EDA, LOO & 80/20 train/test splits | Completed |
| **Seat 3** | Team Member 3 | Baseline implementations (Popularity & Random rankers) | Completed |
| **Seat 4** | Team Member 4 | User-based CF implementation and neighborhood shrinkage tuning | Completed |
| **Seat 5** | Team Member 5 | Item-based CF, SVD matrix factorization, and RMSE evaluation | Completed |
| **Seat 6** | Team Member 6 | Content-based TF-IDF engine on genres and user tags | Completed |
| **Seat 7** | Team Member 7 | Weighted Hybrid engine, z-score normalization, cold-start fallback | Completed |
| **Seat 8** | Team Member 8 | Neural Collaborative Filtering (NCF / NeuMF) in PyTorch | Completed |
| **Seat 9** | Team Member 9 | NCF hyperparameter sweep, tuning logs, and training curve plotting | Completed |
| **Seat 10** | Team Member 10 | Metrics evaluation module (P@10, R@10, NDCG@10, RMSE) | Completed |
| **Seat 11** | Team Member 11 | Benchmark analysis, catalog coverage, and popularity bias audit | Completed |
| **Seat 12** | Team Member 12 | Streamlit interactive web application and live feedback logging | Completed |
| **Seat 13** | Team Member 13 | Cloud deployment configuration, phone testing, demo video script | Completed |
| **Seat 14** | Team Member 14 | Comprehensive academic report compilation & IEEE formatting | Completed |
| **Seat 15** | Team Member 15 | 12-slide presentation deck, oral presentation script, defense Q&A | Completed |

---

## Appendix B: Data Dictionary
*(Refer to `docs/data_dictionary.md` for comprehensive schema documentation).*

| File | Column | Type | Description | Missing Values |
|---|---|---|---|---|
| `ratings.csv` | `userId` | Integer | Anonymized user ID (1 to 610) | 0 (0.0%) |
| `ratings.csv` | `movieId` | Integer | Movie ID (1 to 193609) | 0 (0.0%) |
| `ratings.csv` | `rating` | Float | 5-star rating scale [0.5, 5.0] | 0 (0.0%) |
| `ratings.csv` | `timestamp` | Integer | POSIX timestamp of rating | 0 (0.0%) |
| `movies.csv` | `movieId` | Integer | Unique movie identifier | 0 (0.0%) |
| `movies.csv` | `title` | String | Movie title with release year | 0 (0.0%) |
| `movies.csv` | `genres` | String | Pipe-delimited genre categories | 0 (0.0%) |
| `tags.csv` | `tag` | String | User-applied descriptive keyword | 0 (0.0%) |
| `metrics.csv` | `recall_at_10` | Float | LOO ranking recall on relevant items ($\ge 4.0$) | 0 (0.0%) |
| `metrics.csv` | `ndcg_at_10` | Float | Normalized Discounted Cumulative Gain at rank 10 | 0 (0.0%) |
| `feedback.csv` | `reaction` | String | User feedback indicator (`LIKE` or `DISLIKE`) | 0 (0.0%) |
