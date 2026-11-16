# 🎙️ Technical Presentation Script & Timing Guide

**Project:** Hybrid Movie Recommendation System  
**Course:** Recommender Systems Practical Study, BSc Data Science & Analytics, JKUAT  
**Format:** 12 Slides (16:9) + Live Web Demo  
**Total Presentation Budget:** 12 Minutes (9 Minutes Presentation + 3 Minutes Live Demo)  
**Roles:** Presenter (Orator), Demo Driver (Navigator)

---

### Slide 1: Title & Introduction (0:00 – 0:45 | 45 sec)
- **Visual:** Clean title slide with project title, JKUAT affiliation, group number, and team member list.
- **Talking Points:**
  1. "Good morning/afternoon, panel and colleagues. Today we present our practical study on building an end-to-end Hybrid Movie Recommendation System."
  2. "Recommender systems drive user engagement across Netflix, Spotify, and Amazon. Our core challenge was overcoming the classical trade-offs of sparsity, cold-start, and popularity bias by engineering a production-grade weighted hybrid architecture."
  3. "We will walk you through our methodology, comparative benchmarks across 8 models, deep learning ablation with Neural Collaborative Filtering, and a live deployment demonstration."
- **Cue:** *Presenter nods to Demo Driver to transition to Slide 2.*

---

### Slide 2: Problem Statement & Objectives (0:45 – 1:30 | 45 sec)
- **Visual:** Two-column problem vs solution graphic highlighting Cold-Start, Sparsity, and Monoculture.
- **Talking Points:**
  1. "Collaborative filtering relies on shared co-ratings, which breaks down when interaction matrices are over 98% sparse or when new users arrive."
  2. "Conversely, content-based filtering solves cold-start but risks trapping users in narrow genre echo chambers."
  3. "Our objectives: implement and tune User-CF, Item-CF, SVD, Content-based TF-IDF, and PyTorch NCF; blend them into an optimal weighted hybrid; and deploy a responsive web app with live feedback loops."
- **Cue:** *Next slide.*

---

### Slide 3: Dataset Architecture & Exploratory Data Analysis (1:30 – 2:30 | 60 sec)
- **Visual:** EDA distribution charts (Rating skewness, Sparsity matrix, Power-law long-tail distribution).
- **Talking Points:**
  1. "We utilized MovieLens 100k (`ml-latest-small`), containing 100,836 ratings, 9,742 movies, and 610 users, with a sparsity of 98.30%."
  2. "Key EDA findings: the median rating is 3.5 stars, and 48.2% of ratings are $\ge 4.0$. This empirical threshold dictates our ground-truth relevance criteria."
  3. "Notice the extreme long-tail: the top 100 movies command 16.1% of all ratings, while 6,074 movies (62.5%) have fewer than 5 ratings, illustrating the risk of popularity bias."
- **Cue:** *Next slide.*

---

### Slide 4: System Architecture & Workflow Pipeline (2:30 – 3:30 | 60 sec)
- **Visual:** End-to-end data pipeline flow diagram (Raw data $\rightarrow$ Splits $\rightarrow$ Individual models $\rightarrow$ Hybrid $\rightarrow$ Streamlit).
- **Talking Points:**
  1. "Here is our modular architecture. Raw data undergoes strict chronological Leave-One-Out splitting to hold out each user's latest interaction, plus an independent 80/20 split for RMSE."
  2. "Each model operates as an isolated squad module writing to standard top-10 contracts (`outputs/{model}_top10.csv`)."
  3. "The Weighted Hybrid normalizes all candidate scores via per-user z-scores and blends them before serving top-10 lists to Streamlit and our evaluation engine."
- **Cue:** *Next slide.*

---

### Slide 5: Classical Collaborative Filtering & SVD Matrix Factorization (3:30 – 4:30 | 60 sec)
- **Visual:** Equations and feature comparison table for User-CF, Item-CF, and SVD.
- **Talking Points:**
  1. "For User-CF, we computed cosine similarity across mean-centered ratings with $k=50$ neighbors and shrinkage $\beta=5.0$, dampening sparse noise."
  2. "Item-CF utilizes adjusted cosine similarity along with neighbor co-rating shrinkage ($\gamma=10.0$) to stabilize similarity matrices."
  3. "SVD matrix factorization decomposes interactions into 100 latent factors with user/item biases, achieving our best rating reconstruction error with an RMSE of 0.8831."
- **Cue:** *Next slide.*

---

### Slide 6: Content-Based Filtering & The Weighted Hybrid Engine (4:30 – 5:45 | 75 sec)
- **Visual:** TF-IDF metadata pipeline and validation weight grid search heatmap.
- **Talking Points:**
  1. "Our Content-based engine builds $L_2$-normalized TF-IDF vectors combining pipe-delimited genres and user-applied tags, creating profile centroids from liked movies."
  2. "The core deliverable is our Weighted Hybrid. Because raw model scores inhabit disparate scales, we z-score normalize scores per user and ran an extensive grid search on a validation split ($N_{\text{val}}=5$)."
  3. "The optimal weights converged to $w_{\text{user}}=0.60$ and $w_{\text{item}}=0.40$, achieving a validation recall of 0.0913 and zero leakage into our test set."
- **Cue:** *Next slide.*

---

### Slide 7: Deep Learning: Neural Collaborative Filtering (NCF) (5:45 – 6:45 | 60 sec)
- **Visual:** NeuMF dual-branch diagram (GMF + MLP) and validation recall curves across epochs.
- **Talking Points:**
  1. "To explore deep learning under Topic 3 requirements, we implemented NeuMF in PyTorch, fusing Generalized Matrix Factorization and a 3-layer MLP."
  2. "We treated ratings as implicit interactions and sampled 4 unrated movies as negatives per positive, optimizing Binary Cross-Entropy loss."
  3. "Through systematic tuning across 7 hyperparameter configurations, a compact architecture (dim=16, layers 32-16-8, 10% dropout) prevented overfitting and reached a validation recall of 0.0824 at epoch 16."
- **Cue:** *Next slide.*

---

### Slide 8: Evaluation Protocol & Rigorous Metrics (6:45 – 7:30 | 45 sec)
- **Visual:** Summary diagram of Leave-One-Out protocol, relevance formula ($\ge 4.0$), and NDCG logarithmic discount.
- **Talking Points:**
  1. "Evaluation integrity was paramount. Leave-One-Out measures whether the single held-out movie appears in the top-10."
  2. "Only items rated $\ge 4.0$ by the user are counted as relevant hits (363 users). Recommending a movie the user rated 1 star is considered a failure."
  3. "We report Precision@10, Recall@10, NDCG@10, RMSE, and Catalog Coverage across all 8 models."
- **Cue:** *Next slide.*

---

### Slide 9: Experimental Results vs. Baselines (7:30 – 8:30 | 60 sec)
- **Visual:** Benchmark comparison bar charts and official metrics table.
- **Talking Points:**
  1. "Examining the results: Random achieved 0.0000 recall, while Popularity scored 0.0634 due to blockbuster concentration."
  2. "Our Weighted Hybrid is the decisive winner: Recall@10 of **0.0964**, Precision@10 of **0.0096**, and NDCG@10 of **0.0521**, outperforming every single constituent model."
  3. "NCF performed strongly at 0.0744 Recall, confirming that neural feature interactions capture non-linear collaborative patterns effectively."
- **Cue:** *Presenter turns to Demo Driver: 'Now let us transition to the live application demo.'*

---

### Slide 10: Live Application & Cold-Start Demo (8:30 – 11:00 | 2.5 min DEMO)
- **Visual:** Live Streamlit interface projected on screen.
- **Driver Actions & Presenter Voiceover:**
  1. *(0:00 - 0:45)* **Existing User Tab:** "Driver selects User #10. Observe the user's taste profile: 140 ratings, heavily favoring Action and Crime. Driver switches from Popularity to Weighted Hybrid. Notice how recommendations instantly shift from generic blockbusters to personalized picks like *Heat* and *Fargo*. Driver clicks '👍 Yes' on rank #1; feedback logs instantly to `feedback.csv`."
  2. *(0:45 - 1:45)* **Cold-Start Tab:** "Now we simulate a brand new visitor. Collaborative models fail here. Driver selects the 'Sci-Fi Fan' demo persona and clicks 'Generate Cold-Start Recommendations'. Content-based TF-IDF instantly serves *The Matrix*, *Blade Runner*, and *Star Wars* with 100% genre alignment."
  3. *(1:45 - 2:30)* **Benchmarks & Ethics Tabs:** "Driver shows the live metrics dashboard and the catalog coverage visualizer, demonstrating that our hybrid maintains healthy catalog diversity without sacrificing ranking accuracy."
- **Cue:** *Driver switches back to Slide 11.*

---

### Slide 11: Ethical Considerations, Bias & Limitations (11:00 – 11:45 | 45 sec)
- **Visual:** Coverage comparison graph (Popularity 1.25% vs Hybrid 4.12% vs Content 21.68%) and Privacy summary.
- **Talking Points:**
  1. "Ethically, popularity bias remains the primary hazard: popularity recommendations give 56.8% of slots to 10 blockbusters. Our hybrid mitigates this by expanding unique recommendations to 401 movies."
  2. "Filter bubbles are constrained by blending collaborative discovery with content matching."
  3. "Privacy is safeguarded: MovieLens contains zero PII, and all feedback is stored locally under research compliance."
- **Cue:** *Next slide.*

---

### Slide 12: Conclusion & Q&A Invitation (11:45 – 12:00 | 15 sec)
- **Visual:** Conclusion recap, GitHub repository QR code / link, and team acknowledgment.
- **Talking Points:**
  1. "In conclusion, we delivered a complete, validated hybrid recommender system that beats all individual baselines and provides cold-start robustness and feedback adaptability."
  2. "Thank you for your time and attention. We now welcome questions from the panel."
- **Cue:** *Presenter and team stand ready for Q&A using `docs/qa_document.md`.*
