"""Hybrid Movie Recommendation System - Streamlit Web Application.

Recommender Systems Practical Study - BSc Data Science & Analytics, JKUAT.
Combines User-CF, Item-CF, SVD, Content-based (TF-IDF), and Neural Collaborative Filtering (NCF).
Features interactive recommendations, model comparisons, cold-start onboarding, and feedback logging.
"""
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

# Setup system path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

import content  # Content-based model and cold-start fallback helpers

# Page configuration
st.set_page_config(
    page_title="CineMatch | Hybrid Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #60A5FA;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1E293B;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #334155;
        margin-bottom: 12px;
    }
    .badge-rank {
        background-color: #2563EB;
        color: white;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .genre-tag {
        background-color: #334155;
        color: #E2E8F0;
        padding: 2px 7px;
        border-radius: 6px;
        font-size: 0.78rem;
        margin-right: 4px;
        display: inline-block;
        margin-top: 4px;
    }
    .card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .profile-box {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- Data Loaders
@st.cache_data
def load_movies():
    movies_path = ROOT / "data" / "ml-latest-small" / "movies.csv"
    if not movies_path.exists():
        st.error(f"Movie dataset not found at {movies_path}. Please run `python data/download.py`.")
        return pd.DataFrame(columns=["movieId", "title", "genres"])
    df = pd.read_csv(movies_path)
    return df


@st.cache_data
def load_all_recommendations():
    models = ["popularity", "random", "user_cf", "item_cf", "svd", "content", "hybrid", "ncf"]
    recs = {}
    for m in models:
        f = ROOT / "outputs" / f"{m}_top10.csv"
        if f.exists():
            recs[m] = pd.read_csv(f)
        else:
            recs[m] = pd.DataFrame(columns=["userId", "movieId", "rank", "score"])
    return recs


@st.cache_data
def load_training_data():
    split_path = ROOT / "data" / "splits" / "train_loo.csv"
    if split_path.exists():
        return pd.read_csv(split_path)
    return pd.DataFrame(columns=["userId", "movieId", "rating", "timestamp"])


@st.cache_data
def load_metrics():
    metrics_path = ROOT / "outputs" / "metrics.csv"
    if metrics_path.exists():
        return pd.read_csv(metrics_path)
    return pd.DataFrame()


@st.cache_data
def load_content_features():
    try:
        movies_df, vectorizer, X_matrix = content.build_features()
        return movies_df, vectorizer, X_matrix
    except Exception as e:
        return None, None, None


@st.cache_data
def load_demo_profiles():
    prof_path = ROOT / "app" / "demo_profiles.json"
    if prof_path.exists():
        with open(prof_path, "r", encoding="utf-8") as fp:
            return json.load(fp)
    return []


# ---------------------------------------------------------------- Feedback Logger
def log_user_feedback(user_id, movie_id, movie_title, model_name, reaction, notes=""):
    feedback_file = ROOT / "feedback.csv"
    timestamp = datetime.now().isoformat()
    record = pd.DataFrame([{
        "timestamp": timestamp,
        "userId": user_id,
        "movieId": movie_id,
        "title": movie_title,
        "model": model_name,
        "reaction": reaction,
        "notes": notes
    }])
    if not feedback_file.exists():
        record.to_csv(feedback_file, index=False)
    else:
        record.to_csv(feedback_file, mode="a", header=False, index=False)


# ---------------------------------------------------------------- App Initialization
movies_df = load_movies()
movies_dict = movies_df.set_index("movieId").to_dict(orient="index") if not movies_df.empty else {}
recs_dict = load_all_recommendations()
train_df = load_training_data()
metrics_df = load_metrics()
demo_profiles = load_demo_profiles()

if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = set()

# Header
st.markdown('<div class="main-header">🎬 CineMatch: Hybrid Movie Recommender</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">BSc Data Science & Analytics Practical Study | JKUAT | Hybrid Collaborative & Content Filtering with PyTorch Deep Learning</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🍿 Existing User Recs",
    "❄️ Cold-Start Onboarding",
    "📊 Benchmarks & Evaluation",
    "⚖️ Ethics & Privacy",
    "🏗️ System Architecture"
])

# ---------------------------------------------------------------- TAB 1: Existing User Recommendations
with tab1:
    st.subheader("Personalized Movie Recommendations for Existing Users")
    col_sel1, col_sel2 = st.columns([1, 2])

    with col_sel1:
        user_list = sorted(train_df["userId"].unique()) if not train_df.empty else [1]
        selected_user = st.selectbox("Select User ID (from MovieLens 100k):", options=user_list, index=0)

    with col_sel2:
        model_options = {
            "hybrid": "⭐ Weighted Hybrid (User-CF + Item-CF + SVD + Content)",
            "ncf": "🧠 Neural Collaborative Filtering (NeuMF / PyTorch)",
            "user_cf": "👥 User-based Collaborative Filtering (Cosine)",
            "item_cf": "🎬 Item-based Collaborative Filtering (Adjusted Cosine)",
            "svd": "📐 SVD Matrix Factorization (Latent Factors)",
            "content": "🏷️ Content-based Filtering (TF-IDF Genres & Tags)",
            "popularity": "🔥 Popularity Baseline",
            "random": "🎲 Random Baseline"
        }
        selected_model = st.selectbox("Select Recommendation Model:", options=list(model_options.keys()),
                                      format_func=lambda k: model_options[k], index=0)

    # Show User Profile
    if not train_df.empty:
        user_history = train_df[train_df.userId == selected_user].sort_values("rating", ascending=False)
        user_ratings_count = len(user_history)
        user_avg_rating = user_history["rating"].mean()

        with st.expander(f"👤 View Taste Profile for User #{selected_user} ({user_ratings_count} rated movies, avg rating: {user_avg_rating:.2f} ★)", expanded=False):
            pcol1, pcol2 = st.columns([1, 2])
            with pcol1:
                st.write("**Top Genres Rated:**")
                history_movies = user_history.merge(movies_df, on="movieId")
                genre_counts = history_movies["genres"].str.split("|").explode().value_counts().head(5)
                for g, c in genre_counts.items():
                    st.write(f"- {g}: {c} ratings")
            with pcol2:
                st.write("**Recent / Highest Rated Movies by this User:**")
                for _, row in user_history.head(5).iterrows():
                    m_info = movies_dict.get(row["movieId"], {"title": f"Movie #{row['movieId']}", "genres": ""})
                    st.write(f"★ **{row['rating']:.1f}** - {m_info['title']} *({m_info['genres']})*")

    st.markdown("---")

    # Display Top 10 Recommendations
    model_df = recs_dict.get(selected_model, pd.DataFrame())
    user_recs = model_df[model_df.userId == selected_user].sort_values("rank").head(10)

    if user_recs.empty:
        st.warning(f"No precomputed recommendations found for User #{selected_user} under model `{selected_model}`.")
    else:
        st.markdown(f"#### Top-10 Recommended Movies for User #{selected_user}")
        rcols = st.columns(2)

        for idx, (_, row) in enumerate(user_recs.iterrows()):
            mid = int(row["movieId"])
            rank = int(row["rank"])
            score = float(row["score"])
            movie_info = movies_dict.get(mid, {"title": f"Movie #{mid}", "genres": "Unknown"})
            title = movie_info["title"]
            genres = movie_info["genres"].split("|")

            col = rcols[idx % 2]
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <span class="badge-rank">#{rank}</span>
                    <span class="card-title" style="margin-left: 8px;">{title}</span>
                    <div style="margin-top: 6px;">
                        {' '.join([f'<span class="genre-tag">{g}</span>' for g in genres])}
                    </div>
                    <div style="margin-top: 8px; color: #94A3B8; font-size: 0.85rem;">
                        Model Predicted Score: <strong>{score:.3f}</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Feedback Buttons
                fcol1, fcol2, fcol3 = st.columns([1, 1, 3])
                fb_key_up = f"fb_{selected_user}_{mid}_{selected_model}_up"
                fb_key_down = f"fb_{selected_user}_{mid}_{selected_model}_down"

                with fcol1:
                    if st.button("👍 Yes", key=fb_key_up):
                        log_user_feedback(selected_user, mid, title, selected_model, "LIKE", "User indicated positive interest")
                        st.session_state.feedback_submitted.add(f"{selected_user}_{mid}")
                        st.success("Feedback logged!")

                with fcol2:
                    if st.button("👎 No", key=fb_key_down):
                        log_user_feedback(selected_user, mid, title, selected_model, "DISLIKE", "User indicated disinterest")
                        st.session_state.feedback_submitted.add(f"{selected_user}_{mid}")
                        st.info("Feedback logged!")

    # Model comparison expander
    with st.expander("🔄 Compare All Models for this User Side-by-Side"):
        compare_data = []
        for rk in range(1, 11):
            row_dict = {"Rank": f"#{rk}"}
            for m in ["popularity", "user_cf", "item_cf", "svd", "content", "hybrid", "ncf"]:
                m_df = recs_dict.get(m, pd.DataFrame())
                m_u = m_df[(m_df.userId == selected_user) & (m_df.rank == rk)]
                if not m_u.empty:
                    m_id = int(m_u.iloc[0]["movieId"])
                    row_dict[m] = movies_dict.get(m_id, {}).get("title", f"#{m_id}")
                else:
                    row_dict[m] = "—"
            compare_data.append(row_dict)
        st.dataframe(pd.DataFrame(compare_data), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------- TAB 2: Cold-Start Onboarding (New User)
with tab2:
    st.subheader("❄️ Cold-Start Onboarding: Instant Recommendations for New Users")
    st.write("Demonstration of **Additional Consideration 1**: when a new visitor arrives with zero interaction history, collaborative filtering suffers cold-start failure. The system activates content-based TF-IDF fallback over genres and user-generated tags.")

    cold_mode = st.radio("Choose Cold-Start Mode:", ["🎭 Use a Pre-configured Persona Profile", "🛠️ Build a Custom Preference Profile"], horizontal=True)

    c_movies, c_vec, c_X = load_content_features()

    if c_X is None:
        st.warning("Content features are initializing or unavailable. Please ensure `src/models/content.py` is executable.")
    else:
        chosen_genres = []
        chosen_seeds = []
        persona_name = "Custom New User"

        if cold_mode == "🎭 Use a Pre-configured Persona Profile":
            if demo_profiles:
                p_names = [p["name"] for p in demo_profiles]
                p_idx = st.selectbox("Select a Demo Persona Profile:", range(len(p_names)), format_func=lambda i: p_names[i])
                sel_p = demo_profiles[p_idx]
                persona_name = sel_p["name"]
                st.markdown(f"""
                <div class="profile-box">
                    <strong>Persona:</strong> {sel_p['name']}<br/>
                    <em>{sel_p['bio']}</em><br/>
                    <div style="margin-top: 6px;">
                        <strong>Seed Genres:</strong> {' '.join([f'<span class="genre-tag">{g}</span>' for g in sel_p['target_genres']])}
                    </div>
                    <div style="margin-top: 4px;">
                        <strong>Seed Liked Movies:</strong> {', '.join(sel_p['seed_movie_titles'])}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                chosen_genres = sel_p["target_genres"]
                chosen_seeds = sel_p["seed_movie_ids"]
        else:
            col_b1, col_b2 = st.columns(2)
            all_genres_list = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
            with col_b1:
                chosen_genres = st.multiselect("Select Genres you love:", all_genres_list, default=["Sci-Fi", "Action"])
            with col_b2:
                # Seed movies multiselect
                popular_movies_sample = movies_df[movies_df.movieId.isin([260, 2571, 318, 356, 296, 593, 1, 480, 527, 589])].set_index("movieId")["title"].to_dict()
                seed_selection = st.multiselect("Select Movies you have enjoyed in the past (optional):", options=list(popular_movies_sample.keys()), format_func=lambda k: popular_movies_sample[k])
                chosen_seeds = seed_selection

        if st.button("🚀 Generate Cold-Start Recommendations", type="primary"):
            st.markdown(f"### Recommended Movies for **{persona_name}**")
            # Generate recommendations
            if chosen_seeds:
                cold_recs = content.recommend_from_movies(chosen_seeds, c_movies, c_X, n=10)
            elif chosen_genres:
                cold_recs = content.recommend_from_genres(chosen_genres, c_vec, c_movies, c_X, n=10)
            else:
                cold_recs = pd.DataFrame()

            if cold_recs.empty:
                st.warning("Please select at least one genre or seed movie.")
            else:
                ccols = st.columns(2)
                for idx, (_, crow) in enumerate(cold_recs.iterrows()):
                    cmid = int(crow["movieId"])
                    cscore = float(crow["score"])
                    cinfo = movies_dict.get(cmid, {"title": f"Movie #{cmid}", "genres": "Unknown"})
                    ctitle = cinfo["title"]
                    cgenres = cinfo["genres"].split("|")

                    with ccols[idx % 2]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <span class="badge-rank">#{idx + 1}</span>
                            <span class="card-title" style="margin-left: 8px;">{ctitle}</span>
                            <div style="margin-top: 6px;">
                                {' '.join([f'<span class="genre-tag">{g}</span>' for g in cgenres])}
                            </div>
                            <div style="margin-top: 8px; color: #94A3B8; font-size: 0.85rem;">
                                Content Similarity Score: <strong>{cscore:.3f}</strong>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        fcol1, fcol2, _ = st.columns([1, 1, 3])
                        if fcol1.button("👍", key=f"cold_up_{cmid}_{idx}"):
                            log_user_feedback(999, cmid, ctitle, "cold_start_content", "LIKE", f"Cold start for {persona_name}")
                            st.success("Saved!")
                        if fcol2.button("👎", key=f"cold_down_{cmid}_{idx}"):
                            log_user_feedback(999, cmid, ctitle, "cold_start_content", "DISLIKE", f"Cold start for {persona_name}")
                            st.info("Saved!")


# ---------------------------------------------------------------- TAB 3: Model Benchmarks & Evaluation
with tab3:
    st.subheader("📊 Rigorous Model Evaluation & Comparative Benchmarks")
    st.write(r"""
    **Evaluation Protocol:** Leave-One-Out ranking validation where each user's latest interaction is held out as the test item.
    An item is marked *relevant* if its true rating $\ge 4.0$. RMSE is computed on an independent 80/20 rating split.
    """)

    if not metrics_df.empty:
        st.markdown("#### Official Metrics Table (`outputs/metrics.csv`)")
        display_m = metrics_df.copy()
        display_m["model_name"] = display_m["model"].map({
            "popularity": "Popularity Baseline",
            "random": "Random Baseline",
            "user_cf": "User-based Collaborative Filtering",
            "item_cf": "Item-based Collaborative Filtering",
            "svd": "SVD Matrix Factorization",
            "content": "Content-based Filtering (TF-IDF)",
            "hybrid": "⭐ Weighted Hybrid (Core Deliverable)",
            "ncf": "🧠 Neural Collaborative Filtering (PyTorch NeuMF)"
        })
        st.dataframe(display_m[["model_name", "precision_at_10", "recall_at_10", "ndcg_at_10", "rmse"]], use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### Visual Performance Comparisons")
        vcol1, vcol2 = st.columns(2)
        with vcol1:
            p_img = ROOT / "outputs" / "metrics_comparison.png"
            if p_img.exists():
                st.image(str(p_img), caption="Figure 1: Recall@10, Precision@10, and NDCG@10 across all models", use_container_width=True)
        with vcol2:
            c_img = ROOT / "outputs" / "coverage_diversity.png"
            if c_img.exists():
                st.image(str(c_img), caption="Figure 2: Catalog Coverage vs. Popularity Concentration", use_container_width=True)

        # NCF Training curves
        ncf_img = ROOT / "outputs" / "ncf_training_curves.png"
        if ncf_img.exists():
            st.markdown("#### Deep Learning Training Dynamics")
            st.image(str(ncf_img), caption="Figure 3: NCF NeuMF Training Loss & Validation Recall@10 across Epochs", use_container_width=True)

        # Qualitative Ratings
        qual_path = ROOT / "outputs" / "qualitative_ratings.csv"
        if qual_path.exists():
            with st.expander("📋 View Qualitative Human Evaluation of 20 Recommendations (Menu 2)"):
                st.dataframe(pd.read_csv(qual_path), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------- TAB 4: Ethics & Privacy
with tab4:
    st.subheader("⚖️ Ethical Considerations, Bias Mitigation & Privacy Protection")
    st.write("Demonstration of **Additional Consideration 3**: Recommender systems deployed at scale can distort consumption patterns, reinforce echo chambers, and expose private user signals.")

    ecol1, ecol2 = st.columns(2)
    with ecol1:
        st.markdown("""
        ### 1. Popularity Bias & The Long-Tail Problem
        - **Empirical Evidence:** In our MovieLens evaluation, the **Popularity Baseline** achieved a catalog coverage of only **1.25%** (122 / 9,742 movies), with its top 10 movies hogging **56.8%** of all generated recommendation slots.
        - **The Risk:** Heavy blockbusters (e.g., *Forrest Gump*, *The Matrix*) crowd out niche, indie, and international titles, creating an anti-competitive monoculture.
        - **Our Hybrid Mitigation:** By blending item-based similarity and content metadata, our **Weighted Hybrid** increases unique catalog exposure to **401 movies** while simultaneously achieving the highest ranking accuracy (Recall@10 = **0.0964**).
        """)

    with ecol2:
        st.markdown("""
        ### 2. Filter Bubbles & Feedback Loops
        - **Echo Chamber Risk:** Over-indexing on content-based filtering causes the system to exclusively recommend movies matching past genres (e.g. recommending only 1990s Action films to action fans).
        - **Serendipity by Design:** Collaborative filtering introduces cross-domain serendipity by discovering that users with similar tastes in sci-fi also enjoy historical dramas or anime.
        - **Dynamic Tuning:** Our validation grid search found that balancing $w_{\\text{user}}=0.60$ and $w_{\\text{item}}=0.40$ prevents narrow genre trapping while preserving topical coherence.
        """)

    st.markdown("---")
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        st.markdown("""
        ### 3. Data Privacy & Anonymization
        - **User Identity:** The MovieLens dataset contains strictly anonymized numerical user IDs (`userId` 1 to 610) with no Personally Identifiable Information (PII) such as real names, emails, IP addresses, or location coordinates.
        - **Local Feedback Protection:** Our 👍/👎 feedback mechanism stores interaction signals locally into `feedback.csv` without transmitting user data to third-party ad networks.
        """)
    with pcol2:
        st.markdown("""
        ### 4. Data Licensing & Academic Compliance
        - **Research Terms:** The MovieLens dataset is provided by GroupLens Research under academic research terms.
        - **Zero Redistribution:** Per the dataset terms, the raw data files are excluded via `.gitignore` and downloaded on-demand by each reproducible environment using `data/download.py`.
        """)


# ---------------------------------------------------------------- TAB 5: System Architecture
with tab5:
    st.subheader("🏗️ System Architecture & Engineering Design")
    st.markdown("""
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
    """)

    st.markdown("""
    #### Mathematical Formulations
    1. **User-Based Collaborative Filtering:**
       $$\\hat{r}_{u,i} = \\bar{r}_u + \\frac{\\sum_{v \\in N_k(u)} \\text{sim}(u,v) (r_{v,i} - \\bar{r}_v)}{\\sum_{v \\in N_k(u)} |\\text{sim}(u,v)| + \\beta}$$
    2. **Item-Based Collaborative Filtering:**
       $$\\hat{r}_{u,i} = \\bar{r}_u + \\frac{\\sum_{j \\in N_k(i)} s_{i,j} (r_{u,j} - \\bar{r}_u)}{\\sum_{j \\in N_k(i)} s_{i,j} + \\gamma}$$
    3. **Content-Based TF-IDF Profile:**
       $$\\vec{p}_u = \\text{normalize}\\left( \\sum_{i \\in I_u} (r_{u,i} - \\bar{r}_u) \\vec{x}_i \\right), \\quad \\text{score}(u,i) = \\vec{p}_u \\cdot \\vec{x}_i$$
    4. **Weighted Hybrid Model:**
       $$S_{\\text{hybrid}}(u,i) = \\sum_{m \\in M} w_m \\cdot \\text{z-score}_u(S_m(u,i))$$
    5. **Neural Collaborative Filtering (NeuMF):**
       $$\\hat{y}_{u,i} = \\sigma\\left(\\mathbf{h}^T [\\mathbf{p}_u^G \\odot \\mathbf{q}_i^G, \\phi^{\\text{MLP}}(\\mathbf{p}_u^M, \\mathbf{q}_i^M)]\\right)$$
    """)

    st.markdown("---")
    st.info("🎓 **Academic Project Details:** Recommender Systems Practical Study, BSc Data Science & Analytics, Jomo Kenyatta University of Agriculture and Technology (JKUAT). Full contribution history in `docs/contributions.md`.")
