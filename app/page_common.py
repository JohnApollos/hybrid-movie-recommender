from pathlib import Path
import json
import sys
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MOVIES_PATH = ROOT / "data" / "ml-latest-small" / "movies.csv"
RATINGS_PATH = ROOT / "data" / "ml-latest-small" / "ratings.csv"
OUTPUTS_DIR = ROOT / "outputs"
FEEDBACK_PATH = ROOT / "feedback.csv"
FEEDBACK_COLUMNS = ["userId", "movieId", "feedback", "model", "timestamp"]


def apply_cineverse_style(page_title: str, subtitle: str) -> None:
    st.set_page_config(page_title=page_title, page_icon="🎬", layout="wide")
    st.markdown(
        """
        <style>
        .stApp { background: radial-gradient(ellipse at top left, #21183c 0, #10111b 42rem, #0b0d14 100%); }
        .eyebrow { color: #c9a7ff; font-size: .78rem; font-weight: 700; letter-spacing: .18em; text-transform: uppercase; }
        .hero { padding: 1.7rem 0 1rem; }
        .hero h1 { font-size: clamp(2.5rem, 6vw, 4rem); letter-spacing: -.055em; margin: .25rem 0; }
        .hero p { color: #b8b7c7; font-size: 1.06rem; max-width: 700px; }
        .movie-card { background: linear-gradient(135deg, rgba(39,35,57,.96), rgba(24,25,37,.96));
          border: 1px solid rgba(201,167,255,.16); border-radius: 18px; padding: 1.15rem 1.25rem;
          min-height: 164px; box-shadow: 0 12px 32px rgba(0,0,0,.16); }
        .movie-rank { color: #d5ad55; font-size: .78rem; font-weight: 800; letter-spacing: .13em; text-transform: uppercase; }
        .movie-title { color: #faf8ff; font-size: 1.12rem; font-weight: 700; line-height: 1.35; margin: .45rem 0 .5rem; }
        .movie-meta { color: #aaa9bb; font-size: .88rem; }
        .poster-placeholder { aspect-ratio: 2 / 3; min-height: 190px; display: flex; align-items: center;
          justify-content: center; border: 1px solid rgba(201,167,255,.18); border-radius: 14px;
          background: linear-gradient(145deg, #30264b, #171827 70%); color: #d5ad55; text-align: center;
          padding: .75rem; font-size: .78rem; letter-spacing: .08em; text-transform: uppercase; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="hero"><div class="eyebrow">CineVerse · MovieLens discovery</div>'
        f'<h1>{page_title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_movies() -> pd.DataFrame:
    if not MOVIES_PATH.is_file():
        return pd.DataFrame(columns=["movieId", "title", "genres"])
    movies = pd.read_csv(MOVIES_PATH)
    if "genres" not in movies.columns:
        movies["genres"] = ""
    movies["movieId"] = pd.to_numeric(movies["movieId"], errors="coerce")
    return movies.dropna(subset=["movieId"]).assign(movieId=lambda x: x["movieId"].astype(int))


@st.cache_data(show_spinner=False)
def load_training_data() -> pd.DataFrame:
    if not RATINGS_PATH.is_file():
        return pd.DataFrame(columns=["userId", "movieId", "rating", "timestamp"])
    return pd.read_csv(RATINGS_PATH)


@st.cache_data(show_spinner=False)
def load_metrics() -> pd.DataFrame:
    path = OUTPUTS_DIR / "metrics.csv"
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_demo_profiles() -> list[dict]:
    path = ROOT / "app" / "demo_profiles.json"
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_content_features():
    try:
        sys.path.insert(0, str(ROOT / "src"))
        sys.path.insert(0, str(ROOT / "src" / "models"))
        import content
        return content.build_features(), content
    except (ImportError, OSError, ValueError):
        return (None, None, None), None


def save_feedback(user_id: int, movie_id: int, rating: str, model_name: str) -> None:
    record = {
        "userId": user_id,
        "movieId": movie_id,
        "feedback": rating,
        "model": model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    existing = pd.DataFrame(columns=FEEDBACK_COLUMNS)
    if FEEDBACK_PATH.is_file():
        existing = pd.read_csv(FEEDBACK_PATH)
        for column in FEEDBACK_COLUMNS:
            if column not in existing.columns:
                existing[column] = ""
        keep = ~(
            (pd.to_numeric(existing["userId"], errors="coerce") == user_id)
            & (pd.to_numeric(existing["movieId"], errors="coerce") == movie_id)
            & (existing["model"].astype(str) == model_name)
        )
        existing = existing.loc[keep, FEEDBACK_COLUMNS]
    pd.concat([existing, pd.DataFrame([record])], ignore_index=True).to_csv(FEEDBACK_PATH, index=False)


def render_movie_cards(frame: pd.DataFrame, heading: str, score_label: str = "Score") -> None:
    st.subheader(heading)
    if frame.empty:
        st.info("No recommendations are available for this selection.")
        return
    movies = load_movies().set_index("movieId") if not load_movies().empty else pd.DataFrame()
    for start in range(0, len(frame), 2):
        columns = st.columns(2, gap="large")
        for offset, (column, (_, row)) in enumerate(zip(columns, frame.iloc[start:start + 2].iterrows())):
            movie_id = int(row["movieId"])
            title = movies.loc[movie_id, "title"] if movie_id in movies.index else f"Movie {movie_id}"
            genres = movies.loc[movie_id, "genres"] if movie_id in movies.index else "MovieLens pick"
            genres = " · ".join(str(genres).split("|"))
            with column:
                st.markdown(
                    f'<div class="movie-card"><div class="movie-rank">#{start + offset + 1:02d} recommendation</div>'
                    f'<div class="movie-title">{title}</div><div class="movie-meta">{genres}</div>'
                    f'<div class="movie-meta">MovieLens ID {movie_id} · {score_label} {float(row["score"]):.3f}</div></div>',
                    unsafe_allow_html=True,
                )
