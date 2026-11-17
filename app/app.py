"""Streamlit interface for the MovieLens recommendation outputs."""

from __future__ import annotations

import math
import json
import os
from concurrent.futures import ThreadPoolExecutor
from html import escape
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"
MOVIES_PATH = ROOT / "data" / "ml-latest-small" / "movies.csv"
LINKS_PATH = ROOT / "data" / "ml-latest-small" / "links.csv"
RATINGS_PATH = ROOT / "data" / "ml-latest-small" / "ratings.csv"
FEEDBACK_PATH = ROOT / "feedback.csv"

MODEL_FILES = {
    "Hybrid": "hybrid_top10.csv",
    "User-based collaborative filtering": "user_cf_top10.csv",
    "Item-based collaborative filtering": "item_cf_top10.csv",
    "SVD": "svd_top10.csv",
    "Content-based": "content_top10.csv",
    "Neural collaborative filtering": "ncf_top10.csv",
    "Popularity baseline": "popularity_top10.csv",
    "Random baseline": "random_top10.csv",
}
RECOMMENDATION_COLUMNS = ["userId", "movieId", "rank", "score"]
FEEDBACK_COLUMNS = ["userId", "movieId", "reaction", "title", "notes"]


@st.cache_data(show_spinner=False)
def load_recommendations(csv_path: str, modified_at: float, size: int) -> pd.DataFrame:
    """Read and validate a model's published top-10 CSV."""
    del modified_at, size  # These arguments make Streamlit refresh when the file changes.
    frame = pd.read_csv(csv_path)
    missing = set(RECOMMENDATION_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    try:
        frame = frame[RECOMMENDATION_COLUMNS].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("Recommendation values must be numeric and complete.") from exc
    if frame.isna().any().any():
        raise ValueError("Recommendation values must be numeric and complete.")
    for column in ("userId", "movieId", "rank"):
        if (frame[column] % 1 != 0).any():
            raise ValueError(f"{column} values must be whole numbers.")
        frame[column] = frame[column].astype(int)
    if (frame[["userId", "movieId"]] < 1).any().any():
        raise ValueError("userId and movieId values must be positive.")
    if ((frame["rank"] < 1) | (frame["rank"] > 10)).any():
        raise ValueError("rank values must be between 1 and 10.")
    return frame.sort_values(["userId", "rank", "score"], ascending=[True, True, False])


@st.cache_data(show_spinner=False)
def load_movies(csv_path: str, modified_at: float, size: int) -> pd.DataFrame:
    del modified_at, size
    movies = pd.read_csv(csv_path)
    if not {"movieId", "title"}.issubset(movies.columns):
        raise ValueError("movies.csv must contain movieId and title columns.")
    if "genres" not in movies.columns:
        movies["genres"] = ""
    movies["movieId"] = pd.to_numeric(movies["movieId"], errors="coerce")
    movies = movies.dropna(subset=["movieId"])
    movies["movieId"] = movies["movieId"].astype(int)
    return movies.drop_duplicates("movieId").set_index("movieId")


@st.cache_data(show_spinner=False)
def load_user_ids(csv_path: str, modified_at: float, size: int) -> list[int]:
    """Read anonymous MovieLens user IDs without loading the full ratings table."""
    del modified_at, size
    ratings = pd.read_csv(csv_path, usecols=["userId"])
    users = pd.to_numeric(ratings["userId"], errors="coerce").dropna()
    return sorted(users.astype(int).unique().tolist())


@st.cache_data(show_spinner=False)
def load_tmdb_ids(csv_path: str, modified_at: float, size: int) -> pd.Series:
    """Map MovieLens movie IDs to TMDB IDs from MovieLens links.csv."""
    del modified_at, size
    links = pd.read_csv(csv_path, usecols=["movieId", "tmdbId"])
    links["movieId"] = pd.to_numeric(links["movieId"], errors="coerce")
    links["tmdbId"] = pd.to_numeric(links["tmdbId"], errors="coerce")
    links = links.dropna(subset=["movieId", "tmdbId"])
    links["movieId"] = links["movieId"].astype(int)
    links["tmdbId"] = links["tmdbId"].astype(int)
    return links.drop_duplicates("movieId").set_index("movieId")["tmdbId"]


def read_tmdb_token() -> str:
    """Read a TMDB token from Streamlit secrets or the environment."""
    token = os.environ.get("TMDB_READ_ACCESS_TOKEN", "")
    if token:
        return token.strip()
    try:
        return str(st.secrets["TMDB_READ_ACCESS_TOKEN"]).strip()
    except (KeyError, FileNotFoundError):
        return ""


@st.cache_data(ttl="1d", show_spinner=False)
def load_poster_urls(tmdb_ids: tuple[int, ...], token: str) -> dict[int, str | None]:
    """Fetch poster paths for the visible movies and cache them for one day."""
    def fetch_one(tmdb_id: int) -> tuple[int, str | None]:
        request = Request(
            f"https://api.themoviedb.org/3/movie/{tmdb_id}",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        )
        try:
            with urlopen(request, timeout=4) as response:
                result = json.loads(response.read())
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError):
            return tmdb_id, None
        poster_path = result.get("poster_path")
        if not isinstance(poster_path, str) or not poster_path.startswith("/"):
            return tmdb_id, None
        return tmdb_id, f"https://image.tmdb.org/t/p/w342{poster_path}"

    with ThreadPoolExecutor(max_workers=5) as executor:
        return dict(executor.map(fetch_one, tmdb_ids))


def file_signature(path: Path) -> tuple[str, float, int]:
    stat = path.stat()
    return str(path), stat.st_mtime, stat.st_size


def load_feedback() -> pd.DataFrame:
    """Load feedback using the documented CSV schema, including older local logs."""
    if not FEEDBACK_PATH.is_file():
        return pd.DataFrame(columns=FEEDBACK_COLUMNS)
    try:
        feedback = pd.read_csv(FEEDBACK_PATH)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame(columns=FEEDBACK_COLUMNS)
    if "reaction" not in feedback.columns and "feedback" in feedback.columns:
        feedback = feedback.rename(columns={"feedback": "reaction"})
    for column in FEEDBACK_COLUMNS:
        if column not in feedback.columns:
            feedback[column] = ""
    feedback["userId"] = pd.to_numeric(feedback["userId"], errors="coerce")
    feedback["movieId"] = pd.to_numeric(feedback["movieId"], errors="coerce")
    feedback["reaction"] = feedback["reaction"].fillna("").astype(str).str.lower()
    feedback["title"] = feedback["title"].fillna("").astype(str).str.strip()
    feedback["notes"] = feedback["notes"].fillna("").astype(str)
    feedback = feedback.dropna(subset=["userId", "movieId"]).copy()
    feedback["userId"] = feedback["userId"].astype(int)
    feedback["movieId"] = feedback["movieId"].astype(int)

    missing_titles = feedback["title"].eq("")
    if missing_titles.any() and MOVIES_PATH.is_file():
        try:
            movie_titles = load_movies(*file_signature(MOVIES_PATH))["title"]
            feedback.loc[missing_titles, "title"] = feedback.loc[missing_titles, "movieId"].map(movie_titles)
        except (OSError, ValueError, pd.errors.ParserError):
            pass
    missing_titles = feedback["title"].isna() | feedback["title"].eq("")
    feedback.loc[missing_titles, "title"] = "Movie " + feedback.loc[missing_titles, "movieId"].astype(str)
    return feedback.drop_duplicates(["userId", "movieId"], keep="last")[FEEDBACK_COLUMNS]


def write_feedback(user_id: int, movie_id: int, reaction: str, notes_key: str) -> None:
    """Save or replace the user's reaction for a movie using the documented schema."""
    title = f"Movie {movie_id}"
    if MOVIES_PATH.is_file():
        try:
            movie_titles = load_movies(*file_signature(MOVIES_PATH))["title"]
            title = str(movie_titles.get(movie_id, title))
        except (OSError, ValueError, pd.errors.ParserError):
            pass
    record = {
        "userId": user_id,
        "movieId": movie_id,
        "reaction": reaction,
        "title": title,
        "notes": str(st.session_state.get(notes_key, "")).strip(),
    }
    try:
        old = load_feedback()
        keep = ~((old["userId"] == user_id) & (old["movieId"] == movie_id))
        old = old.loc[keep, FEEDBACK_COLUMNS]
        updated = pd.concat([old, pd.DataFrame([record])], ignore_index=True)
        updated.to_csv(FEEDBACK_PATH, index=False)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.session_state["feedback_error"] = f"Could not update feedback.csv: {exc}"
        st.session_state.pop("feedback_notice", None)
        return
    st.session_state.pop("feedback_error", None)
    st.session_state["feedback_notice"] = (rating, movie_id)


st.set_page_config(page_title="CineVerse | Movie recommendations", page_icon="🎬", layout="wide")
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
    '<div class="hero"><div class="eyebrow">CineVerse · MovieLens discovery</div>'
    '<h1>Find your next favorite.</h1>'
    '<p>Explore the team’s recommendation models, pick a MovieLens profile, and tell us what you think.</p></div>',
    unsafe_allow_html=True,
)

available_models = {
    label: OUTPUTS_DIR / filename
    for label, filename in MODEL_FILES.items()
    if (OUTPUTS_DIR / filename).is_file()
}
if not available_models:
    st.warning("No recommendations are available yet. Add a model CSV to `outputs/` with userId, movieId, rank, and score columns.", title="Recommendations unavailable")
    st.stop()

with st.sidebar:
    st.markdown("## 🎛️ Your session")
    model_name = st.selectbox("Recommendation model", list(available_models))
    recommendation_path = available_models[model_name]
    try:
        recommendations = load_recommendations(*file_signature(recommendation_path))
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Could not read {recommendation_path.name}: {exc}")
        st.stop()
    unavailable_scores = ~recommendations["score"].map(math.isfinite)
    if unavailable_scores.any():
        st.warning(
            f"{int(unavailable_scores.sum())} recommendation scores in this model "
            "are not finite and will be shown as unavailable."
        )
    user_ids = sorted(recommendations["userId"].unique().tolist())
    if RATINGS_PATH.is_file():
        try:
            user_ids = load_user_ids(*file_signature(RATINGS_PATH))
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            st.warning(f"MovieLens user IDs could not be loaded ({exc}); using users found in the recommendation file.")
    if not user_ids:
        st.warning("This model file does not contain any valid user recommendations.", title="No users found")
        st.stop()
    user_id = st.selectbox("MovieLens user ID", user_ids)
    st.caption("MovieLens IDs are anonymous. No personal details are collected.")

user_rows = recommendations.loc[recommendations["userId"] == user_id].head(10).copy()
tmdb_token = read_tmdb_token()
poster_urls: dict[int, str | None] = {}
if LINKS_PATH.is_file():
    try:
        tmdb_ids_by_movie = load_tmdb_ids(*file_signature(LINKS_PATH))
        user_rows["tmdbId"] = user_rows["movieId"].map(tmdb_ids_by_movie)
        if tmdb_token:
            tmdb_ids = tuple(sorted(user_rows["tmdbId"].dropna().astype(int).unique()))
            if tmdb_ids:
                poster_urls = load_poster_urls(tmdb_ids, tmdb_token)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.warning(f"Movie poster links could not be loaded ({exc}); showing cards without posters.")
        user_rows["tmdbId"] = None
else:
    user_rows["tmdbId"] = None

if MOVIES_PATH.is_file():
    try:
        movies = load_movies(*file_signature(MOVIES_PATH))
        user_rows["title"] = user_rows["movieId"].map(movies["title"])
        user_rows["genres"] = user_rows["movieId"].map(movies["genres"])
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.warning(f"Movie metadata could not be read ({exc}); showing movie IDs instead.")
        user_rows["title"] = None
        user_rows["genres"] = None
else:
    user_rows["title"] = None
    user_rows["genres"] = None
    st.info("Movie titles are not installed yet. Run `python data/download.py` to download MovieLens metadata; recommendations are available below by movie ID.")

notice = st.session_state.pop("feedback_notice", None)
feedback_error = st.session_state.pop("feedback_error", None)
if feedback_error:
    st.error(feedback_error)
if notice:
    reaction, movie_id = notice
    st.success(f"Your {reaction} reaction was saved for movie {movie_id}.")

st.markdown(f'<div class="eyebrow">{model_name}</div>', unsafe_allow_html=True)
st.subheader(f"Top picks for user {user_id}")
st.caption("Rankings are loaded from the selected model’s CSV in `outputs/`.")

if user_rows.empty:
    st.info("Try another user or model. This selection has no recommendations.", title="No recommendations for this selection")
else:
    for start in range(0, len(user_rows), 2):
        columns = st.columns(2, gap="large")
        for column, (_, row) in zip(columns, user_rows.iloc[start : start + 2].iterrows()):
            movie_id = int(row["movieId"])
            title = row.get("title")
            if pd.isna(title) or not title:
                title = f"Movie {movie_id}"
            genres = row.get("genres")
            genres = " · ".join(str(genres).split("|")) if pd.notna(genres) and genres else "MovieLens pick"
            score = float(row["score"])
            score_label = f"{score:.3f}" if math.isfinite(score) else "Unavailable"
            title = escape(str(title))
            genres = escape(str(genres))
            with column:
                poster_col, details_col = st.columns([1, 2], gap="medium")
                tmdb_id = row.get("tmdbId")
                poster_url = poster_urls.get(int(tmdb_id)) if pd.notna(tmdb_id) else None
                with poster_col:
                    if poster_url:
                        st.image(poster_url, alt=f"Movie poster for {title}", width="stretch")
                    else:
                        st.markdown('<div class="poster-placeholder">Poster<br>unavailable</div>', unsafe_allow_html=True)
                with details_col:
                    st.markdown(
                        f'<div class="movie-card"><div class="movie-rank">#{int(row["rank"]):02d} recommendation</div>'
                        f'<div class="movie-title">{title}</div><div class="movie-meta">{genres}</div>'
                        f'<div class="movie-meta">MovieLens ID {movie_id} · score {score_label}</div></div>',
                        unsafe_allow_html=True,
                    )
                    notes_key = f"notes-{user_id}-{movie_id}"
                    st.text_input(
                        "Optional note",
                        key=notes_key,
                        max_chars=250,
                        placeholder="Add a short note",
                    )
                up_col, down_col = st.columns(2)
                with up_col:
                    st.button(
                        "Like",
                        icon=":material/thumb_up:",
                        key=f"like-{user_id}-{movie_id}-{model_name}",
                        width="stretch",
                        on_click=write_feedback,
                        args=(int(user_id), movie_id, "up", notes_key),
                    )
                with down_col:
                    st.button(
                        "Dislike",
                        icon=":material/thumb_down:",
                        key=f"dislike-{user_id}-{movie_id}-{model_name}",
                        width="stretch",
                        on_click=write_feedback,
                        args=(int(user_id), movie_id, "down", notes_key),
                    )

with st.expander("About, credits & feedback"):
    st.write("Feedback is saved locally to `feedback.csv` using the documented columns: `userId`, `movieId`, `reaction`, `title`, and `notes`. A new reaction replaces the previous one for the same user and movie.")
    if tmdb_token:
        st.image(
            "https://www.themoviedb.org/assets/2/v4/logos/v2/blue_square_2-d537fb228cf3ded904ef09b136fe3fec72548ebc1fea3fbbd1ad9e36364db38b.svg",
            alt="The Movie Database logo",
            width=56,
        )
        st.caption("This product uses the TMDB API but is not endorsed or certified by TMDB.")
    else:
        st.caption("MovieLens does not include poster artwork. To load posters, set `TMDB_READ_ACCESS_TOKEN` in Streamlit secrets or your environment.")

feedback = load_feedback()
user_feedback = feedback.loc[feedback["userId"] == user_id]
likes = int((user_feedback["reaction"] == "up").sum())
dislikes = int((user_feedback["reaction"] == "down").sum())

st.subheader(f"Feedback summary for user {user_id}")
st.caption("Current saved votes across all recommendation models.")
summary_columns = st.columns(3)
summary_columns[0].metric("Total votes", len(user_feedback), border=True)
summary_columns[1].metric("Likes", likes, border=True)
summary_columns[2].metric("Dislikes", dislikes, border=True)

if not user_feedback.empty:
    latest_feedback = user_feedback.iloc[::-1].head(5).copy()
    st.dataframe(
        latest_feedback[["title", "reaction", "notes"]].rename(
            columns={"title": "Movie", "reaction": "Reaction", "notes": "Notes"}
        ),
        hide_index=True,
        alt="Most recent movie feedback for the selected user",
    )
