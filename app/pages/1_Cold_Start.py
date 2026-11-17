import pandas as pd
import streamlit as st

from page_common import (
    apply_cineverse_style,
    load_content_features,
    load_demo_profiles,
    load_movies,
    save_feedback,
)


apply_cineverse_style(
    "Cold-start discovery",
    "Build a taste profile or choose a CineVerse persona to discover movies without rating history.",
)

movies = load_movies()
(content_data, content_module) = load_content_features()
content_movies, vectorizer, features = content_data

if content_module is None or features is None:
    st.warning("Content-based recommendations are unavailable until the MovieLens metadata is installed.")
    st.code("python data/download.py")
    st.stop()

mode = st.radio(
    "How would you like to start?",
    ["Use a CineVerse persona", "Build my own taste profile"],
    horizontal=True,
)
genres = []
seed_movies = []
persona_name = "New CineVerse explorer"

if mode == "Use a CineVerse persona":
    profiles = load_demo_profiles()
    if not profiles:
        st.info("No demo personas are installed. Choose the custom profile option instead.")
    else:
        names = [profile["name"] for profile in profiles]
        profile = profiles[st.selectbox("Choose a persona", range(len(names)), format_func=lambda i: names[i])]
        persona_name = profile["name"]
        genres = profile.get("target_genres", [])
        seed_movies = profile.get("seed_movie_ids", [])
        st.markdown(
            f'<div class="movie-card"><div class="movie-title">{profile["name"]}</div>'
            f'<div class="movie-meta">{profile.get("bio", "")}</div>'
            f'<div class="movie-meta">Favorite genres: {" · ".join(genres)}</div></div>',
            unsafe_allow_html=True,
        )
else:
    movie_options = movies.set_index("movieId")["title"].to_dict() if not movies.empty else {}
    left, right = st.columns(2)
    with left:
        genres = st.multiselect(
            "Genres you enjoy",
            ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary",
             "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
             "Sci-Fi", "Thriller", "War", "Western"],
            default=["Sci-Fi", "Action"],
        )
    with right:
        seed_movies = st.multiselect(
            "Movies you already enjoy",
            list(movie_options),
            format_func=lambda movie_id: movie_options[movie_id],
        )

if st.button("Discover recommendations", type="primary"):
    if seed_movies:
        results = content_module.recommend_from_movies(seed_movies, content_movies, features, n=10)
    elif genres:
        results = content_module.recommend_from_genres(genres, vectorizer, content_movies, features, n=10)
    else:
        results = pd.DataFrame()
    st.session_state["cold_start_results"] = results
    st.session_state["cold_start_persona"] = persona_name

results = st.session_state.get("cold_start_results", pd.DataFrame())
if not results.empty:
    st.subheader(f"Top picks for {st.session_state.get('cold_start_persona', persona_name)}")
    movies_by_id = movies.set_index("movieId") if not movies.empty else pd.DataFrame()
    for start in range(0, len(results), 2):
        columns = st.columns(2, gap="large")
        for offset, (column, (_, row)) in enumerate(zip(columns, results.iloc[start:start + 2].iterrows())):
            movie_id = int(row["movieId"])
            title = movies_by_id.loc[movie_id, "title"] if movie_id in movies_by_id.index else f"Movie {movie_id}"
            genres_label = movies_by_id.loc[movie_id, "genres"] if movie_id in movies_by_id.index else "MovieLens pick"
            with column:
                st.markdown(
                    f'<div class="movie-card"><div class="movie-rank">#{start + offset + 1:02d} recommendation</div>'
                    f'<div class="movie-title">{title}</div><div class="movie-meta">{" · ".join(str(genres_label).split("|"))}</div>'
                    f'<div class="movie-meta">MovieLens ID {movie_id} · similarity {float(row["score"]):.3f}</div></div>',
                    unsafe_allow_html=True,
                )
                like, dislike = st.columns(2)
                like.button("Like", key=f"cold-like-{movie_id}", on_click=save_feedback,
                            args=(0, movie_id, "up", "cold_start_content"), width="stretch")
                dislike.button("Dislike", key=f"cold-dislike-{movie_id}", on_click=save_feedback,
                               args=(0, movie_id, "down", "cold_start_content"), width="stretch")
