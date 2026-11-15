"""Content-based filtering: TF-IDF over genres + user tags.

Run from the repo root:  python src/models/content.py
Reads  data/ml-latest-small/movies.csv, tags.csv, data/splits/train_loo.csv, test_loo.csv
Writes outputs/content_top10.csv   Columns: userId, movieId, rank, score

Reusable for the hybrid / cold-start fallback and the app:
    movies, vec, X = build_features()
    recommend_from_movies([1, 260], movies, X)           # user names liked movies
    recommend_from_genres(["Sci-Fi", "Thriller"], vec, movies, X)   # user states genres
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "ml-latest-small"
SPLITS = ROOT / "data" / "splits"
OUT = ROOT / "outputs"
K = 10


def _genre_token(g):
    """'Sci-Fi' -> 'g_scifi' (prefix keeps genres separate from tag words)."""
    return "g_" + g.lower().replace("-", "")


def _genre_text(genres):
    if genres == "(no genres listed)":
        return ""
    return " ".join(_genre_token(g) for g in genres.split("|"))


def build_features():
    """Return (movies, vectorizer, X) where X is one L2-normalised TF-IDF row per movie."""
    movies = pd.read_csv(RAW / "movies.csv").sort_values("movieId").reset_index(drop=True)
    tags = pd.read_csv(RAW / "tags.csv").dropna(subset=["tag"])
    tag_text = tags.groupby("movieId").tag.apply(lambda s: " ".join(s.astype(str).str.lower()))
    text = movies.genres.map(_genre_text) + " " + movies.movieId.map(tag_text).fillna("")
    vec = TfidfVectorizer(sublinear_tf=True)
    X = vec.fit_transform(text)
    return movies, vec, X


def _top_n(scores, movies, n, blocked_ids, pop=None):
    scores = scores.copy()
    if pop is not None:  # popularity only breaks exact ties between equally similar movies
        scores = scores + 1e-6 * pop / max(pop.max(), 1)
    scores[movies.movieId.isin(blocked_ids).to_numpy()] = -np.inf
    top = np.argsort(-scores, kind="stable")[:n]
    return pd.DataFrame({"movieId": movies.movieId.to_numpy()[top], "score": scores[top]})


def recommend_from_movies(liked_ids, movies, X, n=K, exclude=(), pop=None):
    """Cold-start: recommend movies similar to a few movies the user says they like."""
    pos = np.flatnonzero(movies.movieId.isin(liked_ids).to_numpy())
    if len(pos) == 0:
        return pd.DataFrame(columns=["movieId", "score"])
    profile = normalize(np.asarray(X[pos].mean(axis=0)))
    scores = np.asarray(X @ profile.T).ravel()
    return _top_n(scores, movies, n, set(liked_ids) | set(exclude), pop)


def recommend_from_genres(genres, vec, movies, X, n=K, exclude=(), pop=None):
    """Cold-start: recommend movies for a stated genre profile, e.g. ['Sci-Fi', 'Thriller']."""
    q = normalize(vec.transform([" ".join(_genre_token(g) for g in genres)]))
    scores = (X @ q.T).toarray().ravel()
    return _top_n(scores, movies, n, set(exclude), pop)


def main():
    OUT.mkdir(exist_ok=True)
    train = pd.read_csv(SPLITS / "train_loo.csv")
    test = pd.read_csv(SPLITS / "test_loo.csv")
    movies, vec, X = build_features()
    # Ratings per movie in training: used only to break exact similarity ties
    pop = train.groupby("movieId").size().reindex(movies.movieId).fillna(0).to_numpy()

    users = np.sort(train.userId.unique())
    u_idx = pd.Series(np.arange(len(users)), index=users)
    m_idx = pd.Series(np.arange(len(movies)), index=movies.movieId.to_numpy())
    means = train.groupby("userId").rating.mean()
    w = train.rating.to_numpy() - means.reindex(train.userId).to_numpy()
    ui = u_idx[train.userId].to_numpy()
    mi = m_idx[train.movieId].to_numpy()

    # User profile = rating-weighted blend of the movie vectors they rated
    W = csr_matrix((w, (ui, mi)), shape=(len(users), len(movies)))
    profiles = normalize(W @ X)
    scores = (profiles @ X.T).toarray()
    scores = scores + 1e-6 * pop[None, :] / pop.max()
    seen = np.zeros(scores.shape, dtype=bool)
    seen[ui, mi] = True
    scores[seen] = -np.inf        # never recommend movies the user already rated

    movie_ids = movies.movieId.to_numpy()
    rows = []
    for i, u in enumerate(users):
        top = np.argsort(-scores[i], kind="stable")[:K]
        for r, j in enumerate(top, start=1):
            rows.append((int(u), int(movie_ids[j]), r, float(scores[i, j])))
    df = pd.DataFrame(rows, columns=["userId", "movieId", "rank", "score"])
    df.to_csv(OUT / "content_top10.csv", index=False)

    seen_pairs = set(zip(train.userId, train.movieId))
    leaked = sum((u, m) in seen_pairs for u, m in zip(df.userId, df.movieId))
    hits = df.merge(test[["userId", "movieId"]], on=["userId", "movieId"])
    print(f"content    rows={len(df)} users={df.userId.nunique()} "
          f"already-rated recs={leaked} hit-rate@10={len(hits) / len(test):.3f}")

    # Demo of the cold-start function: a brand-new user who likes Sci-Fi + Thriller
    demo = recommend_from_genres(["Sci-Fi", "Thriller"], vec, movies, X, n=5, pop=pop)
    print("\nCold-start demo (genres: Sci-Fi, Thriller):")
    print(demo.merge(movies[["movieId", "title"]], on="movieId")[["title", "score"]].to_string(index=False))


if __name__ == "__main__":
    main()
