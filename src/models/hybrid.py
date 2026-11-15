"""Weighted hybrid recommender (user-CF + item-CF + SVD + content) with cold-start fallback.

Run from the repo root:  python src/models/hybrid.py
Reads  data/splits/train_loo.csv, test_loo.csv, data/ml-latest-small/*
Writes outputs/hybrid_top10.csv      Columns: userId, movieId, rank, score
       outputs/hybrid_weights.json   Chosen blend weights + validation scores

Method: each model's scores are z-scored per user, then blended with weights chosen on a
validation split (each user's last N_VAL training ratings). The test set is only used once,
at the end, with the weights already fixed.

Cold start (importable): recommend_new_user(liked_movie_ids=None, genres=None, n=10)
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from surprise import SVD, Dataset, Reader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import content  # noqa: E402  (content-based model + cold-start helpers)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "ml-latest-small"
SPLITS = ROOT / "data" / "splits"
OUT = ROOT / "outputs"
MODELS = ["user_cf", "item_cf", "svd", "content"]
K = 10
N_VAL = 5      # last N training ratings per user form the validation set
STEP = 0.1     # weight grid resolution
SEED = 42


# ---------------------------------------------------------------- model scores
def build_matrices(train, users, movies):
    u_idx = pd.Series(np.arange(len(users)), index=users)
    m_idx = pd.Series(np.arange(len(movies)), index=movies)
    R = np.zeros((len(users), len(movies)), dtype=np.float32)
    R[u_idx[train.userId].to_numpy(), m_idx[train.movieId].to_numpy()] = train.rating.to_numpy()
    B = R > 0
    means = R.sum(axis=1) / B.sum(axis=1)
    Rc = (R - means[:, None]) * B      # mean-centred, 0 where unrated
    return B, Rc


def user_cf_scores(Rc, Bf, k=50, shrink=5.0):
    S = cosine_similarity(Rc)
    np.fill_diagonal(S, 0.0)
    S[S < 0] = 0.0
    drop = np.argsort(-S, axis=1)[:, k:]
    np.put_along_axis(S, drop, 0.0, axis=1)
    num, den = S @ Rc, S @ Bf
    sc = num / (den + shrink)
    sc[den == 0] = np.nan
    return sc


def item_cf_scores(Rc, Bf, k=50, sim_shrink=10.0, shrink=1.0):
    n_items = Rc.shape[1]
    norms = np.sqrt((Rc ** 2).sum(axis=0))
    norms[norms == 0] = 1.0
    rows, cols, vals = [], [], []
    for a in range(0, n_items, 1000):
        b = min(a + 1000, n_items)
        sim = (Rc[:, a:b].T @ Rc) / (norms[a:b, None] * norms[None, :])
        cnt = Bf[:, a:b].T @ Bf                      # users who rated both movies
        sim = sim * cnt / (cnt + sim_shrink)
        sim[np.arange(b - a), np.arange(a, b)] = 0.0
        sim[sim < 0] = 0.0
        top = np.argpartition(-sim, k, axis=1)[:, :k]
        rows.append(np.repeat(np.arange(a, b), k))
        cols.append(top.ravel())
        vals.append(np.take_along_axis(sim, top, axis=1).ravel())
    S = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(n_items, n_items)).tocsr()
    num = np.asarray((S @ Rc.T).T)
    den = np.asarray((S @ Bf.T).T)
    sc = num / (den + shrink)
    sc[den == 0] = np.nan
    return sc


def svd_scores(train, users, movies):
    data = Dataset.load_from_df(train[["userId", "movieId", "rating"]],
                                Reader(rating_scale=(0.5, 5.0)))
    ts = data.build_full_trainset()
    algo = SVD(n_factors=100, n_epochs=20, random_state=SEED)
    algo.fit(ts)
    inner_u = np.array([ts.to_inner_uid(u) for u in users])
    col, inner_i = [], []
    for c, m in enumerate(movies):
        try:
            inner_i.append(ts.to_inner_iid(m))
            col.append(c)
        except ValueError:           # movie has no ratings in this training set
            pass
    col, inner_i = np.array(col), np.array(inner_i)
    sc = np.full((len(users), len(movies)), np.nan, dtype=np.float32)
    sc[:, col] = (ts.global_mean + algo.bu[inner_u][:, None] + algo.bi[inner_i][None, :]
                  + algo.pu[inner_u] @ algo.qi[inner_i].T)
    return sc


def content_scores(Rc, B, movies, cmovies, X):
    pos = np.flatnonzero(cmovies.movieId.isin(movies).to_numpy())
    assert len(pos) == len(movies), "some movies are missing from movies.csv"
    Xu = X[pos]
    profiles = normalize(csr_matrix(Rc) @ Xu)
    sc = (profiles @ Xu.T).toarray().astype(np.float32)
    pop = B.sum(axis=0).astype(np.float32)           # popularity only breaks exact ties
    return sc + 1e-6 * pop[None, :] / max(pop.max(), 1)


def zscore(S, valid):
    """Per-user z-score over the movies the user could be recommended."""
    S = np.where(valid & np.isfinite(S), S, np.nan)
    mu = np.nanmean(S, axis=1, keepdims=True)
    sd = np.nanstd(S, axis=1, keepdims=True) + 1e-9
    return np.nan_to_num((S - mu) / sd, nan=0.0).astype(np.float32)


def model_scores(train, users, movies, cmovies, X):
    B, Rc = build_matrices(train, users, movies)
    Bf = B.astype(np.float32)
    raw = {
        "user_cf": user_cf_scores(Rc, Bf),
        "item_cf": item_cf_scores(Rc, Bf),
        "svd": svd_scores(train, users, movies),
        "content": content_scores(Rc, B, movies, cmovies, X),
    }
    return {m: zscore(s, ~B) for m, s in raw.items()}, B


# ------------------------------------------------------------------- blending
def blend(Z, w, B):
    out = sum(wi * Z[m] for m, wi in zip(MODELS, w) if wi > 0)
    return np.where(B, -np.inf, out)     # never recommend already-rated movies


def top_k(scores, k=K):
    idx = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, idx, axis=1), axis=1, kind="stable")
    return np.take_along_axis(idx, order, axis=1)


def weight_grid(n, step):
    ticks = int(round(1 / step))
    for combo in itertools.product(range(ticks + 1), repeat=n - 1):
        if sum(combo) <= ticks:
            yield tuple(round(c * step, 2) for c in combo) + (round((ticks - sum(combo)) * step, 2),)


# ----------------------------------------------------------------- cold start
def recommend_new_user(liked_movie_ids=None, genres=None, n=K):
    """Fallback for a user with no rating history.

    liked_movie_ids: movies the user says they like -> similar movies (content-based)
    genres:          e.g. ["Sci-Fi", "Thriller"]    -> movies matching those genres
    neither:         the most-rated movies overall
    Popularity only breaks ties between equally similar movies.
    """
    cmovies, vec, X = content.build_features()
    ratings = pd.read_csv(RAW / "ratings.csv")
    pop = ratings.groupby("movieId").size().reindex(cmovies.movieId).fillna(0).to_numpy()
    if liked_movie_ids:
        rec = content.recommend_from_movies(liked_movie_ids, cmovies, X, n=n, pop=pop)
    elif genres:
        rec = content.recommend_from_genres(genres, vec, cmovies, X, n=n, pop=pop)
    else:
        top = np.argsort(-pop, kind="stable")[:n]
        rec = pd.DataFrame({"movieId": cmovies.movieId.to_numpy()[top], "score": pop[top]})
    return rec.merge(cmovies[["movieId", "title"]], on="movieId")


# ----------------------------------------------------------------------- main
def main():
    OUT.mkdir(exist_ok=True)
    train = pd.read_csv(SPLITS / "train_loo.csv")
    test = pd.read_csv(SPLITS / "test_loo.csv")
    users = np.sort(train.userId.unique())
    movies = np.sort(train.movieId.unique())
    u_idx = pd.Series(np.arange(len(users)), index=users)
    m_idx = pd.Series(np.arange(len(movies)), index=movies)
    cmovies, _, X = content.build_features()

    # ---- 1) Choose weights on a validation split carved out of the training data
    ordered = train.sort_values(["userId", "timestamp", "movieId"])
    val = ordered.groupby("userId").tail(N_VAL)
    train_val = ordered.drop(val.index)
    rel = val[val.rating >= 4.0]
    V = np.zeros((len(users), len(movies)), dtype=bool)
    V[u_idx[rel.userId].to_numpy(), m_idx[rel.movieId].to_numpy()] = True
    n_rel = int(V.sum())
    rows = np.arange(len(users))[:, None]

    print("Scoring models on the validation split ...")
    Zv, Bv = model_scores(train_val, users, movies, cmovies, X)
    results = []
    for w in weight_grid(len(MODELS), STEP):
        idx = top_k(blend(Zv, w, Bv))
        results.append((V[rows, idx].sum() / n_rel, w))
    results.sort(key=lambda t: -t[0])
    best_val, best_w = results[0]

    print(f"\nValidation recall@10 ({n_rel} relevant held-out items):")
    for i, m in enumerate(MODELS):
        single = next(r for r, w in results if w[i] == 1.0)
        print(f"  {m:8s} alone  {single:.4f}")
    print(f"  best blend   {best_val:.4f}  weights "
          + ", ".join(f"{m}={w}" for m, w in zip(MODELS, best_w)))
    del Zv, Bv

    # ---- 2) Final model: retrain on the full training set, apply the chosen weights once
    print("\nScoring models on the full training set ...")
    Z, B = model_scores(train, users, movies, cmovies, X)
    scores = blend(Z, best_w, B)
    movie_ids = movies
    out_rows = []
    for i, u in enumerate(users):
        top = np.argsort(-scores[i], kind="stable")[:K]
        for r, j in enumerate(top, start=1):
            out_rows.append((int(u), int(movie_ids[j]), r, float(scores[i, j])))
    df = pd.DataFrame(out_rows, columns=["userId", "movieId", "rank", "score"])
    df.to_csv(OUT / "hybrid_top10.csv", index=False)
    with open(OUT / "hybrid_weights.json", "w") as f:
        json.dump({"weights": dict(zip(MODELS, best_w)),
                   "validation_recall_at_10": round(float(best_val), 4),
                   "normalisation": "per-user z-score over unseen movies"}, f, indent=2)

    seen = set(zip(train.userId, train.movieId))
    leaked = sum((u, m) in seen for u, m in zip(df.userId, df.movieId))
    hits = df.merge(test[["userId", "movieId"]], on=["userId", "movieId"])
    print(f"\nhybrid     rows={len(df)} users={df.userId.nunique()} "
          f"already-rated recs={leaked} hit-rate@10={len(hits) / len(test):.3f}")

    # ---- 3) Cold-start demo
    print("\nCold start, genres Sci-Fi + Thriller:")
    print(recommend_new_user(genres=["Sci-Fi", "Thriller"], n=5)[["title", "score"]].to_string(index=False))
    print("\nCold start, likes Toy Story (1) and Star Wars (260):")
    print(recommend_new_user(liked_movie_ids=[1, 260], n=5)[["title", "score"]].to_string(index=False))


if __name__ == "__main__":
    main()
