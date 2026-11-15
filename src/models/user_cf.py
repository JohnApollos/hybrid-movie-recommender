"""User-based collaborative filtering (cosine similarity on mean-centred ratings).

Run from the repo root:  python src/models/user_cf.py
Reads  data/splits/train_loo.csv, data/splits/test_loo.csv
Writes outputs/user_cf_top10.csv   Columns: userId, movieId, rank, score
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data" / "splits"
OUT = ROOT / "outputs"
K_NEIGHBOURS = 50   # neighbours used per user
SHRINK = 5.0        # damps movies supported by few/weakly similar neighbours
K = 10              # list length


def main():
    OUT.mkdir(exist_ok=True)
    train = pd.read_csv(SPLITS / "train_loo.csv")
    test = pd.read_csv(SPLITS / "test_loo.csv")

    users = np.sort(train.userId.unique())
    movies = np.sort(train.movieId.unique())
    u_idx = pd.Series(np.arange(len(users)), index=users)
    m_idx = pd.Series(np.arange(len(movies)), index=movies)

    # User x movie matrices (610 x ~9,000 is small enough to hold densely)
    R = np.zeros((len(users), len(movies)), dtype=np.float32)
    R[u_idx[train.userId].to_numpy(), m_idx[train.movieId].to_numpy()] = train.rating.to_numpy()
    B = (R > 0).astype(np.float32)               # 1 where the user rated the movie
    means = R.sum(axis=1) / B.sum(axis=1)
    Rc = (R - means[:, None]) * B                # mean-centred, 0 where unrated

    # User-user similarity, keeping each user's top-k positive neighbours
    S = cosine_similarity(Rc)
    np.fill_diagonal(S, 0.0)
    S[S < 0] = 0.0
    drop = np.argsort(-S, axis=1)[:, K_NEIGHBOURS:]
    np.put_along_axis(S, drop, 0.0, axis=1)

    # Score every movie for every user
    num = S @ Rc
    den = S @ B
    scores = num / (den + SHRINK)
    scores[B > 0] = -np.inf      # never recommend movies the user already rated
    scores[den == 0] = -np.inf   # no similar user rated it

    rows = []
    for i, u in enumerate(users):
        top = np.argsort(-scores[i], kind="stable")[:K]
        for r, j in enumerate(top, start=1):
            rows.append((int(u), int(movies[j]), r, float(scores[i, j])))
    df = pd.DataFrame(rows, columns=["userId", "movieId", "rank", "score"])
    df.to_csv(OUT / "user_cf_top10.csv", index=False)

    # Sanity checks
    seen = set(zip(train.userId, train.movieId))
    leaked = sum((u, m) in seen for u, m in zip(df.userId, df.movieId))
    hits = df.merge(test[["userId", "movieId"]], on=["userId", "movieId"])
    print(f"user_cf    rows={len(df)} users={df.userId.nunique()} "
          f"already-rated recs={leaked} hit-rate@10={len(hits) / len(test):.3f}")


if __name__ == "__main__":
    main()
