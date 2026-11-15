"""Item-based collaborative filtering (adjusted cosine, shrunk, top-k neighbours).

Run from the repo root:  python src/models/item_cf.py
Reads  data/splits/train_loo.csv, test_loo.csv, train_rmse.csv, test_rmse.csv
Writes outputs/item_cf_top10.csv   Columns: userId, movieId, rank, score
       outputs/rmse_item_cf.csv    Columns: model, rmse
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data" / "splits"
OUT = ROOT / "outputs"
K_NEIGHBOURS = 50   # similar movies kept per movie
SIM_SHRINK = 10.0   # damps similarities backed by few co-raters
SHRINK = 1.0        # damps top-10 scores backed by little evidence
K = 10              # list length


def fit(train):
    """Build item neighbourhoods and the neighbour sums used for scoring."""
    users = np.sort(train.userId.unique())
    movies = np.sort(train.movieId.unique())
    u_idx = pd.Series(np.arange(len(users)), index=users)
    m_idx = pd.Series(np.arange(len(movies)), index=movies)

    R = np.zeros((len(users), len(movies)), dtype=np.float32)
    R[u_idx[train.userId].to_numpy(), m_idx[train.movieId].to_numpy()] = train.rating.to_numpy()
    B = (R > 0).astype(np.float32)               # 1 where the user rated the movie
    means = R.sum(axis=1) / B.sum(axis=1)
    Rc = (R - means[:, None]) * B                # mean-centred, 0 where unrated

    # Item-item similarity, computed in chunks to keep memory small
    norms = np.sqrt((Rc ** 2).sum(axis=0))
    norms[norms == 0] = 1.0
    n_items = len(movies)
    rows, cols, vals = [], [], []
    for a in range(0, n_items, 1000):
        b = min(a + 1000, n_items)
        dots = Rc[:, a:b].T @ Rc
        sim = dots / (norms[a:b, None] * norms[None, :])
        cnt = B[:, a:b].T @ B                    # number of users who rated both movies
        sim = sim * cnt / (cnt + SIM_SHRINK)
        sim[np.arange(b - a), np.arange(a, b)] = 0.0
        sim[sim < 0] = 0.0
        top = np.argpartition(-sim, K_NEIGHBOURS, axis=1)[:, :K_NEIGHBOURS]
        rows.append(np.repeat(np.arange(a, b), K_NEIGHBOURS))
        cols.append(top.ravel())
        vals.append(np.take_along_axis(sim, top, axis=1).ravel())
    S = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(n_items, n_items)).tocsr()

    num = np.asarray((S @ Rc.T).T)               # sum of sim * centred rating over neighbours
    den = np.asarray((S @ B.T).T)                # sum of sim over neighbours the user rated
    return users, movies, u_idx, m_idx, means, B, num, den


def main():
    OUT.mkdir(exist_ok=True)

    # ---- Top-10 lists (leave-one-out split) ----
    train = pd.read_csv(SPLITS / "train_loo.csv")
    test = pd.read_csv(SPLITS / "test_loo.csv")
    users, movies, _, _, _, B, num, den = fit(train)
    scores = num / (den + SHRINK)
    scores[B > 0] = -np.inf       # never recommend movies the user already rated
    scores[den == 0] = -np.inf    # no evidence for this movie
    rows = []
    for i, u in enumerate(users):
        top = np.argsort(-scores[i], kind="stable")[:K]
        for r, j in enumerate(top, start=1):
            rows.append((int(u), int(movies[j]), r, float(scores[i, j])))
    df = pd.DataFrame(rows, columns=["userId", "movieId", "rank", "score"])
    df.to_csv(OUT / "item_cf_top10.csv", index=False)

    seen = set(zip(train.userId, train.movieId))
    leaked = sum((u, m) in seen for u, m in zip(df.userId, df.movieId))
    hits = df.merge(test[["userId", "movieId"]], on=["userId", "movieId"])
    print(f"item_cf    rows={len(df)} users={df.userId.nunique()} "
          f"already-rated recs={leaked} hit-rate@10={len(hits) / len(test):.3f}")

    # ---- RMSE (80/20 random split) ----
    tr = pd.read_csv(SPLITS / "train_rmse.csv")
    te = pd.read_csv(SPLITS / "test_rmse.csv")
    users, movies, u_idx, m_idx, means, _, num, den = fit(tr)
    u_pos = u_idx.reindex(te.userId).to_numpy()
    m_pos = m_idx.reindex(te.movieId).to_numpy()
    pred = np.full(len(te), tr.rating.mean())     # fallback: global mean
    uk = ~np.isnan(u_pos)
    pred[uk] = means[u_pos[uk].astype(int)]       # fallback: the user's mean
    both = uk & ~np.isnan(m_pos)
    ui, mi = u_pos[both].astype(int), m_pos[both].astype(int)
    d, n = den[ui, mi], num[ui, mi]
    pred[both] = means[ui] + np.where(d > 0, n / np.maximum(d, 1e-9), 0.0)
    pred = np.clip(pred, 0.5, 5.0)
    rmse = float(np.sqrt(np.mean((pred - te.rating.to_numpy()) ** 2)))
    base = float(np.sqrt(np.mean((te.rating - tr.rating.mean()) ** 2)))
    pd.DataFrame([{"model": "item_cf", "rmse": round(rmse, 4)}]).to_csv(
        OUT / "rmse_item_cf.csv", index=False)
    print(f"item_cf    RMSE={rmse:.4f}  (global-mean baseline: {base:.4f})")


if __name__ == "__main__":
    main()
