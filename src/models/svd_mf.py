"""SVD matrix factorisation (scikit-surprise SVD, latent factors plus biases).

Run from the repo root:  python src/models/svd_mf.py
Reads  data/splits/train_loo.csv, test_loo.csv, train_rmse.csv, test_rmse.csv
Writes outputs/svd_top10.csv   Columns: userId, movieId, rank, score
       outputs/rmse_svd.csv    Columns: model, rmse
"""
from pathlib import Path

import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader, accuracy

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data" / "splits"
OUT = ROOT / "outputs"
SEED = 42
K = 10
READER = Reader(rating_scale=(0.5, 5.0))


def fit_svd(train):
    data = Dataset.load_from_df(train[["userId", "movieId", "rating"]], READER)
    trainset = data.build_full_trainset()
    algo = SVD(n_factors=100, n_epochs=20, random_state=SEED)
    algo.fit(trainset)
    return algo, trainset


def main():
    OUT.mkdir(exist_ok=True)

    # ---- Top-10 lists (leave-one-out split) ----
    train = pd.read_csv(SPLITS / "train_loo.csv")
    test = pd.read_csv(SPLITS / "test_loo.csv")
    algo, ts = fit_svd(train)
    users = np.sort(train.userId.unique())
    movies = np.sort(train.movieId.unique())
    iu = np.array([ts.to_inner_uid(u) for u in users])
    ii = np.array([ts.to_inner_iid(m) for m in movies])
    # Predicted rating for every user x movie: mean + user bias + movie bias + taste match
    P = ts.global_mean + algo.bu[iu][:, None] + algo.bi[ii][None, :] + algo.pu[iu] @ algo.qi[ii].T

    u_idx = pd.Series(np.arange(len(users)), index=users)
    m_idx = pd.Series(np.arange(len(movies)), index=movies)
    seen_mask = np.zeros(P.shape, dtype=bool)
    seen_mask[u_idx[train.userId].to_numpy(), m_idx[train.movieId].to_numpy()] = True
    P[seen_mask] = -np.inf        # never recommend movies the user already rated

    rows = []
    for i, u in enumerate(users):
        top = np.argsort(-P[i], kind="stable")[:K]
        for r, j in enumerate(top, start=1):
            rows.append((int(u), int(movies[j]), r, float(P[i, j])))
    df = pd.DataFrame(rows, columns=["userId", "movieId", "rank", "score"])
    df.to_csv(OUT / "svd_top10.csv", index=False)

    seen = set(zip(train.userId, train.movieId))
    leaked = sum((u, m) in seen for u, m in zip(df.userId, df.movieId))
    hits = df.merge(test[["userId", "movieId"]], on=["userId", "movieId"])
    print(f"svd        rows={len(df)} users={df.userId.nunique()} "
          f"already-rated recs={leaked} hit-rate@10={len(hits) / len(test):.3f}")

    # ---- RMSE (80/20 random split) ----
    tr = pd.read_csv(SPLITS / "train_rmse.csv")
    te = pd.read_csv(SPLITS / "test_rmse.csv")
    algo, _ = fit_svd(tr)
    preds = algo.test(list(te[["userId", "movieId", "rating"]].itertuples(index=False, name=None)))
    rmse = accuracy.rmse(preds, verbose=False)
    pd.DataFrame([{"model": "svd", "rmse": round(rmse, 4)}]).to_csv(
        OUT / "rmse_svd.csv", index=False)
    print(f"svd        RMSE={rmse:.4f}")


if __name__ == "__main__":
    main()
