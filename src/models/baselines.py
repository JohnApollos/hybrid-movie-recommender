"""Popularity and random baselines.

Run from the repo root:  python src/models/baselines.py
Reads  data/splits/train_loo.csv, data/splits/test_loo.csv
Writes outputs/popularity_top10.csv and outputs/random_top10.csv
Columns: userId, movieId, rank, score   (rank 1 = best)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data" / "splits"
OUT = ROOT / "outputs"
SEED = 42
K = 10


def popularity_top10(train, users, seen):
    pop = (train.groupby("movieId").size().reset_index(name="n")
           .sort_values(["n", "movieId"], ascending=[False, True]))
    movie_ids, counts = pop.movieId.to_numpy(), pop.n.to_numpy()
    rows = []
    for u in users:
        rank = 0
        for m, c in zip(movie_ids, counts):
            if m in seen[u]:
                continue
            rank += 1
            rows.append((u, int(m), rank, int(c)))
            if rank == K:
                break
    return pd.DataFrame(rows, columns=["userId", "movieId", "rank", "score"])


def random_top10(train, users, seen):
    rng = np.random.default_rng(SEED)
    all_movies = np.sort(train.movieId.unique())
    rows = []
    for u in users:
        cand = all_movies[~np.isin(all_movies, list(seen[u]))]
        scores = rng.random(len(cand))
        top = np.argsort(-scores)[:K]
        for r, i in enumerate(top, start=1):
            rows.append((u, int(cand[i]), r, float(scores[i])))
    return pd.DataFrame(rows, columns=["userId", "movieId", "rank", "score"])


def main():
    OUT.mkdir(exist_ok=True)
    train = pd.read_csv(SPLITS / "train_loo.csv")
    test = pd.read_csv(SPLITS / "test_loo.csv")
    users = sorted(test.userId.unique())
    seen = train.groupby("userId").movieId.apply(set).to_dict()

    results = {
        "popularity": popularity_top10(train, users, seen),
        "random": random_top10(train, users, seen),
    }
    for name, df in results.items():
        df.to_csv(OUT / f"{name}_top10.csv", index=False)
        hits = df.merge(test[["userId", "movieId"]], on=["userId", "movieId"])
        leaked = sum(m in seen[u] for u, m in zip(df.userId, df.movieId))
        print(f"{name:10s} rows={len(df)} users={df.userId.nunique()} "
              f"already-rated recs={leaked} hit-rate@10={len(hits) / len(users):.3f}")


if __name__ == "__main__":
    main()
