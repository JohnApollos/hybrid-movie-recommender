"""Create train/test splits from MovieLens ml-latest-small.

Run from the repo root:  python src/preprocessing.py
Writes to data/splits/:
  train_loo.csv, test_loo.csv    leave-one-out (each user's latest rating held out)
  train_rmse.csv, test_rmse.csv  80/20 random split (seed 42) for RMSE
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "ml-latest-small"
OUT = ROOT / "data" / "splits"
SEED = 42


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(RAW / "ratings.csv")

    # Leave-one-out: latest rating per user is the test item.
    # Ties in timestamp are broken by movieId so the split is reproducible.
    data_sorted = data.sort_values(["userId", "timestamp", "movieId"])
    test_loo = data_sorted.groupby("userId").tail(1)
    train_loo = data_sorted.drop(test_loo.index)

    # 80/20 random split for RMSE
    train_rmse, test_rmse = train_test_split(data, test_size=0.2, random_state=SEED)

    train_loo.to_csv(OUT / "train_loo.csv", index=False)
    test_loo.to_csv(OUT / "test_loo.csv", index=False)
    train_rmse.to_csv(OUT / "train_rmse.csv", index=False)
    test_rmse.to_csv(OUT / "test_rmse.csv", index=False)

    print("LOO   train:", train_loo.shape, "test:", test_loo.shape)
    print("RMSE  train:", train_rmse.shape, "test:", test_rmse.shape)
    print("Saved to", OUT)


if __name__ == "__main__":
    main()
