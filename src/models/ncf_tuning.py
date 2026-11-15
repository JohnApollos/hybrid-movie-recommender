"""NCF hyper-parameter tuning on a validation split + training-curve plot.

Run from the repo root:  python src/models/ncf_tuning.py
Needs  src/models/ncf.py (imported; swaps in each configuration without editing it)
Writes outputs/ncf_tuning_results.csv   one row per configuration
       outputs/ncf_tuning_logs.csv      per-epoch loss and validation recall
       outputs/ncf_training_curves.png  training loss + validation recall curves
       outputs/ncf_tuned_top10.csv      only if a tuned config beats the baseline on validation

Configurations change one setting at a time from the baseline. Selection uses the validation
split only (each user's last N_VAL training ratings); the test set is used once at the end.
"""
import contextlib
import functools
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ncf  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data" / "splits"
OUT = ROOT / "outputs"
MAX_EPOCHS = 30
_ORIGINAL_NEUMF = ncf.NeuMF

BASE = dict(dim=32, layers=(64, 32, 16), dropout=0.1, lr=1e-3, neg=4)
CONFIGS = [
    dict(name="baseline", **BASE),
    dict(name="dim16", **{**BASE, "dim": 16, "layers": (32, 16, 8)}),
    dict(name="dim64", **{**BASE, "dim": 64, "layers": (128, 64, 32)}),
    dict(name="dropout0.3", **{**BASE, "dropout": 0.3}),
    dict(name="lr5e-4", **{**BASE, "lr": 5e-4}),
    dict(name="neg8", **{**BASE, "neg": 8}),
    dict(name="dim64_drop0.3", **{**BASE, "dim": 64, "layers": (128, 64, 32), "dropout": 0.3}),
]


def apply_config(cfg):
    """ncf.fit looks these names up at call time, so replacing them changes the model it builds."""
    ncf.NeuMF = functools.partial(_ORIGINAL_NEUMF, dim=cfg["dim"], layers=cfg["layers"],
                                  dropout=cfg["dropout"])
    ncf.LR = cfg["lr"]
    ncf.N_NEG = cfg["neg"]


def main():
    train = pd.read_csv(SPLITS / "train_loo.csv")
    test = pd.read_csv(SPLITS / "test_loo.csv")
    users = np.sort(train.userId.unique())
    movies = np.sort(train.movieId.unique())
    u_idx = pd.Series(np.arange(len(users)), index=users)
    m_idx = pd.Series(np.arange(len(movies)), index=movies)
    shape = (len(users), len(movies))

    def index_and_mask(df):
        ui, mi = u_idx[df.userId].to_numpy(), m_idx[df.movieId].to_numpy()
        B = np.zeros(shape, dtype=bool)
        B[ui, mi] = True
        return ui, mi, B

    # Validation split: each user's last N_VAL training ratings (same as the hybrid and ncf.py)
    ordered = train.sort_values(["userId", "timestamp", "movieId"])
    val = ordered.groupby("userId").tail(ncf.N_VAL)
    train_val = ordered.drop(val.index)
    rel = val[val.rating >= 4.0]
    V = np.zeros(shape, dtype=bool)
    V[u_idx[rel.userId].to_numpy(), m_idx[rel.movieId].to_numpy()] = True
    n_rel = int(V.sum())
    ui, mi, Bv = index_and_mask(train_val)

    # ---- 1) Run every configuration on the validation split
    print(f"Tuning {len(CONFIGS)} configurations ({n_rel} relevant held-out items) ...\n")
    results, logs = [], []
    for cfg in CONFIGS:
        apply_config(cfg)
        with contextlib.redirect_stdout(io.StringIO()):      # hide the per-epoch prints
            _, log, best_epoch = ncf.fit(ui, mi, Bv, MAX_EPOCHS, val=(V, n_rel))
        best_val = float(log.loc[log.epoch == best_epoch, "val_recall_at_10"].iloc[0])
        print(f"{cfg['name']:15s} best val recall@10 {best_val:.4f} at epoch {best_epoch:2d} "
              f"({len(log)} epochs run)", flush=True)
        log.insert(0, "config", cfg["name"])
        logs.append(log)
        results.append({"config": cfg["name"], "dim": cfg["dim"], "layers": str(cfg["layers"]),
                        "dropout": cfg["dropout"], "lr": cfg["lr"], "negatives": cfg["neg"],
                        "best_epoch": best_epoch, "val_recall_at_10": round(best_val, 4),
                        "epochs_run": len(log)})
    res = pd.DataFrame(results)
    logs = pd.concat(logs, ignore_index=True)
    res.to_csv(OUT / "ncf_tuning_results.csv", index=False)
    logs.to_csv(OUT / "ncf_tuning_logs.csv", index=False)

    best_row = res.sort_values("val_recall_at_10", ascending=False).iloc[0]
    base_val = float(res.loc[res.config == "baseline", "val_recall_at_10"].iloc[0])
    best_name = best_row.config
    print(f"\nBest on validation: {best_name} ({best_row.val_recall_at_10:.4f}) "
          f"vs baseline {base_val:.4f}")

    # ---- 2) Training-curve plot (best configuration drawn thicker)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for name, g in logs.groupby("config", sort=False):
        lw = 2.8 if name == best_name else 1.2
        ax[0].plot(g.epoch, g.train_loss, lw=lw, label=name)
        ax[1].plot(g.epoch, g.val_recall_at_10, lw=lw, label=name)
    ax[0].set(title="Training loss", xlabel="Epoch", ylabel="BCE loss")
    ax[1].set(title="Validation recall@10", xlabel="Epoch", ylabel="Recall@10")
    ax[1].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "ncf_training_curves.png", dpi=150)
    print("Saved outputs/ncf_training_curves.png")

    # ---- 3) If a tuned configuration beats the baseline on validation, retrain it and save top-10
    if best_name == "baseline":
        print("\nVERDICT: KEEP BASELINE. No tuned configuration beat it on validation, "
              "so outputs/ncf_top10.csv stays as it is.")
        return
    cfg = next(c for c in CONFIGS if c["name"] == best_name)
    apply_config(cfg)
    ui, mi, B = index_and_mask(train)
    with contextlib.redirect_stdout(io.StringIO()):
        model, _, _ = ncf.fit(ui, mi, B, int(best_row.best_epoch))
    scores = ncf.score_all(model, *shape)
    scores[B] = -np.inf
    rows = []
    for i, u in enumerate(users):
        top = np.argsort(-scores[i], kind="stable")[: ncf.K]
        for r, j in enumerate(top, start=1):
            rows.append((int(u), int(movies[j]), r, float(scores[i, j])))
    df = pd.DataFrame(rows, columns=["userId", "movieId", "rank", "score"])
    df.to_csv(OUT / "ncf_tuned_top10.csv", index=False)

    seen = set(zip(train.userId, train.movieId))
    leaked = sum((u, m) in seen for u, m in zip(df.userId, df.movieId))
    hits = df.merge(test[["userId", "movieId"]], on=["userId", "movieId"])
    print(f"\nncf_tuned  rows={len(df)} users={df.userId.nunique()} "
          f"already-rated recs={leaked} hit-rate@10={len(hits) / len(test):.3f}")
    print(f"\nVERDICT: tuned config '{best_name}' beat the baseline on validation "
          f"({best_row.val_recall_at_10:.4f} vs {base_val:.4f}).")


if __name__ == "__main__":
    main()
