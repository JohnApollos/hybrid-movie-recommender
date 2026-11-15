"""Neural Collaborative Filtering (NeuMF: GMF + MLP) trained on implicit feedback.

Run from the repo root:  python src/models/ncf.py
Reads  data/splits/train_loo.csv, test_loo.csv
Writes outputs/ncf_top10.csv        Columns: userId, movieId, rank, score
       outputs/ncf_training_log.csv epoch, train_loss, val_recall_at_10 (for plotting)
       outputs/ncf_model.pt         saved weights (gitignored, regenerate by running this script)

Every rated movie is a positive; N_NEG unrated movies are sampled as negatives each epoch.
The number of epochs is chosen on a validation split (each user's last N_VAL training
ratings), then the model is retrained on the full training set. The test set is used once.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data" / "splits"
OUT = ROOT / "outputs"
K = 10
N_VAL = 5          # last N training ratings per user form the validation set
SEED = 42
DIM = 32           # GMF embedding size
LAYERS = (64, 32, 16)
DROPOUT = 0.1
LR = 1e-3
BATCH = 1024
N_NEG = 4          # negatives sampled per positive
MAX_EPOCHS = 20
PATIENCE = 4       # stop if validation recall has not improved for this many epochs


class NeuMF(nn.Module):
    def __init__(self, n_users, n_items, dim=DIM, layers=LAYERS, dropout=DROPOUT):
        super().__init__()
        self.gmf_u = nn.Embedding(n_users, dim)
        self.gmf_i = nn.Embedding(n_items, dim)
        self.mlp_u = nn.Embedding(n_users, layers[0] // 2)
        self.mlp_i = nn.Embedding(n_items, layers[0] // 2)
        blocks, inp = [], layers[0]
        for out in layers[1:]:
            blocks += [nn.Linear(inp, out), nn.ReLU(), nn.Dropout(dropout)]
            inp = out
        self.mlp = nn.Sequential(*blocks)
        self.out = nn.Linear(dim + inp, 1)
        for emb in (self.gmf_u, self.gmf_i, self.mlp_u, self.mlp_i):
            nn.init.normal_(emb.weight, std=0.01)

    def forward(self, u, i):
        g = self.gmf_u(u) * self.gmf_i(i)
        m = self.mlp(torch.cat([self.mlp_u(u), self.mlp_i(i)], dim=-1))
        return self.out(torch.cat([g, m], dim=-1)).squeeze(-1)


def sample_negatives(ui, B, rng, n_items):
    """N_NEG random movies per positive, re-drawn if the user has already rated them."""
    neg = rng.integers(0, n_items, size=(len(ui), N_NEG))
    for _ in range(5):
        bad = B[ui[:, None], neg]
        if not bad.any():
            break
        neg[bad] = rng.integers(0, n_items, size=int(bad.sum()))
    return neg


@torch.no_grad()
def score_all(model, n_users, n_items):
    model.eval()
    items = torch.arange(n_items)
    out = np.empty((n_users, n_items), dtype=np.float32)
    for u in range(n_users):
        out[u] = model(torch.full((n_items,), u, dtype=torch.long), items).numpy()
    return out


def val_recall(model, B, V, n_rel):
    scores = score_all(model, B.shape[0], B.shape[1])
    scores[B] = -np.inf                       # ignore movies already in the training data
    idx = np.argpartition(-scores, K, axis=1)[:, :K]
    return V[np.arange(len(V))[:, None], idx].sum() / n_rel


def fit(ui, mi, B, max_epochs, val=None):
    n_users, n_items = B.shape
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = NeuMF(n_users, n_items)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    log, best, best_epoch, stale = [], -1.0, 0, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        neg = sample_negatives(ui, B, rng, n_items)
        u = torch.from_numpy(np.concatenate([ui, np.repeat(ui, N_NEG)])).long()
        i = torch.from_numpy(np.concatenate([mi, neg.ravel()])).long()
        y = torch.from_numpy(np.concatenate(
            [np.ones(len(ui)), np.zeros(len(ui) * N_NEG)]).astype(np.float32))
        perm = torch.randperm(len(y))
        total = 0.0
        for s in range(0, len(perm), BATCH):
            b = perm[s:s + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(u[b], i[b]), y[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        row = {"epoch": epoch, "train_loss": round(total / len(y), 4)}
        msg = f"epoch {epoch:2d}  loss {row['train_loss']:.4f}"
        if val is not None:
            recall = val_recall(model, B, *val)
            row["val_recall_at_10"] = round(float(recall), 4)
            msg += f"  val recall@10 {recall:.4f}"
            if recall > best:
                best, best_epoch, stale = recall, epoch, 0
            else:
                stale += 1
        log.append(row)
        print(msg, flush=True)
        if val is not None and stale >= PATIENCE:
            break
    return model, pd.DataFrame(log), best_epoch


def main():
    OUT.mkdir(exist_ok=True)
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

    # ---- 1) Pick the number of epochs on a validation split
    ordered = train.sort_values(["userId", "timestamp", "movieId"])
    val = ordered.groupby("userId").tail(N_VAL)
    train_val = ordered.drop(val.index)
    rel = val[val.rating >= 4.0]
    V = np.zeros(shape, dtype=bool)
    V[u_idx[rel.userId].to_numpy(), m_idx[rel.movieId].to_numpy()] = True
    ui, mi, Bv = index_and_mask(train_val)
    print(f"Validation run ({int(V.sum())} relevant held-out items):")
    _, log, best_epoch = fit(ui, mi, Bv, MAX_EPOCHS, val=(V, int(V.sum())))
    log.to_csv(OUT / "ncf_training_log.csv", index=False)
    print(f"\nBest epoch on validation: {best_epoch}")

    # ---- 2) Retrain on the full training set for that many epochs
    print("\nFinal run on the full training set:")
    ui, mi, B = index_and_mask(train)
    model, _, _ = fit(ui, mi, B, best_epoch)
    torch.save(model.state_dict(), OUT / "ncf_model.pt")

    scores = score_all(model, *shape)
    scores[B] = -np.inf                      # never recommend movies the user already rated
    rows = []
    for i, u in enumerate(users):
        top = np.argsort(-scores[i], kind="stable")[:K]
        for r, j in enumerate(top, start=1):
            rows.append((int(u), int(movies[j]), r, float(scores[i, j])))
    df = pd.DataFrame(rows, columns=["userId", "movieId", "rank", "score"])
    df.to_csv(OUT / "ncf_top10.csv", index=False)

    seen = set(zip(train.userId, train.movieId))
    leaked = sum((u, m) in seen for u, m in zip(df.userId, df.movieId))
    hits = df.merge(test[["userId", "movieId"]], on=["userId", "movieId"])
    print(f"\nncf        rows={len(df)} users={df.userId.nunique()} "
          f"already-rated recs={leaked} hit-rate@10={len(hits) / len(test):.3f}")


if __name__ == "__main__":
    main()
