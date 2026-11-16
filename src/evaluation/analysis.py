"""Results tables, plots, baseline comparison, cold-start test and error analysis.

Run from the repo root (after preprocessing and every model script):
    python src/evaluation/analysis.py
Reads  data/splits/train_loo.csv, test_loo.csv, data/ml-latest-small/movies.csv,
       outputs/{model}_top10.csv, outputs/rmse_*.csv
Writes outputs/analysis/*.csv, outputs/analysis/results_table.md
       outputs/figures/*.png
Only the saved top-10 files are evaluated (integration contract); the cold-start test calls
the content-based helpers in src/models/content.py, which the hybrid uses as its fallback.

Protocol (same as metrics.py): leave-one-out, a held-out item is relevant if rating >= 4.0,
so 363 of 610 users are evaluated. With one relevant item per user, Recall@10 equals the
hit rate and Precision@10 is at most 0.1.
"""
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))
import content  # noqa: E402

RAW = ROOT / "data" / "ml-latest-small"
SPLITS = ROOT / "data" / "splits"
OUT = ROOT / "outputs"
TAB = OUT / "analysis"
FIG = OUT / "figures"
MODELS = ["popularity", "random", "user_cf", "item_cf", "svd", "content", "hybrid", "ncf"]
BASELINES = {"popularity", "random"}
LABEL = {"popularity": "Popularity", "random": "Random", "user_cf": "User-based CF",
         "item_cf": "Item-based CF", "svd": "SVD", "content": "Content-based",
         "hybrid": "Hybrid", "ncf": "NCF"}
K = 10
N_BOOT = 5000
SEED = 42

# Reference palette (light surface); emphasis pattern = highlight one, gray the rest
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"     # categorical slots 1-3
GRAY_MODEL, GRAY_BASE = "#a8a7a0", "#d6d5cf"


# --------------------------------------------------------------------- helpers
def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "legend.frameon": False, "savefig.dpi": 160, "savefig.bbox": "tight",
    })


def ndcg_single(rank):
    """NDCG@10 with one relevant item at 1-based `rank` (0 = not in the list)."""
    return 0.0 if rank == 0 else 1.0 / np.log2(rank + 1)


def boot_ci(x, rng, stat=np.mean):
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    b = stat(x[idx], axis=1)
    return np.percentile(b, [2.5, 97.5])


def mcnemar(a, b):
    """Exact McNemar test on paired hit vectors: p-value of the discordant split."""
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    if n10 + n01 == 0:
        return n10, n01, 1.0
    return n10, n01, binomtest(n10, n10 + n01, 0.5).pvalue


def genre_sets(movies):
    return {m: set(g.split("|")) - {"(no genres listed)"}
            for m, g in zip(movies.movieId, movies.genres)}


def bucket(values, edges, labels):
    return pd.cut(values, bins=edges, labels=labels, right=True)


# ------------------------------------------------------------------- load data
def load():
    train = pd.read_csv(SPLITS / "train_loo.csv")
    test = pd.read_csv(SPLITS / "test_loo.csv")
    movies = pd.read_csv(RAW / "movies.csv")
    recs = {}
    for m in MODELS:
        df = pd.read_csv(OUT / f"{m}_top10.csv").sort_values(["userId", "rank"])
        recs[m] = df.groupby("userId").movieId.apply(list).to_dict()
    return train, test, movies, recs


def hit_ranks(recs, rel):
    """rank (1..K) of the held-out relevant item in each user's list, 0 if missed."""
    out = {}
    for m, lists in recs.items():
        out[m] = np.array([lists[u].index(i) + 1 if i in lists[u][:K] else 0
                           for u, i in zip(rel.userId, rel.movieId)])
    return out


# ------------------------------------------------------- 1) results + baselines
def results_table(ranks, rng):
    rmse = {}
    for p in OUT.glob("rmse_*.csv"):
        for _, r in pd.read_csv(p).iterrows():
            rmse[r["model"]] = float(r["rmse"])
    rows = []
    for m in MODELS:
        hit = (ranks[m] > 0).astype(float)
        nd = np.array([ndcg_single(r) for r in ranks[m]])
        lo, hi = boot_ci(hit, rng)
        nlo, nhi = boot_ci(nd, rng)
        rows.append({"model": m, "users": len(hit), "hits": int(hit.sum()),
                     "precision_at_10": hit.mean() / K, "recall_at_10": hit.mean(),
                     "recall_ci_low": lo, "recall_ci_high": hi,
                     "ndcg_at_10": nd.mean(), "ndcg_ci_low": nlo, "ndcg_ci_high": nhi,
                     "mrr_at_10": np.mean([1 / r if r else 0 for r in ranks[m]]),
                     "rmse": rmse.get(m, np.nan)})
    return pd.DataFrame(rows)


def baseline_comparison(ranks, rng):
    """Every model vs. the popularity baseline, and the hybrid vs. every other model."""
    hits = {m: ranks[m] > 0 for m in MODELS}
    rows = []
    pairs = [(m, "popularity") for m in MODELS if m != "popularity"]
    pairs += [("hybrid", m) for m in MODELS if m not in ("hybrid", "popularity")]
    for a, b in pairs:
        d = hits[a].astype(float) - hits[b].astype(float)
        lo, hi = boot_ci(d, rng)
        n10, n01, p = mcnemar(hits[a], hits[b])
        rows.append({"model": a, "vs": b, "recall_diff": d.mean(), "diff_ci_low": lo,
                     "diff_ci_high": hi, "relative_lift": (hits[a].mean() / hits[b].mean() - 1)
                     if hits[b].mean() else np.nan,
                     "only_model_hits": n10, "only_vs_hits": n01, "mcnemar_p": p})
    return pd.DataFrame(rows)


def beyond_accuracy(recs, train, movies):
    """Catalog coverage, popularity bias, intra-list diversity, personalisation (all 610 users)."""
    pop = train.groupby("movieId").size()
    pct = pop.rank(pct=True)                       # popularity percentile of each movie
    n_catalog = train.movieId.nunique()
    gs = genre_sets(movies)
    rng = np.random.default_rng(SEED)
    rows = []
    for m in MODELS:
        lists = list(recs[m].values())
        flat = [i for l in lists for i in l]
        ild = []
        for l in lists:
            d = [1 - len(gs[a] & gs[b]) / max(len(gs[a] | gs[b]), 1) for a, b in combinations(l, 2)]
            ild.append(np.mean(d))
        sample = rng.choice(len(lists), size=(3000, 2))
        overlap = [len(set(lists[a]) & set(lists[b])) / K for a, b in sample if a != b]
        rows.append({"model": m,
                     "catalog_coverage": len(set(flat)) / n_catalog,
                     "unique_items": len(set(flat)),
                     "mean_ratings_of_recs": pop.reindex(flat).fillna(0).mean(),
                     "mean_popularity_percentile": pct.reindex(flat).fillna(0).mean(),
                     "intra_list_genre_diversity": np.mean(ild),
                     "personalisation": 1 - np.mean(overlap)})
    return pd.DataFrame(rows)


def list_overlap(recs):
    """Mean per-user Jaccard similarity between two models' top-10 lists."""
    ms = [m for m in MODELS if m != "random"]
    users = sorted(recs["hybrid"])
    J = pd.DataFrame(1.0, index=ms, columns=ms)
    for a, b in combinations(ms, 2):
        j = np.mean([len(set(recs[a][u]) & set(recs[b][u])) / len(set(recs[a][u]) | set(recs[b][u]))
                     for u in users])
        J.loc[a, b] = J.loc[b, a] = j
    return J


# ----------------------------------------------------------- 2) cold-start test
def cold_start(train, rel, ranks):
    """Simulate a brand-new user: hide the whole history and give the fallback only what a
    sign-up form would collect, then check whether the held-out liked movie is in the top 10.

    popularity        no information           -> most-rated movies
    genres            3 favourite genres       -> content.recommend_from_genres  (shipped)
    liked_n           n most recent 4+ movies  -> content.recommend_from_movies  (shipped)
    genre_popularity  3 favourite genres       -> most-rated movies having >=1 of those genres
                      (an alternative fallback tested here, not in the shipped code)
    """
    cmovies, vec, X = content.build_features()
    pop = train.groupby("movieId").size().reindex(cmovies.movieId).fillna(0).to_numpy()
    gs = genre_sets(cmovies)
    liked = train[train.rating >= 4.0].sort_values(["userId", "timestamp", "movieId"])
    liked_by_user = liked.groupby("userId").movieId.apply(list).to_dict()
    pop_order = cmovies.movieId.to_numpy()[np.argsort(-pop, kind="stable")]

    def fav_genres(u, n=3):
        cnt = pd.Series([g for m in liked_by_user.get(u, []) for g in gs[m]]).value_counts()
        return list(cnt.index[:n])

    scen = {"popularity": [], "genres": [], "genre_popularity": [],
            "liked_1": [], "liked_3": [], "liked_5": []}
    for u, target in zip(rel.userId, rel.movieId):
        scen["popularity"].append(target in pop_order[:K])
        g = fav_genres(u)
        r = content.recommend_from_genres(g, vec, cmovies, X, n=K, pop=pop)
        scen["genres"].append(target in set(r.movieId))
        gp = [m for m in pop_order if gs[m] & set(g)][:K]
        scen["genre_popularity"].append(target in gp)
        for n in (1, 3, 5):
            ids = liked_by_user.get(u, [])[-n:]
            r = content.recommend_from_movies(ids, cmovies, X, n=K, pop=pop)
            scen[f"liked_{n}"].append(target in set(r.movieId))
    desc = {"popularity": "No information - most-rated movies",
            "genres": "3 favourite genres - content (shipped)",
            "genre_popularity": "3 favourite genres - most-rated in genre (alternative)",
            "liked_1": "1 liked movie - content (shipped)",
            "liked_3": "3 liked movies - content (shipped)",
            "liked_5": "5 liked movies - content (shipped)"}
    rows = [{"scenario": s, "description": desc[s], "hits": int(np.sum(v)),
             "hit_rate_at_10": float(np.mean(v))} for s, v in scen.items()]
    rows.append({"scenario": "warm_hybrid", "description": "Full history - hybrid (reference)",
                 "hits": int((ranks["hybrid"] > 0).sum()),
                 "hit_rate_at_10": float((ranks["hybrid"] > 0).mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------- 3) error analysis
ITEM_EDGES = [-1, 0, 5, 20, 50, 10_000]
ITEM_LABELS = ["0 (unseen)", "1-5", "6-20", "21-50", "51+"]
USER_EDGES = [0, 30, 60, 150, 10_000]
USER_LABELS = ["<=30", "31-60", "61-150", "151+"]


def error_analysis(train, rel, ranks, movies, recs):
    item_pop = train.groupby("movieId").size()
    user_n = train.groupby("userId").size()
    df = rel[["userId", "movieId", "rating"]].copy()
    df["item_train_ratings"] = item_pop.reindex(df.movieId).fillna(0).astype(int).to_numpy()
    df["user_train_ratings"] = user_n.reindex(df.userId).to_numpy()
    df["item_bucket"] = bucket(df.item_train_ratings, ITEM_EDGES, ITEM_LABELS)
    df["user_bucket"] = bucket(df.user_train_ratings, USER_EDGES, USER_LABELS)
    for m in MODELS:
        df[f"hit_{m}"] = ranks[m] > 0

    def by(col):
        g = df.groupby(col, observed=False)
        t = g.size().rename("users").to_frame()
        for m in MODELS:
            t[m] = g[f"hit_{m}"].mean()
        return t.reset_index()

    by_item, by_user = by("item_bucket"), by("user_bucket")

    # Genre of the held-out item: hit rate of the hybrid per genre (multi-label)
    gs = genre_sets(movies)
    long = [(g, h) for m, h in zip(df.movieId, df.hit_hybrid) for g in gs.get(m, [])]
    by_genre = (pd.DataFrame(long, columns=["genre", "hit"]).groupby("genre").hit
                .agg(users="size", hybrid_hit_rate="mean").reset_index()
                .sort_values("users", ascending=False))

    # Complementarity: could any single component have found what the hybrid missed?
    comps = ["user_cf", "item_cf", "svd", "content", "ncf"]
    any_comp = np.any([df[f"hit_{m}"] for m in comps], axis=0)
    blend_comps = ["user_cf", "item_cf", "svd", "content"]
    any_blend = np.any([df[f"hit_{m}"] for m in blend_comps], axis=0)
    summary = {
        "users_evaluated": len(df),
        "held_out_items_unseen_in_train": int((df.item_train_ratings == 0).sum()),
        "held_out_items_with_le5_train_ratings": int((df.item_train_ratings <= 5).sum()),
        "median_train_ratings_of_held_out_item": float(df.item_train_ratings.median()),
        "hybrid_hits": int(df.hit_hybrid.sum()),
        "users_hit_by_any_blend_component": int(any_blend.sum()),
        "users_hit_by_any_model_incl_ncf": int((any_comp | df.hit_hybrid | df.hit_popularity).sum()),
        "hybrid_misses_hit_by_some_component": int((any_comp & ~df.hit_hybrid).sum()),
        "users_missed_by_every_model": int((~np.any([df[f"hit_{m}"] for m in MODELS], axis=0)).sum()),
        "median_train_ratings_of_items_hybrid_hits": float(df.loc[df.hit_hybrid, "item_train_ratings"].median()),
        "median_train_ratings_of_items_hybrid_misses": float(df.loc[~df.hit_hybrid, "item_train_ratings"].median()),
    }

    # Qualitative examples (titles) for the write-up
    title = movies.set_index("movieId").title
    trs = train.sort_values(["userId", "timestamp"])
    ex = []
    for hit_flag, label in ((True, "hit"), (False, "miss")):
        sub = df[df.hit_hybrid == hit_flag]
        if not hit_flag:   # pick illustrative misses: one unseen item, one long-tail, one popular
            picks = [sub[sub.item_train_ratings == 0], sub[sub.item_train_ratings.between(1, 5)],
                     sub[sub.item_train_ratings > 50]]
            sub = pd.concat([p.head(1) for p in picks])
        for _, r in sub.head(3).iterrows():
            recent = trs[(trs.userId == r.userId) & (trs.rating >= 4)].movieId.tail(3)
            ex.append({"type": label, "userId": r.userId,
                       "recent_liked": "; ".join(title[m] for m in recent),
                       "held_out": f"{title.get(r.movieId, r.movieId)} ({r.item_train_ratings} train ratings)",
                       "hybrid_top3": "; ".join(title[m] for m in recs["hybrid"][r.userId][:3])})
    return df, by_item, by_user, by_genre, summary, pd.DataFrame(ex)


# ---------------------------------------------------------------------- plots
def bar_with_ci(ax, res, col, lo, hi, title, xlabel):
    res = res.sort_values(col)
    y = np.arange(len(res))
    colors = [BLUE if m == "hybrid" else GRAY_BASE if m in BASELINES else GRAY_MODEL
              for m in res.model]
    ax.barh(y, res[col], height=0.62, color=colors, zorder=2)
    ax.errorbar(res[col], y, xerr=[res[col] - res[lo], res[hi] - res[col]], fmt="none",
                ecolor=INK2, elinewidth=1, capsize=3, zorder=3)
    ax.set_yticks(y, [LABEL[m] + (" (baseline)" if m in BASELINES else "") for m in res.model])
    for yi, (m, v) in enumerate(zip(res.model, res[col])):
        if m in ("hybrid", "popularity"):
            ax.text(res[hi].iloc[yi] + ax.get_xlim()[1] * 0.01, yi, f"{v:.3f}", va="center",
                    color=INK if m == "hybrid" else INK2,
                    fontweight="bold" if m == "hybrid" else "normal")
    ax.grid(axis="y", visible=False)
    ax.set_title(title)
    ax.set_xlabel(xlabel)


def plot_results(res):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    bar_with_ci(axes[0], res, "recall_at_10", "recall_ci_low", "recall_ci_high",
                "Recall@10 (= hit rate)", "share of users whose liked held-out movie is in the top 10")
    bar_with_ci(axes[1], res, "ndcg_at_10", "ndcg_ci_low", "ndcg_ci_high",
                "NDCG@10", "rewards placing the held-out movie near the top")
    for ax in axes:
        ax.set_xlim(0, max(res.recall_ci_high.max(), 0.01) * 1.18 if ax is axes[0]
                    else res.ndcg_ci_high.max() * 1.18)
    fig.suptitle("Hybrid vs. single models and baselines - leave-one-out, 363 users, "
                 "error bars = 95% bootstrap CI", x=0.01, ha="left", color=INK2, fontsize=10, y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "accuracy_comparison.png")
    plt.close(fig)


def plot_lift(cmp):
    d = cmp[cmp.vs == "popularity"].sort_values("recall_diff")
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    y = np.arange(len(d))
    colors = [BLUE if m == "hybrid" else GRAY_BASE if m in BASELINES else GRAY_MODEL for m in d.model]
    ax.barh(y, d.recall_diff * 100, height=0.62, color=colors, zorder=2)
    ax.errorbar(d.recall_diff * 100, y, xerr=[(d.recall_diff - d.diff_ci_low) * 100,
                                              (d.diff_ci_high - d.recall_diff) * 100],
                fmt="none", ecolor=INK2, elinewidth=1, capsize=3, zorder=3)
    ax.axvline(0, color=AXIS, lw=1, zorder=1)
    ax.set_yticks(y, [LABEL[m] for m in d.model])
    ax.grid(axis="y", visible=False)
    ax.set_title("Change in Recall@10 vs. popularity baseline")
    ax.set_xlabel("percentage points (95% paired bootstrap CI) - CI crossing 0 = not significant")
    fig.tight_layout()
    fig.savefig(FIG / "lift_vs_popularity.png")
    plt.close(fig)


def plot_beyond(ba):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    ba = ba.set_index("model").loc[MODELS[::-1]]
    specs = [("catalog_coverage", "Catalog coverage", "share of catalog ever recommended", 100, "%"),
             ("mean_popularity_percentile", "Popularity bias (higher = more mainstream)",
              "mean popularity percentile of recs",
              100, ""),
             ("intra_list_genre_diversity", "Genre diversity", "mean pairwise genre distance in a list",
              1, "")]
    y = np.arange(len(ba))
    colors = [BLUE if m == "hybrid" else GRAY_BASE if m in BASELINES else GRAY_MODEL for m in ba.index]
    for ax, (col, title, xl, scale, unit) in zip(axes, specs):
        ax.barh(y, ba[col] * scale, height=0.62, color=colors, zorder=2)
        v = ba.loc["hybrid", col] * scale
        ax.text(v + ax.get_xlim()[1] * 0.02, list(ba.index).index("hybrid"),
                f"{v:.1f}{unit}" if scale == 100 else f"{v:.2f}", va="center",
                color=INK, fontweight="bold")
        ax.grid(axis="y", visible=False)
        ax.set_title(title)
        ax.set_xlabel(xl)
    axes[0].set_yticks(y, [LABEL[m] for m in ba.index])
    fig.tight_layout()
    fig.savefig(FIG / "beyond_accuracy.png")
    plt.close(fig)


def plot_overlap(J):
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "blue", ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    M = J.to_numpy().copy()
    M[np.triu_indices_from(M)] = np.nan          # lower triangle only: J is symmetric
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=max(np.nanmax(M), 0.3))
    labels = [LABEL[m] for m in J.index]
    ax.set_xticks(range(len(J)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(J)), labels)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(len(J)):
        for j in range(len(J)):
            if i > j:
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if M[i, j] > 0.45 * np.nanmax(M) else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.outline.set_visible(False)
    cb.set_label("mean Jaccard overlap of top-10 lists", color=INK2)
    ax.set_title("How similar are the models' recommendations?")
    fig.tight_layout()
    fig.savefig(FIG / "model_overlap.png")
    plt.close(fig)


def plot_cold(cs):
    cs = cs.iloc[::-1].reset_index(drop=True)
    warm = cs[cs.scenario == "warm_hybrid"].hit_rate_at_10.iloc[0]
    cs = cs[cs.scenario != "warm_hybrid"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    y = np.arange(len(cs))
    colors = [GRAY_BASE if s == "popularity" else BLUE for s in cs.scenario]
    ax.barh(y, cs.hit_rate_at_10 * 100, height=0.62, color=colors, zorder=2)
    for yi, v in zip(y, cs.hit_rate_at_10):
        ax.text(v * 100 + 0.15, yi, f"{v * 100:.1f}%", va="center", color=INK2, fontsize=9)
    ax.axvline(warm * 100, color=INK2, lw=1.5, zorder=3)
    ax.text(warm * 100, len(cs) - 0.45, f" warm hybrid {warm * 100:.1f}%", color=INK2,
            fontsize=9, va="bottom")
    ax.set_yticks(y, cs.description)
    ax.set_ylim(-0.6, len(cs) + 0.2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("hit rate @10 (held-out liked movie in top 10), 363 simulated new users")
    fig.suptitle("Cold-start fallback: what a new user's sign-up info buys", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "cold_start.png")
    plt.close(fig)


def plot_breakdowns(by_item, by_user):
    series = [("hybrid", BLUE), ("popularity", ORANGE), ("content", AQUA)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, t, col, xl in ((axes[0], by_item, "item_bucket",
                            "ratings the held-out movie has in training"),
                           (axes[1], by_user, "user_bucket", "ratings the user has in training")):
        x = np.arange(len(t))
        w = 0.26
        for k, (m, c) in enumerate(series):
            ax.bar(x + (k - 1) * (w + 0.01), t[m] * 100, width=w, color=c, label=LABEL[m], zorder=2)
        ax.set_xticks(x, [f"{b}\nn={n}" for b, n in zip(t[col], t.users)])
        ax.set_xlabel(xl)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("hit rate @10 (%)")
    axes[0].set_title("Misses concentrate on long-tail held-out movies")
    axes[1].set_title("By user activity")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "error_breakdown.png")
    plt.close(fig)


# ----------------------------------------------------------------------- main
def fmt_table(res, cmp):
    vs_pop = cmp[cmp.vs == "popularity"].set_index("model")
    lines = ["| Model | P@10 | R@10 (95% CI) | NDCG@10 | MRR@10 | RMSE | vs. popularity (McNemar p) |",
             "|---|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        name = LABEL[r.model] + (" (baseline)" if r.model in BASELINES else "")
        if r.model == "hybrid":
            name = f"**{name}**"
        rm = "-" if np.isnan(r.rmse) else f"{r.rmse:.4f}"
        if r.model == "popularity":
            vp = "-"
        else:
            v = vs_pop.loc[r.model]
            p = "p<0.01" if v.mcnemar_p < 0.01 else f"p={v.mcnemar_p:.2f}"
            vp = f"{v.recall_diff * 100:+.1f} pp ({p})"
        lines.append(f"| {name} | {r.precision_at_10:.4f} | {r.recall_at_10:.4f} "
                     f"({r.recall_ci_low:.3f}-{r.recall_ci_high:.3f}) | {r.ndcg_at_10:.4f} | "
                     f"{r.mrr_at_10:.4f} | {rm} | {vp} |")
    return "\n".join(lines)


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    style()
    rng = np.random.default_rng(SEED)
    train, test, movies, recs = load()
    rel = test[test.rating >= 4.0].sort_values("userId").reset_index(drop=True)
    ranks = hit_ranks(recs, rel)

    res = results_table(ranks, rng)
    cmp = baseline_comparison(ranks, rng)
    ba = beyond_accuracy(recs, train, movies)
    J = list_overlap(recs)
    print("Running cold-start simulation ...")
    cs = cold_start(train, rel, ranks)
    per_user, by_item, by_user, by_genre, summary, examples = error_analysis(
        train, rel, ranks, movies, recs)

    res.round(4).to_csv(TAB / "results_with_ci.csv", index=False)
    cmp.round(4).to_csv(TAB / "baseline_comparison.csv", index=False)
    ba.round(4).to_csv(TAB / "beyond_accuracy.csv", index=False)
    J.round(4).to_csv(TAB / "model_overlap.csv")
    cs.round(4).to_csv(TAB / "cold_start_results.csv", index=False)
    by_item.round(4).to_csv(TAB / "errors_by_item_popularity.csv", index=False)
    by_user.round(4).to_csv(TAB / "errors_by_user_activity.csv", index=False)
    by_genre.round(4).to_csv(TAB / "errors_by_genre.csv", index=False)
    per_user.to_csv(TAB / "per_user_hits.csv", index=False)
    examples.to_csv(TAB / "error_examples.csv", index=False)
    pd.Series(summary).to_csv(TAB / "error_summary.csv", header=["value"], lineterminator="\n")
    (TAB / "results_table.md").write_text(fmt_table(res, cmp) + "\n", encoding="utf-8")

    plot_results(res)
    plot_lift(cmp)
    plot_beyond(ba)
    plot_overlap(J)
    plot_cold(cs)
    plot_breakdowns(by_item, by_user)

    pd.set_option("display.width", 200)
    print(fmt_table(res, cmp), "\n")
    print(cmp.round(4).to_string(index=False), "\n")
    print(ba.round(4).to_string(index=False), "\n")
    print(J.round(3).to_string(), "\n")
    print(cs.round(4).to_string(index=False), "\n")
    print(by_item.round(3).to_string(index=False), "\n")
    print(by_user.round(3).to_string(index=False), "\n")
    print(by_genre.round(3).to_string(index=False), "\n")
    for k, v in summary.items():
        print(f"{k:45s} {v}")
    print()
    print(examples.to_string(index=False))
    print(f"\nTables -> {TAB}\nFigures -> {FIG}")


if __name__ == "__main__":
    main()
