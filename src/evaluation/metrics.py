import glob
import os
import numpy as np
import pandas as pd


def precision_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    rec_k = recommended[:k]
    if not rec_k:
        return 0.0
    hits = sum(1 for item in rec_k if item in relevant)
    return hits / k


def recall_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    if not relevant:
        return 0.0
    rec_k = recommended[:k]
    hits = sum(1 for item in rec_k if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    if not relevant:
        return 0.0
    rec_k = recommended[:k]
    dcg = sum(
        1.0 / np.log2(idx + 2)
        for idx, item in enumerate(rec_k)
        if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(idx + 2) for idx in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def load_saved_rmse(outputs_dir: str = "outputs") -> dict:
    """Load precomputed RMSE files like outputs/rmse_svd.csv and outputs/rmse_item_cf.csv."""
    rmse_map = {}
    for path in glob.glob(os.path.join(outputs_dir, "rmse_*.csv")):
        df = pd.read_csv(path)
        if "model" in df.columns and "rmse" in df.columns:
            for _, row in df.iterrows():
                rmse_map[str(row["model"]).strip().lower()] = round(float(row["rmse"]), 4)
    return rmse_map


def evaluate_all_models(
    k: int = 10,
    test_path: str = "data/splits/test_loo.csv",
    outputs_dir: str = "outputs",
    save_path: str = "outputs/evaluation_results.csv",
) -> pd.DataFrame:
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Test split not found at '{test_path}'. Run 'python src/preprocessing.py' first."
        )

    print(f"Loading ground-truth test split from: {test_path}")
    test_df = pd.read_csv(test_path)

    # Every row in test_loo.csv is the held-out relevant movie for that userId
    ground_truth = test_df.groupby("userId")["movieId"].apply(set).to_dict()
    print(f"Loaded {len(ground_truth)} test users.\n")

    rmse_lookup = load_saved_rmse(outputs_dir)

    top10_files = sorted(glob.glob(os.path.join(outputs_dir, "*_top10.csv")))
    if not top10_files:
        raise FileNotFoundError(f"No *_top10.csv files found in '{outputs_dir}/'.")

    results = []
    for file_path in top10_files:
        base = os.path.splitext(os.path.basename(file_path))[0]
        model_name = base.replace("_top10", "")
        pred_df = pd.read_csv(file_path)

        if "rank" in pred_df.columns:
            pred_df = pred_df.sort_values(["userId", "rank"], ascending=[True, True])
        elif "score" in pred_df.columns:
            pred_df = pred_df.sort_values(["userId", "score"], ascending=[True, False])

        user_recs = pred_df.groupby("userId")["movieId"].apply(list).to_dict()

        p_scores, r_scores, ndcg_scores = [], [], []
        for uid, rel_items in ground_truth.items():
            if uid in user_recs:
                recs = user_recs[uid]
                p_scores.append(precision_at_k(recs, rel_items, k=k))
                r_scores.append(recall_at_k(recs, rel_items, k=k))
                ndcg_scores.append(ndcg_at_k(recs, rel_items, k=k))

        rmse_val = rmse_lookup.get(model_name.lower(), "N/A")

        results.append({
            "Model": model_name,
            "Users_Evaluated": len(p_scores),
            f"P@{k}": round(float(np.mean(p_scores)), 4) if p_scores else np.nan,
            f"R@{k}": round(float(np.mean(r_scores)), 4) if r_scores else np.nan,
            f"NDCG@{k}": round(float(np.mean(ndcg_scores)), 4) if ndcg_scores else np.nan,
            "RMSE": rmse_val,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(save_path, index=False)
    print("=== Final Evaluation Results ===")
    print(results_df.to_string(index=False))
    print(f"\nSaved results to: {save_path}")
    return results_df


if __name__ == "__main__":
    os.makedirs("src/evaluation", exist_ok=True)
    evaluate_all_models()
