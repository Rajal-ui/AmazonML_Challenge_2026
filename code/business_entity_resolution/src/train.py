"""Business Entity Resolution — Stage 4: LightGBM Training (F0.5 Optimized)."""
import sys
from pathlib import Path
from typing import Dict, List, Set

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score

from config import CACHE_DIR, MODELS_DIR, DATA_RAW, NORM_DIR

sys.path.insert(0, str(Path(__file__).resolve().parent))

TRAIN_FEATURES_PATH = CACHE_DIR / "train_features.parquet"
TEST_FEATURES_PATH = CACHE_DIR / "test_features.parquet"
MODEL_PATH = MODELS_DIR / "lgbm_model.pkl"
THRESHOLD_PATH = MODELS_DIR / "optimal_threshold.txt"

def load_ground_truth(path: Path) -> Dict[str, Set[str]]:
    gt: Dict[str, Set[str]] = {}
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    for _, row in df.iterrows():
        s1_id = row["source1_entity_id"]
        matched_str = row["matched_entity_ids"]
        gt[s1_id] = set(matched_str.split(",")) if matched_str else set()
    return gt

def build_labels(features_path: Path, ground_truth: Dict[str, Set[str]]) -> pd.Series:
    df = pd.read_parquet(features_path)
    labels = []
    for _, row in df.iterrows():
        s1_id = row["source1_entity_id"]
        cand_id = row["candidate_entity_id"]
        is_match = 1 if (s1_id in ground_truth and cand_id in ground_truth[s1_id]) else 0
        labels.append(is_match)
    return pd.Series(labels, dtype=int)

def train_model(features_path: Path, model_path: Path, threshold_path: Path) -> float:
    print(f"\nLoading features from {features_path}")
    features = pd.read_parquet(features_path)
    print(f"  Shape: {features.shape}")
    print(f"  Columns: {list(features.columns)}")

    gt_path = DATA_RAW / "train" / "train_ground_truth.tsv"
    ground_truth = load_ground_truth(gt_path)
    print(f"  Ground truth: {len(ground_truth)} S1 entities")

    print("  Building labels...")
    labels = build_labels(features_path, ground_truth)
    print(f"  Positive rate: {labels.mean():.4f}")

    feature_cols = [c for c in features.columns if c not in ("source1_entity_id", "candidate_entity_id", "source")]
    X = features[feature_cols].fillna(0).values.astype(np.float32)
    y = labels.values

    model_path.parent.mkdir(parents=True, exist_ok=True)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 100,
        "verbose": -1,
        "n_estimators": 1000,
        "random_state": 42,
    }

    print(f"\nTraining LightGBM with 5-fold CV...")
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        models.append(model)
        print(f"  Fold {fold+1}: best_iter={model.best_iteration_}")
        del model, X_train, X_val, y_train, y_val

    # Save all fold models
    import pickle
    with open(model_path, "wb") as f:
        pickle.dump(models, f)
    print(f"\n  Saved {len(models)} fold models to {model_path}")
    return models

def optimize_threshold(models: List, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Optimize decision threshold for F0.5 score."""
    # Average predictions across all fold models
    preds_proba = np.zeros(len(X))
    for model in models:
        preds_proba += model.predict_proba(X)[:, 1]
    preds_proba /= len(models)

    best_thresh, best_f05 = 0.5, 0.0
    for thresh in [t / 100.0 for t in range(50, 95, 2)]:
        score = fbeta_score(y, (preds_proba >= thresh).astype(int), beta=0.5)
        if score > best_f05:
            best_f05, best_thresh = score, thresh

    print(f"Optimal Threshold: {best_thresh} | Train F0.5: {best_f05:.4f}")
    return best_thresh, best_f05

def main():
    print("="*60)
    print("TRAINING — LightGBM Match Scoring (F0.5 Optimized)")
    print("="*60)
    
    features = pd.read_parquet(TRAIN_FEATURES_PATH)
    feature_cols = [c for c in features.columns if c not in ("source1_entity_id", "candidate_entity_id", "source")]
    X = features[feature_cols].fillna(0).values.astype(np.float32)
    
    gt_path = DATA_RAW / "train" / "train_ground_truth.tsv"
    ground_truth = load_ground_truth(gt_path)
    labels = build_labels(TRAIN_FEATURES_PATH, ground_truth)
    y = labels.values
    
    models = train_model(TRAIN_FEATURES_PATH, MODEL_PATH, THRESHOLD_PATH)
    best_thresh, best_f05 = optimize_threshold(models, X, y)
    
    # Save threshold
    THRESHOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(THRESHOLD_PATH, "w") as f:
        f.write(str(best_thresh))
    print(f"  Saved threshold to {THRESHOLD_PATH}")
    
    print("\nTraining complete.")

if __name__ == "__main__":
    main()