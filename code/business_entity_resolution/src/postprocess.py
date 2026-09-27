"""Business Entity Resolution — Stage 5: Postprocessing & Submission Generation."""
import sys
from pathlib import Path
from typing import Dict, Set

import numpy as np
import pandas as pd
import pickle

from config import CACHE_DIR, OUTPUT_DIR, DATA_RAW, NORM_DIR, MODELS_DIR

sys.path.insert(0, str(Path(__file__).resolve().parent))

TEST_FEATURES_PATH = CACHE_DIR / "test_features.parquet"
MODEL_PATH = MODELS_DIR / "lgbm_model.pkl"
THRESHOLD_PATH = MODELS_DIR / "optimal_threshold.txt"
CANDIDATE_PATH = CACHE_DIR / "test_candidates.tsv"

def load_model(model_path: Path):
    with open(model_path, "rb") as f:
        models = pickle.load(f)
    return models

def load_threshold(threshold_path: Path) -> float:
    if threshold_path.exists():
        with open(threshold_path, "r") as f:
            return float(f.read().strip())
    return 0.5  # default fallback

def predict_test(models, test_features_path: Path) -> pd.DataFrame:
    features = pd.read_parquet(test_features_path)
    feature_cols = [c for c in features.columns if c not in ("source1_entity_id", "candidate_entity_id", "source")]
    X = features[feature_cols].fillna(0).values.astype(np.float32)

    # Average predictions across all fold models
    preds = np.zeros(len(X))
    for model in models:
        preds += model.predict_proba(X)[:, 1]
    preds /= len(models)

    features["match_score"] = preds
    return features

def generate_output(features: pd.DataFrame, threshold: float) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate matching_results.tsv
    # Format: left_entity_id, right_entity_id, confidence_score
    matched = features[features["match_score"] >= threshold].copy()
    
    # Output format: left_entity_id, right_entity_id, confidence_score
    output_df = matched[["source1_entity_id", "candidate_entity_id", "match_score"]].copy()
    output_df.columns = ["left_entity_id", "right_entity_id", "confidence_score"]
    output_df = output_df.sort_values(["left_entity_id", "confidence_score"], ascending=[True, False])
    
    output_path = OUTPUT_DIR / "matching_results.tsv"
    output_df.to_csv(output_path, sep="\t", index=False)
    print(f"  Saved matching_results.tsv: {len(output_df):,} pairs")

    # Also generate candidate_pairs.tsv for validation (optional but expected)
    # Format: source1_entity_id, candidate_entity_ids
    if CANDIDATE_PATH.exists():
        candidates = pd.read_csv(CANDIDATE_PATH, sep="\t", dtype=str).fillna("")
        cand_output = OUTPUT_DIR / "candidate_pairs.tsv"
        candidates.to_csv(cand_output, sep="\t", index=False)
        print(f"  Saved candidate_pairs.tsv: {len(candidates):,} S1 entities")
    else:
        print("  Warning: candidate_pairs.tsv not found (test candidates missing)")

    # Print summary stats
    print(f"\n  Matching results summary:")
    print(f"    Total candidate pairs: {len(features):,}")
    print(f"    Matched pairs (score >= {threshold}): {len(output_df):,}")
    print(f"    Unique S1 entities with matches: {output_df['left_entity_id'].nunique()}")
    print(f"    Avg confidence: {output_df['confidence_score'].mean():.4f}")

def main():
    print("="*60)
    print("POSTPROCESSING — Generating Submission Files")
    print("="*60)

    model_path = MODEL_PATH
    threshold_path = THRESHOLD_PATH
    test_features_path = TEST_FEATURES_PATH

    print(f"\nLoading model from {model_path}")
    models = load_model(model_path)
    print(f"  Loaded {len(models)} fold models")

    print(f"\nLoading threshold from {threshold_path}")
    threshold = load_threshold(threshold_path)
    print(f"  Threshold: {threshold}")

    print(f"\nPredicting test features from {test_features_path}")
    features = predict_test(models, test_features_path)
    print(f"  Predicted {len(features):,} pairs")

    print("\nGenerating output files...")
    generate_output(features, threshold)

    print("\nPostprocessing complete.")

if __name__ == "__main__":
    main()