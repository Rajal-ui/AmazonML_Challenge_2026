"""
Business Entity Resolution — Stage 3: Feature Engineering

Generates pairwise features for candidate pairs (S1 entity + S2/S3 candidate).
Features include name similarity, address similarity, meta features from blocking.
"""

from __future__ import annotations

import gc
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# ─── Constants ────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "code" / "business_entity_resolution" / "data"
NORM_DIR = DATA_DIR / "normalized"
CAND_DIR = DATA_DIR / "candidates"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = PROJECT_ROOT / "code" / "business_entity_resolution" / "models"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
EMBEDDING_BATCH_SIZE = 1024

# Feature computation chunk size
FEATURE_CHUNK_SIZE = 50_000

# TF-IDF char n-gram config
TFIDF_NGRAM_RANGE = (3, 5)
TFIDF_MAX_FEATURES = 100_000
TFIDF_MIN_DF = 2

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _jaccard(set_a: Set[str], set_b: Set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


def _token_jaccard(tokens_a: str, tokens_b: str) -> float:
    """Token-level Jaccard on space-separated tokens."""
    if not tokens_a or not tokens_b:
        return 0.0
    return _jaccard(set(tokens_a.split()), set(tokens_b.split()))


def _levenshtein_ratio(a: str, b: str) -> float:
    """Normalized Levenshtein similarity (0-1)."""
    if not a or not b:
        return 0.0
    return fuzz.ratio(a, b) / 100.0


def _token_sort_ratio(a: str, b: str) -> float:
    """Token sort ratio (order-independent token similarity)."""
    if not a or not b:
        return 0.0
    return fuzz.token_sort_ratio(a, b) / 100.0


def _exact_match_flag(a: str, b: str) -> int:
    """1 if normalized strings are identical."""
    return int(a == b and a != "")


def _load_embeddings_cached(split: str, source: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load cached embeddings from blocking stage (float16 mmap)."""
    emb_path = CACHE_DIR / f"embeddings_{split}_{source}.npy"
    id_path = CACHE_DIR / f"ids_{split}_{source}.pkl"
    
    if not emb_path.exists() or not id_path.exists():
        raise FileNotFoundError(f"Embedding cache not found for {split}/{source}. Run blocking.py first.")
    
    embs = np.load(emb_path, mmap_mode="r")
    with open(id_path, "rb") as f:
        ids = pickle.load(f)
    
    return embs, np.array(ids)


def _build_id_to_emb_index(ids: np.ndarray) -> Dict[str, int]:
    """Build mapping from entity_id to embedding row index."""
    return {eid: i for i, eid in enumerate(ids)}


def _get_embedding_cosine(emb_a_idx: int, emb_b_idx: int, 
                          embs_a: np.ndarray, embs_b: np.ndarray) -> float:
    """Cosine similarity between two embeddings (already L2-normalized)."""
    if emb_a_idx < 0 or emb_b_idx < 0:
        return 0.0
    vec_a = embs_a[emb_a_idx].astype(np.float32)
    vec_b = embs_b[emb_b_idx].astype(np.float32)
    return float(np.dot(vec_a, vec_b))


# ─── TF-IDF Vectorizer (fit on combined name corpus, no label leakage) ────────

class NameTfidfVectorizer:
    """TF-IDF on character n-grams of normalized names, fit once on all names."""
    
    def __init__(self):
        self.vectorizer: Optional[TfidfVectorizer] = None
        self._fitted = False
    
    def fit(self, names: List[str]) -> None:
        """Fit on combined corpus of S1 + S2 + S3 names (no labels used)."""
        print(f"  Fitting TF-IDF on {len(names):,} names...")
        t0 = time.time()
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=TFIDF_NGRAM_RANGE,
            max_features=TFIDF_MAX_FEATURES,
            min_df=TFIDF_MIN_DF,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.vectorizer.fit(names)
        self._fitted = True
        print(f"  TF-IDF fitted: {len(self.vectorizer.vocabulary_):,} features in {time.time()-t0:.1f}s")
    
    def transform(self, names: List[str]) -> np.ndarray:
        """Transform names to TF-IDF vectors (returns dense for cosine)."""
        if not self._fitted:
            raise RuntimeError("Vectorizer not fitted")
        return self.vectorizer.transform(names).toarray()
    
    def cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Cosine similarity between two TF-IDF vectors."""
        if vec_a.size == 0 or vec_b.size == 0:
            return 0.0
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


# ─── Main Feature Extraction ──────────────────────────────────────────────────

def build_features_for_split(split: str) -> None:
    """
    Build feature table for candidate pairs.
    Input: candidate_pairs.tsv + normalized source files
    Output: features/{split}_features.parquet (or chunked)
    """
    print(f"\n{'='*80}")
    print(f"FEATURE ENGINEERING — {split.upper()}")
    print(f"{'='*80}")
    
    # Load candidate pairs
    cand_path = CAND_DIR / f"{split}_candidates.tsv"
    if not cand_path.exists():
        raise FileNotFoundError(f"Candidate file not found: {cand_path}. Run blocking.py first.")
    
    candidates = pd.read_csv(cand_path, sep="\t", dtype=str).fillna("")
    print(f"Loaded {len(candidates):,} S1 entities with candidates")
    
    # Load normalized sources (only needed columns)
    print("\n[1/6] Loading normalized sources...")
    s1_df = pd.read_csv(NORM_DIR / f"{split}_source1_normalized.tsv", sep="\t", dtype=str,
                        usecols=["entity_id", "country", "normalized_business_name", 
                                 "business_name_tokens", "normalized_business_address",
                                 "city", "state", "postal_code", "landmark"]).fillna("")
    s2_df = pd.read_csv(NORM_DIR / f"{split}_source2_normalized.tsv", sep="\t", dtype=str,
                        usecols=["entity_id", "country", "normalized_business_name",
                                 "business_name_tokens", "normalized_business_address",
                                 "city", "state", "postal_code", "landmark"]).fillna("")
    s3_df = pd.read_csv(NORM_DIR / f"{split}_source3_normalized.tsv", sep="\t", dtype=str,
                        usecols=["entity_id", "country", "normalized_business_name",
                                 "business_name_tokens", "normalized_business_address",
                                 "city", "state", "postal_code", "landmark"]).fillna("")
    
    print(f"  S1: {len(s1_df):,}, S2: {len(s2_df):,}, S3: {len(s3_df):,}")
    
    # Build lookup dicts for fast access
    print("\n[2/6] Building lookup dictionaries...")
    s1_lookup = s1_df.set_index("entity_id").to_dict("index")
    s2_lookup = s2_df.set_index("entity_id").to_dict("index")
    s3_lookup = s3_df.set_index("entity_id").to_dict("index")
    
    # Load embedding caches
    print("\n[3/6] Loading embedding caches...")
    s1_embs, s1_emb_ids = _load_embeddings_cached(split, "s1")
    s2_embs, s2_emb_ids = _load_embeddings_cached(split, "s2")
    s3_embs, s3_emb_ids = _load_embeddings_cached(split, "s3")
    
    s1_emb_idx = _build_id_to_emb_index(s1_emb_ids)
    s2_emb_idx = _build_id_to_emb_index(s2_emb_ids)
    s3_emb_idx = _build_id_to_emb_index(s3_emb_ids)
    
    print(f"  Embeddings: S1={len(s1_embs):,}, S2={len(s2_embs):,}, S3={len(s3_embs):,}")
    
    # Fit TF-IDF on combined name corpus (no labels)
    print("\n[4/6] Fitting TF-IDF vectorizer...")
    all_names = (
        s1_df["normalized_business_name"].tolist() +
        s2_df["normalized_business_name"].tolist() +
        s3_df["normalized_business_name"].tolist()
    )
    tfidf = NameTfidfVectorizer()
    tfidf.fit(all_names)
    
    # Pre-compute TF-IDF vectors for all entities
    print("  Computing TF-IDF vectors...")
    s1_tfidf = tfidf.transform(s1_df["normalized_business_name"].tolist())
    s2_tfidf = tfidf.transform(s2_df["normalized_business_name"].tolist())
    s3_tfidf = tfidf.transform(s3_df["normalized_business_name"].tolist())
    
    s1_tfidf_dict = dict(zip(s1_df["entity_id"].values, s1_tfidf))
    s2_tfidf_dict = dict(zip(s2_df["entity_id"].values, s2_tfidf))
    s3_tfidf_dict = dict(zip(s3_df["entity_id"].values, s3_tfidf))
    
    # Process candidates in chunks
    print("\n[5/6] Computing pairwise features...")
    
    # Expand candidates to pairwise rows
    pair_rows = []
    for _, row in candidates.iterrows():
        s1_id = row["source1_entity_id"]
        cand_ids = row["candidate_entity_ids"].split(",") if row["candidate_entity_ids"] else []
        strategies = row["source_strategy"].split("|") if row["source_strategy"] else []
        
        for i, cand_id in enumerate(cand_ids):
            source = "S2" if cand_id.startswith("S2-") else "S3"
            pair_rows.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": cand_id,
                "source": source,
                "source_strategy": strategies[i] if i < len(strategies) else "",
            })
    
    pairs_df = pd.DataFrame(pair_rows)
    print(f"  Total candidate pairs: {len(pairs_df):,}")
    
    # Process in chunks
    output_path = CACHE_DIR / f"{split}_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    first_chunk = True
    total_pairs = len(pairs_df)
    
    for start in range(0, total_pairs, FEATURE_CHUNK_SIZE):
        end = min(start + FEATURE_CHUNK_SIZE, total_pairs)
        chunk = pairs_df.iloc[start:end].copy()
        
        print(f"  Processing chunk {start:,}-{end:,} / {total_pairs:,}...")
        features = _compute_chunk_features(
            chunk, s1_lookup, s2_lookup, s3_lookup,
            s1_emb_idx, s2_emb_idx, s3_emb_idx,
            s1_embs, s2_embs, s3_embs,
            s1_tfidf_dict, s2_tfidf_dict, s3_tfidf_dict,
        )
        
        features.to_parquet(output_path, index=False, compression="snappy")
        # For subsequent chunks, append
        if first_chunk:
            first_chunk = False
        else:
            # Read existing and append
            existing = pd.read_parquet(output_path)
            combined = pd.concat([existing, features], ignore_index=True)
            combined.to_parquet(output_path, index=False, compression="snappy")
        
        del features, chunk
        gc.collect()
    
    print(f"\n[6/6] Features saved to {output_path}")
    final_df = pd.read_parquet(output_path)
    print(f"  Final shape: {final_df.shape}")
    print(f"  Columns: {list(final_df.columns)}")


def _compute_chunk_features(
    chunk: pd.DataFrame,
    s1_lookup: Dict,
    s2_lookup: Dict,
    s3_lookup: Dict,
    s1_emb_idx: Dict[str, int],
    s2_emb_idx: Dict[str, int],
    s3_emb_idx: Dict[str, int],
    s1_embs: np.ndarray,
    s2_embs: np.ndarray,
    s3_embs: np.ndarray,
    s1_tfidf: Dict[str, np.ndarray],
    s2_tfidf: Dict[str, np.ndarray],
    s3_tfidf: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """Compute all features for a chunk of candidate pairs."""
    
    results = []
    
    for _, row in chunk.iterrows():
        s1_id = row["source1_entity_id"]
        cand_id = row["candidate_entity_id"]
        source = row["source"]
        strategy = row["source_strategy"]
        
        s1 = s1_lookup.get(s1_id, {})
        cand = (s2_lookup if source == "S2" else s3_lookup).get(cand_id, {})
        
        if not s1 or not cand:
            continue
        
        # ── Name Features ──────────────────────────────────────────────
        s1_name = s1.get("normalized_business_name", "")
        cand_name = cand.get("normalized_business_name", "")
        s1_tokens = s1.get("business_name_tokens", "")
        cand_tokens = cand.get("business_name_tokens", "")
        
        name_exact = _exact_match_flag(s1_name, cand_name)
        name_token_jaccard = _token_jaccard(s1_tokens, cand_tokens)
        name_levenshtein = _levenshtein_ratio(s1_name, cand_name)
        name_token_sort = _token_sort_ratio(s1_name, cand_name)
        
        # TF-IDF cosine
        tfidf_cosine = 0.0
        if s1_id in s1_tfidf and cand_id in (s2_tfidf if source == "S2" else s3_tfidf):
            tfidf_cosine = tfidf.cosine_similarity(
                s1_tfidf[s1_id], 
                (s2_tfidf if source == "S2" else s3_tfidf)[cand_id]
            )
        
        # Embedding cosine (reuse blocking cache)
        emb_cosine = 0.0
        s1_e_idx = s1_emb_idx.get(s1_id, -1)
        cand_e_idx = (s2_emb_idx if source == "S2" else s3_emb_idx).get(cand_id, -1)
        if s1_e_idx >= 0 and cand_e_idx >= 0:
            emb_cosine = _get_embedding_cosine(
                s1_e_idx, cand_e_idx,
                s1_embs, s2_embs if source == "S2" else s3_embs
            )
        
        # ── Address Features ───────────────────────────────────────────
        s1_addr = s1.get("normalized_business_address", "")
        cand_addr = cand.get("normalized_business_address", "")
        s1_city = s1.get("city") or ""
        cand_city = cand.get("city") or ""
        s1_state = s1.get("state") or ""
        cand_state = cand.get("state") or ""
        s1_postal = s1.get("postal_code") or ""
        cand_postal = cand.get("postal_code") or ""
        s1_landmark = s1.get("landmark") or ""
        cand_landmark = cand.get("landmark") or ""
        
        addr_full_sim = _levenshtein_ratio(s1_addr, cand_addr)
        
        # City/state/postal exact match (missing ≠ mismatch, encode presence)
        city_both_present = int(bool(s1_city) and bool(cand_city))
        city_exact = int(s1_city == cand_city and city_both_present)
        state_both_present = int(bool(s1_state) and bool(cand_state))
        state_exact = int(s1_state == cand_state and state_both_present)
        postal_both_present = int(bool(s1_postal) and bool(cand_postal))
        postal_exact = int(s1_postal == cand_postal and postal_both_present)
        
        # Landmark-stripped similarity
        s1_addr_nolandmark = s1_addr.replace(s1_landmark, "").strip()
        cand_addr_nolandmark = cand_addr.replace(cand_landmark, "").strip()
        addr_nolandmark_sim = _levenshtein_ratio(s1_addr_nolandmark, cand_addr_nolandmark)
        
        # ── Meta Features ──────────────────────────────────────────────
        country_match = int(s1.get("country", "") == cand.get("country", "") and s1.get("country", "") != "")
        
        # Blocking strategy flags
        strat_token = int("token" in strategy)
        strat_lsh = int("lsh" in strategy)
        strat_embedding = int("embedding" in strategy)
        
        # Blocking similarity scores (approximations from strategy)
        # Token overlap count
        token_overlap = len(set(s1_tokens.split()) & set(cand_tokens.split())) if s1_tokens and cand_tokens else 0
        
        # MinHash Jaccard estimate (use LSH threshold as proxy if LSH strategy)
        minhash_jaccard_est = LSH_THRESHOLD if strat_lsh else 0.0
        
        # ANN rank/distance proxy (embedding cosine as proxy)
        ann_distance = 1.0 - emb_cosine if strat_embedding else 1.0
        
        # Candidate rank within S1 entity's list (approximate from order)
        # We'll compute this after grouping
        cand_rank = 0  # Will fill in post-processing
        
        results.append({
            "source1_entity_id": s1_id,
            "candidate_entity_id": cand_id,
            "source": source,
            # Name features
            "name_exact_match": name_exact,
            "name_token_jaccard": name_token_jaccard,
            "name_levenshtein_ratio": name_levenshtein,
            "name_token_sort_ratio": name_token_sort,
            "name_tfidf_cosine": tfidf_cosine,
            "name_embedding_cosine": emb_cosine,
            # Address features
            "addr_full_similarity": addr_full_sim,
            "city_both_present": city_both_present,
            "city_exact_match": city_exact,
            "state_both_present": state_both_present,
            "state_exact_match": state_exact,
            "postal_both_present": postal_both_present,
            "postal_exact_match": postal_exact,
            "addr_nolandmark_similarity": addr_nolandmark_sim,
            # Meta features
            "country_match": country_match,
            "block_strategy_token": strat_token,
            "block_strategy_lsh": strat_lsh,
            "block_strategy_embedding": strat_embedding,
            "block_token_overlap": token_overlap,
            "block_minhash_jaccard_est": minhash_jaccard_est,
            "block_ann_distance": ann_distance,
            "candidate_rank": cand_rank,
        })
    
    df = pd.DataFrame(results)
    
    # Add candidate rank within each S1 entity
    if len(df) > 0:
        df["candidate_rank"] = df.groupby("source1_entity_id").cumcount() + 1
    
    return df


# ─── Script Entry Point ───────────────────────────────────────────────────────

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Build features for candidate pairs")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    args = parser.parse_args()
    
    build_features_for_split(args.split)


if __name__ == "__main__":
    main()