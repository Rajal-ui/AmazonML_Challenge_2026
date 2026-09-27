"""
Business Entity Resolution — Step 2: Blocking

Generates high-recall candidate pairs between S1 (reference) and S2/S3
(target) sources via three complementary strategies:

1. Token Inverted-Index Blocking on business_name_tokens
2. Char N-gram MinHash/LSH on normalized_business_name (datasketch)
3. all-MiniLM-L6-v2 Embedding ANN on normalized_business_name via FAISS

Designed for low-resource environments (≤6GB RAM, CPU-only, 2 cores).
Processes data in chunks, uses float16 caches, PQ compression in FAISS.

Outputs:
  data/candidates/{train,test}_candidates.tsv
  Columns: source1_entity_id, candidate_entity_ids, source_strategy

Evaluation (train only):
  Blocking recall, reduction ratio, per-strategy/per-country breakdown.
"""

from __future__ import annotations

import gc
import os
import sys
import time
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# ─── Tunable Constants ────────────────────────────────────────────────────────

# Strategy 1 — Token inverted index
MIN_SHARED_TOKENS: int = 1            # min shared tokens to be a candidate

# Strategy 2 — Char N-gram MinHash/LSH
NGRAM_SIZE: int = 3                    # character n-gram size
LSH_NUM_PERM: int = 64                 # MinHash permutations (lower = less RAM)
LSH_THRESHOLD: float = 0.25           # Jaccard similarity threshold

# Strategy 3 — Embedding ANN (FAISS)
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE: int = 1024      # encoding batch size
ANN_TOP_K: int = 30                    # top-k neighbours per query
FAISS_NPROBE: int = 32                 # IVF nprobe (search quality vs speed)
FAISS_PQ_M: int = 48                   # PQ subquantizer count (dim=384, M divides dim)
FAISS_PQ_NBITS: int = 8               # bits per subquantizer

# Chunk sizes for memory-constrained processing
EMB_ENCODE_CHUNK: int = 200_000        # encode this many texts at a time
FAISS_ADD_CHUNK: int = 200_000         # add to index in chunks
FAISS_QUERY_CHUNK: int = 50_000        # search in chunks
TOKEN_MAX_POSTINGS: int = 500_000      # skip overly-frequent tokens

# General
EMBEDDING_DIM: int = 384               # all-MiniLM-L6-v2 output dim

# ─── Path Configuration ──────────────────────────────────────────────────────

from config import NORM_DIR, CAND_DIR, CACHE_DIR, GROUND_TRUTH  # noqa: E402

GROUND_TRUTH_PATH = GROUND_TRUTH  # kept: name used by main() below


def _ensure_normalized_files(split: str) -> Dict[str, Path]:
    """
    Check if normalized TSVs exist; fail loudly if missing.
    Returns dict of {source_key: norm_path}.
    """
    sources = {}
    missing = []

    for src_num in [1, 2, 3]:
        norm_file = NORM_DIR / f"{split}_source{src_num}_normalized.tsv"
        sources[f"S{src_num}"] = norm_file

        if norm_file.exists():
            print(f"  [norm] Found: {norm_file.name}")
        else:
            missing.append(norm_file.name)

    if missing:
        raise FileNotFoundError(
            f"Normalized files not found for split '{split}': {', '.join(missing)}. "
            f"Run normalize.py first to generate data/normalized/{{train,test}}_source{{1,2,3}}_normalized.tsv"
        )

    return sources


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_normalized_source(path: Path) -> pd.DataFrame:
    """Load a normalized TSV with minimal columns for blocking."""
    df = pd.read_csv(
        path, sep="\t", dtype=str,
        usecols=["entity_id", "country", "normalized_business_name", "business_name_tokens"],
    ).fillna("")
    return df


def load_ground_truth(path: Path) -> Dict[str, Set[str]]:
    """
    Load ground truth as {source1_entity_id: set of matched_entity_ids}.
    Used ONLY for recall evaluation, never for candidate generation.
    """
    gt: Dict[str, Set[str]] = {}
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    for _, row in df.iterrows():
        s1_id = row["source1_entity_id"]
        matched_str = row["matched_entity_ids"]
        gt[s1_id] = set(matched_str.split(",")) if matched_str else set()
    return gt


# ─── Strategy 1: Token Inverted-Index Blocking ───────────────────────────────

def build_token_index(df: pd.DataFrame) -> Dict[str, List[int]]:
    """
    Build inverted index: token → list of row indices.
    Prunes overly-frequent tokens (stop-word-like) to save memory/time.
    """
    index: Dict[str, List[int]] = defaultdict(list)
    tokens_series = df["business_name_tokens"].values

    for idx in range(len(tokens_series)):
        text = tokens_series[idx]
        if not text:
            continue
        for tok in text.split():
            if len(tok) >= 2:
                index[tok].append(idx)

    pruned = 0
    for tok in list(index.keys()):
        if len(index[tok]) > TOKEN_MAX_POSTINGS:
            del index[tok]
            pruned += 1
    if pruned:
        print(f"    → Pruned {pruned} high-freq tokens (>{TOKEN_MAX_POSTINGS:,} postings)")

    return index


def block_by_tokens(
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    min_shared: int = MIN_SHARED_TOKENS,
) -> Dict[str, Set[str]]:
    """
    Token inverted-index blocking.
    For each S1 entity, find target entities sharing ≥ min_shared name tokens.

    Returns:
        {s1_entity_id: set of candidate target entity_ids}
    """
    print("  [token] Building inverted index on target ...")
    t0 = time.time()
    inv_index = build_token_index(target_df)
    target_ids = target_df["entity_id"].values
    print(f"    → index: {len(inv_index):,} unique tokens in {time.time()-t0:.1f}s")

    print("  [token] Querying S1 entities ...")
    t0 = time.time()
    candidates: Dict[str, Set[str]] = {}
    s1_ids = s1_df["entity_id"].values
    s1_tokens_col = s1_df["business_name_tokens"].values

    for i in range(len(s1_ids)):
        text = s1_tokens_col[i]
        if not text:
            candidates[s1_ids[i]] = set()
            continue

        query_tokens = [t for t in text.split() if len(t) >= 2]
        if not query_tokens:
            candidates[s1_ids[i]] = set()
            continue

        hit_counts: Dict[int, int] = defaultdict(int)
        for tok in query_tokens:
            if tok in inv_index:
                for row_idx in inv_index[tok]:
                    hit_counts[row_idx] += 1

        matched = set()
        for row_idx, count in hit_counts.items():
            if count >= min_shared:
                matched.add(target_ids[row_idx])
        candidates[s1_ids[i]] = matched

        if (i + 1) % 500_000 == 0:
            print(f"    ... {i+1:,}/{len(s1_ids):,}")

    elapsed = time.time() - t0
    total_cands = sum(len(v) for v in candidates.values())
    print(f"    → token blocking: {total_cands:,} candidates in {elapsed:.1f}s")
    return candidates


# ─── Strategy 2: Char N-gram MinHash/LSH ─────────────────────────────────────

def _char_ngrams(text: str, n: int = NGRAM_SIZE) -> Set[str]:
    """Generate character n-grams from text."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def block_by_lsh(
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    ngram_size: int = NGRAM_SIZE,
    num_perm: int = LSH_NUM_PERM,
    threshold: float = LSH_THRESHOLD,
) -> Dict[str, Set[str]]:
    """
    MinHash/LSH blocking using character n-grams on normalized_business_name.
    Chunks the target LSH index to prevent memory exhaustion on large datasets.
    """
    from datasketch import MinHash, MinHashLSH

    # 1. Precompute S1 MinHashes into a memory-efficient numpy array
    print(f"  [lsh] Precomputing S1 MinHashes (n={ngram_size}, perm={num_perm}) ...")
    t0 = time.time()
    s1_ids = s1_df["entity_id"].values
    s1_names = s1_df["normalized_business_name"].values
    n_s1 = len(s1_ids)

    s1_hashes = np.zeros((n_s1, num_perm), dtype=np.uint32)
    valid_s1 = np.zeros(n_s1, dtype=bool)

    for i in range(n_s1):
        name = s1_names[i]
        if name:
            ngrams = _char_ngrams(name, ngram_size)
            if ngrams:
                mh = MinHash(num_perm=num_perm)
                for gram in ngrams:
                    mh.update(gram.encode("utf8"))
                s1_hashes[i] = mh.hashvalues
                valid_s1[i] = True
        
        if (i + 1) % 500_000 == 0:
            print(f"    ... {i+1:,}/{n_s1:,} S1 entities hashed")

    print(f"    → S1 hashes computed in {time.time() - t0:.1f}s")

    # 2. Chunk Target and stream through LSH
    print(f"  [lsh] Chunking Target through LSH (thresh={threshold}) ...")
    t0 = time.time()
    
    target_ids = target_df["entity_id"].values
    target_names = target_df["normalized_business_name"].values
    n_target = len(target_ids)
    
    LSH_CHUNK_SIZE = 500_000
    candidates: Dict[str, Set[str]] = defaultdict(set)

    for start in range(0, n_target, LSH_CHUNK_SIZE):
        end = min(start + LSH_CHUNK_SIZE, n_target)
        print(f"    ... processing target chunk {start:,}-{end:,} ...")
        
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        
        # Build LSH for chunk
        for idx in range(start, end):
            name = target_names[idx]
            if not name:
                continue
            ngrams = _char_ngrams(name, ngram_size)
            if not ngrams:
                continue
            mh = MinHash(num_perm=num_perm)
            for gram in ngrams:
                mh.update(gram.encode("utf8"))
            try:
                lsh.insert(target_ids[idx], mh)
            except ValueError:
                pass
                
        # Query S1 against chunk
        for i in range(n_s1):
            if not valid_s1[i]:
                continue
            # Reconstruct MinHash object efficiently
            mh = MinHash(num_perm=num_perm)
            mh.hashvalues = s1_hashes[i]
            
            res = lsh.query(mh)
            if res:
                candidates[s1_ids[i]].update(res)
                
        del lsh
        gc.collect()

    elapsed = time.time() - t0
    total_cands = sum(len(v) for v in candidates.values())
    print(f"    → LSH blocking: {total_cands:,} candidates in {elapsed:.1f}s")
    
    return dict(candidates)


# ─── Strategy 3: Embedding ANN (FAISS) ───────────────────────────────────────

def _get_cache_path(prefix: str, source_name: str, ext: str) -> Path:
    """Return cache file path."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{prefix}_{source_name}.{ext}"


def _encode_to_disk(
    texts: List[str],
    entity_ids: np.ndarray,
    model,
    source_name: str,
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> Tuple[Path, Path]:
    """
    Encode texts to float16 embeddings and save to disk.
    Returns (embeddings_path, ids_path).
    Does NOT keep embeddings in memory — they are memory-mapped on demand.
    """
    emb_path = _get_cache_path("embeddings", source_name, "npy")
    id_path = _get_cache_path("ids", source_name, "pkl")

    if emb_path.exists() and id_path.exists():
        # Validate size
        embs = np.load(emb_path, mmap_mode="r")
        if len(embs) == len(texts):
            print(f"    → Cached: {emb_path.name} ({len(embs):,} vectors)")
            return emb_path, id_path
        print(f"    → Cache mismatch ({len(embs)} vs {len(texts)}), re-encoding")
        del embs

    print(f"    → Encoding {len(texts):,} texts to disk ...")
    t0 = time.time()
    n = len(texts)

    # Create memory-mapped output file
    out = np.lib.format.open_memmap(
        str(emb_path), mode="w+", dtype=np.float16, shape=(n, EMBEDDING_DIM)
    )

    for start in range(0, n, EMB_ENCODE_CHUNK):
        end = min(start + EMB_ENCODE_CHUNK, n)
        chunk_texts = [t if t else " " for t in texts[start:end]]

        chunk_embs = model.encode(
            chunk_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        out[start:end] = chunk_embs.astype(np.float16)
        del chunk_embs
        gc.collect()

        elapsed = time.time() - t0
        print(f"      {end:,}/{n:,} encoded ({elapsed:.0f}s)")

    out.flush()
    del out

    with open(id_path, "wb") as f:
        pickle.dump(entity_ids, f)

    elapsed = time.time() - t0
    print(f"    → Saved {n:,} float16 embeddings in {elapsed:.1f}s")
    return emb_path, id_path


def block_by_embedding(
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    model,
    top_k: int = ANN_TOP_K,
    source_name: str = "",
    s1_source_name: str = "",
) -> Dict[str, Set[str]]:
    """
    Embedding ANN blocking using FAISS IVFPQ index.

    Memory-efficient: encodes to disk (float16), builds a compressed PQ index,
    and queries in chunks. Peak RAM ~1-2GB for the index.

    Returns:
        {s1_entity_id: set of candidate target entity_ids}
    """
    import faiss

    # ── Encode target to disk ──
    print(f"  [emb] Encoding target ({source_name}) ...")
    target_names = target_df["normalized_business_name"].tolist()
    target_ids_arr = target_df["entity_id"].values
    target_emb_path, _ = _encode_to_disk(target_names, target_ids_arr, model, source_name)
    del target_names
    gc.collect()

    # ── Encode S1 to disk ──
    print(f"  [emb] Encoding S1 ({s1_source_name}) ...")
    s1_names = s1_df["normalized_business_name"].tolist()
    s1_ids_arr = s1_df["entity_id"].values
    s1_emb_path, _ = _encode_to_disk(s1_names, s1_ids_arr, model, s1_source_name)
    del s1_names
    gc.collect()

    # ── Build FAISS index (memory-mapped reads) ──
    target_embs_mmap = np.load(target_emb_path, mmap_mode="r")
    n_target = len(target_embs_mmap)
    dim = target_embs_mmap.shape[1]

    print(f"  [emb] Building FAISS IVFPQ index (n={n_target:,}, dim={dim}) ...")
    t0 = time.time()

    if n_target < 10_000:
        # Small dataset: use flat index
        index = faiss.IndexFlatIP(dim)
        index.add(target_embs_mmap[:].astype(np.float32))
    else:
        nlist = max(int(np.sqrt(n_target)), 256)
        nlist = min(nlist, n_target // 20)
        pq_m = FAISS_PQ_M

        print(f"    → IVFPQ: nlist={nlist}, PQ_M={pq_m}, nbits={FAISS_PQ_NBITS}")

        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFPQ(
            quantizer, dim, nlist, pq_m, FAISS_PQ_NBITS,
            faiss.METRIC_INNER_PRODUCT,
        )

        # Train on subset
        train_size = min(n_target, max(nlist * 40, 200_000))
        train_idx = np.random.choice(n_target, train_size, replace=False)
        train_data = target_embs_mmap[train_idx].astype(np.float32)
        index.train(train_data)
        del train_data
        gc.collect()

        # Add in chunks (mmap → float32 conversion per chunk)
        for start in range(0, n_target, FAISS_ADD_CHUNK):
            end = min(start + FAISS_ADD_CHUNK, n_target)
            chunk = target_embs_mmap[start:end].astype(np.float32)
            index.add(chunk)
            del chunk
            gc.collect()

        index.nprobe = FAISS_NPROBE

    build_time = time.time() - t0
    print(f"    → Index built in {build_time:.1f}s")

    del target_embs_mmap
    gc.collect()

    # ── Search S1 in chunks ──
    s1_embs_mmap = np.load(s1_emb_path, mmap_mode="r")
    n_s1 = len(s1_embs_mmap)

    print(f"  [emb] Searching top-{top_k} for {n_s1:,} S1 entities ...")
    t0 = time.time()

    candidates: Dict[str, Set[str]] = {}

    for start in range(0, n_s1, FAISS_QUERY_CHUNK):
        end = min(start + FAISS_QUERY_CHUNK, n_s1)
        q = s1_embs_mmap[start:end].astype(np.float32)
        D, I = index.search(q, top_k)

        for local_i in range(end - start):
            global_i = start + local_i
            matched = set()
            for j in range(top_k):
                idx = int(I[local_i, j])
                if 0 <= idx < n_target:
                    matched.add(target_ids_arr[idx])
            candidates[s1_ids_arr[global_i]] = matched

        if end % 200_000 == 0 or end == n_s1:
            print(f"    ... searched {end:,}/{n_s1:,}")

        del q, D, I
        gc.collect()

    elapsed = time.time() - t0
    total_cands = sum(len(v) for v in candidates.values())
    print(f"    → Embedding blocking: {total_cands:,} candidates in {elapsed:.1f}s")

    del s1_embs_mmap, index
    gc.collect()

    return candidates


# ─── Candidate Merging ────────────────────────────────────────────────────────

def merge_candidates(
    token_cands: Dict[str, Set[str]],
    lsh_cands: Dict[str, Set[str]],
    emb_cands: Dict[str, Set[str]],
) -> Dict[str, Dict[str, Set[str]]]:
    """
    Union candidates from all strategies, tagging each candidate with
    the strategy(ies) that found it.

    Returns:
        {s1_entity_id: {candidate_id: set of strategy names}}
    """
    all_s1_ids = set(token_cands.keys()) | set(lsh_cands.keys()) | set(emb_cands.keys())
    merged: Dict[str, Dict[str, Set[str]]] = {}

    for s1_id in all_s1_ids:
        cand_strategies: Dict[str, Set[str]] = defaultdict(set)
        for cand_id in token_cands.get(s1_id, set()):
            cand_strategies[cand_id].add("token")
        for cand_id in lsh_cands.get(s1_id, set()):
            cand_strategies[cand_id].add("lsh")
        for cand_id in emb_cands.get(s1_id, set()):
            cand_strategies[cand_id].add("embedding")
        merged[s1_id] = dict(cand_strategies)

    return merged


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_blocking_recall(
    merged_cands: Dict[str, Dict[str, Set[str]]],
    ground_truth: Dict[str, Set[str]],
    s1_df: pd.DataFrame,
    token_cands: Dict[str, Set[str]],
    lsh_cands: Dict[str, Set[str]],
    emb_cands: Dict[str, Set[str]],
    total_s2_count: int,
    total_s3_count: int,
) -> None:
    """
    Evaluate blocking recall against ground truth (train split only).

    Prints:
    - Overall recall (fraction of GT pairs present in candidates)
    - Per-country recall
    - Per-strategy recall and unique contributions
    - Reduction ratio vs full cross-product
    """
    s1_country = dict(zip(s1_df["entity_id"].values, s1_df["country"].values))

    gt_total = 0
    gt_found = 0
    gt_by_country: Dict[str, List[int]] = defaultdict(lambda: [0, 0])

    token_found = lsh_found = emb_found = 0
    token_unique = lsh_unique = emb_unique = 0

    for s1_id, gt_matches in ground_truth.items():
        country = s1_country.get(s1_id, "unknown")
        cand_set = set(merged_cands.get(s1_id, {}).keys())
        t_set = token_cands.get(s1_id, set())
        l_set = lsh_cands.get(s1_id, set())
        e_set = emb_cands.get(s1_id, set())

        for gt_id in gt_matches:
            gt_total += 1
            gt_by_country[country][0] += 1

            in_t = gt_id in t_set
            in_l = gt_id in l_set
            in_e = gt_id in e_set

            if gt_id in cand_set:
                gt_found += 1
                gt_by_country[country][1] += 1

            if in_t: token_found += 1
            if in_l: lsh_found += 1
            if in_e: emb_found += 1

            if in_t and not in_l and not in_e: token_unique += 1
            if in_l and not in_t and not in_e: lsh_unique += 1
            if in_e and not in_t and not in_l: emb_unique += 1

    total_candidates = sum(len(v) for v in merged_cands.values())
    total_cross = len(merged_cands) * (total_s2_count + total_s3_count)

    print("\n" + "=" * 80)
    print("BLOCKING EVALUATION (Train Split)")
    print("=" * 80)

    recall = gt_found / gt_total if gt_total > 0 else 0.0
    print(f"\n  Overall Blocking Recall: {gt_found:,}/{gt_total:,} = {recall:.4f} ({recall*100:.2f}%)")

    if recall < 0.90:
        print("  ⚠️  WARNING: Recall < 90%! Consider relaxing blocking parameters.")

    print(f"\n  Total candidates generated:   {total_candidates:,}")
    print(f"  Full cross-product size:      {total_cross:,}")
    rr = 1.0 - (total_candidates / total_cross) if total_cross > 0 else 0.0
    print(f"  Reduction ratio:              {rr:.6f} ({rr*100:.4f}%)")
    avg = total_candidates / max(len(merged_cands), 1)
    print(f"  Candidates/S1 entity (avg):   {avg:.1f}")

    if gt_total > 0:
        print(f"\n  Per-Strategy Recall:")
        print(f"    Token:     {token_found:,}/{gt_total:,} = {token_found/gt_total:.4f}")
        print(f"    LSH:       {lsh_found:,}/{gt_total:,} = {lsh_found/gt_total:.4f}")
        print(f"    Embedding: {emb_found:,}/{gt_total:,} = {emb_found/gt_total:.4f}")

        print(f"\n  Unique Contributions (found ONLY by that strategy):")
        print(f"    Token-only:     {token_unique:,}")
        print(f"    LSH-only:       {lsh_unique:,}")
        print(f"    Embedding-only: {emb_unique:,}")

    print(f"\n  Per-Country Recall:")
    for country in sorted(gt_by_country.keys()):
        total, found = gt_by_country[country]
        r = found / total if total > 0 else 0.0
        print(f"    {country:20s}: {found:,}/{total:,} = {r:.4f}")

    print("=" * 80)


# ─── Output ───────────────────────────────────────────────────────────────────

def save_candidates(
    merged_cands: Dict[str, Dict[str, Set[str]]],
    output_path: Path,
) -> None:
    """
    Save candidate pairs to TSV.
    Columns: source1_entity_id, candidate_entity_ids, source_strategy
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for s1_id in sorted(merged_cands.keys()):
        cand_map = merged_cands[s1_id]
        if not cand_map:
            continue
        cand_ids = sorted(cand_map.keys())
        all_strats: Set[str] = set()
        for cid in cand_ids:
            all_strats.update(cand_map[cid])
        rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(cand_ids),
            "source_strategy": "|".join(sorted(all_strats)),
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep="\t", index=False)
    print(f"  Saved {len(df):,} S1 entities → {output_path}")


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run_blocking_for_split(
    split: str,
    model=None,
    ground_truth: Optional[Dict[str, Set[str]]] = None,
) -> None:
    """
    Run the full blocking pipeline for a given split.

    For each target source (S2, S3 separately):
      1. Token inverted-index blocking
      2. MinHash/LSH blocking
      3. Embedding ANN blocking
    Then merge all candidates and evaluate recall (train only).
    """
    print(f"\n{'='*80}")
    print(f"BLOCKING PIPELINE — {split.upper()}")
    print(f"{'='*80}")

    print(f"\n[1/5] Ensuring normalized files ...")
    source_paths = _ensure_normalized_files(split)

    print(f"\n[2/5] Loading normalized data ...")
    t0 = time.time()
    s1_df = load_normalized_source(source_paths["S1"])
    print(f"  S1: {len(s1_df):,} ({time.time()-t0:.1f}s)")

    all_token_cands: Dict[str, Set[str]] = {}
    all_lsh_cands: Dict[str, Set[str]] = {}
    all_emb_cands: Dict[str, Set[str]] = {}

    total_s2 = 0
    total_s3 = 0

    print(f"\n[3/5] Running blocking strategies ...")

    for target_key in ["S2", "S3"]:
        t_load = time.time()
        target_df = load_normalized_source(source_paths[target_key])
        n_target = len(target_df)
        if target_key == "S2":
            total_s2 = n_target
        else:
            total_s3 = n_target
        print(f"\n  ── Target: {target_key} ({n_target:,} entities, loaded in {time.time()-t_load:.1f}s) ──")

        # Strategy 1: Token blocking
        print(f"\n  Strategy 1: Token (min_shared={MIN_SHARED_TOKENS})")
        tc = block_by_tokens(s1_df, target_df, min_shared=MIN_SHARED_TOKENS)
        for sid, cands in tc.items():
            all_token_cands.setdefault(sid, set()).update(cands)
        del tc
        gc.collect()

        # Strategy 2: LSH
        print(f"\n  Strategy 2: LSH (n={NGRAM_SIZE}, perm={LSH_NUM_PERM}, thresh={LSH_THRESHOLD})")
        lc = block_by_lsh(s1_df, target_df, NGRAM_SIZE, LSH_NUM_PERM, LSH_THRESHOLD)
        for sid, cands in lc.items():
            all_lsh_cands.setdefault(sid, set()).update(cands)
        del lc
        gc.collect()

        # Strategy 3: Embedding ANN
        print(f"\n  Strategy 3: Embedding ANN (top_k={ANN_TOP_K})")
        ec = block_by_embedding(
            s1_df, target_df, model,
            top_k=ANN_TOP_K,
            source_name=f"{split}_{target_key.lower()}",
            s1_source_name=f"{split}_s1",
        )
        for sid, cands in ec.items():
            all_emb_cands.setdefault(sid, set()).update(cands)
        del ec, target_df
        gc.collect()

    # Step 4: Merge
    print(f"\n[4/5] Merging candidates ...")
    merged = merge_candidates(all_token_cands, all_lsh_cands, all_emb_cands)
    total_cands = sum(len(v) for v in merged.values())
    s1_with = sum(1 for v in merged.values() if v)
    print(f"  S1 entities: {len(merged):,} | with ≥1 candidate: {s1_with:,} | total pairs: {total_cands:,}")

    # Step 5: Save
    print(f"\n[5/5] Saving candidates ...")
    save_candidates(merged, CAND_DIR / f"{split}_candidates.tsv")

    # Step 6: Evaluate
    if ground_truth is not None and split == "train":
        evaluate_blocking_recall(
            merged, ground_truth, s1_df,
            all_token_cands, all_lsh_cands, all_emb_cands,
            total_s2, total_s3,
        )


def main():
    """Main entry point: run blocking for train and test splits."""
    overall_t0 = time.time()

    src_dir = Path(__file__).resolve().parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    print("Loading sentence-transformer model ...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"  Model loaded: {EMBEDDING_MODEL_NAME}")

    print(f"\nLoading ground truth ...")
    gt = load_ground_truth(GROUND_TRUTH_PATH)
    print(f"  Ground truth: {len(gt):,} S1 entities")

    run_blocking_for_split("train", model=model, ground_truth=gt)
    gc.collect()

    run_blocking_for_split("test", model=model, ground_truth=None)

    total_time = time.time() - overall_t0
    print(f"\n{'='*80}")
    print(f"BLOCKING COMPLETE — Total time: {total_time/60:.1f} minutes")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
