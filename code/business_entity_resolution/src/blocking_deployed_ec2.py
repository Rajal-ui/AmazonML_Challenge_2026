"""
Business Entity Resolution — Step 2: Blocking (Fixed Architecture)

Generates high-recall candidate pairs between S1 (reference) and S2/S3
(target) sources via geo-partitioned multi-strategy blocking.

Key fixes:
1. Geo-Partitioning: Composite blocking keys (country + city + name tokens, country + postal)
2. Hard Candidate Cap: Max 50 nearest neighbors per entity via FAISS
3. Streaming to Disk: Candidate pairs written as chunked Parquet files (never held in memory)
"""

from __future__ import annotations

import gc
import os
import sys
import time
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Iterator

import numpy as np
import pandas as pd

# ─── Tunable Constants ─────────────────────────────────────────────────────────

# Geo-partitioned blocking
MIN_NAME_TOKENS_FOR_KEY: int = 2       # first N name tokens for composite key
POSTAL_PREFIX_LEN: int = 5             # postal code prefix length

# Strategy 1 — Token inverted index (within geo partitions)
MIN_SHARED_TOKENS: int = 1
TOKEN_MAX_POSTINGS: int = 500_000

# Strategy 2 — Char N-gram MinHash/LSH (within geo partitions)
NGRAM_SIZE: int = 3
LSH_NUM_PERM: int = 64
LSH_THRESHOLD: float = 0.25
LSH_CHUNK_SIZE: int = 500_000

# Strategy 3 — Embedding ANN (FAISS) — HARD CAP: max 50 candidates per S1 entity
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE: int = 1024
ANN_TOP_K: int = 50                    # HARD CAP: max 50 nearest neighbors
FAISS_NPROBE: int = 32
FAISS_PQ_M: int = 48
FAISS_PQ_NBITS: int = 8

# Chunk sizes for memory-constrained processing
EMB_ENCODE_CHUNK: int = 200_000
FAISS_ADD_CHUNK: int = 200_000
FAISS_QUERY_CHUNK: int = 50_000

# Output chunking
CANDIDATE_CHUNK_SIZE: int = 500_000    # write candidate pairs in chunks to Parquet

# General
EMBEDDING_DIM: int = 384

# ─── Path Configuration ────────────────────────────────────────────────────────

from config import NORM_DIR, CAND_DIR, CACHE_DIR, GROUND_TRUTH  # noqa: E402

GROUND_TRUTH_PATH = GROUND_TRUTH


def _ensure_normalized_files(split: str) -> Dict[str, Path]:
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
            f"Normalized files not found for split '{split}': {', '.join(missing)}"
        )
    return sources


# ─── Data Loading ──────────────────────────────────────────────────────────────

def load_normalized_source(path: Path) -> pd.DataFrame:
    """Load normalized TSV with all columns needed for geo-partitioned blocking."""
    df = pd.read_csv(
        path, sep="\t", dtype=str,
        usecols=["entity_id", "country", "normalized_business_name", "business_name_tokens",
                 "city", "state", "postal_code"]
    ).fillna("")
    return df


def load_ground_truth(path: Path) -> Dict[str, Set[str]]:
    gt: Dict[str, Set[str]] = {}
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    for _, row in df.iterrows():
        s1_id = row["source1_entity_id"]
        matched_str = row["matched_entity_ids"]
        gt[s1_id] = set(matched_str.split(",")) if matched_str else set()
    return gt


# ─── Geo-Partitioned Blocking Keys ─────────────────────────────────────────────

def generate_blocking_keys(row: pd.Series) -> List[str]:
    """
    Generate composite blocking keys for a record.
    Keys combine geo + name to restrict comparisons to same locality.
    """
    keys = []
    country = str(row.get('country', '')).lower().strip()
    city = str(row.get('city', '')).lower().strip()
    state = str(row.get('state', '')).lower().strip()
    postal = str(row.get('postal_code', '')).lower().strip()
    name = str(row.get('normalized_business_name', '')).lower().strip()
    
    if not name:
        return keys
    
    name_tokens = name.split()[:MIN_NAME_TOKENS_FOR_KEY]
    if not name_tokens:
        return keys
    
    # Composite Key 1: Country + City + First N name tokens
    if country and city and name_tokens:
        keys.append(f"{country}_{city}_{'_'.join(name_tokens)}")
    
    # Composite Key 2: Country + State + First N name tokens (fallback for missing city)
    if country and state and name_tokens:
        keys.append(f"{country}_{state}_{'_'.join(name_tokens)}")
    
    # Composite Key 3: Country + Postal prefix + First N name tokens
    if country and postal and len(postal) >= 3:
        keys.append(f"{country}_zip_{postal[:POSTAL_PREFIX_LEN]}_{'_'.join(name_tokens)}")
    
    # Composite Key 4: Country only + name tokens (broad fallback)
    if country and name_tokens:
        keys.append(f"{country}_{'_'.join(name_tokens)}")
    
    return keys


def assign_geo_partitions(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Partition a DataFrame by composite geo keys.
    Returns dict: partition_key -> DataFrame subset
    """
    partitions: Dict[str, List[int]] = defaultdict(list)
    
    for idx, row in df.iterrows():
        keys = generate_blocking_keys(row)
        for key in keys:
            partitions[key].append(idx)
    
    # Build partition DataFrames
    result = {}
    for key, indices in partitions.items():
        if len(indices) > 1:  # Need at least 2 entities to form pairs
            result[key] = df.iloc[indices].reset_index(drop=True)
    
    print(f"  [geo] Created {len(result):,} geo partitions")
    return result


# ─── Strategy 1: Token Inverted-Index Blocking (within partition) ──────────────

def build_token_index(df: pd.DataFrame) -> Dict[str, List[int]]:
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


def block_by_tokens_partitioned(
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    min_shared: int = MIN_SHARED_TOKENS,
) -> Iterator[Tuple[str, str, str]]:
    """
    Token blocking within geo partitions. Yields (s1_id, candidate_id, strategy) tuples.
    """
    # Partition both S1 and target by geo keys
    s1_parts = assign_geo_partitions(s1_df)
    target_parts = assign_geo_partitions(target_df)
    
    common_keys = set(s1_parts.keys()) & set(target_parts.keys())
    print(f"    [token] {len(common_keys):,} shared geo partitions")
    
    for part_key in common_keys:
        s1_part = s1_parts[part_key]
        target_part = target_parts[part_key]
        
        if len(s1_part) == 0 or len(target_part) == 0:
            continue
        
        # Build inverted index on target partition
        inv_index = build_token_index(target_part)
        target_ids = target_part["entity_id"].values
        s1_ids = s1_part["entity_id"].values
        s1_tokens_col = s1_part["business_name_tokens"].values
        
        for i in range(len(s1_ids)):
            text = s1_tokens_col[i]
            if not text:
                continue
            query_tokens = [t for t in text.split() if len(t) >= 2]
            if not query_tokens:
                continue
            
            hit_counts: Dict[int, int] = defaultdict(int)
            for tok in query_tokens:
                if tok in inv_index:
                    for row_idx in inv_index[tok]:
                        hit_counts[row_idx] += 1
            
            for row_idx, count in hit_counts.items():
                if count >= min_shared:
                    yield (s1_ids[i], target_ids[row_idx], "token")


# ─── Strategy 2: Char N-gram MinHash/LSH (within partition) ────────────────────

def _char_ngrams(text: str, n: int = NGRAM_SIZE) -> Set[str]:
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def block_by_lsh_partitioned(
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    ngram_size: int = NGRAM_SIZE,
    num_perm: int = LSH_NUM_PERM,
    threshold: float = LSH_THRESHOLD,
) -> Iterator[Tuple[str, str, str]]:
    """MinHash/LSH blocking within geo partitions. Yields (s1_id, candidate_id, strategy)."""
    from datasketch import MinHash, MinHashLSH
    
    s1_parts = assign_geo_partitions(s1_df)
    target_parts = assign_geo_partitions(target_df)
    common_keys = set(s1_parts.keys()) & set(target_parts.keys())
    print(f"    [lsh] {len(common_keys):,} shared geo partitions")
    
    for part_key in common_keys:
        s1_part = s1_parts[part_key]
        target_part = target_parts[part_key]
        
        if len(s1_part) == 0 or len(target_part) == 0:
            continue
        
        s1_ids = s1_part["entity_id"].values
        s1_names = s1_part["normalized_business_name"].values
        n_s1 = len(s1_ids)
        
        # Precompute S1 MinHashes
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
        
        # Chunk target and stream through LSH
        target_ids = target_part["entity_id"].values
        target_names = target_part["normalized_business_name"].values
        n_target = len(target_ids)
        
        for start in range(0, n_target, LSH_CHUNK_SIZE):
            end = min(start + LSH_CHUNK_SIZE, n_target)
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
                mh = MinHash(num_perm=num_perm)
                mh.hashvalues = s1_hashes[i]
                res = lsh.query(mh)
                if res:
                    for cand_id in res:
                        yield (s1_ids[i], cand_id, "lsh")
            
            del lsh
            gc.collect()


# ─── Strategy 3: Embedding ANN (FAISS) with HARD TOP-K CAP ─────────────────────

def _get_cache_path(prefix: str, source_name: str, ext: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{prefix}_{source_name}.{ext}"


def _encode_to_disk(
    texts: List[str],
    entity_ids: np.ndarray,
    model,
    source_name: str,
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> Tuple[Path, Path]:
    emb_path = _get_cache_path("embeddings", source_name, "npy")
    id_path = _get_cache_path("ids", source_name, "pkl")
    
    if emb_path.exists() and id_path.exists():
        embs = np.load(emb_path, mmap_mode="r")
        if len(embs) == len(texts):
            print(f"    → Cached: {emb_path.name} ({len(embs):,} vectors)")
            return emb_path, id_path
        print(f"    → Cache mismatch ({len(embs)} vs {len(texts)}), re-encoding")
        del embs
    
    print(f"    → Encoding {len(texts):,} texts to disk ...")
    t0 = time.time()
    n = len(texts)
    
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


def block_by_embedding_partitioned(
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    model,
    top_k: int = ANN_TOP_K,
    source_name: str = "",
    s1_source_name: str = "",
) -> Iterator[Tuple[str, str, str]]:
    """
    Embedding ANN blocking with HARD CAP: max top_k (50) candidates per S1 entity.
    Yields (s1_id, candidate_id, strategy) tuples.
    """
    import faiss
    
    # Encode to disk (global, not per partition - already memory-efficient)
    print(f"  [emb] Encoding target ({source_name}) ...")
    target_names = target_df["normalized_business_name"].tolist()
    target_ids_arr = target_df["entity_id"].values
    target_emb_path, _ = _encode_to_disk(target_names, target_ids_arr, model, source_name)
    del target_names
    gc.collect()
    
    print(f"  [emb] Encoding S1 ({s1_source_name}) ...")
    s1_names = s1_df["normalized_business_name"].tolist()
    s1_ids_arr = s1_df["entity_id"].values
    s1_emb_path, _ = _encode_to_disk(s1_names, s1_ids_arr, model, s1_source_name)
    del s1_names
    gc.collect()
    
    # Build FAISS index
    target_embs_mmap = np.load(target_emb_path, mmap_mode="r")
    n_target = len(target_embs_mmap)
    dim = target_embs_mmap.shape[1]
    
    print(f"  [emb] Building FAISS IVFPQ index (n={n_target:,}, dim={dim}) ...")
    t0 = time.time()
    
    if n_target < 10_000:
        index = faiss.IndexFlatIP(dim)
        index.add(target_embs_mmap[:].astype(np.float32))
    else:
        nlist = max(int(np.sqrt(n_target)), 256)
        nlist = min(nlist, n_target // 20)
        pq_m = FAISS_PQ_M
        
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFPQ(
            quantizer, dim, nlist, pq_m, FAISS_PQ_NBITS,
            faiss.METRIC_INNER_PRODUCT,
        )
        
        train_size = min(n_target, max(nlist * 40, 200_000))
        train_idx = np.random.choice(n_target, train_size, replace=False)
        train_data = target_embs_mmap[train_idx].astype(np.float32)
        index.train(train_data)
        del train_data
        gc.collect()
        
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
    
    # Search S1 in chunks — HARD CAP: top_k per entity
    s1_embs_mmap = np.load(s1_emb_path, mmap_mode="r")
    n_s1 = len(s1_embs_mmap)
    
    print(f"  [emb] Searching top-{top_k} (HARD CAP) for {n_s1:,} S1 entities ...")
    t0 = time.time()
    
    total_yielded = 0
    
    for start in range(0, n_s1, FAISS_QUERY_CHUNK):
        end = min(start + FAISS_QUERY_CHUNK, n_s1)
        q = s1_embs_mmap[start:end].astype(np.float32)
        D, I = index.search(q, top_k)  # HARD CAP: only top_k returned
        
        for local_i in range(end - start):
            global_i = start + local_i
            s1_id = s1_ids_arr[global_i]
            for j in range(top_k):
                idx = int(I[local_i, j])
                if 0 <= idx < n_target:
                    yield (s1_id, target_ids_arr[idx], "embedding")
                    total_yielded += 1
        
        if end % 200_000 == 0 or end == n_s1:
            print(f"    ... searched {end:,}/{n_s1:,} (yielded {total_yielded:,} pairs)")
        
        del q, D, I
        gc.collect()
    
    elapsed = time.time() - t0
    print(f"    → Embedding blocking: {total_yielded:,} candidates in {elapsed:.1f}s")
    
    del s1_embs_mmap, index
    gc.collect()


# ─── Streaming Candidate Writer ────────────────────────────────────────────────

class StreamingCandidateWriter:
    """
    Writes candidate pairs to chunked Parquet files on disk.
    Never holds all candidates in memory.
    """
    def __init__(self, split: str):
        self.split = split
        self.output_dir = CAND_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_buffer: List[Tuple[str, str, str]] = []
        self.chunk_idx = 0
        self.total_written = 0
    
    def write(self, s1_id: str, cand_id: str, strategy: str):
        self.chunk_buffer.append((s1_id, cand_id, strategy))
        if len(self.chunk_buffer) >= CANDIDATE_CHUNK_SIZE:
            self._flush_chunk()
    
    def _flush_chunk(self):
        if not self.chunk_buffer:
            return
        
        # Convert to DataFrame and aggregate by s1_id
        df = pd.DataFrame(self.chunk_buffer, columns=["source1_entity_id", "candidate_entity_id", "strategy"])
        
        # Aggregate strategies per pair
        agg = df.groupby(["source1_entity_id", "candidate_entity_id"])["strategy"].apply(lambda x: "|".join(sorted(set(x)))).reset_index()
        agg.columns = ["source1_entity_id", "candidate_entity_id", "source_strategy"]
        
        # Write chunk
        chunk_path = self.output_dir / f"{self.split}_candidates_chunk_{self.chunk_idx:04d}.parquet"
        agg.to_parquet(chunk_path, index=False, compression="snappy")
        self.chunk_idx += 1
        self.total_written += len(agg)
        print(f"    → Written chunk {self.chunk_idx}: {len(agg):,} pairs (total: {self.total_written:,})")
        self.chunk_buffer.clear()
    
    def finalize(self) -> List[Path]:
        self._flush_chunk()
        # Return list of chunk files for merging
        chunk_files = sorted(self.output_dir.glob(f"{self.split}_candidates_chunk_*.parquet"))
        print(f"  [writer] Finalized {len(chunk_files)} chunks, {self.total_written:,} total pairs")
        return chunk_files


def merge_candidate_chunks(chunk_files: List[Path], output_path: Path, ground_truth: Optional[Dict[str, Set[str]]] = None, s1_df: Optional[pd.DataFrame] = None, total_s2: int = 0, total_s3: int = 0) -> None:
    """Merge chunked Parquet files into final TSV and evaluate recall."""
    print(f"\n  [merge] Merging {len(chunk_files)} chunks ...")
    
    all_dfs = []
    for cf in chunk_files:
        all_dfs.append(pd.read_parquet(cf))
    
    if not all_dfs:
        print("  [merge] No candidates generated!")
        return
    
    merged = pd.concat(all_dfs, ignore_index=True)
    print(f"  [merge] Combined: {len(merged):,} unique S1-candidate pairs")
    
    # Aggregate by S1 entity (candidates may span multiple chunks)
    grouped = merged.groupby("source1_entity_id").agg({
        "candidate_entity_id": lambda x: ",".join(sorted(x.unique())),
        "source_strategy": lambda x: "|".join(sorted(set("|".join(x).split("|"))))
    }).reset_index()
    grouped.columns = ["source1_entity_id", "candidate_entity_ids", "source_strategy"]
    
    # Save final TSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output_path, sep="\t", index=False)
    print(f"  [merge] Saved {len(grouped):,} S1 entities → {output_path}")
    
    # Evaluate recall (train only)
    if ground_truth is not None and s1_df is not None:
        evaluate_blocking_recall(
            {row["source1_entity_id"]: {cid: set(row["source_strategy"].split("|")) for cid in row["candidate_entity_ids"].split(",")}
             for _, row in grouped.iterrows()},
            ground_truth, s1_df,
            {}, {}, {}, total_s2, total_s3,
        )


# ─── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_blocking_recall(
    merged_cands: Dict[str, Dict[str, Set[str]]],
    ground_truth: Dict[str, Set[str]],
    s1_df: pd.DataFrame,
    token_cands: Dict[str, Set[str]],  # kept for signature compat
    lsh_cands: Dict[str, Set[str]],
    emb_cands: Dict[str, Set[str]],
    total_s2_count: int,
    total_s3_count: int,
) -> None:
    s1_country = dict(zip(s1_df["entity_id"].values, s1_df["country"].values))
    
    gt_total = 0
    gt_found = 0
    gt_by_country: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    
    for s1_id, gt_matches in ground_truth.items():
        country = s1_country.get(s1_id, "unknown")
        cand_set = set(merged_cands.get(s1_id, {}).keys())
        
        for gt_id in gt_matches:
            gt_total += 1
            gt_by_country[country][0] += 1
            if gt_id in cand_set:
                gt_found += 1
                gt_by_country[country][1] += 1
    
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
        print(f"\n  Per-Country Recall:")
        for country in sorted(gt_by_country.keys()):
            total, found = gt_by_country[country]
            r = found / total if total > 0 else 0.0
            print(f"    {country:20s}: {found:,}/{total:,} = {r:.4f}")
    
    print("=" * 80)


# ─── Main Pipeline ─────────────────────────────────────────────────────────────

def run_blocking_for_split(
    split: str,
    model=None,
    ground_truth: Optional[Dict[str, Set[str]]] = None,
) -> None:
    print(f"\n{'='*80}")
    print(f"BLOCKING PIPELINE — {split.upper()} (Geo-Partitioned, Streamed)")
    print(f"{'='*80}")
    
    print(f"\n[1/5] Ensuring normalized files ...")
    source_paths = _ensure_normalized_files(split)
    
    print(f"\n[2/5] Loading normalized data ...")
    t0 = time.time()
    s1_df = load_normalized_source(source_paths["S1"])
    print(f"  S1: {len(s1_df):,} ({time.time()-t0:.1f}s)")
    
    writer = StreamingCandidateWriter(split)
    
    total_s2 = 0
    total_s3 = 0
    
    print(f"\n[3/5] Running blocking strategies (geo-partitioned) ...")
    
    for target_key in ["S2", "S3"]:
        t_load = time.time()
        target_df = load_normalized_source(source_paths[target_key])
        n_target = len(target_df)
        if target_key == "S2":
            total_s2 = n_target
        else:
            total_s3 = n_target
        print(f"\n  ── Target: {target_key} ({n_target:,} entities, loaded in {time.time()-t_load:.1f}s) ──")
        
        # Strategy 1: Token blocking (geo-partitioned)
        print(f"\n  Strategy 1: Token (geo-partitioned, min_shared={MIN_SHARED_TOKENS})")
        token_count = 0
        for s1_id, cand_id, strat in block_by_tokens_partitioned(s1_df, target_df, MIN_SHARED_TOKENS):
            writer.write(s1_id, cand_id, strat)
            token_count += 1
        print(f"    → Token: {token_count:,} pairs")
        gc.collect()
        
        # Strategy 2: LSH blocking (geo-partitioned)
        print(f"\n  Strategy 2: LSH (geo-partitioned, n={NGRAM_SIZE}, perm={LSH_NUM_PERM}, thresh={LSH_THRESHOLD})")
        lsh_count = 0
        for s1_id, cand_id, strat in block_by_lsh_partitioned(s1_df, target_df, NGRAM_SIZE, LSH_NUM_PERM, LSH_THRESHOLD):
            writer.write(s1_id, cand_id, strat)
            lsh_count += 1
        print(f"    → LSH: {lsh_count:,} pairs")
        gc.collect()
        
        # Strategy 3: Embedding ANN (HARD CAP top_k=50)
        print(f"\n  Strategy 3: Embedding ANN (HARD CAP top_k={ANN_TOP_K})")
        emb_count = 0
        for s1_id, cand_id, strat in block_by_embedding_partitioned(
            s1_df, target_df, model,
            top_k=ANN_TOP_K,
            source_name=f"{split}_{target_key.lower()}",
            s1_source_name=f"{split}_s1",
        ):
            writer.write(s1_id, cand_id, strat)
            emb_count += 1
        print(f"    → Embedding: {emb_count:,} pairs")
        gc.collect()
        
        del target_df
        gc.collect()
    
    # Step 4: Finalize chunks and merge
    print(f"\n[4/5] Finalizing and merging candidate chunks ...")
    chunk_files = writer.finalize()
    merge_candidate_chunks(chunk_files, CAND_DIR / f"{split}_candidates.tsv", ground_truth, s1_df, total_s2, total_s3)
    
    # Cleanup chunk files
    for cf in chunk_files:
        cf.unlink()
    
    print(f"\n[5/5] Blocking for {split} complete.")


def main():
    overall_t0 = time.time()
    
    try:
        src_dir = Path(__file__).resolve().parent
    except NameError:
        src_dir = Path.cwd()
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