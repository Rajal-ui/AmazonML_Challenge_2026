# ML Challenge 2026: Business Entity Resolution — Solution

**Team Name:** nids
**Team Members:** RAJAL MISTRY
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

A three-stage **rule-based blocking + gradient-boosted ranking** pipeline. Blocking
reduces 1.73M Source 1 entities against Sources 2/3 to 5.18M candidate pairs using
composite name/geo keys; a 13-feature LightGBM classifier then scores each pair and
emits those above a tuned threshold, cutting 5.18M candidates to 663K predictions.
The key insight is that key-only blocking is precision-limited (measured micro-F1
0.256–0.399 across six tier sets), so a learned ranker is not optional but is the
single largest source of gain — it lifts precision from 0.233 to 0.746 at 0.813 recall.

---

## 2. Methodology

### 2.1 Problem Analysis

- Source 1 contains 1,732,544 entities; Sources 2 and 3 are the candidate pools.
- Ground truth is available only for training, and only for a subset of Source 1
  entities — so recall must be measured on GT-backed entities and precision on
  emitted pairs, and the two cannot be naively combined.
- Business names are dirty (case, punctuation, legal suffixes, token order);
  addresses vary widely and are frequently absent. Geo fields are the most reliable
  signal, but country alone is far too coarse.
- A blocking key that is loose enough to reach high recall admits too many false
  candidates to be emitted directly. Measured ceiling for the T0–T4 candidate set
  is F1 0.586 even with a *perfect* classifier.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier
**Core Innovation:** Tiered composite blocking keys with a per-entity candidate cap,
followed by a gradient-boosted ranker that converts a high-recall, low-precision
candidate set into a high-precision submission.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** exact normalized name + country + city (T0); name-token
  composite keys (T1); normalized name + token key (T2); country + postal5 + token
  (T3); country + state + token (T4). Cap of 12 candidates per Source 1 entity.
- **Candidate pairs generated:** 5,367,170 train / 5,180,852 test.
- **How true matches were protected:** tiers are additive and ordered by
  decreasing strictness, so strict keys are always consumed before loose ones.
  Measured tier recall on held-out GT:

  | tiers | recall | precision |
  |---|---|---|
  | T0 | 0.150 | 0.861 |
  | T0–T2 | 0.340 | 0.483 |
  | T0–T4 | 0.414 | 0.233 |

---

## 4. Matching Model

**Features used (13):** `name_exact`, `name_prefix4` (first 4 chars, length ≥ 4),
`tok_key_exact` (first 2 name tokens), `name_len_absdiff`, `name_len_ratio`,
`country_eq`, `city_eq`, `state_eq`, `postal5_eq`, `city_state_eq`, `addr_exact`,
`addr_len_absdiff`, `city_and_name_exact`.

**Model type:** LightGBM, L2 objective, 300 trees, trained on 5,367,170 pairs
(443,470 positive, `pos_rate` 0.0826).

**Threshold selection method:** swept on the validation split to maximise F1 →
threshold 0.38. Feature gains ranked `tok_key_exact`, `city_eq`,
`name_len_absdiff`, `name_len_ratio`, `addr_exact`.

**Validation result:** F1 0.7776 (precision 0.7455, recall 0.8127) on a
held-out split of the training candidates.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** not measured locally. The metric is defined in the
  challenge README and is macro-averaged per Source 1 entity over all entities
  with singletons included; the figure above is micro-F1, which is a different
  quantity and is optimistic relative to F_0.5 macro.
- **Leaderboard (submitted v2, no ranker): 0.081.** v2 answered only 28.75% of
  Source 1 entities and emitted 10.2 candidates for each. Because an entity with
  a true match but no prediction scores 0, under-coverage dominated the loss.
- **Common false positives (wrong merges):** businesses sharing a postal code and
  a first-two-token name key — franchise/chain locations and multi-site
  businesses at one address. `postal5_eq` is near-universal in the candidate set,
  so it carries little discriminative signal on its own.
- **Common false negatives (missed matches):** entities whose true match differs
  in every blocking field (name reworded, address absent, different country) are
  never generated as candidates and are unrecoverable at this tier set. Widening
  to T5/T6 lifts candidate recall to 0.558 but collapses precision to 0.025,
  which F_0.5 punishes far harder than it rewards.

---

## 6. Conclusion

Blocking plus a learned ranker is the right structure: key-only blocking tops out
near F1 0.40, and the ranker roughly doubles that on held-out data. The main
lesson is that the leaderboard metric is macro-F1 with β=0.5 over all entities, so
per-entity discipline — how many candidates you emit for each Source 1 entity —
matters more than aggregate pair counts. The clear next step is top-1 prediction
per entity, which raises per-entity precision from 2.25 candidates toward 1.0.

---

## Appendix

### A. Code Artefacts

Source ships in `code/business_entity_resolution/`; entry point per stage is the
matching module in `src/`. Stage order is `normalize.py` → `blocking.py` →
`features.py` → `train.py` → `postprocess.py`, with paths centralised in
`config.py` (override the data root via the `SER_DATA_ROOT` environment variable).
`src/blocking_deployed_ec2.py` is retained for reference as the earlier deployed
blocking variant; `src/blocking.py` is canonical. `requirements.txt` pins
dependencies and `tests/` holds 109 passing unit tests.

### B. Additional Results

Submission statistics: 1,732,544 rows, 294,879 entities with a match (17.0%),
663,431 matched pairs (2.25 per matched entity), validator PASS.

Environment: m5.2xlarge (8 vCPU / 32 GB), Python 3.12, pandas 2.x, LightGBM 4.7.0,
pyarrow 25.0.1. The intended MiniLM embedding featurizer was not run at scale
(~24M texts, ~7h) and is not reflected in these results.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
