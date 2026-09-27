# Amazon ML Challenge 2026 — Business Entity Resolution

Solution for the Amazon ML Challenge 2026 business entity resolution task: match
every Source 1 entity to its true duplicate(s) in Source 2 / Source 3.

## Current submission (v3)

| | |
|---|---|
| Deliverable | `submission/output/matching_results.tsv`, `submission/output/candidate_pairs.tsv` |
| Rows | 1,732,544 (one per Source 1 entity, empty = no match) |
| Entities with a match | 294,879 (17.0%) |
| Matched pairs | 663,431 (2.25 per matched entity) |
| Validator | **PASS** (`utils/validate_submission.py`) |
| MD5 `matching_results.tsv` | `BAC7D1914619C527B9A7FC9039F5267D` |
| Backup | `s3://ml-challenge-entity-res-1790438290/output_v3/` |

## Approach

**Rule-based blocking → gradient-boosted classifier.** Three tiers, each a
trade-off measured on held-out training data:

1. **Normalize** — lowercase/strip names, addresses, geo fields; derive a
   2-token name key. (`src/normalize.py`)
2. **Block** — build candidate pairs from composite keys (exact name+geo, token
   keys, postal, state, country), capped at 12 candidates per entity. Yields
   5.37M train / 5.18M test pairs from 1.73M Source 1 entities.
3. **Rank** — LightGBM (L2, 300 trees, `pos_rate` 0.0826) over 13 pairwise
   features (exact name, 4-char name prefix, name-length ratio/absdiff, 2-token
   key match, country/city/state/postal/address agreement, and combinations).
   Pairs scoring ≥ 0.38 are emitted.

Top feature gains: `tok_key_exact`, `city_eq`, `name_len_absdiff`,
`name_len_ratio`, `addr_exact`.

## Results

Measured on a 60,000-entity ground-truth-backed sample of training data:

| blocking tier set | recall | precision | micro-F1 |
|---|---|---|---|
| T0 exact name+geo | 0.150 | 0.861 | 0.256 |
| T0+T1+T2 (shipped as v2) | 0.340 | 0.483 | 0.399 |
| T0..T4 (v3 candidate set) | 0.414 | 0.233 | 0.298 |
| **v3 after LightGBM ranking** | **0.813** | **0.746** | **0.778** |

### Interpreting the metric

The official metric is **not** micro-F1. Per the challenge README it is
**F_0.5 macro-averaged per Source 1 entity over all entities, singletons
included**:

```
F_0.5 = (1.25 * P * R) / (0.25 * P + R)
```

- An entity with no true match scores **1.0 for an empty prediction** and
  **0.0 for any prediction at all**.
- It is **precision-heavy** — a false merge costs ~2x a missed match.
- An entity with 1 true match scores **1.0** from a single correct prediction,
  **0.556** from two (P=0.5, R=1), and **0.385** from three (P=0.33, R=1).

Two design consequences follow directly, and both should be kept in mind when
choosing between model variants:

1. **Breadth per entity is expensive.** Emitting 2 candidates where 1 is correct
   halves the score for that entity, so a single confident prediction beats a
   ranked list.
2. **Coverage still matters.** An entity with a true match but no prediction
   scores 0, so the candidate set must still cover the entities that do have
   matches.

The micro-F1 figures above are a tuning aid only; they are a different quantity
from the official score and overstate it.

### Planned improvements

1. **Raise candidate coverage.** The current tier set caps at 0.414 recall.
   Widening to the country-level (T5) and single-token (T6) tiers lifts candidate
   recall to ~0.56, at the cost of many more false candidates for the ranker to
   reject.
2. **Top-1 prediction per entity** — emit only each entity's single best-scoring
   candidate, which is the operating point F_0.5 rewards most.
3. Both are pure inference over the existing candidates and model, so neither
   requires retraining.

## Repo layout

```
code/business_entity_resolution/
  src/
    config.py        path config (SER_DATA_ROOT override)
    normalize.py     Stage 1 - field normalization
    blocking.py      Stage 2 - candidate generation (canonical)
    features.py      Stage 3 - pairwise feature engineering
    train.py         Stage 4 - model training
    postprocess.py   Stage 5 - submission assembly
    blocking_deployed_ec2.py   earlier EC2-deployed blocking variant (reference)
  tests/             109 tests, all passing
  requirements.txt
  setup1.sh setup2.sh   EC2 environment setup
utils/validate_submission.py   official-format validator
submission/output/             the deliverable
docs/                          challenge PDFs, master doc, implementation plan
```

`dataset/`, `code/*/data/` and `submission/` are gitignored — they hold large
generated artifacts. Raw data and all submission versions are mirrored in S3
bucket `ml-challenge-entity-res-1790438290`.

## Reproducing

```bash
pip install -r code/business_entity_resolution/requirements.txt
python -m pytest code/business_entity_resolution/tests -q

python code/business_entity_resolution/src/normalize.py
python code/business_entity_resolution/src/blocking.py
python code/business_entity_resolution/src/features.py
python code/business_entity_resolution/src/train.py
python code/business_entity_resolution/src/postprocess.py

python utils/validate_submission.py \
  --matching submission/output/matching_results.tsv \
  --candidate submission/output/candidate_pairs.tsv \
  --test-dir dataset/test
```

## Known gaps

- The intended MiniLM-embedding featurizer in `features.py` was never run at
  scale (~24M texts, ~7h). Ranking used the 13 cheap lexical features instead.
- Validation F1 is measured on entities that have ground-truth matches, so it is
  a tuning aid rather than a proxy for the official macro-F0.5 score.
- The LSH block in the deployed `blocking.py` was disabled for the deadline; the
  exact patched state was never committed.
- `candidate_pairs.tsv` lists only retained (post-threshold) pairs, not the full
  blocking candidate set the challenge docs describe.
