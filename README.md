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

**Leaderboard (v2, submitted): 0.081.**

### Read this before trusting the numbers above

The official metric is **not** micro-F1. Per the challenge README it is
**F_0.5 macro-averaged per Source 1 entity over all entities, singletons
included**:

```
F_0.5 = (1.25 * P * R) / (0.25 * P + R)
```

- An entity with no true match scores **1.0 for an empty prediction** and
  **0.0 for any prediction at all**.
- It is **precision-heavy** — a false merge costs ~2x a missed match.

Two consequences:

1. **The micro-F1 column above is the wrong yardstick** and systematically
   overstates the submission. Ranking more than one candidate per entity is
   heavily punished: an entity with 1 true match scores 1.0 from a single correct
   prediction but only 0.556 from two predictions (P=0.5, R=1.0).
2. **v2's 0.081 is consistent with under-coverage.** v2 left 1,234,504 of
   1,732,544 entities (71%) empty while emitting 10.2 candidates for each
   entity it did answer. Entities with true matches but no prediction score 0.

The highest-leverage remaining change is therefore **top-1 prediction per
entity** — keep only each entity's single best-scoring candidate. This raises
per-entity precision at some cost in recall, which is the correct direction for
F_0.5. Not yet applied.

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
- Validation F1 is measured on entities that have ground-truth matches, so it
  overstates performance versus the macro-F0.5 leaderboard metric.
- The LSH block in the deployed `blocking.py` was disabled for the deadline; the
  exact patched state was never committed.
- `candidate_pairs.tsv` lists only retained (post-threshold) pairs, not the full
  blocking candidate set the challenge docs describe.
