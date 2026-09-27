import os
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PKG_DIR = SRC_DIR.parent
PROJECT_ROOT = PKG_DIR.parents[1]

# Data now lives at <root>/dataset/. SER_DATA_ROOT overrides it if relocated.
_env = os.environ.get("SER_DATA_ROOT")
DATA_RAW = Path(_env) if _env else (PROJECT_ROOT / "dataset")

DATA = PKG_DIR / "data"
NORM_DIR = DATA / "normalized"
CAND_DIR = DATA / "candidates"
CACHE_DIR = DATA / "cache"
OUTPUT_DIR = PKG_DIR / "output"
MODELS_DIR = PKG_DIR / "models"

RAW_TRAIN = DATA_RAW / "train"
RAW_TEST = DATA_RAW / "test"
GROUND_TRUTH = RAW_TRAIN / "train_ground_truth.tsv"
