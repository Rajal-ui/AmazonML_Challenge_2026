#!/bin/bash
set -e
cd /home/ubuntu
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install faiss-cpu sentence-transformers lightgbm pandas numpy scikit-learn
mkdir -p dataset/raw dataset/normalized dataset/candidates dataset/features dataset/models dataset/output
aws s3 sync s3://ml-challenge-entity-res-1790438290/raw/ ./dataset/raw/
echo "=== STEP 2 COMPLETE ==="
