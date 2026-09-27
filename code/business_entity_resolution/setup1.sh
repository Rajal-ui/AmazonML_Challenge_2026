#!/bin/bash
set -e
cd /home/ubuntu
sudo apt-get update -y
sudo apt-get install -y python3-pip python3-venv git
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
echo "=== STEP 1 COMPLETE ==="
