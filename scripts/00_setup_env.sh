#!/usr/bin/env bash
set -euo pipefail

# Recommended on Vast AI:
#   conda create -n qwen_fp4 python=3.10 -y
#   conda activate qwen_fp4

python -m pip install --upgrade pip
pip install -r requirements.txt

# Optional, depending on your container:
# pip install nvidia-modelopt tensorrt_llm
