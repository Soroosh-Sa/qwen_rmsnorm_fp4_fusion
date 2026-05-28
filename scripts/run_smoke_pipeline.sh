#!/usr/bin/env bash
set -euo pipefail

bash scripts/01_verify_folding.sh
bash scripts/02_fold_checkpoint.sh
bash scripts/03_verify_model_logits.sh
bash scripts/05_summarize_metrics.sh
