#!/usr/bin/env bash
cat <<'EOM'
Available MODEL_PROFILE values:

  qwen25_05b     -> Qwen/Qwen2.5-0.5B-Instruct
  qwen25_15b     -> Qwen/Qwen2.5-1.5B-Instruct   [default]
  qwen25_7b      -> Qwen/Qwen2.5-7B-Instruct
  qwen3_17b      -> Qwen/Qwen3-1.7B
  qwen3_4b       -> Qwen/Qwen3-4B
  qwen3_8b       -> Qwen/Qwen3-8B
  prequant_nvfp4 -> nvidia/Qwen3-Coder-480B-A35B-Instruct-NVFP4
  custom         -> use MODEL_ID=...

Examples:

  MODEL_PROFILE=qwen25_05b bash scripts/download_selected_model.sh
  MODEL_PROFILE=qwen3_8b bash scripts/download_selected_model.sh
  MODEL_PROFILE=custom MODEL_ID=Qwen/Qwen2.5-3B-Instruct bash scripts/download_selected_model.sh

Then fold:

  MODEL_PROFILE=qwen3_8b bash scripts/fold_selected_model_sharded.sh
EOM
