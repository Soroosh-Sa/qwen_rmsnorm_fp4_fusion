# Vast AI Notes

## Recommended first test

Use a smaller Qwen model first:

```text
Qwen/Qwen2.5-1.5B-Instruct
Qwen/Qwen2.5-7B-Instruct
Qwen/Qwen3-8B
```

4x RTX PRO 6000 Blackwell GPUs should be more than enough for folding and verification on these small models.

## Suggested setup

```bash
git clone <your-private-repo-or-upload-folder>
cd qwen_rmsnorm_fp4_fusion
conda create -n qwen_fp4 python=3.10 -y
conda activate qwen_fp4
bash scripts/00_setup_env.sh
```

Then:

```bash
bash scripts/run_smoke_pipeline.sh
```

## Expected outputs

```text
metrics/folding_layer_checks.csv
metrics/folding_layer_checks.json
metrics/fold_checkpoint_records.csv
metrics/fold_checkpoint_records.json
metrics/model_logits_check.json
metrics/run_manifest.json
logs/*.log
outputs/qwen_folded_rmsnorm/
```

## GPU count guidance

For the small model experiment, 1 GPU is usually enough. Using 4 GPUs is fine and lets `device_map=auto` shard the model if needed.

For 460B/480B:

```text
folding only: use shard-by-shard processing or many 80GB-class GPUs
FP4 quantization: likely needs 8-16 GPUs depending on model, sequence length, calibration, and toolchain
```
