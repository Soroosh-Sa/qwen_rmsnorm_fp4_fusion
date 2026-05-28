# Shard-by-shard RMSNorm folding

This path is for the real large-model workflow. It avoids `AutoModelForCausalLM.from_pretrained(...)` and never loads the full model.

## What it changes

For each transformer layer:

```text
input_layernorm.weight -> self_attn.q_proj.weight
input_layernorm.weight -> self_attn.k_proj.weight
input_layernorm.weight -> self_attn.v_proj.weight

post_attention_layernorm.weight -> mlp.gate_proj.weight
post_attention_layernorm.weight -> mlp.up_proj.weight
```

The rewrite is:

```python
W_fused = W * gamma[None, :]
```

Then the corresponding RMSNorm weights are set to `1.0`, so normal Qwen code no longer applies `gamma` again.

## Small test

```bash
bash scripts/00_setup_env.sh
bash scripts/00b_download_small_checkpoint.sh
export INPUT_DIR=checkpoints/qwen_small_original
export OUTPUT_DIR=outputs/qwen_small_folded_sharded
bash scripts/02c_dry_run_sharded_folding.sh
bash scripts/02b_fold_checkpoint_sharded_small.sh
```

Then run the regular logit check if desired by pointing `configs/qwen_small.yaml` to the folded output.

## Large model

```bash
export INPUT_DIR=/path/to/original/qwen-460b-or-480b-hf-checkpoint
export OUTPUT_DIR=/path/to/folded/qwen-460b-or-480b-hf-checkpoint
export METRICS_DIR=metrics/qwen_460b_fold
bash scripts/06_fold_checkpoint_sharded_large.sh
```

Optional debug mode:

```bash
export MAX_SHARDS=2
bash scripts/06_fold_checkpoint_sharded_large.sh
```

Validate a small sample:

```bash
export ORIGINAL_DIR=/path/to/original/qwen-460b-or-480b-hf-checkpoint
export FOLDED_DIR=/path/to/folded/qwen-460b-or-480b-hf-checkpoint
export MAX_TENSORS=32
bash scripts/07_validate_sharded_large_sample.sh
```

## GPU requirement

The sharded folding step is CPU/offline and does not require GPUs. GPUs are needed later for logit/perplexity checks, FP4/NVFP4 quantization calibration, TensorRT-LLM engine build, and inference benchmarking.
