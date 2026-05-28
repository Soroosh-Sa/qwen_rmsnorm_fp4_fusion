# Vast AI TensorRT-LLM integration notes

This version reuses the serving and benchmarking style from the previous TensorRT-LLM Qwen assignment.

## What was integrated

- `trtllm-serve serve --backend pytorch` workflow.
- OpenAI-compatible `/v1/chat/completions` and `/v1/completions` benchmark client.
- Server health wait loop with early failure diagnostics.
- Safe context/concurrency planner based on `max_seq_len`, `max_num_tokens`, GPU memory, KV dtype, and tensor parallel size.
- CSV metrics format compatible with the previous assignment: TTFT, TPS, aggregate TPS, VRAM idle/load, KV-cache growth, GPU utilization, and runtime stability.

## Main comparison targets

The preferred comparison avoids slow BF16 full-model serving and instead compares quantized targets:

1. `prequant_nvfp4`: NVIDIA pre-quantized NVFP4 Qwen baseline.
2. `folded_nvfp4`: our RMSNorm-folded checkpoint quantized to NVFP4.
3. `folded_gptq_int4`: our RMSNorm-folded checkpoint quantized to GPTQ INT4.

GPTQ INT4 is not FP4. It is included only as a practical 4-bit baseline.

## Typical Vast AI run

```bash
cd qwen_rmsnorm_fp4_fusion
bash scripts/setup_vast_trtllm_env.sh

# Run one target.
TARGET=prequant_nvfp4 \
TP_SIZE=4 \
MAX_SEQ_LEN=32768 \
MAX_NUM_TOKENS=32768 \
CONTEXTS="1024 8192 32768" \
CONCURRENCIES="1 2" \
bash scripts/run_one_target_server_and_benchmark.sh

# Run all comparison targets, if all model paths exist.
TARGETS="prequant_nvfp4 folded_nvfp4 folded_gptq_int4" \
TP_SIZE=4 \
MAX_SEQ_LEN=32768 \
MAX_NUM_TOKENS=32768 \
bash scripts/run_comparison_suite.sh
```

## Important knobs

- `TARGET`: `prequant_nvfp4`, `folded_nvfp4`, `folded_gptq_int4`, or `small_folded_nvfp4`.
- `TP_SIZE`: tensor parallel degree, for example 4, 8, or 16.
- `MAX_SEQ_LEN`: TensorRT-LLM server max sequence length.
- `MAX_NUM_TOKENS`: important for long context; keep it aligned with `MAX_SEQ_LEN` unless the server needs a smaller limit.
- `KV_DTYPE`: default `fp8`, matching the previous assignment style.
- `KV_MEMORY_FRACTION`: default `0.70`.
- `CONTEXTS`: context lengths to test.
- `CONCURRENCIES`: concurrency levels to test.

## Relation to fusion

The folding scripts create compatibility-folded checkpoints:

```text
W_fused = W * gamma
RMSNorm gamma = 1
```

This makes the checkpoint mathematically equivalent, but it does not by itself remove the RMSNorm kernel or intermediate activation traffic. Real compute reduction requires a runtime/kernel change that computes:

```text
inv_rms = rsqrt(mean(x*x) + eps)
y_raw = x @ W_fused.T
y = y_raw * inv_rms
```

The benchmark integration is prepared so that once a TensorRT-LLM plugin or graph rewrite exists, it can be exposed as another `TARGET` and compared using the same CSV metrics.
