# Stage 6: Folded-NVFP4 Base vs Plugin Benchmark

This is the current main benchmark path for the RMSNorm + SwiGLU fusion project.

## What is compared

The benchmark compares two TensorRT-LLM **engine** directories built from the same folded-NVFP4 checkpoint:

1. **Folded NVFP4 base engine**
   - checkpoint: `${FOLDED_NVFP4_PATH}`
   - engine: `${ENGINE_ROOT}/${MODEL_TAG}-FOLDED-NVFP4-engine-tp${TP_SIZE}`
   - no custom plugin enabled during build

2. **Folded NVFP4 plugin engine**
   - checkpoint: `${FOLDED_NVFP4_PATH}`
   - engine: `${ENGINE_ROOT}/${MODEL_TAG}-FOLDED-NVFP4-rms-scale-swiglu-plugin-engine-tp${TP_SIZE}`
   - custom `QwenRmsScaleSwiglu` / `QwenRmsScaleSwigluGated` plugin enabled during build

This is the correct comparison for the current milestone because TensorRT-LLM still handles NVFP4 GEMM quantization internally, while our plugin only replaces the folded RMS-scale + SwiGLU part and returns BF16/FP16 intermediate tensors.

## Main command

```bash
cd ~/workspace/qwen_rmsnorm_fp4_fusion

export MODEL_PROFILE=qwen25_05b
export TP_SIZE=2

# Quick benchmark defaults. Override these for full benchmark.
export CONTEXTS="1024 2048"
export CONCURRENCIES="1 2"
export NUM_REQUESTS=20
export MAX_NEW_TOKENS=64
export OPENAI_API_MODE=completion
export COMPLETION_STREAM=0

bash scripts/run_folded_nvfp4_base_vs_plugin_benchmark.sh
```

Backward-compatible entrypoint:

```bash
bash scripts/run_selected_nvfp4_pair_benchmark.sh
```

This now delegates to the folded-NVFP4 base-vs-plugin benchmark. It no longer serves local TensorRT-LLM checkpoints directly.

## Outputs

The script writes:

```text
results/${MODEL_TAG}_folded_nvfp4_base_engine_benchmark.csv
results/${MODEL_TAG}_folded_nvfp4_plugin_engine_benchmark.csv
results/${MODEL_TAG}_folded_nvfp4_base_vs_plugin_summary.csv
```

The summary CSV includes per-context/concurrency percentage changes for `tps_mean` and `aggregate_tps`.

## Optional original NVFP4 engine

To also benchmark an original-NVFP4 engine:

```bash
RUN_ORIGINAL_ENGINE=1 bash scripts/run_selected_nvfp4_pair_engine_benchmark.sh
```

The default is `RUN_ORIGINAL_ENGINE=0` because the most relevant comparison is folded base vs folded plugin.
