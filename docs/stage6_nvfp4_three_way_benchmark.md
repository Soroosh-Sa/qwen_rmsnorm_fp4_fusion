# Stage 6: final NVFP4 three-way benchmark

After Stage 5 passes, the main final benchmark is no longer only folded-base vs plugin.
It must compare the plugin engine against the standard NVFP4 TensorRT-LLM implementation.

## Targets

```text
A. normal/original NVFP4 engine
   original weights -> NVFP4 quantization -> TensorRT-LLM engine

B. folded NVFP4 base engine
   folded weights -> NVFP4 quantization -> TensorRT-LLM engine, no plugin

C. folded NVFP4 plugin engine
   folded weights -> NVFP4 quantization -> TensorRT-LLM engine with QwenRmsScaleSwiglu plugin
```

## Comparisons reported

```text
C vs A: final real comparison against normal NVFP4 TensorRT-LLM
C vs B: plugin/graph benefit after folding
B vs A: folding + requantization effect without plugin
```

## Main command

```bash
export MODEL_PROFILE=qwen25_05b
export TP_SIZE=2

export CONTEXTS="1024 2048"
export CONCURRENCIES="1 2"
export NUM_REQUESTS=20
export MAX_NEW_TOKENS=64
export OPENAI_API_MODE=completion
export COMPLETION_STREAM=0

bash scripts/run_nvfp4_normal_vs_folded_vs_plugin_benchmark.sh
```

## Outputs

```text
results/<MODEL_TAG>_normal_nvfp4_engine_benchmark.csv
results/<MODEL_TAG>_folded_nvfp4_base_engine_benchmark.csv
results/<MODEL_TAG>_folded_nvfp4_plugin_engine_benchmark.csv
results/<MODEL_TAG>_nvfp4_normal_vs_folded_vs_plugin_summary.csv
```

The summary CSV includes `plugin_vs_normal_*`, `plugin_vs_folded_base_*`, and
`folded_base_vs_normal_*` percentage-change columns.

## Build behavior

By default, the script builds missing checkpoints and engines:

```bash
BUILD_MISSING_CHECKPOINTS=1
BUILD_ENGINES=1
```

If the original and folded NVFP4 checkpoints and all three engines already exist,
you can skip rebuilding:

```bash
BUILD_ENGINES=0 bash scripts/run_nvfp4_normal_vs_folded_vs_plugin_benchmark.sh
```

If you only want the ablation B vs C, use:

```bash
bash scripts/run_folded_nvfp4_base_vs_plugin_benchmark.sh
```
