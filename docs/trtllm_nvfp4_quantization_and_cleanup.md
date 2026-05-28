# TensorRT-LLM NVFP4 quantization and cleanup notes

## Does quantization save a checkpoint?

Yes. `scripts/quantize_trtllm_nvfp4.sh` writes the quantized TensorRT-LLM/ModelOpt checkpoint to `OUTPUT_DIR`.

For the selected folded model:

```bash
MODEL_PROFILE=qwen25_05b bash scripts/quantize_selected_folded_nvfp4.sh
```

writes:

```text
$FOLDED_NVFP4_PATH
```

For a large folded 480B/460B BF16 checkpoint, set the paths explicitly:

```bash
export INPUT_MODEL=/workspace/models/Qwen3-Coder-480B-A35B-Instruct-FOLDED-BF16
export OUTPUT_DIR=/workspace/models/Qwen3-Coder-480B-A35B-Instruct-FOLDED-NVFP4
export TP_SIZE=16
export CALIB_SIZE=512
export CALIB_MAX_SEQ_LENGTH=2048
bash scripts/quantize_trtllm_nvfp4.sh
```

The wrapper now repairs output metadata after quantization:

```text
src/repair_trtllm_quantized_checkpoint.py
```

This copies missing tokenizer/generation files and fixes cases where the exported config uses an unrecognized `model_type: qwen` even though the source model was `qwen2` or `qwen3`.

## Cleanup

Use this before/after repeated benchmarks:

```bash
bash scripts/cleanup_trtllm_runtime.sh
```

The benchmark runner now calls cleanup automatically before starting and after exiting through a shell trap.

For aggressive cleanup of logs:

```bash
CLEAN_RUNTIME_LOGS=1 bash scripts/cleanup_trtllm_runtime.sh
```
