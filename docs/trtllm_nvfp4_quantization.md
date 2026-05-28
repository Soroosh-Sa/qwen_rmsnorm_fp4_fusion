# TensorRT-LLM / ModelOpt NVFP4 quantization workflow

This repo uses NVIDIA's TensorRT-LLM quantization example script:

```bash
/app/tensorrt_llm/examples/quantization/quantize.py
```

The wrapper script is:

```bash
scripts/quantize_trtllm_nvfp4.sh
```

It exports a TensorRT-LLM checkpoint, not a normal Hugging Face checkpoint.

## Controlled small-model experiment

Use the same Qwen model for both paths:

1. original BF16 checkpoint -> NVFP4
2. folded BF16 checkpoint -> NVFP4

Example:

```bash
export MODEL_ROOT=/workspace/models
export MODEL_PROFILE=qwen25_05b
export TP_SIZE=4
export CALIB_SIZE=128
export CALIB_MAX_SEQ_LENGTH=512

bash scripts/download_selected_model.sh
bash scripts/fold_selected_model_sharded.sh
bash scripts/run_small_nvfp4_quantization_pair.sh
```

Outputs:

```text
/workspace/models/<MODEL_TAG>-NVFP4
/workspace/models/<MODEL_TAG>-FOLDED-NVFP4
```

For the Qwen2.5-0.5B profile, this becomes:

```text
/workspace/models/Qwen-Qwen2.5-0.5B-Instruct-NVFP4
/workspace/models/Qwen-Qwen2.5-0.5B-Instruct-FOLDED-NVFP4
```

## Useful environment variables

```bash
INPUT_MODEL=/path/to/hf_or_folded_checkpoint
OUTPUT_DIR=/path/to/trtllm_nvfp4_output
QFORMAT=nvfp4
DTYPE=bfloat16
TP_SIZE=4
PP_SIZE=1
CP_SIZE=1
KV_CACHE_DTYPE=fp8
CALIB_DATASET=cnn_dailymail
CALIB_SIZE=128
CALIB_BATCH_SIZE=1
CALIB_MAX_SEQ_LENGTH=512
TOKENIZER_MAX_SEQ_LENGTH=2048
DEVICE=cuda
DEVICE_MAP=auto
EXTRA_QUANT_ARGS="..."
```

For a quick smoke test, reduce calibration cost:

```bash
CALIB_SIZE=16 CALIB_MAX_SEQ_LENGTH=256 bash scripts/run_small_nvfp4_quantization_pair.sh
```

For better quality, increase calibration:

```bash
CALIB_SIZE=512 CALIB_MAX_SEQ_LENGTH=1024 bash scripts/run_small_nvfp4_quantization_pair.sh
```

## Benchmarking the pair

After both NVFP4 outputs exist:

```bash
export MODEL_PROFILE=qwen25_05b
export TP_SIZE=4
export MAX_SEQ_LEN=4096
export MAX_NUM_TOKENS=4096
export CONTEXTS="1024 2048"
export CONCURRENCIES="1"
export NUM_REQUESTS=4

bash scripts/run_selected_nvfp4_pair_benchmark.sh
```

## Notes

- This quantization path is for true NVFP4/FP4 through TensorRT-LLM/ModelOpt.
- GPTQ and AWQ are INT4 baselines, not FP4.
- RMSNorm folding prepares the checkpoint. Actual speedup still requires the runtime to avoid materializing RMSNorm output or to use a fused TensorRT-LLM/CUDA path.
