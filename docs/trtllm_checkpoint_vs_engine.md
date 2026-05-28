# TensorRT-LLM checkpoint vs engine path

`examples/quantization/quantize.py` exports a TensorRT-LLM checkpoint, not a normal Hugging Face model folder.
The output usually looks like:

```text
config.json
rank0.safetensors
rank1.safetensors
...
```

This checkpoint is intended to be consumed by `trtllm-build` to build TensorRT engines.
Serving it directly with `trtllm-serve --backend pytorch` can fail because the PyTorch backend loader expects a different checkpoint layout for fused QKV / fused gate-up modules.

Correct flow for locally quantized NVFP4 checkpoints:

```text
HF BF16 checkpoint
  -> RMSNorm folding
  -> TensorRT-LLM / ModelOpt quantize.py
  -> TensorRT-LLM checkpoint: rank*.safetensors
  -> trtllm-build
  -> TensorRT engine directory
  -> trtllm-serve engine directory
```

For the selected small model pair:

```bash
export MODEL_PROFILE=qwen25_05b
export TP_SIZE=2
export MAX_SEQ_LEN=4096
export MAX_NUM_TOKENS=4096
export CONTEXTS="1024 2048"
export CONCURRENCIES="1"
export NUM_REQUESTS=4

bash scripts/run_selected_nvfp4_pair_engine_benchmark.sh
```

Use `BUILD_ENGINES=0` if the engines are already built and you only want to re-run benchmarks.
