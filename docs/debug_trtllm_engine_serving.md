# Debugging TensorRT-LLM engine serving

If `trtllm-serve` reports:

```text
No weight files found in /workspace/engines/...engine-tpX
```

then the engine directory is being treated like a checkpoint directory, or it does not actually contain serialized engine files.

A valid TensorRT-LLM engine directory should contain serialized engine files such as:

```text
rank0.engine
rank1.engine
config.json
```

or, depending on version, equivalent `.plan` files.

Do not serve an engine directory unless validation passes:

```bash
python3 src/validate_trtllm_engine_dir.py \
  --engine-dir /workspace/engines/Qwen-Qwen2.5-0.5B-Instruct-NVFP4-engine-tp2 \
  --expected-tp 2
```

The engine server script now forces the TensorRT backend:

```bash
--backend tensorrt
TLLM_USE_TRT_ENGINE=1
```

and validates the engine directory before starting the server.

Recommended clean rebuild:

```bash
rm -rf /workspace/engines/Qwen-Qwen2.5-0.5B-Instruct-NVFP4-engine-tp2
rm -rf /workspace/engines/Qwen-Qwen2.5-0.5B-Instruct-FOLDED-NVFP4-engine-tp2

export MODEL_PROFILE=qwen25_05b
export TP_SIZE=2
export CLEAN_ENGINE_DIR=1
export BUILD_ENGINES=1

bash scripts/build_selected_nvfp4_pair_engines.sh
bash scripts/validate_selected_engine_dirs.sh
bash scripts/run_selected_nvfp4_pair_engine_benchmark.sh
```
