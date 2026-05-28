# Consistent selected-model TensorRT-LLM pipeline

Use this checklist for every new model profile.

1. Select model profile and TP size.

```bash
export MODEL_PROFILE=qwen25_05b
export TP_SIZE=2
export AUTO_ADJUST_TP_SIZE=1
```

2. Download and fold.

```bash
bash scripts/download_selected_model.sh
bash scripts/fold_selected_model_sharded.sh
```

3. Quantize original and folded checkpoints using the same resolved TP size.

```bash
export CLEAN_OUTPUT_DIR=1
bash scripts/run_small_nvfp4_quantization_pair.sh
bash scripts/validate_selected_nvfp4_checkpoints.sh
```

4. Build engines. The build script now reads the TP size from the quantized checkpoint config and uses that as the source of truth.

```bash
export CLEAN_ENGINE_DIR=1
bash scripts/build_selected_nvfp4_pair_engines.sh
bash scripts/validate_selected_engine_dirs.sh
```

5. Benchmark engines only. Do not directly serve local rank*.safetensors checkpoints with the PyTorch backend.

```bash
export BUILD_ENGINES=0
bash scripts/run_selected_nvfp4_pair_engine_benchmark.sh
```

6. Audit whenever switching models.

```bash
bash scripts/audit_selected_pipeline.sh
```

Key invariant: the same effective TP size must be used for quantized checkpoint rank layout, engine directory naming, engine build validation, and benchmark serving.
