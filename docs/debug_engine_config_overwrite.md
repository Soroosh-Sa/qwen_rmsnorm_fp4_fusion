# Engine config overwrite bug

Older repo versions copied `CHECKPOINT_DIR/config.json` into `ENGINE_DIR/config.json` after `trtllm-build`.
That is wrong: `trtllm-build` writes an engine-specific `config.json`. If it is overwritten
with the checkpoint config, `trtllm-serve` may see `rank0.engine` and `rank1.engine`, but still
try to load checkpoint weights and fail with:

```text
RuntimeError: No weight files found in /workspace/engines/...-engine-tp2
```

Fix:

```bash
export CLEAN_ENGINE_DIR=1
export TRTLLM_ENGINE_BACKEND=tensorrt
bash scripts/build_selected_nvfp4_pair_engines.sh
bash scripts/validate_selected_engine_dirs.sh
bash scripts/run_selected_nvfp4_pair_engine_benchmark.sh
```

The v13 build script preserves `ENGINE_DIR/config.json` and saves the source checkpoint config as
`ENGINE_DIR/checkpoint_config.json` instead.
