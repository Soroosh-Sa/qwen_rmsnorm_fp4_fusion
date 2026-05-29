# Qwen RMSNorm Folding + FP4/NVFP4 Fusion Testbed

This project is a small, controlled testbed for validating the idea:

```text
RMSNorm(x, gamma) -> Linear(W)
```

can be rewritten as:

```text
RMSOnly(x) -> Linear(W * gamma)
```

before FP4/NVFP4 quantization.

The first goal is to test this on a smaller Qwen model on 4x RTX PRO 6000 Blackwell GPUs. If correctness and quantization are stable, the same workflow can be scaled to a very large Qwen model such as 460B/480B.

## Directory layout

```text
qwen_rmsnorm_fp4_fusion/
  configs/                  # experiment configuration files
  scripts/                  # bash entry points
  src/                      # Python implementation
  metrics/                  # CSV/JSON metrics are written here
  logs/                     # run logs
  outputs/                  # folded/quantized model outputs
  checkpoints/              # optional local checkpoints/cache
  docs/                     # notes and experiment plan
```

## Main workflow

### 1. Install environment

```bash
bash scripts/00_setup_env.sh
```

### 2. Configure the model

Edit:

```bash
configs/qwen_small.yaml
```

Example model choices:

```text
Qwen/Qwen2.5-0.5B-Instruct
Qwen/Qwen2.5-1.5B-Instruct
Qwen/Qwen2.5-7B-Instruct
Qwen/Qwen3-8B
```

### 3. Verify RMSNorm folding without saving a full checkpoint

```bash
bash scripts/01_verify_folding.sh
```

This compares the original module output against the folded equivalent on random activations.

### 4. Fold the checkpoint

```bash
bash scripts/02_fold_checkpoint.sh
```

This creates a modified checkpoint where:

```text
q/k/v/gate/up Linear weights are multiplied by their preceding RMSNorm gamma
corresponding RMSNorm gamma vectors are set to 1.0
```

### 5. Verify original model vs folded model text logits

```bash
bash scripts/03_verify_model_logits.sh
```

This loads the original and folded checkpoints and compares logits on short prompts.

### 6. Quantize to FP4/NVFP4

```bash
bash scripts/04_quantize_modelopt_placeholder.sh
```

This is intentionally a placeholder because the exact ModelOpt/TensorRT-LLM command depends on the installed versions and company target format. Use this script as the place to add the final quantization command.

## Important correctness rule

After folding:

```text
W_fused = W * gamma
```

you must not also apply the old RMSNorm gamma. This project handles that by setting the RMSNorm weights to all ones in the folded checkpoint.



## Shard-by-shard folding path for large models

For large Qwen checkpoints, do **not** load the whole model just to fold `gamma` into `W`. Use the safetensors shard processor instead:

```bash
# small local checkpoint test
bash scripts/00b_download_small_checkpoint.sh
export INPUT_DIR=checkpoints/qwen_small_original
export OUTPUT_DIR=outputs/qwen_small_folded_sharded
bash scripts/02c_dry_run_sharded_folding.sh
bash scripts/02b_fold_checkpoint_sharded_small.sh

# real 460B/480B-style checkpoint
export INPUT_DIR=/path/to/original/qwen-460b-or-480b-hf-checkpoint
export OUTPUT_DIR=/path/to/folded/qwen-460b-or-480b-hf-checkpoint
bash scripts/06_fold_checkpoint_sharded_large.sh
```

This path reads one `.safetensors` shard at a time, folds only the target tensors, writes a new checkpoint, and saves reports under `metrics/`. See `docs/sharded_folding.md`.

## Metrics

Scripts write outputs to:

```text
metrics/folding_layer_checks.csv
metrics/model_logits_check.json
metrics/run_manifest.json
```


## v3 update: compatibility folding vs actual fused compute

The current folding scripts mainly implement **compatibility folding**:

```text
W_fused = W * gamma
RMSNorm gamma = 1
```

This is required for correctness and for quantizing the folded checkpoint, but it
is not by itself a full kernel-fusion speedup. A generic Qwen runtime may still
execute the RMSNorm kernel and multiply by `1`.

The actual optimized target is documented in:

```bash
python src/fused_runtime_notes.py
bash scripts/08_test_fused_compute_math.sh
```

The target computation is:

```text
inv_rms = rsqrt(mean(x*x) + eps)
y = (x @ W_fused.T) * inv_rms
```

That path avoids materializing the normalized activation and should be
implemented later as a TensorRT-LLM graph rewrite/plugin/custom fused kernel.

## v3 update: quantized comparison targets

The requested main comparison now avoids original BF16 runtime benchmarking and
uses:

```text
1. pre-quantized NVFP4 baseline
2. folded GPTQ INT4 baseline
3. folded TensorRT-LLM NVFP4 target
```

Generate the comparison manifest with:

```bash
bash scripts/11_write_comparison_manifest.sh
```

Placeholders for quantization hooks:

```bash
bash scripts/09_quantize_gptq_int4_placeholder.sh
bash scripts/10_quantize_modelopt_nvfp4_placeholder.sh
bash scripts/12_compare_quantized_targets_placeholder.sh
```

Remember: GPTQ INT4 is not FP4. NVFP4 is the true FP4 path for Blackwell.

## v4 update: TensorRT-LLM benchmark workflow from previous Vast AI assignment

This repo now includes a benchmark/serving workflow aligned with the previous `trtllm-qwen-benchmark` assignment repo.

New important files:

```text
benchmark/benchmark_openai_stream.py
benchmark/plan_safe_tests.py
scripts/common_env.sh
scripts/setup_vast_trtllm_env.sh
scripts/serve_quantized_target.sh
scripts/run_quantized_target_benchmark.sh
scripts/run_one_target_server_and_benchmark.sh
scripts/run_comparison_suite.sh
src/summarize_quantized_comparison.py
docs/vast_trtllm_benchmark_integration.md
```

The main comparison is now:

```text
pre-quantized NVFP4 baseline
vs folded + GPTQ INT4
vs folded + TensorRT-LLM/ModelOpt NVFP4
```

This intentionally avoids full BF16 Qwen 460B/480B serving as the main baseline, because that can be too slow and memory-heavy.

Example:

```bash
bash scripts/setup_vast_trtllm_env.sh

TARGET=prequant_nvfp4 \
TP_SIZE=4 \
MAX_SEQ_LEN=32768 \
MAX_NUM_TOKENS=32768 \
CONTEXTS="1024 8192 32768" \
CONCURRENCIES="1 2" \
bash scripts/run_one_target_server_and_benchmark.sh
```

To run all configured quantized targets:

```bash
TARGETS="prequant_nvfp4 folded_nvfp4 folded_gptq_int4" \
TP_SIZE=4 \
MAX_SEQ_LEN=32768 \
MAX_NUM_TOKENS=32768 \
bash scripts/run_comparison_suite.sh
```

See `docs/vast_trtllm_benchmark_integration.md` for details.

## v5 update: choose Qwen model by profile

You can now choose the Qwen model without editing scripts.

List available profiles:

```bash
bash scripts/list_model_profiles.sh
```

Download and fold a small model first:

```bash
MODEL_PROFILE=qwen25_05b bash scripts/download_selected_model.sh
MODEL_PROFILE=qwen25_05b bash scripts/fold_selected_model_sharded.sh
```

Try a larger small model:

```bash
MODEL_PROFILE=qwen3_8b bash scripts/download_selected_model.sh
MODEL_PROFILE=qwen3_8b bash scripts/fold_selected_model_sharded.sh
```

Use any custom Qwen model:

```bash
MODEL_PROFILE=custom \
MODEL_ID=Qwen/Qwen2.5-3B-Instruct \
bash scripts/download_selected_model.sh
```

The selected-model paths are controlled by:

```bash
MODEL_ROOT=/workspace/models
MODEL_PROFILE=qwen3_8b
```

See `docs/model_selection.md`.

## v6 update: TensorRT-LLM / ModelOpt NVFP4 quantization scripts

This version adds actual wrapper scripts around the TensorRT-LLM release-container quantizer:

```bash
/app/tensorrt_llm/examples/quantization/quantize.py
```

New scripts:

```text
scripts/quantize_trtllm_nvfp4.sh
scripts/quantize_selected_original_nvfp4.sh
scripts/quantize_selected_folded_nvfp4.sh
scripts/run_small_nvfp4_quantization_pair.sh
scripts/run_selected_nvfp4_pair_benchmark.sh
```

Controlled small-model flow:

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

This creates:

```text
/workspace/models/<MODEL_TAG>-NVFP4
/workspace/models/<MODEL_TAG>-FOLDED-NVFP4
```

For a faster smoke test:

```bash
CALIB_SIZE=16 CALIB_MAX_SEQ_LENGTH=256 bash scripts/run_small_nvfp4_quantization_pair.sh
```

After quantization, benchmark both selected small NVFP4 targets:

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

This is the correct next stage before moving to the 480B model: compare **small original NVFP4** vs **small folded NVFP4** first.

### Engine serving note

For locally quantized TensorRT-LLM NVFP4 checkpoints, build and validate engines before serving:

```bash
bash scripts/build_selected_nvfp4_pair_engines.sh
bash scripts/validate_selected_engine_dirs.sh
bash scripts/run_selected_nvfp4_pair_engine_benchmark.sh
```

The engine serving path forces `--backend tensorrt` and validates that `.engine`/`.plan` files exist before starting the server.

## Engine serving note

For built TensorRT-LLM engine directories, `scripts/serve_engine_target.sh` intentionally does **not** pass `--backend` by default. In recent TensorRT-LLM containers, the engine path should use the default C++/engine path. Passing `--backend pytorch` forces checkpoint loading, and passing `--backend tensorrt` can still fail in some builds with `assert os.path.isfile(weights_path)`. Leave `TRTLLM_ENGINE_BACKEND` unset unless debugging.


## v13 note: engine config must not be overwritten

If `trtllm-serve` reports `No weight files found in /workspace/engines/...-engine-tp*`, rebuild the engines with v13 scripts:

```bash
export CLEAN_ENGINE_DIR=1
export TRTLLM_ENGINE_BACKEND=tensorrt
bash scripts/build_selected_nvfp4_pair_engines.sh
bash scripts/validate_selected_engine_dirs.sh
bash scripts/run_selected_nvfp4_pair_engine_benchmark.sh
```

Older scripts copied the checkpoint `config.json` over the engine `config.json`. v13 preserves the engine config and stores the checkpoint config as `checkpoint_config.json`.

### Note: engine server works but benchmark says Unknown TARGET

If the log shows `Server health check passed` and then `Unknown TARGET=selected_original_nvfp4_engine`, the TensorRT-LLM engine launched correctly. The failure is only in the benchmark wrapper target mapping. Use repo v14 or newer, where `selected_original_nvfp4_engine` and `selected_folded_nvfp4_engine` are supported benchmark targets.

## v15 selected-model consistency notes

For locally quantized TensorRT-LLM NVFP4 outputs, use the engine path:

```bash
bash scripts/run_small_nvfp4_quantization_pair.sh
bash scripts/build_selected_nvfp4_pair_engines.sh
bash scripts/run_selected_nvfp4_pair_engine_benchmark.sh
```

Before switching to a different model, run:

```bash
bash scripts/audit_selected_pipeline.sh
```

The scripts now keep `TP_SIZE` consistent across quantization, engine build, validation, and benchmark serving. If a quantized checkpoint already exists, its `config.json -> mapping.tp_size` is treated as the source of truth for engine naming and benchmark execution.

## Folded NVFP4 dual-plugin path

After creating the folded BF16 checkpoint, the current safe NVFP4 path is:

```bash
export MODEL_PROFILE=qwen25_05b
export TP_SIZE=2
export CALIB_SIZE=16
export CALIB_MAX_SEQ_LENGTH=256
export CALIB_BATCH_SIZE=1
bash scripts/run_folded_nvfp4_dual_plugin_fast.sh
```

This uses `TRTLLM_QWEN_RMS_SCALE_SWIGLU_PLUGIN_MODE=bf16_intermediate`: the RMS-scale-SwiGLU plugin returns BF16/FP16 intermediate, and TensorRT-LLM's existing NVFP4 projection linear quantizes that intermediate internally.

See `docs/nvfp4_contract_and_next_steps.md` for the explicit FP4-output plan and why that mode requires replacing the projection with a prequant NVFP4 GEMM.
