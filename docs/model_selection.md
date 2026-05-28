# Model selection

The repo now has a central model selector based on `MODEL_PROFILE` and `MODEL_ID`.

List profiles:

```bash
bash scripts/list_model_profiles.sh
```

Download a small model:

```bash
MODEL_PROFILE=qwen25_05b bash scripts/download_selected_model.sh
```

Other examples:

```bash
MODEL_PROFILE=qwen25_15b bash scripts/download_selected_model.sh
MODEL_PROFILE=qwen25_7b bash scripts/download_selected_model.sh
MODEL_PROFILE=qwen3_8b bash scripts/download_selected_model.sh
```

Use any custom Qwen model:

```bash
MODEL_PROFILE=custom \
MODEL_ID=Qwen/Qwen2.5-3B-Instruct \
bash scripts/download_selected_model.sh
```

Fold the selected checkpoint shard-by-shard:

```bash
MODEL_PROFILE=qwen3_8b bash scripts/fold_selected_model_sharded.sh
```

By default, paths are generated under `MODEL_ROOT`:

```bash
export MODEL_ROOT=/workspace/models
```

For `MODEL_PROFILE=qwen3_8b`, this creates paths like:

```text
/workspace/models/Qwen-Qwen3-8B
/workspace/models/Qwen-Qwen3-8B-FOLDED-BF16
/workspace/models/Qwen-Qwen3-8B-FOLDED-NVFP4
/workspace/models/Qwen-Qwen3-8B-FOLDED-GPTQ-INT4
```

Serving selected quantized targets:

```bash
MODEL_PROFILE=qwen3_8b \
TARGET=selected_folded_nvfp4 \
TP_SIZE=4 \
bash scripts/run_one_target_server_and_benchmark.sh
```

For GPTQ INT4:

```bash
MODEL_PROFILE=qwen3_8b \
TARGET=selected_folded_gptq_int4 \
TP_SIZE=4 \
bash scripts/run_one_target_server_and_benchmark.sh
```

The old target names still work:

```bash
TARGET=prequant_nvfp4 bash scripts/run_one_target_server_and_benchmark.sh
```
