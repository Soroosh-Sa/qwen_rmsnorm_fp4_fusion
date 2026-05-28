# Tensor parallel size compatibility

TensorRT-LLM requires tensor parallel size to divide the model attention heads.
For Qwen small models, `TP_SIZE=4` may be invalid even on a 4-GPU machine.

Example: Qwen2.5-0.5B has 14 attention heads and 2 KV heads, so valid TP sizes are usually `1` and `2`, not `4`.

Check a selected model:

```bash
export MODEL_PROFILE=qwen25_05b
export MODEL_ROOT=/workspace/models
export TP_SIZE=4
bash scripts/check_selected_tp_compatibility.sh
```

Safe benchmark run:

```bash
export MODEL_PROFILE=qwen25_05b
export TP_SIZE=2   # or leave TP_SIZE=4 with AUTO_ADJUST_TP_SIZE=1
export AUTO_ADJUST_TP_SIZE=1
bash scripts/run_selected_nvfp4_pair_benchmark_safe.sh
```

For the 480B model, TP sizes are different. Always check the model config before serving/quantizing.
