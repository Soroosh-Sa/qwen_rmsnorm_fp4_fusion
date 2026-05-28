# Debugging TensorRT-LLM NVFP4 checkpoints

TensorRT-LLM/ModelOpt NVFP4 export writes rank-sharded checkpoints such as:

```text
rank0.safetensors
rank1.safetensors
config.json
```

These are not Hugging Face-style tensor names. For Qwen, MLP tensors usually look like:

```text
transformer.layers.0.mlp.gate.weight
transformer.layers.0.mlp.fc.weight
transformer.layers.0.mlp.proj.weight
```

## Important TP rule

The quantized output depends on `TP_SIZE`. If a checkpoint was exported with
`TP_SIZE=4`, its MLP gate/fc tensor first dimension for Qwen2.5-0.5B is
`intermediate_size / 4 = 1216`. If you later change config/serving to `TP_SIZE=2`,
TensorRT-LLM expects `intermediate_size / 2 = 2432` rows instead. This mismatch can
trigger weight-loading errors such as `load_weights_fused_gate_up_linear` assertions.

The quantization wrapper now removes the output directory before export by default:

```bash
CLEAN_OUTPUT_DIR=1 bash scripts/run_small_nvfp4_quantization_pair.sh
```

To validate existing outputs:

```bash
TP_SIZE=2 MODEL_PROFILE=qwen25_05b bash scripts/validate_selected_nvfp4_checkpoints.sh
```

To clean stale outputs:

```bash
CONFIRM_CLEAN=1 MODEL_PROFILE=qwen25_05b bash scripts/clean_selected_nvfp4_outputs.sh
```
