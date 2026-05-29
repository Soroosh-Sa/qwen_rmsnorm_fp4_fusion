# Folded NVFP4 RMS-scale SwiGLU plugin workflow

This repo now includes two TensorRT plugins in one shared library:

1. `QwenRmsScaleSwiglu`
   - For `FusedGatedMLP` / `fused_fc` path.
   - Inputs: `hidden_states`, `raw_gate_up = [raw_gate | raw_inter]`.

2. `QwenRmsScaleSwigluGated`
   - For plain `GatedMLP` path.
   - Inputs: `hidden_states`, `raw_gate`, `raw_inter`.
   - This avoids a TensorRT `concat` before the plugin and is the preferred path when NVFP4 builds keep `fc` and `gate` separate.

Both plugins compute the same folded-weight runtime expression:

```text
rstd = rsqrt(mean(hidden_states^2) + eps)
gate = raw_gate * rstd
inter = raw_inter * rstd
out = silu(inter) * gate
```

The RMSNorm gamma must already be folded into the MLP gate/up weights before quantization.

## Correct order

```text
original BF16/FP16 HF weights
→ fold RMSNorm gamma into MLP gate/up weights
→ save folded BF16 checkpoint
→ quantize folded BF16 checkpoint to folded NVFP4 TensorRT-LLM checkpoint
→ build TensorRT-LLM engine with plugin patch enabled
```

Do not start from an already-NVFP4 checkpoint unless it was produced from the folded weights.

## Fast path

```bash
cd ~/workspace/qwen_rmsnorm_fp4_fusion
export MODEL_PROFILE=qwen25_05b
export TP_SIZE=2
export CALIB_SIZE=16
export CALIB_MAX_SEQ_LENGTH=256
export CALIB_BATCH_SIZE=1
bash scripts/run_folded_nvfp4_plugin_pipeline.sh
```

## Step-by-step

```bash
bash scripts/quantize_selected_folded_nvfp4.sh
bash scripts/build_qwen_rms_scale_swiglu_plugin.sh
bash scripts/build_selected_folded_nvfp4_plugin_engine.sh
```

## Validation

Check plugin registration:

```bash
bash scripts/build_qwen_rms_scale_swiglu_plugin.sh
```

The script should print both creators:

```text
QwenRmsScaleSwiglu
QwenRmsScaleSwigluGated
```

Check the engine:

```bash
timeout 30s strings "$FOLDED_NVFP4_PLUGIN_ENGINE/rank0.engine" | grep -i "QwenRmsScaleSwiglu" | head
```

For Nsight, the kernel names should include either:

```text
qwenRmsScaleSwigluFusedKernel
qwenRmsScaleSwigluGatedKernel
```
