# Folded RMS-scale SwiGLU + NVFP4 contract

This note records the current understanding of TensorRT-LLM NVFP4 in this repo and defines the safe path for the folded RMSNorm experiment.

## What the checkpoint looks like

After folding BF16 weights and quantizing the folded checkpoint to NVFP4, the TensorRT-LLM checkpoint contains separate MLP projections:

- `transformer.layers.*.mlp.fc.weight`: packed `uint8` FP4, shape `[local_intermediate, hidden_size / 2]`
- `transformer.layers.*.mlp.gate.weight`: packed `uint8` FP4, shape `[local_intermediate, hidden_size / 2]`
- `transformer.layers.*.mlp.proj.weight`: packed `uint8` FP4, shape `[hidden_size, local_intermediate / 2]`
- each linear has:
  - `activation_scaling_factor`: scalar `float32`
  - `weights_scaling_factor`: per-block scale, often `float8_e4m3fn`
  - `weights_scaling_factor_2`: scalar `float32`

For the 0.5B TP=2 inspection, the model is dense Qwen2, not MoE, with `quant_algo=NVFP4` and `dtype=bfloat16`.

## What TensorRT-LLM NVFP4 linear does

TensorRT-LLM's Python AutoDeploy reference path shows that `torch_quant_nvfp4_linear(input, weight_fp4, input_scale, weight_scale, alpha)` receives **unquantized BF16/FP16 input**, internally calls `torch.ops.trtllm.fp4_quantize`, and then calls `torch.ops.trtllm.nvfp4_gemm`.

So for the legacy TensorRT-LLM Qwen engine path used by this repo, it is valid for our RMS-scale-SwiGLU plugin to output BF16/FP16 intermediate. The following `mlp.proj` layer can still quantize that intermediate inside the NVFP4 linear/GEMM path.

## Current safe plugin mode

The implemented safe mode is:

```text
hidden_states BF16/FP16
raw_gate/raw_inter BF16/FP16 from NVFP4 GEMMs
    -> QwenRmsScaleSwigluGated plugin
intermediate BF16/FP16
    -> existing NVFP4 proj linear quantizes internally
output BF16/FP16
```

This is controlled by:

```bash
export TRTLLM_QWEN_FOLDED_RMSNORM_MLP_FUSION=1
export TRTLLM_QWEN_RMS_SCALE_SWIGLU_PLUGIN=1
export TRTLLM_QWEN_RMS_SCALE_SWIGLU_PLUGIN_MODE=bf16_intermediate
export TRTLLM_QWEN_FOLDED_RMSNORM_MLP_FUSION_ALLOW_QUANTIZED=1
```

## Why explicit FP4-output plugin is not added yet

An explicit FP4-output plugin would need to output:

```text
intermediate_fp4: uint8 packed FP4
intermediate_sf: uint8/FP8 scale-factor tensor in TensorRT-LLM/CUTLASS layout
```

and then replace the following `mlp.proj(intermediate)` with a prequantized NVFP4 GEMM equivalent to:

```python
torch.ops.trtllm.nvfp4_gemm(intermediate_fp4, proj_weight_fp4, intermediate_sf, proj_weight_scale, proj_alpha, out_dtype)
```

That means the patch must also take ownership of the `proj` GEMM or call a TensorRT plugin that consumes prequantized activations. Merely changing the SwiGLU plugin output to FP4 is not enough.

## Native files still needed for a true explicit FP4 plugin

Before implementing the true explicit FP4-output plugin, keep these native TensorRT-LLM files in the repo or upload them:

```text
third_party/tensorrt_llm_native/kernels/fusedGatedRMSNormQuant.cu
third_party/tensorrt_llm_native/kernels/fusedGatedRMSNormQuant.cuh
```

Those files define NVIDIA's native fused gated RMSNorm + NVFP4 quantization kernel style. The current repo does not include them in the patched zip.

## Recommended step order

1. Verify folded BF16 exists.
2. Quantize folded BF16 to folded NVFP4.
3. Inspect the folded NVFP4 checkpoint contract.
4. Build the dual plugin.
5. Build the folded NVFP4 engine using `bf16_intermediate` plugin mode.
6. Run one generation sanity check.
7. Only after this passes, implement explicit FP4-output + prequant-proj fusion.
