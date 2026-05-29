# NVFP4 Understanding and Step-by-Step Implementation Plan

This document is intentionally conservative. It records what we know from the
local TensorRT-LLM installation and what we still need to inspect before adding a
true FP4-output plugin.

## Current repo status

The repo already has a working TensorRT plugin path for folded-weight RMS-scale
SwiGLU:

- `QwenRmsScaleSwiglu` for `FusedGatedMLP` / `fused_fc`.
- `QwenRmsScaleSwigluGated` for plain `GatedMLP` / `fc + gate`.

These plugins take BF16/FP16 activations, compute the RMS denominator from the
un-normalized hidden state, scale gate/up outputs, apply SwiGLU, and output
BF16/FP16 intermediate activations.

This is useful for NVFP4-weight engines **only if** TensorRT-LLM keeps the
activation interface around this point in BF16/FP16. It is not yet a true
FP4-output plugin.

## What the local grep output tells us

The local TensorRT-LLM package contains two relevant NVFP4 paths:

1. `_torch/auto_deploy/custom_ops/linear/swiglu.py`
   - contains `torch_nvfp4_swiglu_mlp`
   - contains `fused_nvfp4_swiglu_mlp`
   - comments describe a fused NVFP4 SwiGLU MLP with packed FP4 gate/up/down
     weights.

2. `_torch/auto_deploy/custom_ops/fused_moe/trtllm_moe.py`
   - uses `torch.ops.trtllm.fp4_quantize` before FP4 MoE runner calls.
   - comments say FP4 pairs are packed in `uint8` and groups of 16 FP4 weight
     elements can be viewed as `int64` for the MoE runner.
   - importantly, the MoE fused path can output in the model dtype
     (`bf16`/`fp16`), not necessarily FP4.

Therefore, before writing an FP4-output plugin, we need to know which path the
classic TensorRT-LLM engine builder uses for our target model:

- Classic `models/qwen/model.py` path with `GatedMLP` / `FusedGatedMLP`.
- AutoDeploy NVFP4 `fused_nvfp4_swiglu_mlp` path.
- MoE NVFP4 path using `fp4_quantize` + `fp4_block_scale_moe_runner`.

## Correct folded NVFP4 workflow

Do not quantize the original checkpoint directly for our method. The correct
workflow is:

1. Start from original BF16/FP16 checkpoint.
2. Fold RMSNorm weight into gate/up/fused_fc weights.
3. Save folded BF16 checkpoint.
4. Quantize the folded BF16 checkpoint to NVFP4.
5. Build the TensorRT-LLM engine with plugin mode enabled.
6. Confirm whether the plugin is inserted and whether TensorRT uses BF16/FP16
   or a quantized NVFP4 interface around the MLP.

## Stage 0: collect NVFP4 references from the target machine

Run:

```bash
cd ~/workspace/qwen_rmsnorm_fp4_fusion
bash scripts/collect_nvfp4_references.sh
```

This creates:

```text
runtime_logs/nvfp4_refs/
```

Please share that folder or at least these files if further patching is needed:

```text
runtime_logs/nvfp4_refs/swiglu.py
runtime_logs/nvfp4_refs/quant.py
runtime_logs/nvfp4_refs/torch_quant.py
runtime_logs/nvfp4_refs/trtllm_moe.py
runtime_logs/nvfp4_refs/default.yaml
runtime_logs/nvfp4_refs/fusedGatedRMSNormQuant.cu
runtime_logs/nvfp4_refs/fusedGatedRMSNormQuant.cuh
runtime_logs/nvfp4_refs/summary.txt
```

## Stage 1: create folded BF16 checkpoint

For the 0.5B smoke model, if it does not already exist:

```bash
export MODEL_PROFILE=qwen25_05b
export TP_SIZE=2
bash scripts/fold_selected_model_sharded.sh
```

Expected output:

```text
/workspace/models/Qwen-Qwen2.5-0.5B-Instruct-FOLDED-BF16
```

## Stage 2: quantize folded BF16 to folded NVFP4

```bash
export MODEL_PROFILE=qwen25_05b
export TP_SIZE=2
export CALIB_SIZE=16
export CALIB_MAX_SEQ_LENGTH=256
export CALIB_BATCH_SIZE=1
bash scripts/quantize_selected_folded_nvfp4.sh
```

Expected output:

```text
/workspace/models/Qwen-Qwen2.5-0.5B-Instruct-FOLDED-NVFP4
```

## Stage 3: inspect folded NVFP4 checkpoint

```bash
python src/inspect_nvfp4_checkpoint.py \
  --checkpoint /workspace/models/Qwen-Qwen2.5-0.5B-Instruct-FOLDED-NVFP4 \
  --max-keys 300 \
  --output runtime_logs/folded_nvfp4_checkpoint_inspection.txt
```

This tells us whether the checkpoint has packed FP4 weights, scale tensors,
alpha/global-scale tensors, and whether gate/up/down weights are separate or
fused.

## Stage 4: build current dual plugin and try folded NVFP4 engine

```bash
bash scripts/build_qwen_rms_scale_swiglu_plugin.sh
bash scripts/build_selected_folded_nvfp4_plugin_engine.sh
```

If this succeeds, run:

```bash
timeout 20s strings "$FOLDED_NVFP4_PLUGIN_ENGINE/rank0.engine" \
  | grep -Ei "QwenRmsScaleSwiglu|QwenRmsScaleSwigluGated" \
  | head
```

This tells us whether the current BF16/FP16 activation plugin is still attachable
in the NVFP4 engine.

## Stage 5: only then implement true NVFP4 plugin

A true NVFP4 plugin should not be added blindly. It likely needs one of these
interfaces:

### Candidate A: BF16/FP16 output plugin

This is the current dual plugin. It lets the next TensorRT-LLM quantized GEMM
perform its own activation quantization internally.

### Candidate B: FP4-quantizing activation plugin

Inputs:

```text
hidden_states: BF16/FP16
raw_gate_up or raw_gate/raw_inter: BF16/FP16
act_global_scale: FP32/FP16 scalar or tensor
```

Outputs:

```text
intermediate_fp4_packed: UINT8, shape [num_tokens, inter_size / 2]
intermediate_sf: INT8/scale tensor, shape determined by TRTLLM_NVFP4_SCALING_VECTOR_SIZE
```

This only works if the following `down_proj` GEMM can consume the packed FP4
activation and scale tensor directly.

### Candidate C: full fused NVFP4 MLP plugin

Inputs:

```text
hidden_states
packed gate/up/down weights
all weight scales/global scales
```

Output:

```text
MLP output in BF16/FP16
```

This is closer to TensorRT-LLM AutoDeploy `fused_nvfp4_swiglu_mlp`, and may be
the correct target for the 460B MoE model.

## Decision rule

Do not implement Candidate B or C until we inspect:

1. Full `swiglu.py` NVFP4 custom op signatures.
2. Full `quant.py` and `torch_quant.py` helpers.
3. Full `trtllm_moe.py` NVFP4 MoE comments and call signatures.
4. The actual folded NVFP4 checkpoint keys/shapes/scales.
5. Whether the target 460B model is dense MLP or MoE in the TensorRT-LLM build path.
