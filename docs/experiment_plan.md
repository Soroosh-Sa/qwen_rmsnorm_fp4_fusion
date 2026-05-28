# Experiment Plan

## Goal

Validate RMSNorm gamma folding before FP4/NVFP4 quantization:

```text
RMSNorm(x, gamma) -> Linear(W)
```

becomes:

```text
RMSOnly(x) -> Linear(W * gamma)
```

No retraining is required.

## Correctness checks

### Layer-level check

`src/verify_folding.py` creates random hidden states and compares:

```text
Linear(RMSNorm(x, gamma), W)
```

against:

```text
Linear(RMSOnly(x), W * gamma)
```

Expected error: very small, mostly dtype-level noise.

### Model-level check

`src/verify_model_logits.py` compares logits from:

```text
original model
folded model with RMSNorm weights set to 1.0
```

Expected error: small but not necessarily exactly zero because model loading, dtype, device placement, and kernels may introduce tiny differences.

## Which weights are folded?

For each decoder layer:

```text
input_layernorm.weight -> self_attn.q_proj.weight
input_layernorm.weight -> self_attn.k_proj.weight
input_layernorm.weight -> self_attn.v_proj.weight

post_attention_layernorm.weight -> mlp.gate_proj.weight
post_attention_layernorm.weight -> mlp.up_proj.weight
```

Usually not folded:

```text
self_attn.o_proj
mlp.down_proj
```

because they do not directly consume the RMS-normalized hidden state.

## After folding

The corresponding RMSNorm weights are set to all ones so standard Qwen code does not apply gamma twice.

## Scaling to 460B/480B

For very large models, do not load the entire BF16 checkpoint into a single process unless you have enough GPU/CPU memory. A safer future path is shard-by-shard safetensors rewriting:

```text
load one shard
load needed gamma tensors
fold matching Linear tensors
save rewritten shard
free memory
repeat
```

That is not implemented in this first small-model scaffold, but the metadata and naming conventions from this test will make it easier to add later.
