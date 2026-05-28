# Fusion modes and final comparison plan

## What the current folding scripts do

The checkpoint folding scripts perform **compatibility folding**:

```text
W_fused = W * gamma[None, :]
RMSNorm gamma = 1
```

This is useful because it lets the original Qwen model code run without applying
`gamma` twice. It is mainly for correctness, checkpoint preparation, and
quantization experiments.

However, this mode usually **does not remove the RMSNorm kernel**. The generic
runtime may still compute:

```text
x_rms = x / rms(x)
x_rms = x_rms * 1
linear(x_rms, W_fused)
```

So it is not the final optimized computation.

## Actual fused-computation target

The real optimization should replace `RMSNorm -> Linear` with:

```text
inv_rms = rsqrt(mean(x*x) + eps)
y_raw = x @ W_fused.T
y = y_raw * inv_rms
```

This avoids materializing the normalized activation and can remove global-memory
read/write traffic between RMSNorm and GEMM. This likely requires a TensorRT-LLM
graph rewrite, plugin, or custom kernel. The file `src/fused_runtime_notes.py`
contains reference math for this path.

## Requested final comparison

Do not use original BF16 as the main runtime baseline because it may be too slow.
Use this comparison instead:

| Target | Folded? | Quantization | True FP4? | Runtime |
|---|---:|---|---:|---|
| Pre-quantized NVFP4 | No | NVFP4 | Yes | TensorRT-LLM |
| Folded GPTQ INT4 | Yes | GPTQ INT4 | No | GPTQ-compatible runtime |
| Folded TensorRT-LLM NVFP4 | Yes | NVFP4 | Yes | TensorRT-LLM |

Notes:

- GPTQ INT4 is a useful 4-bit integer baseline, but it is not FP4.
- NVFP4 is the main Blackwell FP4 target.
- The pre-quantized NVFP4 checkpoint is the practical baseline replacing original BF16.
