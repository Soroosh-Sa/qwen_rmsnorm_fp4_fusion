"""
Reference math for the *actual* optimized computation path.

This file is intentionally not a production TensorRT-LLM plugin. It documents and
unit-tests the computation that a TRT-LLM plugin / graph rewrite should implement.

Compatibility-only folding does:
    W_fused = W * gamma[None, :]
    RMSNorm gamma = 1
    run normal RMSNorm + Linear code

That is correct but usually does not reduce much computation.

Actual fused computation should avoid materializing x_norm:
    inv_rms = rsqrt(mean(x*x) + eps)       # [tokens, 1]
    y = (x @ W_fused.T) * inv_rms          # row-scale GEMM output

This works because inv_rms is a scalar per token row.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def rmsnorm_linear_reference(x: torch.Tensor, gamma: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    """Original RMSNorm+Linear reference with no bias."""
    inv_rms = torch.rsqrt(torch.mean(x.float() * x.float(), dim=-1, keepdim=True) + eps).to(x.dtype)
    x_norm = x * inv_rms * gamma.to(x.dtype)
    return x_norm @ weight.t()


@torch.no_grad()
def folded_compatibility_path(x: torch.Tensor, gamma: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    """What the checkpoint-only folded model effectively computes with RMSNorm gamma set to 1."""
    w_fused = weight * gamma.to(weight.dtype).view(1, -1)
    inv_rms = torch.rsqrt(torch.mean(x.float() * x.float(), dim=-1, keepdim=True) + eps).to(x.dtype)
    x_rms_only = x * inv_rms
    return x_rms_only @ w_fused.t()


@torch.no_grad()
def folded_fused_compute_path(x: torch.Tensor, gamma: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    """Target compute path for a real kernel/plugin: no normalized activation materialization."""
    w_fused = weight * gamma.to(weight.dtype).view(1, -1)
    inv_rms = torch.rsqrt(torch.mean(x.float() * x.float(), dim=-1, keepdim=True) + eps).to(x.dtype)
    y_raw = x @ w_fused.t()
    return y_raw * inv_rms


def main():
    torch.manual_seed(0)
    x = torch.randn(7, 128, dtype=torch.float16, device="cuda" if torch.cuda.is_available() else "cpu")
    gamma = torch.randn(128, dtype=torch.float16, device=x.device)
    weight = torch.randn(256, 128, dtype=torch.float16, device=x.device)

    y0 = rmsnorm_linear_reference(x, gamma, weight)
    y1 = folded_compatibility_path(x, gamma, weight)
    y2 = folded_fused_compute_path(x, gamma, weight)

    def report(name, y):
        diff = (y0.float() - y.float()).abs()
        print(f"{name}: max_abs={diff.max().item():.6e}, mean_abs={diff.mean().item():.6e}")

    report("compatibility_path", y1)
    report("fused_compute_path", y2)


if __name__ == "__main__":
    main()
