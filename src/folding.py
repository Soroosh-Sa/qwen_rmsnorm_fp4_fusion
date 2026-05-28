from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import torch


@dataclass
class FoldRecord:
    layer_idx: int
    norm_name: str
    linear_name: str
    weight_shape: str
    gamma_shape: str
    weight_dtype: str
    max_abs_weight_before: float
    max_abs_gamma: float
    max_abs_weight_after: float


def _get_module_attr(obj: Any, path: str):
    cur = obj
    for part in path.split("."):
        cur = getattr(cur, part)
    return cur


def _fold_gamma_into_linear(linear: torch.nn.Module, gamma: torch.Tensor) -> Dict[str, Any]:
    if not hasattr(linear, "weight"):
        raise TypeError("Target linear module has no weight")
    weight = linear.weight
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D Linear weight, got shape {tuple(weight.shape)}")
    if gamma.ndim != 1:
        raise ValueError(f"Expected 1D RMSNorm gamma, got shape {tuple(gamma.shape)}")
    if weight.shape[1] != gamma.shape[0]:
        raise ValueError(
            f"Shape mismatch: weight shape {tuple(weight.shape)} but gamma shape {tuple(gamma.shape)}"
        )

    before = weight.detach().abs().max().float().item()
    gamma_max = gamma.detach().abs().max().float().item()
    gamma_cast = gamma.to(device=weight.device, dtype=weight.dtype).view(1, -1)
    weight.data.mul_(gamma_cast)
    after = weight.detach().abs().max().float().item()
    return {
        "max_abs_weight_before": before,
        "max_abs_gamma": gamma_max,
        "max_abs_weight_after": after,
    }


@torch.no_grad()
def fold_qwen_rmsnorms(model: torch.nn.Module, cfg: Dict[str, Any]) -> List[FoldRecord]:
    """Fold Qwen-style RMSNorm gamma into adjacent Linear weights.

    Expected Qwen structure:
      model.model.layers[i].input_layernorm -> self_attn.q_proj/k_proj/v_proj
      model.model.layers[i].post_attention_layernorm -> mlp.gate_proj/up_proj

    After folding, optionally set the corresponding norm weights to 1.0 so standard
    model code becomes RMS-only and does not double-count gamma.
    """
    folding_cfg = cfg.get("folding", {})
    attn_names: Iterable[str] = folding_cfg.get("attention_linears", ["q_proj", "k_proj", "v_proj"])
    mlp_names: Iterable[str] = folding_cfg.get("mlp_linears", ["gate_proj", "up_proj"])
    set_one: bool = bool(folding_cfg.get("set_norm_weights_to_one", True))

    layers = model.model.layers
    records: List[FoldRecord] = []

    for layer_idx, layer in enumerate(layers):
        if hasattr(layer, "input_layernorm") and hasattr(layer, "self_attn"):
            norm = layer.input_layernorm
            gamma = norm.weight.detach().clone()
            for proj_name in attn_names:
                if hasattr(layer.self_attn, proj_name):
                    linear = getattr(layer.self_attn, proj_name)
                    stats = _fold_gamma_into_linear(linear, gamma)
                    records.append(
                        FoldRecord(
                            layer_idx=layer_idx,
                            norm_name="input_layernorm",
                            linear_name=f"self_attn.{proj_name}",
                            weight_shape=str(tuple(linear.weight.shape)),
                            gamma_shape=str(tuple(gamma.shape)),
                            weight_dtype=str(linear.weight.dtype),
                            **stats,
                        )
                    )
            if set_one:
                norm.weight.data.fill_(1.0)

        if hasattr(layer, "post_attention_layernorm") and hasattr(layer, "mlp"):
            norm = layer.post_attention_layernorm
            gamma = norm.weight.detach().clone()
            for proj_name in mlp_names:
                if hasattr(layer.mlp, proj_name):
                    linear = getattr(layer.mlp, proj_name)
                    stats = _fold_gamma_into_linear(linear, gamma)
                    records.append(
                        FoldRecord(
                            layer_idx=layer_idx,
                            norm_name="post_attention_layernorm",
                            linear_name=f"mlp.{proj_name}",
                            weight_shape=str(tuple(linear.weight.shape)),
                            gamma_shape=str(tuple(gamma.shape)),
                            weight_dtype=str(linear.weight.dtype),
                            **stats,
                        )
                    )
            if set_one:
                norm.weight.data.fill_(1.0)

    return records


def rms_only(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


@torch.no_grad()
def direct_rmsnorm_linear_check(
    x: torch.Tensor,
    gamma: torch.Tensor,
    linear: torch.nn.Linear,
    eps: float,
) -> Dict[str, float]:
    """Check algebraic equivalence for one RMSNorm -> Linear pair."""
    gamma_cast = gamma.to(device=x.device, dtype=x.dtype)
    x_norm = rms_only(x, eps) * gamma_cast.view(1, 1, -1)
    y_original = linear(x_norm)

    w = linear.weight.detach()
    b = linear.bias.detach() if linear.bias is not None else None
    w_fused = w * gamma.to(device=w.device, dtype=w.dtype).view(1, -1)
    y_fused = torch.nn.functional.linear(rms_only(x, eps), w_fused, b)

    diff = (y_original - y_fused).detach().float()
    denom = y_original.detach().float().abs().clamp_min(1e-8)
    return {
        "max_abs_error": diff.abs().max().item(),
        "mean_abs_error": diff.abs().mean().item(),
        "max_rel_error": (diff.abs() / denom).max().item(),
        "mean_rel_error": (diff.abs() / denom).mean().item(),
    }
