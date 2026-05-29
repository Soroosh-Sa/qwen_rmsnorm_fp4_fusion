import torch
import qwen_rms_scale_swiglu_cuda

torch.manual_seed(0)

device = "cuda"
dtype = torch.bfloat16

num_tokens = 128
hidden_size = 896
inter_size = 2432
eps = 1e-6

hidden = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
raw_gate = torch.randn(num_tokens, inter_size, device=device, dtype=dtype)
raw_inter = torch.randn(num_tokens, inter_size, device=device, dtype=dtype)
raw_gate_up = torch.cat([raw_gate, raw_inter], dim=-1)

# FusedGatedMLP path
out_fused = qwen_rms_scale_swiglu_cuda.qwen_rms_scale_swiglu_bf16(
    hidden,
    raw_gate_up,
    eps,
)

# Plain GatedMLP path
out_gated = qwen_rms_scale_swiglu_cuda.qwen_rms_scale_swiglu_gated_bf16(
    hidden,
    raw_gate,
    raw_inter,
    eps,
)

hidden_fp32 = hidden.float()
rstd = torch.rsqrt(torch.mean(hidden_fp32 * hidden_fp32, dim=-1, keepdim=True) + eps)
ref = torch.nn.functional.silu(raw_inter.float() * rstd) * (raw_gate.float() * rstd)
ref = ref.to(dtype)

for name, out in [("fused", out_fused), ("gated", out_gated)]:
    max_err = (out.float() - ref.float()).abs().max().item()
    mean_err = (out.float() - ref.float()).abs().mean().item()
    print(name, "max_err:", max_err)
    print(name, "mean_err:", mean_err)
    assert max_err < 0.06, (name, max_err)
    assert mean_err < 0.006, (name, mean_err)

fused_gated_err = (out_fused.float() - out_gated.float()).abs().max().item()
print("fused_vs_gated_max_err:", fused_gated_err)
assert fused_gated_err == 0.0, fused_gated_err

print("PASS")
