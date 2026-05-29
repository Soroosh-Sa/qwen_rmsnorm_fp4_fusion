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
raw_gate_up = torch.randn(num_tokens, 2 * inter_size, device=device, dtype=dtype)

out = qwen_rms_scale_swiglu_cuda.qwen_rms_scale_swiglu_bf16(
    hidden,
    raw_gate_up,
    eps,
)

hidden_fp32 = hidden.float()
raw_fp32 = raw_gate_up.float()

rstd = torch.rsqrt(torch.mean(hidden_fp32 * hidden_fp32, dim=-1, keepdim=True) + eps)

gate = raw_fp32[:, :inter_size] * rstd
up = raw_fp32[:, inter_size:] * rstd

ref = torch.nn.functional.silu(up) * gate
ref = ref.to(dtype)

max_err = (out.float() - ref.float()).abs().max().item()
mean_err = (out.float() - ref.float()).abs().mean().item()

print("max_err:", max_err)
print("mean_err:", mean_err)

assert max_err < 0.06, max_err
assert mean_err < 0.006, mean_err

print("PASS")
