import time
import torch
import qwen_rms_scale_swiglu_cuda

torch.manual_seed(0)

device = "cuda"
dtype = torch.bfloat16

hidden_size = 896
inter_size = 2432
eps = 1e-6
iters = 1000

for num_tokens in [1, 8, 32, 128, 512, 1024]:
    hidden = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    raw_gate_up = torch.randn(num_tokens, 2 * inter_size, device=device, dtype=dtype)

    def ref_fn():
        hidden_fp32 = hidden.float()
        raw_fp32 = raw_gate_up.float()
        rstd = torch.rsqrt(torch.mean(hidden_fp32 * hidden_fp32, dim=-1, keepdim=True) + eps)
        gate = raw_fp32[:, :inter_size] * rstd
        up = raw_fp32[:, inter_size:] * rstd
        return (torch.nn.functional.silu(up) * gate).to(dtype)

    def custom_fn():
        return qwen_rms_scale_swiglu_cuda.qwen_rms_scale_swiglu_bf16(
            hidden,
            raw_gate_up,
            eps,
        )

    for _ in range(50):
        ref_fn()
        custom_fn()
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        ref_fn()
    torch.cuda.synchronize()
    ref_ms = (time.time() - t0) * 1000.0 / iters

    t0 = time.time()
    for _ in range(iters):
        custom_fn()
    torch.cuda.synchronize()
    custom_ms = (time.time() - t0) * 1000.0 / iters

    print(
        f"num_tokens={num_tokens:4d} "
        f"ref_ms={ref_ms:.6f} "
        f"custom_ms={custom_ms:.6f} "
        f"speedup={ref_ms / custom_ms:.3f}x"
    )
