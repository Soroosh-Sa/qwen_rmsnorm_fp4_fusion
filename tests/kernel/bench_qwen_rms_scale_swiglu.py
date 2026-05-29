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
    raw_gate = torch.randn(num_tokens, inter_size, device=device, dtype=dtype)
    raw_inter = torch.randn(num_tokens, inter_size, device=device, dtype=dtype)
    raw_gate_up = torch.cat([raw_gate, raw_inter], dim=-1)

    def ref_fn():
        rstd = torch.rsqrt(torch.mean(hidden.float() * hidden.float(), dim=-1, keepdim=True) + eps)
        return (torch.nn.functional.silu(raw_inter.float() * rstd) * (raw_gate.float() * rstd)).to(dtype)

    def fused_fn():
        return qwen_rms_scale_swiglu_cuda.qwen_rms_scale_swiglu_bf16(
            hidden,
            raw_gate_up,
            eps,
        )

    def gated_fn():
        return qwen_rms_scale_swiglu_cuda.qwen_rms_scale_swiglu_gated_bf16(
            hidden,
            raw_gate,
            raw_inter,
            eps,
        )

    for _ in range(50):
        ref_fn()
        fused_fn()
        gated_fn()
    torch.cuda.synchronize()

    times = {}
    for name, fn in [("ref", ref_fn), ("fused", fused_fn), ("gated", gated_fn)]:
        t0 = time.time()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        times[name] = (time.time() - t0) * 1000.0 / iters

    print(
        f"num_tokens={num_tokens:4d} "
        f"ref_ms={times['ref']:.6f} "
        f"fused_ms={times['fused']:.6f} "
        f"gated_ms={times['gated']:.6f} "
        f"fused_speedup={times['ref'] / times['fused']:.3f}x "
        f"gated_speedup={times['ref'] / times['gated']:.3f}x"
    )
