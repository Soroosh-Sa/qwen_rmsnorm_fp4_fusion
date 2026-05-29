#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "kernel/qwenRmsScaleSwigluKernel.h"

namespace
{

void checkCommon(torch::Tensor hiddenStates, double eps)
{
    TORCH_CHECK(hiddenStates.is_cuda(), "hiddenStates must be CUDA");
    TORCH_CHECK(hiddenStates.scalar_type() == torch::kBFloat16, "test binding currently expects BF16");
    TORCH_CHECK(hiddenStates.dim() == 2, "hiddenStates must be [numTokens, hiddenSize]");
    TORCH_CHECK(eps > 0.0, "eps must be positive");
}

} // namespace

torch::Tensor qwen_rms_scale_swiglu_bf16(
    torch::Tensor hiddenStates,
    torch::Tensor rawGateUp,
    double eps)
{
    checkCommon(hiddenStates, eps);
    TORCH_CHECK(rawGateUp.is_cuda(), "rawGateUp must be CUDA");
    TORCH_CHECK(rawGateUp.scalar_type() == torch::kBFloat16, "rawGateUp must be BF16");
    TORCH_CHECK(rawGateUp.dim() == 2, "rawGateUp must be [numTokens, 2 * interSize]");
    TORCH_CHECK(hiddenStates.size(0) == rawGateUp.size(0), "numTokens mismatch");
    TORCH_CHECK(rawGateUp.size(1) % 2 == 0, "rawGateUp last dimension must be even");

    int32_t numTokens = static_cast<int32_t>(hiddenStates.size(0));
    int32_t hiddenSize = static_cast<int32_t>(hiddenStates.size(1));
    int32_t interSize = static_cast<int32_t>(rawGateUp.size(1) / 2);

    auto output = torch::empty(
        {numTokens, interSize},
        torch::TensorOptions().dtype(torch::kBFloat16).device(hiddenStates.device()));

    tensorrt_llm::kernels::QwenRmsScaleSwigluParams params;
    params.hiddenStates = hiddenStates.data_ptr();
    params.rawGateUp = rawGateUp.data_ptr();
    params.output = output.data_ptr();
    params.numTokens = numTokens;
    params.hiddenSize = hiddenSize;
    params.interSize = interSize;
    params.eps = static_cast<float>(eps);
    params.stream = at::cuda::getCurrentCUDAStream().stream();

    tensorrt_llm::kernels::invokeQwenRmsScaleSwigluBf16(params);

    return output;
}

torch::Tensor qwen_rms_scale_swiglu_gated_bf16(
    torch::Tensor hiddenStates,
    torch::Tensor rawGate,
    torch::Tensor rawInter,
    double eps)
{
    checkCommon(hiddenStates, eps);
    TORCH_CHECK(rawGate.is_cuda(), "rawGate must be CUDA");
    TORCH_CHECK(rawInter.is_cuda(), "rawInter must be CUDA");
    TORCH_CHECK(rawGate.scalar_type() == torch::kBFloat16, "rawGate must be BF16");
    TORCH_CHECK(rawInter.scalar_type() == torch::kBFloat16, "rawInter must be BF16");
    TORCH_CHECK(rawGate.dim() == 2, "rawGate must be [numTokens, interSize]");
    TORCH_CHECK(rawInter.dim() == 2, "rawInter must be [numTokens, interSize]");
    TORCH_CHECK(hiddenStates.size(0) == rawGate.size(0), "numTokens mismatch for gate");
    TORCH_CHECK(hiddenStates.size(0) == rawInter.size(0), "numTokens mismatch for inter");
    TORCH_CHECK(rawGate.size(1) == rawInter.size(1), "interSize mismatch");

    int32_t numTokens = static_cast<int32_t>(hiddenStates.size(0));
    int32_t hiddenSize = static_cast<int32_t>(hiddenStates.size(1));
    int32_t interSize = static_cast<int32_t>(rawGate.size(1));

    auto output = torch::empty(
        {numTokens, interSize},
        torch::TensorOptions().dtype(torch::kBFloat16).device(hiddenStates.device()));

    tensorrt_llm::kernels::QwenRmsScaleSwigluGatedParams params;
    params.hiddenStates = hiddenStates.data_ptr();
    params.rawGate = rawGate.data_ptr();
    params.rawInter = rawInter.data_ptr();
    params.output = output.data_ptr();
    params.numTokens = numTokens;
    params.hiddenSize = hiddenSize;
    params.interSize = interSize;
    params.eps = static_cast<float>(eps);
    params.stream = at::cuda::getCurrentCUDAStream().stream();

    tensorrt_llm::kernels::invokeQwenRmsScaleSwigluGatedBf16(params);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("qwen_rms_scale_swiglu_bf16", &qwen_rms_scale_swiglu_bf16,
          "Qwen fused_fc RMS denominator scale + SwiGLU BF16");
    m.def("qwen_rms_scale_swiglu_gated_bf16", &qwen_rms_scale_swiglu_gated_bf16,
          "Qwen plain GatedMLP RMS denominator scale + SwiGLU BF16");
}
