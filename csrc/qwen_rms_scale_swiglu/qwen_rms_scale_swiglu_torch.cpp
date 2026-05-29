#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "kernel/qwenRmsScaleSwigluKernel.h"

torch::Tensor qwen_rms_scale_swiglu_bf16(
    torch::Tensor hiddenStates,
    torch::Tensor rawGateUp,
    double eps)
{
    TORCH_CHECK(hiddenStates.is_cuda(), "hiddenStates must be CUDA");
    TORCH_CHECK(rawGateUp.is_cuda(), "rawGateUp must be CUDA");

    TORCH_CHECK(hiddenStates.scalar_type() == torch::kBFloat16, "hiddenStates must be BF16");
    TORCH_CHECK(rawGateUp.scalar_type() == torch::kBFloat16, "rawGateUp must be BF16");

    TORCH_CHECK(hiddenStates.dim() == 2, "hiddenStates must be [numTokens, hiddenSize]");
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("qwen_rms_scale_swiglu_bf16", &qwen_rms_scale_swiglu_bf16,
          "Qwen RMS denominator scale + SwiGLU BF16");
}
