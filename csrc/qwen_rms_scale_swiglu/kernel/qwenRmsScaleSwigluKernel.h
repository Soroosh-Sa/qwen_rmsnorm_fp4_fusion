#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace tensorrt_llm
{
namespace kernels
{

struct QwenRmsScaleSwigluParams
{
    void const* hiddenStates;   // [numTokens, hiddenSize]
    void const* rawGateUp;      // [numTokens, 2 * interSize]
    void* output;               // [numTokens, interSize]

    int32_t numTokens;
    int32_t hiddenSize;
    int32_t interSize;

    float eps;
    cudaStream_t stream;
};

void invokeQwenRmsScaleSwigluBf16(QwenRmsScaleSwigluParams const& params);
void invokeQwenRmsScaleSwigluFp16(QwenRmsScaleSwigluParams const& params);

} // namespace kernels
} // namespace tensorrt_llm
