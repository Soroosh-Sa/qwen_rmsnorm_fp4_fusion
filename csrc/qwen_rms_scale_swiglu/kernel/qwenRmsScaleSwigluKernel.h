#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace tensorrt_llm
{
namespace kernels
{

// FusedGatedMLP path:
//   inputs: hiddenStates, rawGateUp = [raw_gate | raw_inter]
//   output: silu(raw_inter * rstd) * (raw_gate * rstd)
struct QwenRmsScaleSwigluParams
{
    void const* hiddenStates;   // [numTokens, hiddenSize]
    void const* rawGateUp;      // [numTokens, 2 * interSize], layout [gate | inter]
    void* output;               // [numTokens, interSize]

    int32_t numTokens;
    int32_t hiddenSize;
    int32_t interSize;

    float eps;
    cudaStream_t stream;
};

// Plain GatedMLP path:
//   inputs: hiddenStates, rawGate, rawInter
//   output: silu(rawInter * rstd) * (rawGate * rstd)
// This avoids materializing concat([rawGate, rawInter]) in TensorRT.
struct QwenRmsScaleSwigluGatedParams
{
    void const* hiddenStates;   // [numTokens, hiddenSize]
    void const* rawGate;        // [numTokens, interSize]
    void const* rawInter;       // [numTokens, interSize]
    void* output;               // [numTokens, interSize]

    int32_t numTokens;
    int32_t hiddenSize;
    int32_t interSize;

    float eps;
    cudaStream_t stream;
};

void invokeQwenRmsScaleSwigluBf16(QwenRmsScaleSwigluParams const& params);
void invokeQwenRmsScaleSwigluFp16(QwenRmsScaleSwigluParams const& params);

void invokeQwenRmsScaleSwigluGatedBf16(QwenRmsScaleSwigluGatedParams const& params);
void invokeQwenRmsScaleSwigluGatedFp16(QwenRmsScaleSwigluGatedParams const& params);

} // namespace kernels
} // namespace tensorrt_llm
