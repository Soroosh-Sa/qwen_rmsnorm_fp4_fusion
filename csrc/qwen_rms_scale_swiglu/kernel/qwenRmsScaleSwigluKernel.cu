#include "qwenRmsScaleSwigluKernel.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

namespace tensorrt_llm
{
namespace kernels
{

namespace
{

// Keep this intentionally close to TensorRT-LLM native kernel style:
// - one CTA owns one token row
// - FP32 accumulation for RMS denominator
// - warp/block reductions with shuffle instructions
// - no global workspace
// - fused denominator scaling + SiLU + multiply in one kernel

__device__ __forceinline__ float warpReduceSum(float val)
{
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
    {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

__device__ __forceinline__ float blockReduceSum(float val, float* shared)
{
    int const lane = threadIdx.x & 31;
    int const wid = threadIdx.x >> 5;

    val = warpReduceSum(val);

    if (lane == 0)
    {
        shared[wid] = val;
    }

    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0.0f;

    if (wid == 0)
    {
        val = warpReduceSum(val);
    }

    return val;
}

__device__ __forceinline__ float fastSilu(float x)
{
    return x / (1.0f + __expf(-x));
}

template <typename T>
__device__ __forceinline__ float toFloat(T v);

template <>
__device__ __forceinline__ float toFloat<__nv_bfloat16>(__nv_bfloat16 v)
{
    return __bfloat162float(v);
}

template <>
__device__ __forceinline__ float toFloat<half>(half v)
{
    return __half2float(v);
}

template <typename T>
__device__ __forceinline__ T fromFloat(float v);

template <>
__device__ __forceinline__ __nv_bfloat16 fromFloat<__nv_bfloat16>(float v)
{
    return __float2bfloat16(v);
}

template <>
__device__ __forceinline__ half fromFloat<half>(float v)
{
    return __float2half(v);
}

template <typename T, int BLOCK_SIZE>
__device__ __forceinline__ float computeRstd(
    T const* __restrict__ hiddenRow,
    int32_t hiddenSize,
    float eps)
{
    float localSum = 0.0f;

    // Hidden size is usually large and aligned for Qwen. A simple strided loop
    // is robust across BF16/FP16 and avoids alignment assumptions for now.
    for (int i = threadIdx.x; i < hiddenSize; i += BLOCK_SIZE)
    {
        float const v = toFloat<T>(hiddenRow[i]);
        localSum += v * v;
    }

    __shared__ float reduceShared[32];
    float const sum = blockReduceSum(localSum, reduceShared);

    __shared__ float sharedRstd;
    if (threadIdx.x == 0)
    {
        sharedRstd = rsqrtf(sum / static_cast<float>(hiddenSize) + eps);
    }
    __syncthreads();

    return sharedRstd;
}

template <typename T, int BLOCK_SIZE>
__global__ void qwenRmsScaleSwigluFusedKernel(
    T const* __restrict__ hiddenStates,
    T const* __restrict__ rawGateUp,
    T* __restrict__ output,
    int32_t numTokens,
    int32_t hiddenSize,
    int32_t interSize,
    float eps)
{
    int const tokenIdx = blockIdx.x;
    if (tokenIdx >= numTokens)
    {
        return;
    }

    T const* hiddenRow = hiddenStates + tokenIdx * hiddenSize;
    T const* gateRow = rawGateUp + tokenIdx * (2 * interSize);
    T const* interRow = gateRow + interSize;
    T* outRow = output + tokenIdx * interSize;

    float const rstd = computeRstd<T, BLOCK_SIZE>(hiddenRow, hiddenSize, eps);

    for (int j = threadIdx.x; j < interSize; j += BLOCK_SIZE)
    {
        float const gate = toFloat<T>(gateRow[j]) * rstd;
        float const inter = toFloat<T>(interRow[j]) * rstd;
        outRow[j] = fromFloat<T>(fastSilu(inter) * gate);
    }
}

template <typename T, int BLOCK_SIZE>
__global__ void qwenRmsScaleSwigluGatedKernel(
    T const* __restrict__ hiddenStates,
    T const* __restrict__ rawGate,
    T const* __restrict__ rawInter,
    T* __restrict__ output,
    int32_t numTokens,
    int32_t hiddenSize,
    int32_t interSize,
    float eps)
{
    int const tokenIdx = blockIdx.x;
    if (tokenIdx >= numTokens)
    {
        return;
    }

    T const* hiddenRow = hiddenStates + tokenIdx * hiddenSize;
    T const* gateRow = rawGate + tokenIdx * interSize;
    T const* interRow = rawInter + tokenIdx * interSize;
    T* outRow = output + tokenIdx * interSize;

    float const rstd = computeRstd<T, BLOCK_SIZE>(hiddenRow, hiddenSize, eps);

    for (int j = threadIdx.x; j < interSize; j += BLOCK_SIZE)
    {
        float const gate = toFloat<T>(gateRow[j]) * rstd;
        float const inter = toFloat<T>(interRow[j]) * rstd;
        outRow[j] = fromFloat<T>(fastSilu(inter) * gate);
    }
}

template <typename T, int BLOCK_SIZE>
void launchFusedTyped(QwenRmsScaleSwigluParams const& params)
{
    auto const* hiddenStates = reinterpret_cast<T const*>(params.hiddenStates);
    auto const* rawGateUp = reinterpret_cast<T const*>(params.rawGateUp);
    auto* output = reinterpret_cast<T*>(params.output);

    dim3 grid(params.numTokens);
    dim3 block(BLOCK_SIZE);

    qwenRmsScaleSwigluFusedKernel<T, BLOCK_SIZE><<<grid, block, 0, params.stream>>>(
        hiddenStates,
        rawGateUp,
        output,
        params.numTokens,
        params.hiddenSize,
        params.interSize,
        params.eps);
}

template <typename T>
void invokeFusedTyped(QwenRmsScaleSwigluParams const& params)
{
    // Larger CTAs help the big hidden/intermediate sizes used by larger Qwen models.
    if (params.hiddenSize >= 4096 || params.interSize >= 8192)
    {
        launchFusedTyped<T, 512>(params);
    }
    else
    {
        launchFusedTyped<T, 256>(params);
    }
}

template <typename T, int BLOCK_SIZE>
void launchGatedTyped(QwenRmsScaleSwigluGatedParams const& params)
{
    auto const* hiddenStates = reinterpret_cast<T const*>(params.hiddenStates);
    auto const* rawGate = reinterpret_cast<T const*>(params.rawGate);
    auto const* rawInter = reinterpret_cast<T const*>(params.rawInter);
    auto* output = reinterpret_cast<T*>(params.output);

    dim3 grid(params.numTokens);
    dim3 block(BLOCK_SIZE);

    qwenRmsScaleSwigluGatedKernel<T, BLOCK_SIZE><<<grid, block, 0, params.stream>>>(
        hiddenStates,
        rawGate,
        rawInter,
        output,
        params.numTokens,
        params.hiddenSize,
        params.interSize,
        params.eps);
}

template <typename T>
void invokeGatedTyped(QwenRmsScaleSwigluGatedParams const& params)
{
    if (params.hiddenSize >= 4096 || params.interSize >= 8192)
    {
        launchGatedTyped<T, 512>(params);
    }
    else
    {
        launchGatedTyped<T, 256>(params);
    }
}

} // namespace

void invokeQwenRmsScaleSwigluBf16(QwenRmsScaleSwigluParams const& params)
{
    invokeFusedTyped<__nv_bfloat16>(params);
}

void invokeQwenRmsScaleSwigluFp16(QwenRmsScaleSwigluParams const& params)
{
    invokeFusedTyped<half>(params);
}

void invokeQwenRmsScaleSwigluGatedBf16(QwenRmsScaleSwigluGatedParams const& params)
{
    invokeGatedTyped<__nv_bfloat16>(params);
}

void invokeQwenRmsScaleSwigluGatedFp16(QwenRmsScaleSwigluGatedParams const& params)
{
    invokeGatedTyped<half>(params);
}

} // namespace kernels
} // namespace tensorrt_llm
