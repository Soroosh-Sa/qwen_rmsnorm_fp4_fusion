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
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

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
__global__ void qwenRmsScaleSwigluKernel(
    T const* __restrict__ hiddenStates,
    T const* __restrict__ rawGateUp,
    T* __restrict__ output,
    int32_t numTokens,
    int32_t hiddenSize,
    int32_t interSize,
    float eps)
{
    int tokenIdx = blockIdx.x;
    int tid = threadIdx.x;

    if (tokenIdx >= numTokens)
    {
        return;
    }

    T const* hiddenRow = hiddenStates + tokenIdx * hiddenSize;
    T const* gateRow = rawGateUp + tokenIdx * (2 * interSize);
    T const* upRow = gateRow + interSize;
    T* outRow = output + tokenIdx * interSize;

    float localSum = 0.0f;

    for (int i = tid; i < hiddenSize; i += BLOCK_SIZE)
    {
        float v = toFloat<T>(hiddenRow[i]);
        localSum += v * v;
    }

    __shared__ float reduceShared[32];
    float sum = blockReduceSum(localSum, reduceShared);

    __shared__ float sharedRstd;

    if (tid == 0)
    {
        sharedRstd = rsqrtf(sum / static_cast<float>(hiddenSize) + eps);
    }

    __syncthreads();

    float rstd = sharedRstd;

    for (int j = tid; j < interSize; j += BLOCK_SIZE)
    {
        float gate = toFloat<T>(gateRow[j]) * rstd;
        float up = toFloat<T>(upRow[j]) * rstd;
        float y = fastSilu(up) * gate;
        outRow[j] = fromFloat<T>(y);
    }
}

template <typename T>
void invokeTyped(QwenRmsScaleSwigluParams const& params)
{
    constexpr int BLOCK_SIZE = 256;

    auto const* hiddenStates = reinterpret_cast<T const*>(params.hiddenStates);
    auto const* rawGateUp = reinterpret_cast<T const*>(params.rawGateUp);
    auto* output = reinterpret_cast<T*>(params.output);

    dim3 grid(params.numTokens);
    dim3 block(BLOCK_SIZE);

    qwenRmsScaleSwigluKernel<T, BLOCK_SIZE><<<grid, block, 0, params.stream>>>(
        hiddenStates,
        rawGateUp,
        output,
        params.numTokens,
        params.hiddenSize,
        params.interSize,
        params.eps);
}

} // namespace

void invokeQwenRmsScaleSwigluBf16(QwenRmsScaleSwigluParams const& params)
{
    invokeTyped<__nv_bfloat16>(params);
}

void invokeQwenRmsScaleSwigluFp16(QwenRmsScaleSwigluParams const& params)
{
    invokeTyped<half>(params);
}

} // namespace kernels
} // namespace tensorrt_llm
