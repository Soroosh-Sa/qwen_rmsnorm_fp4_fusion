#include "qwenRmsScaleSwigluKernel.h"

#include <cuda_bf16.h>
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

__device__ __forceinline__ float blockReduceSum(float val)
{
    static __shared__ float shared[32];

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

template <int BLOCK_SIZE>
__global__ void qwenRmsScaleSwigluBf16Kernel(
    __nv_bfloat16 const* __restrict__ hiddenStates,
    __nv_bfloat16 const* __restrict__ rawGateUp,
    __nv_bfloat16* __restrict__ output,
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

    __nv_bfloat16 const* hiddenRow = hiddenStates + tokenIdx * hiddenSize;
    __nv_bfloat16 const* gateRow = rawGateUp + tokenIdx * (2 * interSize);
    __nv_bfloat16 const* upRow = gateRow + interSize;
    __nv_bfloat16* outRow = output + tokenIdx * interSize;

    float localSum = 0.0f;

    for (int i = tid; i < hiddenSize; i += BLOCK_SIZE)
    {
        float v = __bfloat162float(hiddenRow[i]);
        localSum += v * v;
    }

    float sum = blockReduceSum(localSum);

    __shared__ float sharedRstd;

    if (tid == 0)
    {
        sharedRstd = rsqrtf(sum / static_cast<float>(hiddenSize) + eps);
    }

    __syncthreads();

    float rstd = sharedRstd;

    for (int j = tid; j < interSize; j += BLOCK_SIZE)
    {
        float gate = __bfloat162float(gateRow[j]) * rstd;
        float up = __bfloat162float(upRow[j]) * rstd;

        float y = fastSilu(up) * gate;

        outRow[j] = __float2bfloat16(y);
    }
}

} // namespace

void invokeQwenRmsScaleSwigluBf16(QwenRmsScaleSwigluParams const& params)
{
    constexpr int BLOCK_SIZE = 256;

    auto const* hiddenStates = reinterpret_cast<__nv_bfloat16 const*>(params.hiddenStates);
    auto const* rawGateUp = reinterpret_cast<__nv_bfloat16 const*>(params.rawGateUp);
    auto* output = reinterpret_cast<__nv_bfloat16*>(params.output);

    dim3 grid(params.numTokens);
    dim3 block(BLOCK_SIZE);

    qwenRmsScaleSwigluBf16Kernel<BLOCK_SIZE><<<grid, block, 0, params.stream>>>(
        hiddenStates,
        rawGateUp,
        output,
        params.numTokens,
        params.hiddenSize,
        params.interSize,
        params.eps);
}

} // namespace kernels
} // namespace tensorrt_llm
