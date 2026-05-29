#include "qwenRmsScaleSwigluPlugin.h"
#include "kernel/qwenRmsScaleSwigluKernel.h"

#include <cstring>
#include <iostream>
#include <string>

using namespace nvinfer1;

namespace tensorrt_llm
{
namespace plugins
{

namespace
{

char const* PLUGIN_NAME = "QwenRmsScaleSwiglu";
char const* PLUGIN_VERSION = "1";

template <typename T>
void writeToBuffer(char*& buffer, T const& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

template <typename T>
T readFromBuffer(char const*& buffer)
{
    T val = *reinterpret_cast<T const*>(buffer);
    buffer += sizeof(T);
    return val;
}

bool isSupportedActivationType(DataType type)
{
    return type == DataType::kBF16 || type == DataType::kHALF;
}

} // namespace

QwenRmsScaleSwigluPlugin::QwenRmsScaleSwigluPlugin(int hiddenSize, int interSize, float eps)
    : mHiddenSize(hiddenSize)
    , mInterSize(interSize)
    , mEps(eps)
{
}

QwenRmsScaleSwigluPlugin::QwenRmsScaleSwigluPlugin(void const* data, size_t length)
{
    char const* d = reinterpret_cast<char const*>(data);
    mHiddenSize = readFromBuffer<int>(d);
    mInterSize = readFromBuffer<int>(d);
    mEps = readFromBuffer<float>(d);
}

char const* QwenRmsScaleSwigluPlugin::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}

char const* QwenRmsScaleSwigluPlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

int QwenRmsScaleSwigluPlugin::getNbOutputs() const noexcept
{
    return 1;
}

DimsExprs QwenRmsScaleSwigluPlugin::getOutputDimensions(
    int outputIndex,
    DimsExprs const* inputs,
    int nbInputs,
    IExprBuilder& exprBuilder) noexcept
{
    DimsExprs out;
    out.nbDims = 2;
    out.d[0] = inputs[0].d[0];
    out.d[1] = exprBuilder.constant(mInterSize);
    return out;
}

bool QwenRmsScaleSwigluPlugin::supportsFormatCombination(
    int pos,
    PluginTensorDesc const* inOut,
    int nbInputs,
    int nbOutputs) noexcept
{
    if (nbInputs != 2 || nbOutputs != 1)
    {
        return false;
    }

    PluginTensorDesc const& desc = inOut[pos];
    if (desc.format != TensorFormat::kLINEAR)
    {
        return false;
    }

    // inputs[0]: hidden states, inputs[1]: fused raw gate/up, output[0]: intermediate.
    // All three must use the same activation dtype. NVFP4 checkpoints usually still
    // expose BF16/FP16 activation tensors here; weights are quantized inside GEMMs.
    if (pos == 0)
    {
        return isSupportedActivationType(desc.type);
    }

    return desc.type == inOut[0].type;
}

void QwenRmsScaleSwigluPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in,
    int nbInputs,
    DynamicPluginTensorDesc const* out,
    int nbOutputs) noexcept
{
}

size_t QwenRmsScaleSwigluPlugin::getWorkspaceSize(
    PluginTensorDesc const* inputs,
    int nbInputs,
    PluginTensorDesc const* outputs,
    int nbOutputs) const noexcept
{
    return 0;
}

int QwenRmsScaleSwigluPlugin::enqueue(
    PluginTensorDesc const* inputDesc,
    PluginTensorDesc const* outputDesc,
    void const* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept
{
    if (inputDesc[0].dims.nbDims < 2)
    {
        return 1;
    }

    int numTokens = inputDesc[0].dims.d[0];

    tensorrt_llm::kernels::QwenRmsScaleSwigluParams params;
    params.hiddenStates = inputs[0];
    params.rawGateUp = inputs[1];
    params.output = outputs[0];
    params.numTokens = numTokens;
    params.hiddenSize = mHiddenSize;
    params.interSize = mInterSize;
    params.eps = mEps;
    params.stream = stream;

    if (inputDesc[0].type == DataType::kBF16)
    {
        tensorrt_llm::kernels::invokeQwenRmsScaleSwigluBf16(params);
    }
    else if (inputDesc[0].type == DataType::kHALF)
    {
        tensorrt_llm::kernels::invokeQwenRmsScaleSwigluFp16(params);
    }
    else
    {
        return 2;
    }

    return static_cast<int>(cudaPeekAtLastError());
}

size_t QwenRmsScaleSwigluPlugin::getSerializationSize() const noexcept
{
    return sizeof(int) + sizeof(int) + sizeof(float);
}

void QwenRmsScaleSwigluPlugin::serialize(void* buffer) const noexcept
{
    char* d = reinterpret_cast<char*>(buffer);
    writeToBuffer<int>(d, mHiddenSize);
    writeToBuffer<int>(d, mInterSize);
    writeToBuffer<float>(d, mEps);
}

void QwenRmsScaleSwigluPlugin::destroy() noexcept
{
    delete this;
}

IPluginV2DynamicExt* QwenRmsScaleSwigluPlugin::clone() const noexcept
{
    auto* plugin = new QwenRmsScaleSwigluPlugin(mHiddenSize, mInterSize, mEps);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

void QwenRmsScaleSwigluPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace ? pluginNamespace : "";
}

char const* QwenRmsScaleSwigluPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

DataType QwenRmsScaleSwigluPlugin::getOutputDataType(
    int index,
    DataType const* inputTypes,
    int nbInputs) const noexcept
{
    return inputTypes[0];
}

int QwenRmsScaleSwigluPlugin::initialize() noexcept
{
    return 0;
}

void QwenRmsScaleSwigluPlugin::terminate() noexcept
{
}

void QwenRmsScaleSwigluPlugin::attachToContext(
    cudnnContext* cudnnContext,
    cublasContext* cublasContext,
    IGpuAllocator* gpuAllocator) noexcept
{
}

void QwenRmsScaleSwigluPlugin::detachFromContext() noexcept
{
}

QwenRmsScaleSwigluPluginCreator::QwenRmsScaleSwigluPluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("hidden_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("inter_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("eps", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = static_cast<int>(mPluginAttributes.size());
    mFC.fields = mPluginAttributes.data();
}

char const* QwenRmsScaleSwigluPluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

char const* QwenRmsScaleSwigluPluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

PluginFieldCollection const* QwenRmsScaleSwigluPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* QwenRmsScaleSwigluPluginCreator::createPlugin(
    char const* name,
    PluginFieldCollection const* fc) noexcept
{
    int hiddenSize = 0;
    int interSize = 0;
    float eps = 1e-6f;

    for (int i = 0; i < fc->nbFields; ++i)
    {
        std::string fieldName(fc->fields[i].name);
        if (fieldName == "hidden_size")
        {
            hiddenSize = *static_cast<int const*>(fc->fields[i].data);
        }
        else if (fieldName == "inter_size")
        {
            interSize = *static_cast<int const*>(fc->fields[i].data);
        }
        else if (fieldName == "eps")
        {
            eps = *static_cast<float const*>(fc->fields[i].data);
        }
    }

    return new QwenRmsScaleSwigluPlugin(hiddenSize, interSize, eps);
}

IPluginV2* QwenRmsScaleSwigluPluginCreator::deserializePlugin(
    char const* name,
    void const* serialData,
    size_t serialLength) noexcept
{
    return new QwenRmsScaleSwigluPlugin(serialData, serialLength);
}

void QwenRmsScaleSwigluPluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace ? pluginNamespace : "";
}

char const* QwenRmsScaleSwigluPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

} // namespace plugins
} // namespace tensorrt_llm

// TensorRT's registration macro concatenates the class token into a static
// variable name, so it cannot receive a namespace-qualified type directly.
using QwenRmsScaleSwigluPluginCreatorAlias =
    tensorrt_llm::plugins::QwenRmsScaleSwigluPluginCreator;

REGISTER_TENSORRT_PLUGIN(QwenRmsScaleSwigluPluginCreatorAlias);
