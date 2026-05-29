#include "qwenRmsScaleSwigluPlugin.h"
#include "kernel/qwenRmsScaleSwigluKernel.h"

#include <cstring>
#include <iostream>
#include <numeric>
#include <string>

using namespace nvinfer1;

namespace tensorrt_llm
{
namespace plugins
{

namespace
{

char const* FUSED_PLUGIN_NAME = "QwenRmsScaleSwiglu";
char const* GATED_PLUGIN_NAME = "QwenRmsScaleSwigluGated";
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

bool isSupportedDataType(DataType type)
{
    // NVFP4-weight TensorRT-LLM GEMMs usually expose BF16/FP16 activation tensors
    // around custom plugins. If TensorRT sends FP8/packed FP4 here, the build should
    // fail loudly instead of silently running wrong code.
    return type == DataType::kBF16 || type == DataType::kHALF;
}

int64_t volume(Dims const& dims)
{
    int64_t v = 1;
    for (int i = 0; i < dims.nbDims; ++i)
    {
        v *= static_cast<int64_t>(dims.d[i]);
    }
    return v;
}

int computeNumTokens(Dims const& hiddenDims, int hiddenSize)
{
    int64_t const total = volume(hiddenDims);
    if (hiddenSize <= 0 || total <= 0)
    {
        return 0;
    }
    return static_cast<int>(total / hiddenSize);
}

void parseFields(PluginFieldCollection const* fc, int& hiddenSize, int& interSize, float& eps)
{
    hiddenSize = 0;
    interSize = 0;
    eps = 1e-6f;

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
}

void initFields(std::vector<PluginField>& attrs, PluginFieldCollection& fc)
{
    attrs.emplace_back(PluginField("hidden_size", nullptr, PluginFieldType::kINT32, 1));
    attrs.emplace_back(PluginField("inter_size", nullptr, PluginFieldType::kINT32, 1));
    attrs.emplace_back(PluginField("eps", nullptr, PluginFieldType::kFLOAT32, 1));

    fc.nbFields = static_cast<int>(attrs.size());
    fc.fields = attrs.data();
}

size_t serializedSize()
{
    return sizeof(int) + sizeof(int) + sizeof(float);
}

void serializeCommon(void* buffer, int hiddenSize, int interSize, float eps)
{
    char* d = reinterpret_cast<char*>(buffer);
    writeToBuffer<int>(d, hiddenSize);
    writeToBuffer<int>(d, interSize);
    writeToBuffer<float>(d, eps);
}

void deserializeCommon(void const* data, int& hiddenSize, int& interSize, float& eps)
{
    char const* d = reinterpret_cast<char const*>(data);
    hiddenSize = readFromBuffer<int>(d);
    interSize = readFromBuffer<int>(d);
    eps = readFromBuffer<float>(d);
}

} // namespace

// ============================= FusedGatedMLP plugin =============================

QwenRmsScaleSwigluPlugin::QwenRmsScaleSwigluPlugin(int hiddenSize, int interSize, float eps)
    : mHiddenSize(hiddenSize)
    , mInterSize(interSize)
    , mEps(eps)
{
}

QwenRmsScaleSwigluPlugin::QwenRmsScaleSwigluPlugin(void const* data, size_t length)
{
    deserializeCommon(data, mHiddenSize, mInterSize, mEps);
}

char const* QwenRmsScaleSwigluPlugin::getPluginType() const noexcept
{
    return FUSED_PLUGIN_NAME;
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
    // Output follows rawGateUp shape with the last dimension replaced by interSize.
    DimsExprs out = inputs[1];
    out.d[out.nbDims - 1] = exprBuilder.constant(mInterSize);
    return out;
}

bool QwenRmsScaleSwigluPlugin::supportsFormatCombination(
    int pos,
    PluginTensorDesc const* inOut,
    int nbInputs,
    int nbOutputs) noexcept
{
    if (inOut[pos].format != TensorFormat::kLINEAR)
    {
        return false;
    }
    if (pos == 0)
    {
        return isSupportedDataType(inOut[pos].type);
    }
    return inOut[pos].type == inOut[0].type;
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
    int const numTokens = computeNumTokens(inputDesc[0].dims, mHiddenSize);

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
        return 1;
    }

    return 0;
}

size_t QwenRmsScaleSwigluPlugin::getSerializationSize() const noexcept
{
    return serializedSize();
}

void QwenRmsScaleSwigluPlugin::serialize(void* buffer) const noexcept
{
    serializeCommon(buffer, mHiddenSize, mInterSize, mEps);
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

void QwenRmsScaleSwigluPlugin::terminate() noexcept {}

void QwenRmsScaleSwigluPlugin::attachToContext(
    cudnnContext* cudnnContext,
    cublasContext* cublasContext,
    IGpuAllocator* gpuAllocator) noexcept
{
}

void QwenRmsScaleSwigluPlugin::detachFromContext() noexcept {}

// ============================= Plain GatedMLP plugin =============================

QwenRmsScaleSwigluGatedPlugin::QwenRmsScaleSwigluGatedPlugin(int hiddenSize, int interSize, float eps)
    : mHiddenSize(hiddenSize)
    , mInterSize(interSize)
    , mEps(eps)
{
}

QwenRmsScaleSwigluGatedPlugin::QwenRmsScaleSwigluGatedPlugin(void const* data, size_t length)
{
    deserializeCommon(data, mHiddenSize, mInterSize, mEps);
}

char const* QwenRmsScaleSwigluGatedPlugin::getPluginType() const noexcept
{
    return GATED_PLUGIN_NAME;
}

char const* QwenRmsScaleSwigluGatedPlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

int QwenRmsScaleSwigluGatedPlugin::getNbOutputs() const noexcept
{
    return 1;
}

DimsExprs QwenRmsScaleSwigluGatedPlugin::getOutputDimensions(
    int outputIndex,
    DimsExprs const* inputs,
    int nbInputs,
    IExprBuilder& exprBuilder) noexcept
{
    // Output follows rawGate shape.
    return inputs[1];
}

bool QwenRmsScaleSwigluGatedPlugin::supportsFormatCombination(
    int pos,
    PluginTensorDesc const* inOut,
    int nbInputs,
    int nbOutputs) noexcept
{
    if (inOut[pos].format != TensorFormat::kLINEAR)
    {
        return false;
    }
    if (pos == 0)
    {
        return isSupportedDataType(inOut[pos].type);
    }
    return inOut[pos].type == inOut[0].type;
}

void QwenRmsScaleSwigluGatedPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in,
    int nbInputs,
    DynamicPluginTensorDesc const* out,
    int nbOutputs) noexcept
{
}

size_t QwenRmsScaleSwigluGatedPlugin::getWorkspaceSize(
    PluginTensorDesc const* inputs,
    int nbInputs,
    PluginTensorDesc const* outputs,
    int nbOutputs) const noexcept
{
    return 0;
}

int QwenRmsScaleSwigluGatedPlugin::enqueue(
    PluginTensorDesc const* inputDesc,
    PluginTensorDesc const* outputDesc,
    void const* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept
{
    int const numTokens = computeNumTokens(inputDesc[0].dims, mHiddenSize);

    tensorrt_llm::kernels::QwenRmsScaleSwigluGatedParams params;
    params.hiddenStates = inputs[0];
    params.rawGate = inputs[1];
    params.rawInter = inputs[2];
    params.output = outputs[0];
    params.numTokens = numTokens;
    params.hiddenSize = mHiddenSize;
    params.interSize = mInterSize;
    params.eps = mEps;
    params.stream = stream;

    if (inputDesc[0].type == DataType::kBF16)
    {
        tensorrt_llm::kernels::invokeQwenRmsScaleSwigluGatedBf16(params);
    }
    else if (inputDesc[0].type == DataType::kHALF)
    {
        tensorrt_llm::kernels::invokeQwenRmsScaleSwigluGatedFp16(params);
    }
    else
    {
        return 1;
    }

    return 0;
}

size_t QwenRmsScaleSwigluGatedPlugin::getSerializationSize() const noexcept
{
    return serializedSize();
}

void QwenRmsScaleSwigluGatedPlugin::serialize(void* buffer) const noexcept
{
    serializeCommon(buffer, mHiddenSize, mInterSize, mEps);
}

void QwenRmsScaleSwigluGatedPlugin::destroy() noexcept
{
    delete this;
}

IPluginV2DynamicExt* QwenRmsScaleSwigluGatedPlugin::clone() const noexcept
{
    auto* plugin = new QwenRmsScaleSwigluGatedPlugin(mHiddenSize, mInterSize, mEps);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

void QwenRmsScaleSwigluGatedPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace ? pluginNamespace : "";
}

char const* QwenRmsScaleSwigluGatedPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

DataType QwenRmsScaleSwigluGatedPlugin::getOutputDataType(
    int index,
    DataType const* inputTypes,
    int nbInputs) const noexcept
{
    return inputTypes[0];
}

int QwenRmsScaleSwigluGatedPlugin::initialize() noexcept
{
    return 0;
}

void QwenRmsScaleSwigluGatedPlugin::terminate() noexcept {}

void QwenRmsScaleSwigluGatedPlugin::attachToContext(
    cudnnContext* cudnnContext,
    cublasContext* cublasContext,
    IGpuAllocator* gpuAllocator) noexcept
{
}

void QwenRmsScaleSwigluGatedPlugin::detachFromContext() noexcept {}

// ============================= Creators =============================

QwenRmsScaleSwigluPluginCreator::QwenRmsScaleSwigluPluginCreator()
{
    initFields(mPluginAttributes, mFC);
}

char const* QwenRmsScaleSwigluPluginCreator::getPluginName() const noexcept
{
    return FUSED_PLUGIN_NAME;
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
    parseFields(fc, hiddenSize, interSize, eps);
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

QwenRmsScaleSwigluGatedPluginCreator::QwenRmsScaleSwigluGatedPluginCreator()
{
    initFields(mPluginAttributes, mFC);
}

char const* QwenRmsScaleSwigluGatedPluginCreator::getPluginName() const noexcept
{
    return GATED_PLUGIN_NAME;
}

char const* QwenRmsScaleSwigluGatedPluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

PluginFieldCollection const* QwenRmsScaleSwigluGatedPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* QwenRmsScaleSwigluGatedPluginCreator::createPlugin(
    char const* name,
    PluginFieldCollection const* fc) noexcept
{
    int hiddenSize = 0;
    int interSize = 0;
    float eps = 1e-6f;
    parseFields(fc, hiddenSize, interSize, eps);
    return new QwenRmsScaleSwigluGatedPlugin(hiddenSize, interSize, eps);
}

IPluginV2* QwenRmsScaleSwigluGatedPluginCreator::deserializePlugin(
    char const* name,
    void const* serialData,
    size_t serialLength) noexcept
{
    return new QwenRmsScaleSwigluGatedPlugin(serialData, serialLength);
}

void QwenRmsScaleSwigluGatedPluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace ? pluginNamespace : "";
}

char const* QwenRmsScaleSwigluGatedPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

} // namespace plugins
} // namespace tensorrt_llm

// TensorRT's registration macro concatenates the class token into a static
// variable name, so it cannot receive a namespace-qualified type directly.
using QwenRmsScaleSwigluPluginCreatorAlias =
    tensorrt_llm::plugins::QwenRmsScaleSwigluPluginCreator;
using QwenRmsScaleSwigluGatedPluginCreatorAlias =
    tensorrt_llm::plugins::QwenRmsScaleSwigluGatedPluginCreator;

REGISTER_TENSORRT_PLUGIN(QwenRmsScaleSwigluPluginCreatorAlias);
REGISTER_TENSORRT_PLUGIN(QwenRmsScaleSwigluGatedPluginCreatorAlias);
