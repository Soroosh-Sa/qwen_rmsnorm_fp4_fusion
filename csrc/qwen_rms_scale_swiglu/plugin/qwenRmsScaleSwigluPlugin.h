#pragma once

#include "NvInfer.h"
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace tensorrt_llm
{
namespace plugins
{

class QwenRmsScaleSwigluPlugin : public nvinfer1::IPluginV2DynamicExt
{
public:
    QwenRmsScaleSwigluPlugin(int hiddenSize, int interSize, float eps);
    QwenRmsScaleSwigluPlugin(void const* data, size_t length);

    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;

    nvinfer1::DimsExprs getOutputDimensions(
        int outputIndex,
        nvinfer1::DimsExprs const* inputs,
        int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    bool supportsFormatCombination(
        int pos,
        nvinfer1::PluginTensorDesc const* inOut,
        int nbInputs,
        int nbOutputs) noexcept override;

    void configurePlugin(
        nvinfer1::DynamicPluginTensorDesc const* in,
        int nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out,
        int nbOutputs) noexcept override;

    size_t getWorkspaceSize(
        nvinfer1::PluginTensorDesc const* inputs,
        int nbInputs,
        nvinfer1::PluginTensorDesc const* outputs,
        int nbOutputs) const noexcept override;

    int enqueue(
        nvinfer1::PluginTensorDesc const* inputDesc,
        nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs,
        void* const* outputs,
        void* workspace,
        cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;

    void destroy() noexcept override;
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;

    nvinfer1::DataType getOutputDataType(
        int index,
        nvinfer1::DataType const* inputTypes,
        int nbInputs) const noexcept override;

    int initialize() noexcept override;
    void terminate() noexcept override;

    void attachToContext(
        cudnnContext* cudnnContext,
        cublasContext* cublasContext,
        nvinfer1::IGpuAllocator* gpuAllocator) noexcept override;

    void detachFromContext() noexcept override;

private:
    int mHiddenSize;
    int mInterSize;
    float mEps;
    std::string mNamespace;
};

class QwenRmsScaleSwigluGatedPlugin : public nvinfer1::IPluginV2DynamicExt
{
public:
    QwenRmsScaleSwigluGatedPlugin(int hiddenSize, int interSize, float eps);
    QwenRmsScaleSwigluGatedPlugin(void const* data, size_t length);

    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;

    nvinfer1::DimsExprs getOutputDimensions(
        int outputIndex,
        nvinfer1::DimsExprs const* inputs,
        int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    bool supportsFormatCombination(
        int pos,
        nvinfer1::PluginTensorDesc const* inOut,
        int nbInputs,
        int nbOutputs) noexcept override;

    void configurePlugin(
        nvinfer1::DynamicPluginTensorDesc const* in,
        int nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out,
        int nbOutputs) noexcept override;

    size_t getWorkspaceSize(
        nvinfer1::PluginTensorDesc const* inputs,
        int nbInputs,
        nvinfer1::PluginTensorDesc const* outputs,
        int nbOutputs) const noexcept override;

    int enqueue(
        nvinfer1::PluginTensorDesc const* inputDesc,
        nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs,
        void* const* outputs,
        void* workspace,
        cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;

    void destroy() noexcept override;
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;

    nvinfer1::DataType getOutputDataType(
        int index,
        nvinfer1::DataType const* inputTypes,
        int nbInputs) const noexcept override;

    int initialize() noexcept override;
    void terminate() noexcept override;

    void attachToContext(
        cudnnContext* cudnnContext,
        cublasContext* cublasContext,
        nvinfer1::IGpuAllocator* gpuAllocator) noexcept override;

    void detachFromContext() noexcept override;

private:
    int mHiddenSize;
    int mInterSize;
    float mEps;
    std::string mNamespace;
};

class QwenRmsScaleSwigluPluginCreator : public nvinfer1::IPluginCreator
{
public:
    QwenRmsScaleSwigluPluginCreator();

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(
        char const* name,
        nvinfer1::PluginFieldCollection const* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        char const* name,
        void const* serialData,
        size_t serialLength) noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;

private:
    std::string mNamespace;
    std::vector<nvinfer1::PluginField> mPluginAttributes;
    nvinfer1::PluginFieldCollection mFC;
};

class QwenRmsScaleSwigluGatedPluginCreator : public nvinfer1::IPluginCreator
{
public:
    QwenRmsScaleSwigluGatedPluginCreator();

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(
        char const* name,
        nvinfer1::PluginFieldCollection const* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        char const* name,
        void const* serialData,
        size_t serialLength) noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;

private:
    std::string mNamespace;
    std::vector<nvinfer1::PluginField> mPluginAttributes;
    nvinfer1::PluginFieldCollection mFC;
};

} // namespace plugins
} // namespace tensorrt_llm
