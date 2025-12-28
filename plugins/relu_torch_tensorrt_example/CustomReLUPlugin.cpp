#include "CustomReLUPlugin.h"
#include "CustomReLUPluginKernel.h"
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <mutex>
#include <memory>
#include <cstring>

namespace nvinfer1
{
namespace plugin
{
constexpr char const* kPluginName = "CustomReLUPlugin";
constexpr char const* kPluginVersion = "1";
constexpr char const* kPluginNamespace = "";

CustomReLUPlugin::CustomReLUPlugin()
    : IPluginBase(kPluginName, kPluginVersion, kPluginNamespace){}

IPluginCapability* CustomReLUPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    try
    {
        if (type == PluginCapabilityType::kBUILD)
        {
            return static_cast<IPluginV3OneBuild*>(this);
        }
        if (type == PluginCapabilityType::kRUNTIME)
        {
            return static_cast<IPluginV3OneRuntime*>(this);
        }
        PLUGIN_ASSERT(type == PluginCapabilityType::kCORE);
        return static_cast<IPluginV3OneCore*>(this);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV3* CustomReLUPlugin::clone() noexcept
{
    try
    {
        auto plugin = std::make_unique<CustomReLUPlugin>(*this);
        return plugin.release();
    }
    catch(const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

int32_t CustomReLUPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t CustomReLUPlugin::configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInput,
                                              DynamicPluginTensorDesc const* out, int32_t nbOutput) noexcept
{
    PLUGIN_ASSERT(in != nullptr);
    PLUGIN_ASSERT(out != nullptr);
    PLUGIN_ASSERT(nbInput == 1);
    PLUGIN_ASSERT(nbOutput == 1);

    return 0;
}

bool CustomReLUPlugin::supportsFormatCombination(int32_t pos,
    nvinfer1::DynamicPluginTensorDesc const* inOut,
    int32_t nbInputs, int32_t nbOutputs) noexcept
{
    PLUGIN_ASSERT(inOut != nullptr);
    PLUGIN_ASSERT(nbInputs == 1);
    PLUGIN_ASSERT(nbOutputs == 1);
    PLUGIN_ASSERT(pos >= 0 && pos < nbInputs + nbOutputs);

    const auto& desc = inOut[pos].desc;

    // All tensors must be in LINEAR format
    if (desc.format != nvinfer1::TensorFormat::kLINEAR)
        return false;

    // Support FP32 and FP16
    if (pos == 0) // Input
    {
        return (desc.type == nvinfer1::DataType::kFLOAT ||
                desc.type == nvinfer1::DataType::kHALF);
    }
    else // Output (pos == 1)
    {
        // Output must match input type
        return desc.type == inOut[0].desc.type;
    }

    return false;
}

int32_t CustomReLUPlugin::getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs,
                                                  DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    PLUGIN_ASSERT(outputTypes != nullptr);
    PLUGIN_ASSERT(inputTypes != nullptr);
    PLUGIN_ASSERT(nbInputs == 1);
    PLUGIN_ASSERT(nbOutputs == 1);

    outputTypes[0] = inputTypes[0];
    return 0;
}

int32_t CustomReLUPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
                                               DimsExprs const* shapeInputs, int32_t nbShapeInputs,
                                               DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept
{
    PLUGIN_ASSERT(inputs != nullptr);
    PLUGIN_ASSERT(outputs != nullptr);
    PLUGIN_ASSERT(nbInputs == 1);
    PLUGIN_ASSERT(nbOutputs == 1);

    outputs[0] = inputs[0];

    return 0;
}

int32_t CustomReLUPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
                                        void const* const* inputs, void* const* outputs, void* workspace,
                                        cudaStream_t stream) noexcept
{
    PLUGIN_ASSERT(inputDesc != nullptr && outputDesc != nullptr && inputs != nullptr && outputs != nullptr);

    int n = inputDesc[0].dims.d[0];
    for (int i = 1; i < inputDesc[0].dims.nbDims; ++i)
    {
        n *= inputDesc[0].dims.d[i];
    }

    const auto type = inputDesc[0].type;
    if (type == nvinfer1::DataType::kFLOAT)
    {
        customReLU(static_cast<const float*>(inputs[0]),
            static_cast<float*>(outputs[0]), n, stream);
    }
    else if (type == nvinfer1::DataType::kHALF)
    {
        customReLUFP16(static_cast<const __half*>(inputs[0]),
            static_cast<__half*>(outputs[0]), n, stream);
    }
    else
    {
        PLUGIN_ASSERT(false);
        return -1;
    }

    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int32_t CustomReLUPlugin::onShapeChange(PluginTensorDesc const* in, int32_t nbInput,
    PluginTensorDesc const* out, int32_t nbOutput) noexcept
{
    PLUGIN_ASSERT(in != nullptr);
    PLUGIN_ASSERT(out != nullptr);
    PLUGIN_ASSERT(nbInput == 1);
    PLUGIN_ASSERT(nbOutput == 1);

    // Verify output shape matches input shape (element-wise operation)
    PLUGIN_ASSERT(in[0].dims.nbDims == out[0].dims.nbDims);
    for (int32_t i = 0; i < in[0].dims.nbDims; ++i)
    {
    PLUGIN_ASSERT(in[0].dims.d[i] == out[0].dims.d[i]);
    }

    // Verify data types match
    PLUGIN_ASSERT(in[0].type == out[0].type);

    // Verify format
    PLUGIN_ASSERT(in[0].format == nvinfer1::TensorFormat::kLINEAR);
    PLUGIN_ASSERT(out[0].format == nvinfer1::TensorFormat::kLINEAR);

    return 0;
}

IPluginV3* CustomReLUPlugin::attachToContext(IPluginResourceContext* context) noexcept
{
    return clone();
}

PluginFieldCollection const* CustomReLUPlugin::getFieldsToSerialize() noexcept
{

    m_FCToSerialize.nbFields = m_DataToSerialize.size();
    m_FCToSerialize.fields = m_DataToSerialize.data();

    return &m_FCToSerialize;
}

size_t CustomReLUPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
                                                DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

// Plugin Creator
CustomReLUPluginCreator::CustomReLUPluginCreator()
    : IPluginCreatorBase(kPluginName, kPluginVersion, kPluginNamespace)
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> guard(sMutex);
    m_pluginAttributes.clear();

    m_FC.nbFields = m_pluginAttributes.size();
    m_FC.fields = m_pluginAttributes.data();
}

PluginFieldCollection const* CustomReLUPluginCreator::getFieldNames() noexcept
{
    return &m_FC;
}

IPluginV3* CustomReLUPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc,
                                                      TensorRTPhase phase) noexcept
{
    try
    {
        PLUGIN_ASSERT(fc != nullptr);

        return new CustomReLUPlugin();
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }

    return nullptr;
}

} // namespace plugin
} // namespace nvinfer1

DEFINE_TRT_PLUGIN_CREATOR(nvinfer1::plugin::CustomReLUPluginCreator);