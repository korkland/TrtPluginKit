#include "IdentityConvPluginV3.h"
#include <cuda_runtime_api.h>
#include <mutex>
#include <memory>
#include <cstring>
#include <array>

namespace nvinfer1
{
namespace plugin
{
constexpr char const* kPluginName = "IdentityConv";
constexpr char const* kPluginVersion = "1";
constexpr char const* kPluginNamespace = "";

IdentityConvPluginV3::IdentityConvPluginV3(const IdentityConvParameters& params)
    : IPluginBase(kPluginName, kPluginVersion, kPluginNamespace)
    , m_params(params)
{
    PLUGIN_ASSERT(params.group > 0);

    // Here we can check device parameters and set threads and blocks for example
}

IPluginCapability* IdentityConvPluginV3::getCapabilityInterface(PluginCapabilityType type) noexcept
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

IPluginV3* IdentityConvPluginV3::clone() noexcept
{
    try
    {
        auto plugin = std::make_unique<IdentityConvPluginV3>(*this);
        return plugin.release();
    }
    catch(const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

int32_t IdentityConvPluginV3::getNbOutputs() const noexcept
{
    return 1;
}

int32_t IdentityConvPluginV3::configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInput,
                                              DynamicPluginTensorDesc const* out, int32_t nbOutput) noexcept
{
    PLUGIN_ASSERT(nbInput == 2);
    PLUGIN_ASSERT(nbOutput == 1);
    PLUGIN_ASSERT(in[0].desc.dims.nbDims == 4);
    PLUGIN_ASSERT(in[1].desc.dims.nbDims == 4);
    PLUGIN_ASSERT(out[0].desc.dims.nbDims == 4);
    PLUGIN_ASSERT(in[0].desc.dims.d[0] == out[0].desc.dims.d[0]);
    PLUGIN_ASSERT(in[0].desc.dims.d[1] == out[0].desc.dims.d[1]);
    PLUGIN_ASSERT(in[0].desc.dims.d[2] == out[0].desc.dims.d[2]);
    PLUGIN_ASSERT(in[0].desc.dims.d[3] == out[0].desc.dims.d[3]);
    PLUGIN_ASSERT(in[0].desc.type == out[0].desc.type);

    m_params.dtype = in[0].desc.type;
    m_params.channelSize = in[0].desc.dims.d[0];
    m_params.height = in[0].desc.dims.d[1];
    m_params.width = in[0].desc.dims.d[2];
    if (m_params.dtype == DataType::kINT8)
    {
        m_params.dtypeBytes = 1;
    }
    else if (m_params.dtype == DataType::kHALF)
    {
        m_params.dtypeBytes = 2;
    }
    else if (m_params.dtype == DataType::kFLOAT)
    {
        m_params.dtypeBytes = 4;
    }
    else
    {
        PLUGIN_ASSERT(false);
    }

    return 0;
}

bool IdentityConvPluginV3::supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* inOut,
                                                      int32_t nbInputs, int32_t nbOutputs) noexcept
{
    PLUGIN_ASSERT(inOut != nullptr);
    PLUGIN_ASSERT(nbInputs == 2);
    PLUGIN_ASSERT(nbOutputs == 1);
    PLUGIN_ASSERT(pos < nbInputs + nbOutputs);
    bool isValidCombination = false;

    // Suppose we support only a limited number of format configurations.
    isValidCombination |=
        (inOut[pos].desc.format == nvinfer1::TensorFormat::kLINEAR &&
         inOut[pos].desc.type == nvinfer1::DataType::kFLOAT);
    isValidCombination |=
        (inOut[pos].desc.format == nvinfer1::TensorFormat::kLINEAR &&
         inOut[pos].desc.type == nvinfer1::DataType::kHALF);
    // Make sure the input tensor and output tensor types and formats are same.
    isValidCombination &=
        (pos < nbInputs || (inOut[pos].desc.format == inOut[0].desc.format &&
                            inOut[pos].desc.type == inOut[0].desc.type));

    return isValidCombination;
}

int32_t IdentityConvPluginV3::getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs,
                                                  DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    PLUGIN_ASSERT(outputTypes != nullptr);
    PLUGIN_ASSERT(inputTypes != nullptr);
    PLUGIN_ASSERT(nbOutputs == 1);
    PLUGIN_ASSERT(nbInputs == 2);

    outputTypes[0] = inputTypes[0];
    return 0;
}

int32_t IdentityConvPluginV3::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
                                               DimsExprs const* shapeInputs, int32_t nbShapeInputs,
                                               DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept
{
    PLUGIN_ASSERT(inputs != nullptr);
    PLUGIN_ASSERT(outputs != nullptr);
    PLUGIN_ASSERT(nbInputs == 2);
    PLUGIN_ASSERT(nbOutputs == 1);
    PLUGIN_ASSERT(inputs[0].nbDims == 4);

    outputs[0].nbDims = inputs[0].nbDims;
    for (int32_t i = 0; i < inputs[0].nbDims; ++i)
    {
        outputs[0].d[i] = inputs[0].d[i];
    }
    return 0;
}

int32_t IdentityConvPluginV3::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
                                        void const* const* inputs, void* const* outputs, void* workspace,
                                        cudaStream_t stream) noexcept
{
    PLUGIN_ASSERT(inputDesc != nullptr);
    PLUGIN_ASSERT(outputDesc != nullptr);
    PLUGIN_ASSERT(inputs != nullptr);
    PLUGIN_ASSERT(outputs != nullptr);

    // Here we can call the kernel to do the computation
    // For example:
    // launchKernel(stream, m_params, inputs[0], outputs[0]);
    const auto inputSize = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1] *
                           inputDesc[0].dims.d[2] * inputDesc[0].dims.d[3] *
                           m_params.dtypeBytes;

    PLUGIN_CUASSERT(cudaMemcpyAsync(outputs[0], inputs[0], inputSize, cudaMemcpyDeviceToDevice, stream));
    return 0;
}

int32_t IdentityConvPluginV3::onShapeChange(PluginTensorDesc const* in, int32_t nbInput,
                                             PluginTensorDesc const* out, int32_t nbOutput) noexcept
{
    PLUGIN_ASSERT(in != nullptr);
    PLUGIN_ASSERT(out != nullptr);
    PLUGIN_ASSERT(nbInput == 2);
    PLUGIN_ASSERT(nbOutput == 1);
    PLUGIN_ASSERT(in[0].dims.nbDims == 4);
    PLUGIN_ASSERT(in[1].dims.nbDims == 4);
    PLUGIN_ASSERT(out[0].dims.nbDims == 4);

    return 0;
}

IPluginV3* IdentityConvPluginV3::attachToContext(IPluginResourceContext* context) noexcept
{
    return clone();
}

PluginFieldCollection const* IdentityConvPluginV3::getFieldsToSerialize() noexcept
{
    m_DataToSerialize.clear();
    m_DataToSerialize.emplace_back(PluginField("parameters", &m_params, PluginFieldType::kUNKNOWN, sizeof(m_params)));

    m_FCToSerialize.nbFields = m_DataToSerialize.size();
    m_FCToSerialize.fields = m_DataToSerialize.data();

    return &m_FCToSerialize;
}

size_t IdentityConvPluginV3::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
                                                DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

// Plugin Creator
IdentityConvPluginV3Creator::IdentityConvPluginV3Creator()
    : IPluginCreatorBase(kPluginName, kPluginVersion, kPluginNamespace)
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> guard(sMutex);
    m_pluginAttributes.clear();
    m_pluginAttributes.emplace_back(PluginField("kernel_shape", nullptr, PluginFieldType::kINT32, 2));
    m_pluginAttributes.emplace_back(PluginField("strides", nullptr, PluginFieldType::kINT32, 2));
    m_pluginAttributes.emplace_back(PluginField("pads", nullptr, PluginFieldType::kINT32, 4));
    m_pluginAttributes.emplace_back(PluginField("group", nullptr, PluginFieldType::kINT32, 1));

    m_FC.nbFields = m_pluginAttributes.size();
    m_FC.fields = m_pluginAttributes.data();
}

PluginFieldCollection const* IdentityConvPluginV3Creator::getFieldNames() noexcept
{
    return &m_FC;
}

IPluginV3* IdentityConvPluginV3Creator::createPlugin(char const* name, PluginFieldCollection const* fc,
                                                      TensorRTPhase phase) noexcept
{
    if (phase == TensorRTPhase::kBUILD)
    {
        // The attributes from the ONNX node will be parsed and passed via fc.
        try
        {
            PLUGIN_ASSERT(fc != nullptr);
            logInfo(("number of fields: " + std::to_string(fc->nbFields)).c_str());
            logInfo(("field name: " + std::string(fc->fields[0].name)).c_str());
            PLUGIN_ASSERT(fc->nbFields == 4);
            PluginField const* fields = fc->fields;

            std::array<int32_t, 2> kernelShape;
            std::array<int32_t, 2> strides;
            std::array<int32_t, 4> pads;
            int32_t group = 1;
            for (int32_t i = 0; i < fc->nbFields; ++i)
            {
                if (!strcmp(fields[i].name, "kernel_shape"))
                {
                    PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
                    PLUGIN_ASSERT(fields[i].length == static_cast<int32_t>(kernelShape.size()));
                    kernelShape[0] = static_cast<int32_t const*>(fields[i].data)[0];
                    kernelShape[1] = static_cast<int32_t const*>(fields[i].data)[1];
                    logInfo(("kernel_shape: " + std::to_string(kernelShape[0]) + ", " +
                            std::to_string(kernelShape[1])).c_str());
                }
                else if (!strcmp(fields[i].name, "strides"))
                {
                    PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
                    PLUGIN_ASSERT(fields[i].length == static_cast<int32_t>(strides.size()));
                    strides[0] = static_cast<int32_t const*>(fields[i].data)[0];
                    strides[1] = static_cast<int32_t const*>(fields[i].data)[1];
                    logInfo(("strides: " + std::to_string(strides[0]) + ", " +
                            std::to_string(strides[1])).c_str());
                }
                else if (!strcmp(fields[i].name, "pads"))
                {
                    PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
                    PLUGIN_ASSERT(fields[i].length == static_cast<int32_t>(pads.size()));
                    pads[0] = static_cast<int32_t const*>(fields[i].data)[0];
                    pads[1] = static_cast<int32_t const*>(fields[i].data)[1];
                    pads[2] = static_cast<int32_t const*>(fields[i].data)[2];
                    pads[3] = static_cast<int32_t const*>(fields[i].data)[3];
                    logInfo(("pads: " + std::to_string(pads[0]) + ", " +
                             std::to_string(pads[1]) + ", " + std::to_string(pads[2]) + ", " +
                             std::to_string(pads[3])).c_str());
                }
                else if (!strcmp(fields[i].name, "group"))
                {
                    PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
                    PLUGIN_ASSERT(fields[i].length == 1);
                    group = *static_cast<int32_t const*>(fields[i].data);
                    logInfo(("group: " + std::to_string(group)).c_str());
                }
            }
            IdentityConvParameters params{.group = group};
            return new IdentityConvPluginV3(params);
        }
        catch (std::exception const& e)
        {
            caughtError(e);
        }
    }
    else if (phase == TensorRTPhase::kRUNTIME)
    {
        // The attributes from the serialized plugin will be passed via fc.
        try
        {
            PLUGIN_ASSERT(fc != nullptr);
            PLUGIN_ASSERT(fc->nbFields == 1);
            PluginField const* fields = fc->fields;

            PLUGIN_ASSERT(!strcmp(fields[0].name, "parameters"));
            PLUGIN_ASSERT(fields[0].type == PluginFieldType::kUNKNOWN);
            PLUGIN_ASSERT(fields[0].length == sizeof(IdentityConvParameters));
            auto params = *static_cast<IdentityConvParameters const*>(fields[0].data);

            return new IdentityConvPluginV3(params);
        }
        catch (std::exception const& e)
        {
            caughtError(e);
        }
    }

    return nullptr;
}

} // namespace plugin
} // namespace nvinfer1

DEFINE_TRT_PLUGIN_CREATOR(nvinfer1::plugin::IdentityConvPluginV3Creator);