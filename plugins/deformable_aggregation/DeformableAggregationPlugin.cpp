#include "DeformableAggregationPlugin.h"
#include "DeformableAggregationParameters.h"
#include "DeformableAggregationKernel.h"
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <mutex>
#include <memory>
#include <cstring>

namespace nvinfer1
{
namespace plugin
{
constexpr char const* kPluginName = "DeformableAggregationPlugin";
constexpr char const* kPluginVersion = "1";
constexpr char const* kPluginNamespace = "";

DeformableAggregationPlugin::DeformableAggregationPlugin()
    : IPluginBase(kPluginName, kPluginVersion, kPluginNamespace){}

IPluginCapability* DeformableAggregationPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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

IPluginV3* DeformableAggregationPlugin::clone() noexcept
{
    try
    {
        auto plugin = std::make_unique<DeformableAggregationPlugin>(*this);
        return plugin.release();
    }
    catch(const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

int32_t DeformableAggregationPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t DeformableAggregationPlugin::configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInput,
                                              DynamicPluginTensorDesc const* out, int32_t nbOutput) noexcept
{
    PLUGIN_ASSERT(in != nullptr);
    PLUGIN_ASSERT(out != nullptr);
    PLUGIN_ASSERT(nbInput == NUM_OF_INPUTS);
    PLUGIN_ASSERT(nbOutput == NUM_OF_OUTPUTS);

    // VERIFY RANKS
    PLUGIN_ASSERT(in[MC_MS_FEAT_IDX].desc.dims.nbDims == MC_MS_FEAT_RANK);
    PLUGIN_ASSERT(in[SPATIAL_SHAPE_IDX].desc.dims.nbDims == SPATIAL_SHAPE_RANK);
    PLUGIN_ASSERT(in[SCALE_START_IDX].desc.dims.nbDims == SCALE_START_RANK);
    PLUGIN_ASSERT(in[SAMPLE_LOCATION_IDX].desc.dims.nbDims == SAMPLE_LOCATION_RANK);
    PLUGIN_ASSERT(in[WEIGHTS_IDX].desc.dims.nbDims == WEIGHTS_RANK);
    PLUGIN_ASSERT(out[OUTPUT_IDX].desc.dims.nbDims == OUTPUT_RANK);

    // VERIFY DIMS
    auto batchSize = in[MC_MS_FEAT_IDX].desc.dims.d[0];
    PLUGIN_ASSERT(in[SAMPLE_LOCATION_IDX].desc.dims.d[0] == batchSize &&
                 in[WEIGHTS_IDX].desc.dims.d[0] == batchSize &&
                 in[OUTPUT_IDX].desc.dims.d[0] == batchSize);

    auto numCams = in[SPATIAL_SHAPE_IDX].desc.dims.d[0];
    PLUGIN_ASSERT(in[SCALE_START_IDX].desc.dims.d[0] == numCams &&
                 in[SAMPLE_LOCATION_IDX].desc.dims.d[3] == numCams &&
                 in[WEIGHTS_IDX].desc.dims.d[3] == numCams);

    auto numEmbeds = in[MC_MS_FEAT_IDX].desc.dims.d[2];
    PLUGIN_ASSERT(out[OUTPUT_IDX].desc.dims.d[2] == numEmbeds);

    auto numScale = in[SPATIAL_SHAPE_IDX].desc.dims.d[1];
    PLUGIN_ASSERT(in[SCALE_START_IDX].desc.dims.d[1] == numScale &&
                 in[WEIGHTS_IDX].desc.dims.d[4] == numScale);

    auto numAnchors = in[SAMPLE_LOCATION_IDX].desc.dims.d[1];
    PLUGIN_ASSERT(in[WEIGHTS_IDX].desc.dims.d[1] == numAnchors &&
                 in[OUTPUT_IDX].desc.dims.d[1] == numAnchors);

    auto numPts = in[SAMPLE_LOCATION_IDX].desc.dims.d[2];
    PLUGIN_ASSERT(in[WEIGHTS_IDX].desc.dims.d[2] == numPts);

    PLUGIN_ASSERT(in[SPATIAL_SHAPE_IDX].desc.dims.d[2] == 2 &&
                 in[SAMPLE_LOCATION_IDX].desc.dims.d[4] == 2);

    return 0;
}

bool DeformableAggregationPlugin::supportsFormatCombination(int32_t pos,
    nvinfer1::DynamicPluginTensorDesc const* inOut,
    int32_t nbInputs, int32_t nbOutputs) noexcept
{
    PLUGIN_ASSERT(inOut != nullptr);
    PLUGIN_ASSERT(nbInputs == NUM_OF_INPUTS);
    PLUGIN_ASSERT(nbOutputs == NUM_OF_OUTPUTS);
    PLUGIN_ASSERT(pos < nbInputs + nbOutputs);

    const auto& desc = inOut[pos].desc;

    // All tensors must be in LINEAR format
    if (desc.format != nvinfer1::TensorFormat::kLINEAR)
        return false;

    // Validate floating-point group (inputs MC_MS_FEAT_IDX, SAMPLE_LOCATION_IDX, WEIGHTS_IDX and output)
    const auto refFloatType = inOut[MC_MS_FEAT_IDX].desc.type;
    if (pos == MC_MS_FEAT_IDX)
    {
        return (desc.type == nvinfer1::DataType::kFLOAT || desc.type == nvinfer1::DataType::kHALF);
    }
    else if (pos == SAMPLE_LOCATION_IDX || pos == WEIGHTS_IDX || pos == NUM_OF_INPUTS /*output idx*/)
    {
        return desc.type == refFloatType &&
               (refFloatType == nvinfer1::DataType::kFLOAT || refFloatType == nvinfer1::DataType::kHALF);
    }


    // Validate integer group (inputs SPATIAL_SHAPE_IDX and SCALE_START_IDX)
    const auto refIndexType = inOut[SPATIAL_SHAPE_IDX].desc.type;
    if (pos == SPATIAL_SHAPE_IDX)
    {
        return desc.type == nvinfer1::DataType::kINT64;
    }
    else if (pos == SCALE_START_IDX)
    {
        return desc.type == refIndexType && refIndexType == nvinfer1::DataType::kINT64;
    }

    return false;
}

int32_t DeformableAggregationPlugin::getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs,
                                                  DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    PLUGIN_ASSERT(outputTypes != nullptr);
    PLUGIN_ASSERT(inputTypes != nullptr);
    PLUGIN_ASSERT(nbInputs == NUM_OF_INPUTS);
    PLUGIN_ASSERT(nbOutputs == NUM_OF_OUTPUTS);

    outputTypes[OUTPUT_IDX] = inputTypes[MC_MS_FEAT_IDX];
    return 0;
}

int32_t DeformableAggregationPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
                                               DimsExprs const* shapeInputs, int32_t nbShapeInputs,
                                               DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept
{
    PLUGIN_ASSERT(inputs != nullptr);
    PLUGIN_ASSERT(outputs != nullptr);
    PLUGIN_ASSERT(nbInputs == NUM_OF_INPUTS);
    PLUGIN_ASSERT(nbOutputs == NUM_OF_OUTPUTS);
    PLUGIN_ASSERT(outputs[OUTPUT_IDX].nbDims == OUTPUT_RANK);

    // batch size
    outputs[OUTPUT_IDX].d[0] = inputs[MC_MS_FEAT_IDX].d[0];
    // num anchors
    outputs[OUTPUT_IDX].d[1] = inputs[SAMPLE_LOCATION_IDX].d[1];
    // num embeds
    outputs[OUTPUT_IDX].d[2] = inputs[MC_MS_FEAT_IDX].d[2];

    return 0;
}

int32_t DeformableAggregationPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
                                        void const* const* inputs, void* const* outputs, void* workspace,
                                        cudaStream_t stream) noexcept
{
    PLUGIN_ASSERT(inputDesc != nullptr && outputDesc != nullptr && inputs != nullptr && outputs != nullptr);

    auto batchSize = inputDesc[MC_MS_FEAT_IDX].dims.d[0];
    auto numCams = inputDesc[SPATIAL_SHAPE_IDX].dims.d[0];
    auto numFeat = inputDesc[MC_MS_FEAT_IDX].dims.d[1];
    auto numEmbeds = inputDesc[MC_MS_FEAT_IDX].dims.d[2];
    auto numScale = inputDesc[SPATIAL_SHAPE_IDX].dims.d[1];
    auto numAnchors = inputDesc[SAMPLE_LOCATION_IDX].dims.d[1];
    auto numPts = inputDesc[SAMPLE_LOCATION_IDX].dims.d[2];
    auto numGroups = inputDesc[WEIGHTS_IDX].dims.d[5];
    auto spatialShape = static_cast<int64_t const*>(inputs[SPATIAL_SHAPE_IDX]);
    auto scaleStart = static_cast<int64_t const*>(inputs[SCALE_START_IDX]);

    const auto type = inputDesc[MC_MS_FEAT_IDX].type;
    if (type == nvinfer1::DataType::kFLOAT)
    {
        auto mcMsFeat = static_cast<float const*>(inputs[MC_MS_FEAT_IDX]);
        auto sampleLocation = static_cast<float const*>(inputs[SAMPLE_LOCATION_IDX]);
        auto weights = static_cast<float const*>(inputs[WEIGHTS_IDX]);
        auto output = static_cast<float*>(outputs[OUTPUT_IDX]);
        PLUGIN_CUASSERT(cudaMemsetAsync(output, 0, batchSize * numAnchors * numEmbeds * sizeof(float), stream));
        return deformableAggregationImpl<float, int64_t>(output, mcMsFeat, spatialShape, scaleStart, sampleLocation, weights,
                                                        batchSize, numCams, numFeat, numEmbeds, numScale, numAnchors,
                                                        numPts, numGroups, stream);
    }
    else if (type == nvinfer1::DataType::kHALF)
    {
        auto mcMsFeat = static_cast<__half const*>(inputs[MC_MS_FEAT_IDX]);
        auto sampleLocation = static_cast<__half const*>(inputs[SAMPLE_LOCATION_IDX]);
        auto weights = static_cast<__half const*>(inputs[WEIGHTS_IDX]);
        auto output = static_cast<__half*>(outputs[OUTPUT_IDX]);
        PLUGIN_CUASSERT(cudaMemsetAsync(output, 0, batchSize * numAnchors * numEmbeds * sizeof(__half), stream));
        return deformableAggregationImpl<__half, int64_t>(output, mcMsFeat, spatialShape, scaleStart, sampleLocation, weights,
                                                        batchSize, numCams, numFeat, numEmbeds, numScale, numAnchors,
                                                        numPts, numGroups, stream);
    }
    else
    {
        PLUGIN_ASSERT(false);
        return -1;
    }

    return 0;
}

int32_t DeformableAggregationPlugin::onShapeChange(PluginTensorDesc const* in, int32_t nbInput,
                                             PluginTensorDesc const* out, int32_t nbOutput) noexcept
{
    PLUGIN_ASSERT(in != nullptr);
    PLUGIN_ASSERT(out != nullptr);
    PLUGIN_ASSERT(nbInput == NUM_OF_INPUTS);
    PLUGIN_ASSERT(nbOutput == NUM_OF_OUTPUTS);

    // VERIFY RANKS
    PLUGIN_ASSERT(in[MC_MS_FEAT_IDX].dims.nbDims == MC_MS_FEAT_RANK);
    PLUGIN_ASSERT(in[SPATIAL_SHAPE_IDX].dims.nbDims == SPATIAL_SHAPE_RANK);
    PLUGIN_ASSERT(in[SCALE_START_IDX].dims.nbDims == SCALE_START_RANK);
    PLUGIN_ASSERT(in[SAMPLE_LOCATION_IDX].dims.nbDims == SAMPLE_LOCATION_RANK);
    PLUGIN_ASSERT(in[WEIGHTS_IDX].dims.nbDims == WEIGHTS_RANK);
    PLUGIN_ASSERT(out[OUTPUT_IDX].dims.nbDims == OUTPUT_RANK);

    // VERIFY DIMS
    auto batchSize = in[MC_MS_FEAT_IDX].dims.d[0];
    PLUGIN_ASSERT(in[SAMPLE_LOCATION_IDX].dims.d[0] == batchSize &&
                 in[WEIGHTS_IDX].dims.d[0] == batchSize &&
                 in[OUTPUT_IDX].dims.d[0] == batchSize);

    auto numCams = in[SPATIAL_SHAPE_IDX].dims.d[0];
    PLUGIN_ASSERT(in[SCALE_START_IDX].dims.d[0] == numCams &&
                 in[SAMPLE_LOCATION_IDX].dims.d[3] == numCams &&
                 in[WEIGHTS_IDX].dims.d[3] == numCams);

    auto numEmbeds = in[MC_MS_FEAT_IDX].dims.d[2];
    PLUGIN_ASSERT(out[OUTPUT_IDX].dims.d[2] == numEmbeds);

    auto numScale = in[SPATIAL_SHAPE_IDX].dims.d[1];
    PLUGIN_ASSERT(in[SCALE_START_IDX].dims.d[1] == numScale &&
                 in[WEIGHTS_IDX].dims.d[4] == numScale);

    auto numAnchors = in[SAMPLE_LOCATION_IDX].dims.d[1];
    PLUGIN_ASSERT(in[WEIGHTS_IDX].dims.d[1] == numAnchors &&
                 in[OUTPUT_IDX].dims.d[1] == numAnchors);

    auto numPts = in[SAMPLE_LOCATION_IDX].dims.d[2];
    PLUGIN_ASSERT(in[WEIGHTS_IDX].dims.d[2] == numPts);

    PLUGIN_ASSERT(in[SPATIAL_SHAPE_IDX].dims.d[2] == 2 &&
                 in[SAMPLE_LOCATION_IDX].dims.d[4] == 2);

    return 0;
}

IPluginV3* DeformableAggregationPlugin::attachToContext(IPluginResourceContext* context) noexcept
{
    return clone();
}

PluginFieldCollection const* DeformableAggregationPlugin::getFieldsToSerialize() noexcept
{

    m_FCToSerialize.nbFields = m_DataToSerialize.size();
    m_FCToSerialize.fields = m_DataToSerialize.data();

    return &m_FCToSerialize;
}

size_t DeformableAggregationPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
                                                DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

// Plugin Creator
DeformableAggregationCreator::DeformableAggregationCreator()
    : IPluginCreatorBase(kPluginName, kPluginVersion, kPluginNamespace)
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> guard(sMutex);
    m_pluginAttributes.clear();

    m_FC.nbFields = m_pluginAttributes.size();
    m_FC.fields = m_pluginAttributes.data();
}

PluginFieldCollection const* DeformableAggregationCreator::getFieldNames() noexcept
{
    return &m_FC;
}

IPluginV3* DeformableAggregationCreator::createPlugin(char const* name, PluginFieldCollection const* fc,
                                                      TensorRTPhase phase) noexcept
{
    try
    {
        PLUGIN_ASSERT(fc != nullptr);

        return new DeformableAggregationPlugin();
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }

    return nullptr;
}

} // namespace plugin
} // namespace nvinfer1

DEFINE_TRT_PLUGIN_CREATOR(nvinfer1::plugin::DeformableAggregationCreator);