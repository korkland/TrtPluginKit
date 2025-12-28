#pragma once

#include <cuda_runtime.h>
#include "IPluginBase.h"
#include "common.h"
#include <vector>

namespace nvinfer1
{
namespace plugin
{

class CustomReLUPlugin : public IPluginBase,
                         public IPluginV3OneBuild,
                         public IPluginV3OneRuntime
{
public:
    explicit CustomReLUPlugin();
    ~CustomReLUPlugin() override = default;

    // IPluginV3 methods
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override;
    IPluginV3* clone() noexcept override;

    // IPluginV3OneBuild methods
    int32_t getNbOutputs() const noexcept override;
    int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInput,
                            DynamicPluginTensorDesc const* out, int32_t nbOutput) noexcept override;
    bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* inOut,
                                   int32_t nbInputs, int32_t nbOutputs) noexcept override;
    int32_t getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs,
                               DataType const* inputTypes, int32_t nbInputs) const noexcept override;
    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
                            DimsExprs const* shapeInputs, int32_t nbShapeInputs,
                            DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept override;

    // IPluginV3OneRuntime methods
    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
                    void const* const* inputs, void* const* outputs, void* workspace,
                    cudaStream_t stream) noexcept override;
    int32_t onShapeChange(PluginTensorDesc const* in, int32_t nbInput,
                         PluginTensorDesc const* out, int32_t nbOutput) noexcept override;
    IPluginV3* attachToContext(IPluginResourceContext* context) noexcept override;
    PluginFieldCollection const* getFieldsToSerialize() noexcept override;
    size_t getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
                            DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;

private:
    std::vector<nvinfer1::PluginField> m_DataToSerialize;
    nvinfer1::PluginFieldCollection m_FCToSerialize;
};

class CustomReLUPluginCreator : public IPluginCreatorBase
{
public:
    CustomReLUPluginCreator();
    ~CustomReLUPluginCreator() override = default;

    PluginFieldCollection const* getFieldNames() noexcept override;
    IPluginV3* createPlugin(char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept override;

private:
    PluginFieldCollection m_FC{};
    std::vector<PluginField> m_pluginAttributes{};
};

} //namespace plugin
} // namespace nvinfer1