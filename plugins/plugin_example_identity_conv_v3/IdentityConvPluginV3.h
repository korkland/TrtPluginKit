#pragma once

#include <cuda_runtime.h>
#include "IPluginBase.h"
#include "common.h"
#include <vector>

namespace nvinfer1
{
namespace plugin
{
 /*
 * In our simple use case, even though there is no parameter used for this plugin.
 * we deserialize and serialize some attributes for demonstration purposes.
 */
struct IdentityConvParameters
{
    int32_t group; // attribute passed from ONNX
    int32_t channelSize;
    int32_t height;
    int32_t width;
    int32_t dtypeBytes;
    nvinfer1::DataType dtype;
};

class IdentityConvPluginV3 : public IPluginBase,
                             public IPluginV3OneBuild,
                             public IPluginV3OneRuntime
{
public:
    explicit IdentityConvPluginV3(const IdentityConvParameters& params);
    ~IdentityConvPluginV3() override = default;

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
    IdentityConvParameters m_params{};
    std::vector<nvinfer1::PluginField> m_DataToSerialize;
    nvinfer1::PluginFieldCollection m_FCToSerialize;
};

class IdentityConvPluginV3Creator : public IPluginCreatorBase
{
public:
    IdentityConvPluginV3Creator();
    ~IdentityConvPluginV3Creator() override = default;

    PluginFieldCollection const* getFieldNames() noexcept override;
    IPluginV3* createPlugin(char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept override;

private:
    PluginFieldCollection m_FC{};
    std::vector<PluginField> m_pluginAttributes{};
};

} //namespace plugin
} // namespace nvinfer1