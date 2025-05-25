#pragma once

#include <NvInferRuntime.h>

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

} //namespace plugin
} // namespace nvinfer1