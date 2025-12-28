# Custom ReLU Plugin with Torch-TensorRT

Example demonstrating custom TensorRT plugin integration with Torch-TensorRT for Python deployment.

## Overview

This example shows how to replace PyTorch operations with custom TensorRT plugins during torch_tensorrt compilation. The plugin implements a simple ReLU activation to demonstrate the integration pattern.

## Important Note

Torch-TensorRT produces TorchScript modules with embedded TensorRT engines, not standalone `.engine` files. This is suitable for Python or LibTorch C++ deployment, but not for pure TensorRT C++ applications. For pure TensorRT deployment, use the ONNX workflow shown in other examples.

## Files

- `CustomReLUPlugin.h/cpp` - Plugin implementation (IPluginV3)
- `CustomReLUPluginKernel.cu/h` - CUDA kernel
- `python/convert_to_trt.py` - Torch-TensorRT integration example

## Building

```bash
cd /path/to/TrtPluginKit/build
make
```

Output: `build/plugins/relu_torch_tensorrt_example/libtrt_plugin_kit_relu_torch_tensorrt_example.so`

## Usage

```bash
cd plugins/relu_torch_tensorrt_example/python
python convert_to_trt.py
```

This will:
1. Load the custom ReLU plugin
2. Register a converter to replace torch.relu operations
3. Compile a PyTorch model to TensorRT
4. Run inference and validate accuracy
5. Save the model as TorchScript

## How It Works

The key steps for Torch-TensorRT integration:

### 1. Load Plugin

```python
registry = trt.get_plugin_registry()
registry.load_library(plugin_path)
```

Critical: Use `registry.load_library()`, not `ctypes.CDLL()`.

### 2. Register Converter

```python
from torch_tensorrt.dynamo.conversion import (
    dynamo_tensorrt_converter,
    ConversionContext,
    ConverterPriority
)

@dynamo_tensorrt_converter(
    torch.ops.aten.relu.default,
    priority=ConverterPriority.HIGH,
    supports_dynamic_shapes=True
)
def convert_relu_to_plugin(ctx, target, args, kwargs, name):
    creator = trt.get_plugin_registry().get_creator("CustomReLUPlugin", "1")
    plugin = creator.create_plugin(
        "custom_relu",
        trt.PluginFieldCollection([]),
        trt.TensorRTPhase.BUILD
    )
    layer = ctx.net.add_plugin_v3([args[0]], [], plugin)
    return layer.get_output(0)
```

### 3. Compile Model

```python
trt_model = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input(shape=(1, 3, 224, 224), dtype=torch.float32)],
    enabled_precisions={torch.float32},
)
```

### 4. Save and Load

```python
# Save
torch_tensorrt.save(trt_model, "model.ts", inputs=[input])

# Load (plugin must be loaded first)
registry.load_library(plugin_path)
model = torch_tensorrt.load("model.ts")
```

## Key Points

- IPluginV3 `create_plugin()` requires 3 parameters including `TensorRTPhase`
- Use `ConverterPriority.HIGH` to override default converters
- Plugin must be loaded before model loading for deserialization
- Output is TorchScript module, not pure TensorRT engine

## Deployment

The resulting `.ts` file can be deployed in:
- Python with PyTorch runtime
- C++ with LibTorch

It cannot be used with standalone TensorRT C++ runtime.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Torch-TensorRT 2.9+
- TensorRT 10.0+
- CUDA 11.0+

## Notes

This example uses ReLU for demonstration. In practice, custom plugins are most useful for operations not natively supported by TensorRT or for specialized optimizations.
