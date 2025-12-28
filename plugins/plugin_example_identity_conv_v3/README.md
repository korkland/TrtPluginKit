# Identity Convolution Plugin (IPluginV3 Example)

Reference implementation demonstrating TensorRT IPluginV3 interface basics.

## Overview

This is a minimal example plugin that performs identity convolution. It serves as a template for understanding the IPluginV3 interface structure and requirements.

## Files

- `IdentityConvPluginV3.h/cpp` - Plugin implementation
- `IdentityConvParameters.h` - Plugin parameters
- `python_test/create_plugin_network.py` - ONNX export example

## Building

Built as part of the main project:

```bash
cd /path/to/TrtPluginKit/build
make
```

Output: `build/plugins/plugin_example_identity_conv_v3/libtrt_plugin_kit_plugin_example_identity_conv_v3.so`

## Usage

This plugin is designed for ONNX workflow. The test script shows how to:

1. Define custom module in PyTorch
2. Export to ONNX with custom operation
3. Load plugin when building TensorRT engine

See `python_test/create_plugin_network.py` for complete example.

## Key Features

- Demonstrates IPluginV3OneBuild interface
- Demonstrates IPluginV3OneRuntime interface
- Shows proper plugin creator implementation
- Example of plugin parameter handling

## Notes

This is an educational example. For production use, standard TensorRT convolution layers are recommended.

## Requirements

- TensorRT 10.0+
- CUDA 11.0+

