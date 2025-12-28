# TensorRT Plugin Kit

A collection of custom TensorRT plugin examples demonstrating various plugin implementations and integration patterns.

## Overview

This repository provides working examples of TensorRT custom plugins using the IPluginV3 interface. Each example includes complete source code, build configuration, and usage demonstrations.

## Structure

```
TrtPluginKit/
├── base/              # Base plugin utilities and interfaces
├── common/            # Shared helper functions
├── cmake/             # CMake modules for finding dependencies
└── plugins/           # Plugin implementations
    ├── deformable_aggregation/
    ├── plugin_example_identity_conv_v3/
    └── relu_torch_tensorrt_example/
```

## Quick Start

### Using Docker (Recommended)

The easiest way to get started is using the provided Docker environment:

```bash
# Build the Docker image
./docker_build.sh

# Run container and build project
./docker_run.sh
```

The container includes all dependencies (TensorRT, CUDA, PyTorch, Torch-TensorRT) and automatically builds the plugins. Your local directory is mounted at `/workspace/TrtPluginKit`.

### Manual Build

#### Prerequisites

- TensorRT 10.0+
- CUDA 11.0+
- CMake 3.22+
- GCC/G++ with C++17 support

#### Compilation

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

Compiled plugins will be placed in `build/plugins/[plugin_name]/`.

## Plugin Examples

### deformable_aggregation
Implements deformable aggregation operation commonly used in vision transformers and detection models. Includes ONNX integration example.

### plugin_example_identity_conv_v3
Reference implementation showing IPluginV3 interface basics with a simple identity convolution operation.

### relu_torch_tensorrt_example
Demonstrates integration with Torch-TensorRT, including custom converter registration to replace PyTorch operations with custom plugins.

## Usage Patterns

### ONNX Workflow

Most plugins follow this pattern:
1. Export PyTorch model to ONNX
2. Load plugin library with TensorRT
3. Build engine from ONNX
4. Run inference in C++

See individual plugin directories for specific examples.

### Torch-TensorRT Workflow

For Python deployment using PyTorch runtime:
1. Load plugin with `registry.load_library()`
2. Register custom converter
3. Compile with `torch_tensorrt.compile()`
4. Run inference or serialize to TorchScript

Note: This produces TorchScript modules with embedded TensorRT engines, not standalone engine files.

## Plugin Interface

All plugins use the TensorRT IPluginV3 interface:

- `IPluginV3OneBuild` - Configuration and build-time methods
- `IPluginV3OneRuntime` - Inference-time methods
- `IPluginCreatorV3One` - Plugin factory

## Testing

Each plugin directory contains test scripts demonstrating usage:

```bash
cd plugins/[plugin_name]/python_test
python create_plugin_network.py
```

## Development

To add a new plugin:

1. Create directory in `plugins/`
2. Implement plugin class inheriting from `IPluginBase`
3. Implement plugin creator inheriting from `IPluginCreatorBase`
4. Add `DEFINE_TRT_PLUGIN_CREATOR` macro
5. Create `CMakeLists.txt` following existing examples
6. Add CUDA kernels as needed

The build system automatically discovers and compiles all plugins in the `plugins/` directory.

## Platform Support

- Linux x86_64 with NVIDIA GPUs
- CUDA compute capability 8.6+ (configurable in CMakeLists.txt)

## License

See LICENSE file for details.

