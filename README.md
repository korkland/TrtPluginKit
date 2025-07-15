# TensorRT Plugin Kit

A comprehensive library for TensorRT custom plugins with future expansion planned for end-to-end neural network productization from PyTorch to TensorRT engines via ONNX.

## Project Vision

**Current State**: A robust TensorRT plugin development framework that simplifies the creation, testing, and deployment of custom TensorRT plugins.

**Future Vision**: An all-in-one neural network productization library that provides:
- Seamless PyTorch → ONNX → TensorRT pipeline
- Automated optimization and quantization
- Production-ready deployment utilities
- Performance benchmarking and profiling tools

## Features

### Current Features
- **Plugin V3 API Support**: Full implementation of TensorRT's latest plugin interface
- **Modular Architecture**: Clean separation of concerns with base classes and common utilities
- **CUDA Kernel Integration**: Optimized CUDA implementations for high-performance inference
- **Type Safety**: Comprehensive data type validation and format checking
- **Example Plugins**: Ready-to-use implementations including:
  - DeformableAggregation: Advanced deformable convolution aggregation
  - IdentityConv: Example plugin demonstrating plugin development patterns

### Planned Features
- **PyTorch Integration**: Direct model conversion utilities
- **ONNX Optimization**: Graph optimization and simplification tools
- **Quantization Support**: INT8/FP16 quantization pipelines
- **Deployment Tools**: Production-ready inference utilities
- **Performance Profiling**: Comprehensive benchmarking suite

## Project Structure

```
TrtPluginKit/
├── base/                          # Base plugin interfaces
│   └── IPluginBase.h             # Common plugin base classes
├── common/                        # Shared utilities
│   ├── common.h                  # Common macros and utilities
│   └── common.cpp                # Logger and error handling
├── plugins/                       # Plugin implementations
│   ├── deformable_aggregation/   # Deformable aggregation plugin
│   │   ├── DeformableAggregationPlugin.h
│   │   ├── DeformableAggregationPlugin.cpp
│   │   ├── DeformableAggregationKernel.cu
│   │   └── python_test/          # Python test utilities
│   └── plugin_example_identity_conv_v3/  # Example plugin
├── cmake/                         # CMake modules
│   └── FindTensorRT.cmake        # TensorRT detection
└── CMakeLists.txt                # Main build configuration
```

## Prerequisites

- **CUDA Toolkit**: Version 11.0 or higher
- **TensorRT**: Version 10.1.0.27 or higher
- **CMake**: Version 3.22 or higher
- **C++ Compiler**: Supporting C++17 standard
- **Python**: 3.8+ (for testing utilities)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/korkland/TrtPluginKit.git
cd TrtPluginKit
```

### 2. Setup TensorRT
```bash
# Create external directory and place TensorRT
mkdir -p external
# Download and extract TensorRT to external/TensorRT-10.1.0.27
```

### 3. Build the Project
```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

## Testing

### Python Test Environment
```bash
# Create Python environment
python -m venv venv
source venv/bin/activate
pip install torch torchvision onnx
```

### Running Plugin Tests
```bash
# Generate test ONNX models
cd plugins/deformable_aggregation/python_test
python create_plugin_network.py

cd plugins/plugin_example_identity_conv_v3/python_test
python create_plugin_network.py
```

## Plugin Development Guide

### Creating a New Plugin

1. **Create Plugin Directory**
```bash
mkdir plugins/your_plugin_name
cd plugins/your_plugin_name
```

2. **Implement Plugin Classes**
```cpp
// YourPlugin.h
#include "IPluginBase.h"

class YourPlugin : public IPluginBase,
                   public IPluginV3OneBuild,
                   public IPluginV3OneRuntime
{
    // Implement required methods
};

class YourPluginCreator : public IPluginCreatorBase
{
    // Implement plugin creation logic
};
```

3. **Add CMakeLists.txt**
```cmake
get_filename_component(PLUGIN_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(TARGET_NAME ${CMAKE_PROJECT_NAME}_${PLUGIN_NAME})
# ... rest of CMake configuration
```

4. **Register Plugin**
```cpp
DEFINE_TRT_PLUGIN_CREATOR(YourPluginCreator);
```

### Plugin Architecture

- **IPluginBase**: Common base class providing plugin metadata
- **IPluginV3OneBuild**: Build-time configuration and validation
- **IPluginV3OneRuntime**: Runtime execution interface
- **IPluginCreatorBase**: Plugin factory for creation and serialization

## Example Usage

### DeformableAggregation Plugin
```python
import torch
import torch.nn as nn

class DeformableAggregation(nn.Module):
    def forward(self, mc_ms_feat, spatial_shape, scale_start_index,
                sampling_location, weights):
        # Plugin implementation
        return result
```

### ONNX Export
```python
torch.onnx.export(
    model=model,
    args=inputs,
    f="model.onnx",
    opset_version=17,
    export_modules_as_functions={DeformableAggregation}
)
```

## Roadmap

### Phase 1: Plugin Framework (Current)
- [x] Plugin V3 API implementation
- [x] CUDA kernel integration
- [x] Example plugins
- [ ] Comprehensive testing suite
- [ ] Documentation improvements

### Phase 2: ONNX Integration
- [ ] ONNX graph optimization
- [ ] Custom op registration
- [ ] Model validation utilities
- [ ] Conversion pipeline

### Phase 3: Productization Suite
- [ ] PyTorch model analysis
- [ ] Automated optimization
- [ ] Quantization support
- [ ] Deployment utilities
- [ ] Performance benchmarking

### Phase 4: Advanced Features
- [ ] Multi-precision support
- [ ] Dynamic shape handling
- [ ] Batch processing optimization
- [ ] Cloud deployment tools

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.