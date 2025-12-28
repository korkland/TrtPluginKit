# Deformable Aggregation Plugin

Custom TensorRT plugin implementing deformable aggregation operation for vision models.

## Overview

This plugin provides an optimized CUDA implementation of deformable aggregation, commonly used in deformable attention mechanisms and modern detection architectures.

## Files

- `DeformableAggregationPlugin.h/cpp` - Plugin implementation
- `DeformableAggregationKernel.cu/h` - CUDA kernels
- `DeformableAggregationParameters.h` - Plugin parameters
- `python_test/` - Test scripts and examples

## Building

Built automatically as part of the main project:

```bash
cd /path/to/TrtPluginKit/build
make
```

Output: `build/plugins/deformable_aggregation/libtrt_plugin_kit_deformable_aggregation.so`

## Usage

### With ONNX

1. Export PyTorch model to ONNX with deformable aggregation ops
2. Load plugin library before building engine
3. Build TensorRT engine from ONNX
4. Run inference

Example workflow is in `python_test/create_plugin_network.py`.

### Direct TensorRT API

```python
import tensorrt as trt

# Load plugin
registry = trt.get_plugin_registry()
registry.load_library("libtrt_plugin_kit_deformable_aggregation.so")

# Get plugin creator
creator = registry.get_creator("DeformableAggregation", "1")

# Create plugin with parameters
plugin = creator.create_plugin(name, field_collection, trt.TensorRTPhase.BUILD)

# Add to network
layer = network.add_plugin_v3(inputs, shape_inputs, plugin)
```

## Parameters

See `DeformableAggregationParameters.h` for configuration options.

## Requirements

- TensorRT 10.0+
- CUDA 11.0+
- Compatible with FP32 and FP16 precision

