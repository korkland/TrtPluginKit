import torch
import torch.nn as nn
import torch_tensorrt
from torch_tensorrt.dynamo.conversion import dynamo_tensorrt_converter, ConversionContext
from typing import Tuple, Dict, Union, Sequence, Any
import tensorrt as trt
from pathlib import Path

# Simple model using standard ReLU
class SimpleModel(nn.Module):
    """Simple model using standard ReLU"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)  # Will be replaced with CustomReLUPlugin

        x = self.conv2(x)
        x = torch.relu(x)  # Will be replaced with CustomReLUPlugin

        x = self.conv3(x)
        x = torch.relu(x)  # Will be replaced with CustomReLUPlugin

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def load_plugin_library():
    """Load the custom ReLU plugin library"""

    plugin_path = Path(__file__).parent.parent.parent.parent / \
                  "build/plugins/relu_torch_tensorrt_example/libtrt_plugin_kit_relu_torch_tensorrt_example.so"

    if not plugin_path.exists():
        raise RuntimeError(f"Plugin not found at {plugin_path}")

    # Load the library using TensorRT's registry
    registry = trt.get_plugin_registry()

    # Use load_library to register plugins from the shared library
    result = registry.load_library(str(plugin_path))

    if not result:
        raise RuntimeError(f"Failed to load plugin library from {plugin_path}")

    print(f" Loaded plugin library: {plugin_path.name}")

    # Verify plugin is registered
    creator = registry.get_creator("CustomReLUPlugin", "1")
    if creator:
        print(f" CustomReLUPlugin registered successfully!")
    else:
        raise RuntimeError("Failed to register CustomReLUPlugin")

def register_custom_relu_converter():
    """Register converter to replace torch.relu with CustomReLUPlugin"""

    # This decorator will register the converter when the function is defined
    # Use HIGH priority to override the default ReLU converter
    from torch_tensorrt.dynamo.conversion import ConverterPriority

    @dynamo_tensorrt_converter(
        torch.ops.aten.relu.default,
        priority=ConverterPriority.HIGH,
        supports_dynamic_shapes=True
    )
    def convert_relu_to_custom_plugin(
        ctx: ConversionContext,
        target: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        name: str,
    ) -> Union[trt.ITensor, Sequence[trt.ITensor]]:
        """Convert torch.relu to CustomReLUPlugin"""

        # Get the input tensor (first argument)
        input_tensor = args[0]

        # Get the TensorRT network from context
        network = ctx.net

        # Get plugin creator from registry
        plugin_registry = trt.get_plugin_registry()
        plugin_creator = plugin_registry.get_creator("CustomReLUPlugin", "1")

        if plugin_creator is None:
            print(f" Warning: CustomReLUPlugin not found, using standard ReLU")
            # Fallback to standard ReLU
            layer = network.add_activation(input_tensor, trt.ActivationType.RELU)
            layer.name = name
            return layer.get_output(0)

        # Create plugin instance
        # IPluginCreatorV3One requires phase parameter
        plugin_field_collection = trt.PluginFieldCollection([])
        plugin = plugin_creator.create_plugin(
            "custom_relu",
            plugin_field_collection,
            trt.TensorRTPhase.BUILD
        )

        if plugin is None:
            print(f" Warning: Failed to create plugin, using standard ReLU")
            layer = network.add_activation(input_tensor, trt.ActivationType.RELU)
            layer.name = name
            return layer.get_output(0)

        # Add plugin layer to network
        plugin_layer = network.add_plugin_v3([input_tensor], [], plugin)
        plugin_layer.name = f"{name}_custom_relu_plugin"

        print(f"   Converted {name} to CustomReLUPlugin")

        return plugin_layer.get_output(0)

    print(" Registered CustomReLU converter with HIGH priority")

def main():
    """Convert PyTorch model to TensorRT with custom plugin"""

    # 1. Load plugin
    print("Loading plugin...")
    load_plugin_library()

    # 2. Register converter
    print("\nRegistering converter...")
    register_custom_relu_converter()

    # 3. Create model
    print("\nCreating model...")
    model = SimpleModel().eval().cuda()

    # 4. Create sample input
    dummy_input = torch.randn(1, 3, 224, 224, device='cuda')

    # 5. Get PyTorch baseline
    print("\nPyTorch inference...")
    with torch.no_grad():
        pytorch_out = model(dummy_input)
    print(f"Output: {pytorch_out[0, :5]}")

    # 6. Compile to TensorRT (will use custom plugin)
    print("\nCompiling to TensorRT with CustomReLUPlugin...")

    try:
        trt_model = torch_tensorrt.compile(
            model,
            inputs=[torch_tensorrt.Input(
                shape=(1, 3, 224, 224),
                dtype=torch.float32
            )],
            enabled_precisions={torch.float32},
            truncate_long_and_double=True,
            debug=True,  # Enable debug output
        )

        print("\n Compilation successful!")

        # 7. Test TensorRT model
        print("\nTensorRT inference...")
        with torch.no_grad():
            trt_out = trt_model(dummy_input)
        print(f"Output: {trt_out[0, :5]}")

        # 8. Compare
        max_diff = torch.abs(pytorch_out - trt_out).max().item()
        print(f"\nMax difference: {max_diff:.6f}")

        if max_diff < 1e-3:
            print(" Results match!")
        else:
            print(f" Warning: difference is {max_diff}")

        # 9. Save the TensorRT model
        print("\nSaving TensorRT model...")
        output_path = Path("model_trt_custom_plugin.ts")
        torch_tensorrt.save(trt_model, output_path, inputs=[dummy_input])
        print(f" Saved to {output_path}")
        print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

        return True

    except Exception as e:
        print(f"\n✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)