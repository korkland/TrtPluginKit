import os
import torch
import torch.nn as nn
from typing import List, ClassVar

class TraditionalConv(nn.Module):
    """
    A traditional convolutional layer with a specified number of input and output channels.
    """
    def __init__(self, channels: int):
        super(TraditionalConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=channels,
                              out_channels=channels,
                             kernel_size=(1, 1),
                             stride=(1, 1),
                             padding=(0, 0),
                             groups=channels,
                             bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class IdentityConv(TraditionalConv):
    __constants__ = ["kernel_shape", "strides", "pads", "group"]
    kernel_shape: ClassVar[List[int]]
    strides: ClassVar[List[int]]
    pads: ClassVar[List[int]]
    group: int
    def __init__(self, channels):
        super(IdentityConv, self).__init__(channels)
        self.kernel_shape = list(self.conv.kernel_size)
        self.strides = list(self.conv.stride)
        self.pads = list(self.conv.padding)
        # ONNX expects a list of 4 pad values whereas PyTorch uses a list of 2 pad values.
        self.pads = self.pads + self.pads
        self.group = self.conv.groups

    def forward(self, x):
        # Call the parent class method.
        x = super(IdentityConv, self).forward(x)
        # Apply the identity operation.
        return x

class IdentityNetwork(nn.Module):
    def __init__(self, channels: int):
        super(IdentityNetwork, self).__init__()
        self.conv_in = TraditionalConv(channels)
        self.conv_modified = IdentityConv(channels)
        self.conv_out = TraditionalConv(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        x = self.conv_modified(x)
        x = self.conv_out(x)
        return x

def create_onnx(onnx_path):
    # Create a random input tensor with the shape (1, 3, 224, 224)
    x = torch.randn(1, 3, 224, 224)

    # Create an instance of the IdentityNetwork
    model = IdentityNetwork(channels=3)

    # Set the model to evaluation mode
    model.eval()

    # Export the model to ONNX format
    torch.onnx.export(model=model,
                      args=(x,),
                      f=onnx_path,
                      opset_version=17,
                      input_names=['input'],
                      output_names=['output'],
                      export_modules_as_functions={IdentityConv})

if __name__ == "__main__":
    # Specify the path to save the ONNX model
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # build_dir = os.path.join(current_dir, "/../build")
    # if not os.path.exists(build_dir):
    #     os.makedirs(build_dir)
    # full_onnx_path = os.path.join(build_dir, "identity_network.onnx")

    # Create the ONNX model
    create_onnx("identity_network.onnx")

    # print(f"ONNX model saved to {full_onnx_path}")