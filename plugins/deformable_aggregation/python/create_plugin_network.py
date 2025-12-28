import torch
import torch.nn as nn

class DeformableAggregation(nn.Module):
    """
    A plugin that performs deformable aggregation on the input tensor.
    """
    def __init__(self):
        super(DeformableAggregation, self).__init__()

    def forward(self,
                mc_ms_feat: torch.Tensor,
                spatial_shape: torch.Tensor,
                scale_start_index: torch.Tensor,
                sampling_location: torch.Tensor,
                weights: torch.Tensor
                ) -> torch.Tensor:
        return torch.randn((mc_ms_feat.shape[0], sampling_location.shape[1], mc_ms_feat.shape[2]),
                           dtype=mc_ms_feat.dtype, device=mc_ms_feat.device)

def create_onnx(onnx_path):

    batch_size = 1
    num_cams = 6
    num_feat = 89760
    num_embeds = 256
    num_scales = 4
    num_anchors = 900
    num_pts = 13
    num_groups = 8
    mc_ms_feat = torch.randn((batch_size, num_feat, num_embeds))
    spatial_shape = torch.zeros((num_cams, num_scales, 2), dtype=torch.int32)
    scale_start_index = torch.zeros((num_cams, num_scales), dtype=torch.int32)
    sampling_location = torch.randn((batch_size, num_anchors, num_pts, num_cams, 2))
    weights = torch.randn((batch_size, num_anchors, num_pts, num_cams, num_scales, num_groups))

    model = DeformableAggregation()
    model.eval()

    # Export the model to ONNX format
    torch.onnx.export(model=model,
                      args=(mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights),
                      f=onnx_path,
                      opset_version=17,
                      input_names=['mc_ms_feat',
                                   'spatial_shape',
                                   'scale_start_index',
                                   'sampling_location',
                                   'weights'],
                      output_names=['output'],
                      export_modules_as_functions={DeformableAggregation})

if __name__ == "__main__":

    # Create the ONNX model
    create_onnx("deformable_aggregation.onnx")

    # print(f"ONNX model saved to {full_onnx_path}")
