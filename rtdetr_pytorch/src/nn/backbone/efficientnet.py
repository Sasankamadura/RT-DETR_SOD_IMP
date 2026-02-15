
import torch
import torch.nn as nn 
import torchvision

from src.core import register

__all__ = ['EfficientNet']

@register
class EfficientNet(nn.Module):
    def __init__(self, model_name='efficientnet_b2', pretrained=True, freeze_at=-1, freeze_norm=False):
        super().__init__()
        
        # Load the requested efficientnet model from torchvision
        if model_name == 'efficientnet_b2':
            backbone = torchvision.models.efficientnet_b2(pretrained=pretrained)
            # Correct indices for EfficientNet-B2 (Torchvision)
            # 0: Conv3x3 (s2)
            # 1: MBConv1 (s1, 2 layers)
            # 2: MBConv6 (s2, 2 layers) -> Output Stride 4
            # 3: MBConv6 (s2, 3 layers) -> Output Stride 8 (Start)
            # 4: MBConv6 (s1, 3 layers) -> Output Stride 8 (End) -> C3 (48 channels)
            # 5: MBConv6 (s2, 4 layers) -> Output Stride 16 (Start)
            # 6: MBConv6 (s1, 4 layers) -> Output Stride 16 (End) -> C4 (120 channels)
            # 7: MBConv6 (s2, 5 layers) -> Output Stride 32 (End) -> C5 (352 channels)
            # 8: Conv1x1 (s1) -> Output Stride 32 (1408 channels)
            
            self.return_idx = [4, 6, 7] 
            self.out_channels = [48, 120, 352]
            
        elif model_name == 'efficientnet_b0':
             backbone = torchvision.models.efficientnet_b0(pretrained=pretrained)
             # B0: 3 -> 40 (s8), 6 -> 112 (s16), 8 -> 320 (s32) -- checking indices
             # This is a placeholder, strictly mostly implementing for B2 as requested
             # Indices might differ slightly per model depth, assuming B2 for now.
             self.return_idx = [3, 5, 7] # This might need adjustment for B0
             self.out_channels = [40, 112, 320]
        else:
            raise ValueError(f"Model {model_name} not explicitly supported in this wrapper yet.")

        # Extract features container
        self.features = backbone.features
        
        # Freeze parameters if requested
        if freeze_at >= 0:
            for i in range(min(freeze_at + 1, len(self.features))):
                for p in self.features[i].parameters():
                    p.requires_grad = False
                    
        if freeze_norm:
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    m.eval()
                    # Also freeze affine parameters
                    for param in m.parameters():
                        param.requires_grad = False

    def forward(self, x):
        outs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.return_idx:
                outs.append(x)
        return outs
