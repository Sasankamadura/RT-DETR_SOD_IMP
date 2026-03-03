import torch
import torch.nn as nn 
import torchvision

from src.core import register

__all__ = ['EfficientNet']

@register
class EfficientNet(nn.Module):
    def __init__(self, model_name='efficientnet_b2', pretrained=True, freeze_at=-1, freeze_norm=False, return_idx=None, out_channels=None):
        super().__init__()
        
        # Load the requested efficientnet model from torchvision
        if model_name == 'efficientnet_b2':
            backbone = torchvision.models.efficientnet_b2(pretrained=pretrained)
            if return_idx is not None and out_channels is not None:
                self.return_idx = return_idx
                self.out_channels = out_channels
            else:
                self.return_idx = [2, 3, 5, 7] 
                self.out_channels = [24, 48, 120, 352]
            
        elif model_name == 'efficientnet_b0':
             backbone = torchvision.models.efficientnet_b0(pretrained=pretrained)
             if return_idx is not None and out_channels is not None:
                self.return_idx = return_idx
                self.out_channels = out_channels
             else:
                self.return_idx = [2, 3, 5, 7] # This might need adjustment for B0
                self.out_channels = [24, 40, 112, 320]
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
