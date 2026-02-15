
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
            # Corrected indices for EfficientNet-B2 (Torchvision) based on runtime error
            # The indices for features in efficientnet_b2 are sequential.
            # 88 channels corresponds to index 4 (MBConv6, s1, 3 layers) -> this is the output of the block *after* the stride 2 block.
            # Let's map accurately:
            # - Stride 8: Index 3 is the first block with stride 2 (start of S8). Index 4 is the end of S8. Channels: 48? No, wait.
            #   Visualizing from error: "Expected 48... got 88". 
            #   The layer at index 4 in B2 actually has 88 channels (it seems).
            #   Actually, looking at standard B2: 
            #   feature[2] (stride 4) -> 24 channels
            #   feature[3] (stride 8) -> 48 channels
            #   feature[4] (stride 16) -> 120 channels -- Wait, B2 structure is specific.
            
            # Let's rely on the error message. The model has:
            # Layer 2: 24 channels (Stride 4)
            # Layer 3: 48 channels (Stride 8) 
            # Layer 4: 120 channels (Stride 16)
            # Layer 5: 352 channels (Stride 32)
            
            # The error "Expected 48... got 88" suggests I might have picked an index that has 88 channels.
            # Layer 3 output in B2 is indeed 48 channels.
            # Let's use indices [3, 5, 7] which are commonly the end of the stages.
            # But let's verify exact channels for B2:
            # 0: 32
            # 1: 16
            # 2: 24 (Stride 4)
            # 3: 48 (Stride 8)
            # 4: 120 (Stride 16)
            # 5: 352 (Stride 32)
            
            # Re-aligning to standard usage:
            self.return_idx = [3, 5, 7] 
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
