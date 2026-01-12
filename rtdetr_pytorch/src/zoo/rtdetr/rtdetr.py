"""by lyuwenyu
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 

from src.core import register


__all__ = ['RTDETR', ]


@register
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale
        
    def forward(self, x, targets=None):
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])
            
        x = self.backbone(x)
        
        # NOTE: Research - Extract P2 if available
        # Backbone usually returns [P3, P4, P5] or [P2, P3, P4, P5] depending on config.
        # We need to ensure backbone returns P2. 
        # If backbone returns 3 feats, we assume they are P3, P4, P5.
        # If 4, likely P2, P3, P4, P5.
        
        p2_feat = None
        if len(x) == 4:
             p2_feat = x[0]
             encoder_feats = x[1:]
        else:
             encoder_feats = x

        x_enc = self.encoder(encoder_feats)        
        x_dec = self.decoder(x_enc, targets, p2_feat=p2_feat) # Pass P2

        return x_dec
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
