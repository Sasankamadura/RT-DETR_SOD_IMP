'''
HybridEncoder with "Slack Fast Hybrid" strategy.
- P2 (Stride 4): RepVggBlock + SLIM Channels (e.g. 128)
- P3-P5 (Stride 8+): GnConvFusionBlock + Standard Channels (e.g. 256)
'''

import copy
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from .utils import get_activation

from src.core import register
from .gnconv_module import gnConv, LayerNorm

__all__ = ['HybridEncoderSlimFast']


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias 

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class GnConvFusionBlock(nn.Module):
    """
    Fusion Block using gnConv.
    """
    def __init__(self, dim, order=4):
        super().__init__()
        self.norm = LayerNorm(dim, data_format='channels_first')
        self.gnconv = gnConv(dim, order=order)
        
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.gnconv(x)
        x = x + residual
        return x


class CSPHybridLayer(nn.Module):
    """
    CSP Layer that switches between RepVggBlock and GnConvFusionBlock.
    Also handles input/output channels which might not be uniform in 'Slim' mode.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu",
                 gnconv_order=4,
                 use_gnconv=False):
        super(CSPHybridLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        
        # CSP projection: splitting input into two branches
        # Input 'x' has 'in_channels'.
        
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        
        if use_gnconv:
            self.bottlenecks = nn.Sequential(*[
                GnConvFusionBlock(hidden_channels, order=gnconv_order) for _ in range(num_blocks)
            ])
        else:
            # Use Standard RepVggBlock
            self.bottlenecks = nn.Sequential(*[
                RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
            ])
        
        if hidden_channels != out_channels:
            # Note: This is usually 1x1 conv to merge branches back if needed, 
            # In CSP, branches are merged by concatenation, THEN projected?
            # Wait, standard CSP: 
            # y1 = conv1(x) -> bottleneck(y1)
            # y2 = conv2(x)
            # y = conv3(concat(y1, y2))
            
            # Here implementation seems to be:
            # forward: return self.conv3(x_1 + x_2) ?? 
            # Wait, in original 'CSPRepLayer', it was x_1 + x_2. 
            # Let's check 'forward' in standard implementation.
            # Yes: return self.conv3(x_1 + x_2).
            # So hidden_channels MUST be equal for x_1 and x_2.
            # And output of conv3 is out_channels.
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


# transformer (Standard)
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation) 

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)
        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)
        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)
        if self.norm is not None:
            output = self.norm(output)
        return output


@register
class HybridEncoderSlimFast(nn.Module):
    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward = 1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None,
                 gnconv_order=4,
                 min_stride_for_gnconv=8,
                 gnconv_rule='greater_or_equal',  # New arg: 'greater_or_equal' (default) or 'less_than'
                 p2_hidden_dim=128):   # New arg for Slim P2
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.gnconv_rule = gnconv_rule
        
        # Calculate dimensions per level
        # Logic to determine if a level uses GnConv or not, and thus if it is "slim" or not?
        # Actually dims_per_level depends on "slim" logic which was tied to P2 (stride < 8).
        # We should probably keep the dim logic consistent with the original "Slim P2" intent 
        # (reduce dim for high-res), but the USER request specifically asked for GnConv on P2.
        # Implied: P2 is (Stride 4). 
        # If gnconv_rule='less_than' and min_stride=8 => P2 uses GnConv.
        
        self.dims_per_level = []
        for s in feat_strides:
            # We keep the dimension reduction logic TIED to the Stride < 8 assumption for "Slim" nature, 
            # OR we should update this too? 
            # The user request said "SLIM P2... vs GnConv". 
            # Usually GnConv is heavier, so we might want 128 dim for it if it's on P2.
            # Let's keep the dimension logic as "Stride < 8 => p2_hidden_dim" regardless of block type.
            if s < 8: # Hardcoded 8 or derived? Original used min_stride_for_gnconv=8. 
                      # But if we change rule, we shouldn't break dim logic.
                      # Let's assume Stride 4 is ALWAYS the "Slim" layer candidate.
                self.dims_per_level.append(p2_hidden_dim)
            else:
                self.dims_per_level.append(hidden_dim)
        
        self.hidden_dim = hidden_dim # Used for Transformer (AIFI) on P5
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size

        # Output channels for Decoder to consume
        self.out_channels = self.dims_per_level
        self.out_strides = feat_strides
        
        # 1. Channel Projection
        self.input_proj = nn.ModuleList()
        for i, in_channel in enumerate(in_channels):
            # Project to the specific dim for this level
            target_dim = self.dims_per_level[i]
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, target_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(target_dim)
                )
            )

        # 2. Encoder Transformer (AIFI) - Applies to P5 (index 3 usually)
        # Note: AIFI is usually applied to the highest level (P5).
        # We assume P5 uses 'hidden_dim' (256), so this remains standard.
        encoder_layer = TransformerEncoderLayer(
            hidden_dim, 
            nhead=nhead,
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation=enc_act)

        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])

        # 3. Top-Down FPN
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        
        # Processing P5->P4->P3->P2
        # idx loops over: 3, 2, 1
        for idx in range(len(in_channels) - 1, 0, -1):
            
            high_level_idx = idx
            low_level_idx = idx - 1
            
            high_dim = self.dims_per_level[high_level_idx]
            low_dim = self.dims_per_level[low_level_idx]
            
            # lateral_conv: transforms feature from high_level to low_level dimensions for fusion
            self.lateral_convs.append(ConvNormLayer(high_dim, low_dim, 1, 1, act=act))
            
            # Fusion Logic:
            # upsample(lateral_conv(high)) -> shape matches low_dim
            # contact(upsample, low) -> 2 * low_dim input
            
            dest_stride = feat_strides[low_level_idx]
            
            # GnConv Logic Selection
            if self.gnconv_rule == 'less_than':
                use_gnconv = dest_stride < min_stride_for_gnconv
            else: # default 'greater_or_equal'
                use_gnconv = dest_stride >= min_stride_for_gnconv
            
            self.fpn_blocks.append(
                CSPHybridLayer(in_channels = low_dim * 2, 
                               out_channels = low_dim, 
                               num_blocks = round(3 * depth_mult), 
                               act=act, 
                               expansion=expansion, 
                               gnconv_order=gnconv_order, 
                               use_gnconv=use_gnconv)
            )

        # 4. Bottom-Up PAN
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        
        # Processing P2->P3->P4->P5
        # idx loops over: 0, 1, 2
        for idx in range(len(in_channels) - 1):
            
            low_level_idx = idx
            high_level_idx = idx + 1
            
            low_dim = self.dims_per_level[low_level_idx]
            high_dim = self.dims_per_level[high_level_idx]
            
            # downsample: transforms feature from low_level to high_level dimensions
            # Usually 3x3 s2. Input channels = low_dim, Output = high_dim.
            self.downsample_convs.append(
                ConvNormLayer(low_dim, high_dim, 3, 2, act=act)
            )
            
            # Fusion:
            # concat(downsample, high) -> high_dim + high_dim = 2*high_dim
            
            dest_stride = feat_strides[high_level_idx]
            
            # GnConv Logic Selection
            if self.gnconv_rule == 'less_than':
                use_gnconv = dest_stride < min_stride_for_gnconv
            else: # default 'greater_or_equal'
                use_gnconv = dest_stride >= min_stride_for_gnconv

            self.pan_blocks.append(
                CSPHybridLayer(in_channels = high_dim * 2,
                               out_channels = high_dim,
                               num_blocks = round(3 * depth_mult), 
                               act=act, 
                               expansion=expansion, 
                               gnconv_order=gnconv_order, 
                               use_gnconv=use_gnconv)
            )

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        
        # encoder (AIFI) - Applies to P5 (index 3, assume hidden_dim 256)
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        
        # Top-Down
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            
            # lateral_conv adapts high channels to low channels (e.g 256->128)
            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_high)
            # inner_outs[0] = feat_high  <-- BUG FIX: Do NOT overwrite the high-level feature (256) with the projected one (128).
            # We need the original 256-ch feature for the Bottom-Up PAN path later.
            
            upsample_feat = F.interpolate(feat_high, scale_factor=2., mode='nearest')
            
            # concat (low + upsample_high) -> 128 + 128 = 256
            inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        # Bottom-Up
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            
            # downsample adapts low channels to high channels (e.g. 128->256)
            downsample_feat = self.downsample_convs[idx](feat_low)
            
            # concat (downsample_low + high) -> 256 + 256 = 512
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_high], dim=1))
            outs.append(out)

        return outs
