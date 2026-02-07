import sys
import os
import torch
import torch.nn as nn

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock dependencies to avoid cryptic ImportErrors during verification
import types
import sys
from unittest.mock import MagicMock

# Mock pycocotools
sys.modules['pycocotools'] = MagicMock()
sys.modules['pycocotools.mask'] = MagicMock()

# Mock src.data to avoid loading dataset code
sys.modules['src.data'] = MagicMock()
sys.modules['src.data.coco'] = MagicMock()
sys.modules['src.data.coco.coco_dataset'] = MagicMock()

import torchvision
if not hasattr(torchvision, 'datapoints'):
    torchvision.datapoints = types.SimpleNamespace()

from src.zoo.rtdetr.hybrid_encoder_slim_fast import HybridEncoderSlimFast, GnConvFusionBlock, RepVggBlock, CSPHybridLayer

def verify_model():
    print("Verifying HybridEncoderSlimFast with Inverted GnConv Logic...")
    
    # Configuration mirroring 'rtdetr_r18vd_p2_slim_gnconv_reversed.yml'
    config = {
        'in_channels': [64, 128, 256, 512],
        'feat_strides': [4, 8, 16, 32],
        'hidden_dim': 256,
        'use_encoder_idx': [3],
        'num_encoder_layers': 1,
        'nhead': 8,
        'dim_feedforward': 1024,
        'dropout': 0.,
        'enc_act': 'gelu',
        'expansion': 1.0,
        'depth_mult': 1.0,
        'act': 'silu',
        'gnconv_order': 4,
        'min_stride_for_gnconv': 8,
        'gnconv_rule': 'less_than', # <--- The key parameter
        'p2_hidden_dim': 128 
    }

    try:
        model = HybridEncoderSlimFast(**config)
    except Exception as e:
        print(f"Failed to instantiate model: {e}")
        return

    print(f"\nModel Config: gnconv_rule='{model.gnconv_rule}', min_stride_for_gnconv={config['min_stride_for_gnconv']}")
    
    # We want to check P2 (stride 4), P3 (stride 8), P4 (stride 16), P5 (stride 32)
    # The encoder separates "Top-Down" FPN blocks and "Bottom-Up" PAN blocks.
    
    # FPN Blocks (Top-Down): P5 -> P4 -> P3 -> P2
    # The loop in init was: range(len - 1, 0, -1) -> 3, 2, 1
    # block 0 correlates to processing for P4? No.
    # Let's inspect the blocks directly in the order they were added.
    
    # Helper to identify block type
    def get_block_type(csp_layer):
        if not isinstance(csp_layer, CSPHybridLayer):
            return "Unknown"
        
        # Check first bottleneck
        if len(csp_layer.bottlenecks) > 0:
            first_block = csp_layer.bottlenecks[0]
            if isinstance(first_block, GnConvFusionBlock):
                return "GnConv"
            elif isinstance(first_block, RepVggBlock):
                return "RepVGG"
            else:
                return str(type(first_block))
        return "Empty"

    print("\n--- Top-Down FPN Blocks ---")
    # Blocks correspond to logic in: for idx in range(len(in_channels) - 1, 0, -1)
    # idx=3 (P5->P4), idx=2 (P4->P3), idx=1 (P3->P2)
    # So fpn_blocks[0] is dest P4 (stride 16)
    # fpn_blocks[1] is dest P3 (stride 8)
    # fpn_blocks[2] is dest P2 (stride 4)
    
    fpn_dest_strides = [16, 8, 4] 
    
    for i, block in enumerate(model.fpn_blocks):
        stride = fpn_dest_strides[i]
        btype = get_block_type(block)
        print(f"Block {i} (Dest Stride {stride}): {btype}")

    print("\n--- Bottom-Up PAN Blocks ---")
    # Blocks correspond to logic in: for idx in range(len(in_channels) - 1)
    # idx=0 (P2->P3), idx=1 (P3->P4), idx=2 (P4->P5)
    # pan_blocks[0] is dest P3 (stride 8)
    # pan_blocks[1] is dest P4 (stride 16)
    # pan_blocks[2] is dest P5 (stride 32)
    
    pan_dest_strides = [8, 16, 32]
    
    for i, block in enumerate(model.pan_blocks):
        stride = pan_dest_strides[i]
        btype = get_block_type(block)
        print(f"Block {i} (Dest Stride {stride}): {btype}")

    print("\n---------------------------------------------------")
    print("Summary Verification:")
    print("Feature Level | Stride | Expected | Actual")
    print("---------------------------------------------------")
    
    # Mapping results
    # P2 (Stride 4) is fpn_blocks[2]
    p2_actual = get_block_type(model.fpn_blocks[2])
    print(f"P2            | 4      | GnConv   | {p2_actual}")
    
    # P3 (Stride 8) is fpn_blocks[1] AND pan_blocks[0]
    p3_fpn = get_block_type(model.fpn_blocks[1])
    p3_pan = get_block_type(model.pan_blocks[0])
    print(f"P3 (FPN)      | 8      | RepVGG   | {p3_fpn}")
    print(f"P3 (PAN)      | 8      | RepVGG   | {p3_pan}")

    # P4 (Stride 16) is fpn_blocks[0] AND pan_blocks[1]
    p4_fpn = get_block_type(model.fpn_blocks[0])
    p4_pan = get_block_type(model.pan_blocks[1])
    print(f"P4 (FPN)      | 16     | RepVGG   | {p4_fpn}")
    print(f"P4 (PAN)      | 16     | RepVGG   | {p4_pan}")
    
    # P5 (Stride 32) is pan_blocks[2]
    p5_actual = get_block_type(model.pan_blocks[2])
    print(f"P5            | 32     | RepVGG   | {p5_actual}")
    print("---------------------------------------------------\n")

if __name__ == "__main__":
    verify_model()
