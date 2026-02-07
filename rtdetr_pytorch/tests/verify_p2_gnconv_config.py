
import sys
import os
import torch
import torch.nn as nn
from unittest.mock import MagicMock

# Mock torchvision.datapoints to avoid ImportError
import torchvision
if not hasattr(torchvision, 'datapoints'):
    torchvision.datapoints = MagicMock()

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core import YAMLConfig

def verify_config():
    config_path = 'configs/rtdetr/rtdetr_r18vd_p2_gnconv_repvgg.yml'
    try:
        cfg = YAMLConfig(config_path)
        model = cfg.model
        encoder = model.encoder
        
        print(f"Encoder Class: {type(encoder).__name__}")
        
        # P2 corresponds to index 2 in `fpn_blocks` (since it processes P5->P4->P3->P2)
        # Let's inspect the blocks
        # in_channels=[256, 512, 1024], we process high to low.
        # fpn_blocks[0]: P5 input -> P4 output
        # fpn_blocks[1]: P4 input -> P3 output
        # fpn_blocks[2]: P3 input -> P2 output
        
        # Actually proper mapping:
        # P5 is top.
        # The FPN loop goes:
        # top_down P5 -> P4 (idx 0)
        # top_down P4 -> P3 (idx 1)
        # top_down P3 -> P2 (idx 2)
        
        # Let's check the bottlenecks in each CSPHybridLayer
        
        # Helper to check block type
        def check_block_type(layer, name):
            bottleneck = layer.bottlenecks[0]
            block_type = type(bottleneck).__name__
            print(f"{name} Block Type: {block_type}")
            return block_type

        print("\n--- Top-Down Path (FPN) ---")
        check_block_type(encoder.fpn_blocks[0], "P5->P4")
        check_block_type(encoder.fpn_blocks[1], "P4->P3")
        p2_block = check_block_type(encoder.fpn_blocks[2], "P3->P2 (Target: GnConv)")
        
        print("\n--- Bottom-Up Path (PAN) ---")
        # PAN loop:
        # P2 -> P3 (idx 0)
        # P3 -> P4 (idx 1)
        # P4 -> P5 (idx 2)
        
        check_block_type(encoder.pan_blocks[0], "P2->P3")
        check_block_type(encoder.pan_blocks[1], "P3->P4")
        check_block_type(encoder.pan_blocks[2], "P4->P5")
        
        if p2_block == 'GnConvFusionBlock':
            print("\nSUCCESS: P2 layer is using GnConvFusionBlock as requested.")
        else:
            print(f"\nFAILURE: P2 layer is using {p2_block}, expected GnConvFusionBlock.")
            
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_config()
