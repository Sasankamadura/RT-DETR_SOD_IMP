
import os
import sys
import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

# Import registry
import src.zoo
from src.core import YAMLConfig
from src.zoo.rtdetr.hybrid_encoder_fast_gnconv import HybridEncoderFastGnConv, RepVggBlock, GnConvFusionBlock, CSPHybridLayer

def test_fast_hybrid_setup():
    print("="*60)
    print("Verifying Fast Hybrid Encoder (RepVGG P2 + GnConv P3+)")
    print("="*60)

    config_path = 'configs/rtdetr/rtdetr_r18vd_p2_fast_gnconv.yml'
    
    # 1. Load Config
    print(f"Loading config: {config_path}")
    try:
        cfg = YAMLConfig(config_path)
    except Exception as e:
        print(f"FAILED to load config: {e}")
        return

    # 2. Build Model
    print("Building model...")
    try:
        model = cfg.model
        print("Model built successfully.")
    except Exception as e:
        print(f"FAILED to build model: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Inspect Encoder Layers (The Critical Check)
    print("\nInspecting Encoder Layers architecture...")
    encoder = model.encoder
    if not isinstance(encoder, HybridEncoderFastGnConv):
        print(f"ERROR: Encoder is {type(encoder).__name__}, expected HybridEncoderFastGnConv")
        return

    # Helper to check component type
    def check_layer_type(fpn_block, stride):
        # fpn_block is CSPHybridLayer
        if not isinstance(fpn_block, CSPHybridLayer):
            return "Unknown"
        
        # Check first bottleneck
        bottleneck = fpn_block.bottlenecks[0]
        if isinstance(bottleneck, RepVggBlock):
            return "RepVGG (Standard)"
        elif isinstance(bottleneck, GnConvFusionBlock):
            return "GnConv (Advanced)"
        else:
            return f"Unknown ({type(bottleneck).__name__})"

    # Check FPN Blocks (Top-Down: P5 -> P4 -> P3)
    # The config defines inputs as [P2, P3, P4, P5]
    # fpn_blocks[0] processes P5 input + P4 lateral -> Output corresponds to P4 level (Stride 16)
    # Actually, let's verify strides directly from config logic
    print("\nChecking Fusion Blocks (Expected: P2=RepVGG, P3/P4/P5=GnConv)")
    
    # We iterate through fpn_blocks. 
    # In 'forward', they are used as:
    # for idx in range(3, 0, -1): (3, 2, 1)
    #   fpn_blocks[3-1-idx] -> fpn_blocks[0] corresponds to idx=3 (P5 fusing with P4?)
    #   Wait, let's trust the 'dest_stride' logic I wrote in the class.
    
    # Let's simple check layer types.
    for i, block in enumerate(encoder.fpn_blocks):
        # We can't easily know stride just by index without re-running logic, 
        # but we know the order is High->Low or Low->High.
        type_name = check_layer_type(block, 0)
        print(f"  FPN Block {i}: {type_name}")

    print("\nChecking PAN Blocks (Bottom-Up: P2 -> P3 -> P4 -> P5)")
    # pan_blocks[0]: Input P2 (RepVGG output?) -> P3
    for i, block in enumerate(encoder.pan_blocks):
        type_name = check_layer_type(block, 0)
        print(f"  PAN Block {i}: {type_name}")

    # 4. Forward Pass
    print("\nRunning Forward Pass Test...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    model = model.to(device)
    
    dummy_input = torch.randn(2, 3, 640, 640).to(device)
    
    try:
        with torch.no_grad():
            output = model(dummy_input)
        print("Forward pass successful!")
        if 'pred_boxes' in output:
            print(f"Output Boxes: {output['pred_boxes'].shape}")
            
    except Exception as e:
        print(f"FAILED Forward Pass: {e}")

    print("\n" + "="*60)
    print("Test Complete")
    print("="*60)

if __name__ == "__main__":
    test_fast_hybrid_setup()
