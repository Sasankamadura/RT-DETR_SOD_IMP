
import os
import sys
import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

# Import registry via src.zoo to ensure everything is registered
import src.zoo
from src.core import YAMLConfig

def test_p2_gnconv_setup():
    print("="*60)
    print("Verifying P2 + GnConv Integration")
    print("="*60)

    config_path = 'configs/rtdetr/rtdetr_r18vd_p2_gnconv.yml'
    
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
        return

    # 3. Check Components
    print("\nComponent Checks:")
    
    # Check Backbone
    backbone = model.backbone
    print(f"Backbone: {type(backbone).__name__}")
    if hasattr(backbone, 'return_idx'):
        print(f"  Return Indices: {backbone.return_idx} (Should be [0, 1, 2, 3])")
    
    # Check Encoder
    encoder = model.encoder
    print(f"Encoder: {type(encoder).__name__}")
    if hasattr(encoder, 'in_channels'):
        print(f"  In Channels: {encoder.in_channels}")
        print(f"  Feature Strides: {encoder.feat_strides}")
        
    # Check Decoder
    decoder = model.decoder
    print(f"Decoder: {type(decoder).__name__}")
    if hasattr(decoder, 'num_levels'):
        print(f"  Num Levels: {decoder.num_levels} (Should be 4)")

    # 4. Forward Pass Test
    print("\nRunning Forward Pass Test (Dry Run)...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    dummy_input = torch.randn(2, 3, 640, 640).to(device)
    
    try:
        with torch.no_grad():
            output = model(dummy_input)
            
        print("Forward pass successful!")
        
        if 'pred_boxes' in output:
            print(f"Output Boxes Shape: {output['pred_boxes'].shape} (Batch, Queries, 4)")
        if 'pred_logits' in output:
            print(f"Output Logits Shape: {output['pred_logits'].shape} (Batch, Queries, Classes)")
            
    except RuntimeError as e:
        print(f"FAILED Forward Pass: {e}")
        if "out of memory" in str(e):
            print("  -> GPU OOM. P2 fusion is memory intensive.")
    except Exception as e:
        print(f"FAILED Forward Pass: {e}")

    print("\n" + "="*60)
    print("Test Complete")
    print("="*60)

if __name__ == "__main__":
    test_p2_gnconv_setup()
