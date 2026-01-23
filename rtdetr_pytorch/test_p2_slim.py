
import os
import sys
import torch

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

# Import registry
import src.zoo
from src.core import YAMLConfig
from src.zoo.rtdetr.hybrid_encoder_slim import HybridEncoderSlim

def test_p2_slim_setup():
    print("="*60)
    print("Verifying Slim P2 (Standard RepVGG + 128ch P2)")
    print("="*60)

    config_path = 'configs/rtdetr/rtdetr_r18vd_p2_slim.yml'
    
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

    # 3. Inspect Encoder
    print("\nInspecting Encoder...")
    encoder = model.encoder
    if not isinstance(encoder, HybridEncoderSlim):
        print(f"ERROR: Encoder is {type(encoder).__name__}, expected HybridEncoderSlim")
        return

    # Check the critical dimensions
    # P2 should be 128 (Slim), others 256
    expected_channels = [128, 256, 256, 256]
    actual_channels = encoder.out_channels
    print(f"Encoder Output Channels: {actual_channels}")
    
    if actual_channels != expected_channels:
        print(f"ERROR: Channel mismatch. Expected {expected_channels}, got {actual_channels}")
        return
    else:
        print("✓ Channel widths are correct (Slim P2 confirmed).")

    # 4. Forward Pass
    print("\nRunning Forward Pass Test...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    model = model.to(device)
    
    # Batch size 4 test
    dummy_input = torch.randn(4, 3, 640, 640).to(device)
    
    try:
        with torch.no_grad():
            output = model(dummy_input)
        print("Forward pass successful!")
        if 'pred_boxes' in output:
            print(f"Output Boxes: {output['pred_boxes'].shape} (Batch 4 worked!)")
            
    except Exception as e:
        print(f"FAILED Forward Pass: {e}")

    print("\n" + "="*60)
    print("Test Complete")
    print("="*60)

if __name__ == "__main__":
    test_p2_slim_setup()
