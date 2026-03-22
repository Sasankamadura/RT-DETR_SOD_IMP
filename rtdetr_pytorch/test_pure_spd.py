import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

try:
    from src.zoo.rtdetr.hybrid_encoder_sod_novel import HybridEncoderNovelCoordGnConv, CSPCoordGnLayer
    from src.zoo.rtdetr.hybrid_encoder import CSPRepLayer
    from src.zoo.rtdetr.coord_gnconv import SPDConv
    print("✓ Project modules imported successfully.")
except ImportError as e:
    print(f"✗ Import Error: {e}")
    sys.exit(1)

def test_pure_spd():
    print("\n" + "="*60)
    print("TESTING: Exp 4: Pure SPD (SPD + Standard Neck)")
    print("="*60)

    # Instantiate model with SPD=True, Gating=False
    model = HybridEncoderNovelCoordGnConv(
        use_spd=True, 
        use_coord_gating=False,
        hidden_dim=256,
        expansion=0.5
    )
    
    # 1. Verify Downsampling Logic (Should be SPDConv)
    is_spd = any(isinstance(m, SPDConv) for m in model.downsample_convs)
    if is_spd:
        print("✓ Verified: Downsampling is using SPDConv.")
    else:
        print("✗ Failed: Downsampling is NOT using SPDConv.")

    # 2. Verify Neck Blocks (Should be CSPRepLayer, NOT CSPCoordGnLayer)
    is_gated = any(isinstance(m, CSPCoordGnLayer) for m in model.fpn_blocks)
    if not is_gated:
        print("✓ Verified: Neck is using Standard CSP blocks (No Gating).")
    else:
        print("✗ Failed: Neck is still using gated blocks.")

    # 3. Forward Pass
    dummy_input = [torch.randn(1, 64, 160, 160), torch.randn(1, 128, 80, 80), 
                   torch.randn(1, 256, 40, 40), torch.randn(1, 512, 20, 20)]
    try:
        with torch.no_grad():
            output = model(dummy_input)
        print("✓ Forward pass successful.")
        print(f"  Output feature shapes: {[f.shape for f in output]}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")

if __name__ == "__main__":
    test_pure_spd()
    print("\n" + "="*60)
    print("PURE SPD VERIFICATION COMPLETE")
    print("="*60)
