import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

try:
    from src.zoo.rtdetr.hybrid_encoder_sod_novel import HybridEncoderNovelCoordGnConv, CSPCoordGnLayer
    from src.zoo.rtdetr.hybrid_encoder import CSPRepLayer, ConvNormLayer
    from src.zoo.rtdetr.coord_gnconv import SPDConv
    print("✓ Project modules imported successfully.")
except ImportError as e:
    print(f"✗ Import Error: {e}")
    sys.exit(1)

def test_pure_gating():
    print("\n" + "="*60)
    print("TESTING: Exp 5: Pure Gating (Coord-GnConv + Standard Downsampling)")
    print("="*60)

    # Instantiate model with SPD=False, Gating=True
    model = HybridEncoderNovelCoordGnConv(
        use_spd=False, 
        use_coord_gating=True,
        hidden_dim=256,
        expansion=0.5
    )
    
    # 1. Verify Downsampling Logic (Should NOT be SPDConv)
    is_spd = any(isinstance(m, SPDConv) for m in model.downsample_convs)
    if not is_spd:
        print("✓ Verified: Downsampling is using Standard Stride-2 Convs.")
    else:
        print("✗ Failed: Downsampling is STILL using SPDConv.")

    # 2. Verify Neck Blocks (Should be CSPCoordGnLayer)
    is_gated = any(isinstance(m, CSPCoordGnLayer) for m in model.fpn_blocks)
    if is_gated:
        print("✓ Verified: Neck is using Coordinate-Gated blocks.")
    else:
        print("✗ Failed: Neck is NOT using gated blocks.")

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
    test_pure_gating()
    print("\n" + "="*60)
    print("PURE GATING VERIFICATION COMPLETE")
    print("="*60)
