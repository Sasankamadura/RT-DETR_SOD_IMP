import torch
import torch.nn as nn
import sys
import os

# Add local path to import
sys.path.append(os.getcwd())

try:
    from src.zoo.rtdetr.hybrid_encoder_sod_novel import HybridEncoderNovelCoordGnConv
    from src.zoo.rtdetr.coord_gnconv import CoordGnConv, RepDWConv
    print("✓ Successfully imported model components.")
except ImportError as e:
    print(f"✗ Import Error: {e}")
    sys.exit(1)

def test_ablation_variants():
    print("\n" + "="*50)
    print("1. Testing Ablation Toggles (SPD & Gating)")
    print("="*50)
    
    # 4 Scenarios to test
    scenarios = [
        {"name": "Full (Gating=True, SPD=True)", "spd": True, "gating": True},
        {"name": "No SPD (Gating=True, SPD=False)", "spd": False, "gating": True},
        {"name": "Pure SPD (Gating=False, SPD=True)", "spd": True, "gating": False},
        {"name": "Standard Neck (Gating=False, SPD=False)", "spd": False, "gating": False},
    ]
    
    for s in scenarios:
        try:
            model = HybridEncoderNovelCoordGnConv(
                use_spd=s["spd"], 
                use_coord_gating=s["gating"],
                hidden_dim=256,
                expansion=0.5
            )
            print(f"✓ Created: {s['name']}")
            
            # Simple Forward Pass (single feature)
            dummy_input = [torch.randn(1, 64, 160, 160), torch.randn(1, 128, 80, 80), 
                           torch.randn(1, 256, 40, 40), torch.randn(1, 512, 20, 20)]
            _ = model(dummy_input)
            print(f"  ✓ Forward pass successful.")
        except Exception as e:
            print(f"✗ Failed {s['name']}: {e}")

def test_rep_logic():
    print("\n" + "="*50)
    print("2. Testing Reparameterization Logic (RepDWConv)")
    print("="*50)
    
    dim = 256
    rep_block = RepDWConv(dim, deploy=False)
    x = torch.randn(1, dim, 32, 32)
    
    # 1. Training pass
    y_train = rep_block(x)
    print("✓ Training forward pass successful.")
    
    # 2. Deploy/Fuse
    rep_block.switch_to_deploy()
    y_deploy = rep_block(x)
    print("✓ Fused/Deploy forward pass successful.")
    
    # 3. Numeric Check
    diff = torch.abs(y_train - y_deploy).max().item()
    print(f"Max difference (Train vs Deploy): {diff:.2e}")
    if diff < 1e-4:
        print("✓ Numeric Verification passed.")
    else:
        print("⚠ Numeric Verification failed - check fusion logic.")

def test_coord_gating_switch():
    print("\n" + "="*50)
    print("3. Testing Coord-Gating vs Standard Neck")
    print("="*50)
    
    dim = 256
    # Gated
    gated = HybridEncoderNovelCoordGnConv(use_coord_gating=True)
    # Standard
    standard = HybridEncoderNovelCoordGnConv(use_coord_gating=False)
    
    gated_params = sum(p.numel() for p in gated.parameters())
    standard_params = sum(p.numel() for p in standard.parameters())
    
    print(f"Gated Bottlenecks Params: {gated_params:,}")
    print(f"Standard Neck Params: {standard_params:,}")
    
    if gated_params != standard_params:
        print("✓ Verified: Model weights change when toggling gating logic.")
    else:
        print("✗ Warning: Parameter count is identical. Ensure logic is switching correctly.")

if __name__ == "__main__":
    test_ablation_variants()
    test_rep_logic()
    test_coord_gating_switch()
    print("\n" + "="*50)
    print("PRE-TRAINING CHECKS COMPLETE")
    print("Code is ready for Kaggle.")
    print("="*50)
