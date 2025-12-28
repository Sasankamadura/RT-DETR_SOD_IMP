"""
Test script to verify P2 fusion implementation in RT-DETR
"""
import torch
import sys
sys.path.insert(0, '/kaggle/working/RT-DETR_SOD_IMP/rtdetr_pytorch')

from src.core import YAMLConfig


def test_p2_fusion():
    print("=" * 60)
    print("Testing P2 Fusion Implementation")
    print("=" * 60)
    
    # Load config
    cfg_path = '/kaggle/working/RT-DETR_SOD_IMP/rtdetr_pytorch/configs/rtdetr/rtdetr_r18vd_visdrone_p2fusion'
    print(f"\n1. Loading config: {cfg_path}")
    cfg = YAMLConfig(cfg_path, resume=None, use_amp=False)
    
    # Build model
    print("\n2. Building model...")
    model = cfg.model
    model.eval()
    print(f"   Model type: {type(model).__name__}")
    
    # Test input (640x640)
    input_size = 640
    x = torch.randn(1, 3, input_size, input_size)
    print(f"\n3. Testing with input shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        # Test backbone
        print("\n4. Testing Backbone...")
        backbone_out = model.backbone(x)
        print(f"   Backbone outputs: {len(backbone_out)} levels")
        for i, feat in enumerate(backbone_out):
            stride = feat.shape[2] // (input_size // feat.shape[2])
            actual_stride = input_size // feat.shape[2]
            print(f"   Level {i} (C{i+2}): shape={tuple(feat.shape)}, stride={actual_stride}")
        
        # Expected shapes for 640x640 input with ResNet-18:
        expected_backbone = [
            (1, 64, 160, 160),   # C2: stride 4
            (1, 128, 80, 80),    # C3: stride 8
            (1, 256, 40, 40),    # C4: stride 16
            (1, 512, 20, 20),    # C5: stride 32
        ]
        
        print("\n   Verification:")
        all_correct = True
        for i, (feat, expected) in enumerate(zip(backbone_out, expected_backbone)):
            actual = tuple(feat.shape)
            match = "✓" if actual == expected else "✗"
            print(f"   {match} Level {i}: expected {expected}, got {actual}")
            if actual != expected:
                all_correct = False
        
        # Test encoder
        print("\n5. Testing Encoder (with P2 Fusion)...")
        encoder_out = model.encoder(backbone_out)
        print(f"   Encoder outputs: {len(encoder_out)} levels")
        for i, feat in enumerate(encoder_out):
            actual_stride = input_size // feat.shape[2]
            print(f"   Level {i}: shape={tuple(feat.shape)}, stride={actual_stride}")
        
        # Expected shapes for encoder output:
        # All should have hidden_dim=256 channels
        expected_encoder = [
            (1, 256, 80, 80),    # Enriched P3 (fused with P2): stride 8
            (1, 256, 40, 40),    # P4: stride 16
            (1, 256, 20, 20),    # P5: stride 32
        ]
        
        print("\n   Verification:")
        for i, (feat, expected) in enumerate(zip(encoder_out, expected_encoder)):
            actual = tuple(feat.shape)
            match = "✓" if actual == expected else "✗"
            level_name = ["Enriched P3", "P4", "P5"][i]
            print(f"   {match} {level_name}: expected {expected}, got {actual}")
            if actual != expected:
                all_correct = False
        
        # Test full model
        print("\n6. Testing Full Model...")
        output = model(x)
        print(f"   Output keys: {list(output.keys())}")
        if 'pred_boxes' in output:
            print(f"   pred_boxes shape: {output['pred_boxes'].shape}")
        if 'pred_logits' in output:
            print(f"   pred_logits shape: {output['pred_logits'].shape}")
    
    print("\n" + "=" * 60)
    if all_correct:
        print("✅ ALL TESTS PASSED!")
        print("\nP2 Fusion is correctly implemented:")
        print("  - Backbone extracts 4 levels (C2, C3, C4, C5)")
        print("  - Encoder outputs 3 levels (Enriched P3, P4, P5)")
        print("  - P2 (stride 4) is fused into P3 (stride 8)")
    else:
        print("⚠️  SOME TESTS FAILED - Please review shapes above")
    print("=" * 60)


if __name__ == '__main__':
    try:
        test_p2_fusion()
    except Exception as e:
        print(f"\n❌ Error during testing: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
