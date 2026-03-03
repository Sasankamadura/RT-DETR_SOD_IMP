import torch
import torchvision
import sys
import os

def test_efficientnet_backbone():
    print("Testing EfficientNet Backbone Strides and Channels...")
    from src.nn.backbone.efficientnet import EfficientNet
    
    # Initialize our custom wrapper
    backbone = EfficientNet(model_name='efficientnet_b2', pretrained=False)
    
    # Dummy input
    x = torch.randn(2, 3, 640, 640)
    
    # Forward pass
    features = backbone(x)
    
    print(f"Num feature maps extracted: {len(features)}")
    assert len(features) == 4, "Expected 4 feature maps (P2, P3, P4, P5)"
    
    expected_shapes = [
        (2, 24, 160, 160), # P2 (Stride 4)
        (2, 48, 80, 80),   # P3 (Stride 8)
        (2, 120, 40, 40),  # P4 (Stride 16)
        (2, 352, 20, 20)   # P5 (Stride 32)
    ]
    
    for i, feat in enumerate(features):
        print(f"Feature P{i+2} Shape: {feat.shape}")
        assert feat.shape == expected_shapes[i], f"Expected {expected_shapes[i]} for P{i+2}, got {feat.shape}"

    print("Success: Backbone correctly extracts P2, P3, P4, P5!\n")


if __name__ == '__main__':
    # Ensure src is in python path
    sys.path.append(os.getcwd())
    try:
        test_efficientnet_backbone()
    except Exception as e:
        print(f"Test Failed: {e}")
