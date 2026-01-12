
import sys
import os
import torch
import types

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

# WORKAROUND: Mock missing dependencies for isolated unit testing
# This prevents ImportErrors from modules not relevant to the encoder logic
# (like yaml, transformers, or data loading components)

# 1. Mock yaml
mock_yaml = types.ModuleType('yaml')
sys.modules['yaml'] = mock_yaml

# 2. Mock transformers (triggered by regnet import if present in backbone init)
mock_transformers = types.ModuleType('transformers')
class DummyRegNetModel: pass
mock_transformers.RegNetModel = DummyRegNetModel
sys.modules['transformers'] = mock_transformers

# 3. Mock src.data to avoid loading COCO api or torchvision datapoints
sys.modules['src.data'] = types.ModuleType('src.data')
sys.modules['src.data.coco'] = types.ModuleType('src.data.coco')
sys.modules['src.data.coco.coco_dataset'] = types.ModuleType('src.data.coco.coco_dataset')

# Now import the NECK-ONLY Encoder
from src.zoo.rtdetr.hybrid_encoder_gnconv_ccfm import HybridEncoderGnConvCCFM

def test_gnconv_neck_encoder():
    print("Testing HybridEncoderGnConvCCFM (Neck Integration)...")
    
    # 1. Instantiate the Encoder
    # Using typical RT-DETR channel dimensions
    # gnconv_order=4 is the recommended setting
    encoder = HybridEncoderGnConvCCFM(
        in_channels=[128, 256, 512],
        feat_strides=[8, 16, 32],
        hidden_dim=256,
        use_encoder_idx=[2], 
        num_encoder_layers=1,
        gnconv_order=4 
    )
    print("Encoder instantiated successfully. (Using GnConvFusionBlock in CCFM)")

    # 2. Create Dummy Inputs (Simulating ResNet18 Output)
    # Shapes for strides 8, 16, 32 with batch size 1
    # Note: ResNet18 outputs might need projection, but the Encoder handles that.
    # We pass tensors matching 'in_channels' above.
    feats = [
        torch.randn(1, 128, 80, 80), # Stride 8 (640/8)
        torch.randn(1, 256, 40, 40), # Stride 16 (640/16)
        torch.randn(1, 512, 20, 20)  # Stride 32 (640/32)
    ]
    print("Dummy inputs created (Simulating ResNet features).")

    # 3. Forward Pass
    try:
        outs = encoder(feats)
        print("Forward pass successful.")
    except Exception as e:
        print(f"Forward pass FAILED: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Output Checks
    print("\nChecking outputs:")
    # The encoder outputs features at the same strides as input [8, 16, 32]
    # Expected channels are all hidden_dim (256)
    expected_strides = [8, 16, 32]
    
    for i, out in enumerate(outs):
        print(f"Output {i} shape: {out.shape}")
        
        # Check if output is a Tensor
        if not isinstance(out, torch.Tensor):
             print(f"ERROR: Output {i} is not a tensor.")
             continue

        # Basic shape check (B, C, H, W) -> (1, 256, H, W)
        if out.shape[1] != 256: 
            print(f"ERROR: Output {i} channel mismatch. Expected 256, got {out.shape[1]}")
    
    print("\nTest Complete.")

if __name__ == "__main__":
    test_gnconv_neck_encoder()
