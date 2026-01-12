
import sys
import os
import torch
from unittest.mock import MagicMock

# Hack to bypass src.data import which triggers pycocotools dependency
sys.modules['src.data'] = MagicMock()
sys.modules['src.data.coco'] = MagicMock()
sys.modules['pycocotools'] = MagicMock()
sys.modules['pycocotools.mask'] = MagicMock()
sys.modules['yaml'] = MagicMock() # Mock yaml to bypass config utils

# Add root to path
sys.path.append(os.getcwd())

# Now we can safely import model components
try:
    from src.zoo.rtdetr.rtdetr import RTDETR
    from src.zoo.rtdetr.hybrid_encoder import HybridEncoder
    from src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer
except ImportError as e:
    print(f"Import Error even with mocks: {e}")
    # Fallback: Maybe we need to modify src/__init__.py to not import data by default?
    # No, let's try to proceed.
    raise e

def test_sparse_p2():
    print("Initializing RT-DETR components with Sparse P2...")
    
    # Mock Backbone output: P2, P3, P4, P5
    # Batch size 1, Channels 256
    # P2: 160x160 (Stride 4)
    # P3: 80x80 (Stride 8) 
    # P4: 40x40 (Stride 16)
    # P5: 20x20 (Stride 32)
    
    backbone_feats = [
        torch.randn(1, 256, 160, 160), # P2
        torch.randn(1, 256, 80, 80),   # P3
        torch.randn(1, 256, 40, 40),   # P4
        torch.randn(1, 256, 20, 20)    # P5
    ]
    
    # Encoder expects P3, P4, P5 (3 inputs) if we configure it standardly
    # We simulate the backbone wrapper behavior
    encoder = HybridEncoder(in_channels=[256, 256, 256], feat_strides=[8, 16, 32], hidden_dim=256)
    
    # Decoder
    # feat_channels should match encoder output (256)
    decoder = RTDETRTransformer(
        num_classes=80, 
        feat_channels=[256, 256, 256], 
        feat_strides=[8, 16, 32],
        num_levels=3
    ) 
    
    # Model
    model = RTDETR(lambda x: x, encoder, decoder) # Dummy backbone
    
    print("Running forward pass...")
    
    # Manually mimic the forward flow in rtdetr.py
    # We pass the list of 4 feats to our modified logic
    
    # Case 1: With P2
    p2_feat = backbone_feats[0]
    encoder_feats = backbone_feats[1:]
    
    print(f"P2 shape: {p2_feat.shape}")
    print(f"Encoder inputs: {[f.shape for f in encoder_feats]}")
    
    # Run Encoder
    x_enc = encoder(encoder_feats)
    print("Encoder output computed.")
    
    # Run Decoder via Model Wrapper specific logic or direct call
    # We want to test logic inside RTDETR.forward, specifically the p2_feat extraction
    # But RTDETR.forward takes 'x'. Let's wrap backbone to return our list.
    
    class MockBackbone(torch.nn.Module):
        def forward(self, x):
            return backbone_feats
            
    model.backbone = MockBackbone()
    
    input_dummy = torch.randn(1, 3, 640, 640)
    
    try:
        x_dec = model(input_dummy)
        print("Model forward pass successful!")
        print(f"Output Boxes Shape: {x_dec['pred_boxes'].shape}")
        
        # Verify Sparse Selector was executed
        if decoder.use_sparse_p2:
            print("Verified: use_sparse_p2 flag is True.")
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sparse_p2()
