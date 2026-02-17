
import sys
import os
import torch

# Ensure src is in python path
sys.path.append(os.getcwd())

try:
    from src.core import YAMLConfig
except ImportError:
    print("Error: Could not import YAMLConfig. Ensure you are running this script from the project root.")
    sys.exit(1)

def verify_p2_architecture():
    print("Loading config: configs/rtdetr/include/rtdetr_eb2_p2.yml")
    try:
        cfg = YAMLConfig('configs/rtdetr/include/rtdetr_eb2_p2.yml')
    except Exception as e:
        print(f"Failed to load config: {e}")
        return

    model = cfg.model
    
    print("\n--- Verifying Model Architecture ---")
    
    # Check Backbone
    print("Checking Backbone...")
    if hasattr(model.backbone, 'return_idx'):
        print(f"Backbone return_idx: {model.backbone.return_idx}")
        # Expecting P2 (stride 4) to be included. EfficientNet B2 strides: 2, 4, 8, 16, 32.
        # Indices [2, 3, 5, 7] correspond to strides 4, 8, 16, 32.
    
    # Check Encoder
    print("\nChecking Encoder...")
    if hasattr(model.encoder, 'feat_strides'):
        print(f"Encoder feat_strides: {model.encoder.feat_strides}")
        if len(model.encoder.feat_strides) == 4 and model.encoder.feat_strides[0] == 4:
            print("SUCCESS: Encoder uses 4 feature levels starting with stride 4 (P2).")
        else:
            print("WARNING: Encoder feature strides do not match P2 configuration.")

    # Check Transformer
    print("\nChecking Transformer/Decoder...")
    if hasattr(model.decoder, 'num_levels'):
        print(f"Decoder num_levels: {model.decoder.num_levels}")
        if model.decoder.num_levels == 4:
            print("SUCCESS: Decoder uses 4 feature levels.")
        else:
            print("WARNING: Decoder num_levels is not 4.")

    # Dummy Forward Pass
    print("\nRunning dummy forward pass to verify graph capability...")
    x = torch.randn(1, 3, 640, 640)
    if torch.cuda.is_available():
        x = x.cuda()
        model = model.cuda()
        print("Using CUDA.")
    
    try:
        output = model(x)
        if 'pred_logits' in output and 'pred_boxes' in output:
            print("Forward pass successful!")
            print(f"Pred logits shape: {output['pred_logits'].shape}")
            print(f"Pred boxes shape: {output['pred_boxes'].shape}")
        else:
            print("Forward pass output missing expected keys.")
    except Exception as e:
        print(f"Forward pass failed: {e}")

if __name__ == "__main__":
    verify_p2_architecture()
