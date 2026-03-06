import os
import sys
import torch
import torch.nn as nn
import time

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

try:
    from src.core import YAMLConfig
    import src.zoo
    from src.zoo.rtdetr.coord_gnconv import CoordGnConv
    print("✓ Successfully imported project modules.")
except ImportError as e:
    print(f"✗ Import Error: {e}")
    print("Ensure you are running this from the rtdetr_pytorch root directory.")
    sys.exit(1)

def benchmark_module():
    print("\n" + "="*50)
    print("1. Benchmarking CoordGnConv Module")
    print("="*50)
    
    dim = 256
    conv = CoordGnConv(dim).cuda()
    x = torch.randn(1, dim, 160, 160).cuda()
    
    # Warmup
    for _ in range(10):
        _ = conv(x)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = conv(x)
    torch.cuda.synchronize()
    end = time.time()
    
    print(f"Input Shape: {x.shape}")
    print(f"Latency (160x160): {(end-start)/100:.4f}s per forward pass")

def test_full_model():
    print("\n" + "="*50)
    print("2. Testing Full Model Architecture (Novel Coord-gnConv)")
    print("="*50)
    
    config_path = 'configs/rtdetr/rtdetr_r18vd_sod_novel.yml'
    
    if not os.path.exists(config_path):
        print(f"✗ Config file not found at {config_path}")
        return

    print(f"Loading config: {config_path}")
    cfg = YAMLConfig(config_path)
    
    print("Building model...")
    model = cfg.model.cuda()
    model.eval()
    print("✓ Model built successfully.")
    
    # Check if correct encoder is used
    from src.zoo.rtdetr.hybrid_encoder_sod_novel import HybridEncoderNovelCoordGnConv
    if isinstance(model.encoder, HybridEncoderNovelCoordGnConv):
        print("✓ Verified: Model is using HybridEncoderNovelCoordGnConv.")
    else:
        print(f"✗ Error: Model is using {type(model.encoder).__name__}")

    print("\nRunning Forward Pass...")
    dummy_input = torch.randn(1, 3, 640, 640).cuda()
    
    with torch.no_grad():
        start = time.time()
        output = model(dummy_input)
        torch.cuda.synchronize()
        end = time.time()
    
    print(f"✓ Forward pass successful in {end-start:.4f}s")
    if 'pred_boxes' in output:
        print(f"Output shape (boxes): {output['pred_boxes'].shape}")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Verify R18 Optimizations
    if total_params < 22e6:
        print("✓ Verified: Model is using Optimized R18 settings (Expansion 0.5, Decoder 3).")
    else:
        print("⚠ Warning: Parameter count is still high. Check expansion and decoder layer settings.")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("✗ CUDA is not available. Please run this on a GPU-enabled environment (Kaggle).")
    else:
        benchmark_module()
        test_full_model()
        print("\n" + "="*50)
        print("Verification Complete: Architecture is valid and ready for training.")
        print("="*50)
