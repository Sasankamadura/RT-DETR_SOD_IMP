
import sys
import os
import torch
import torch.nn as nn 
import torchvision

# Ensure src is in python path
sys.path.append(os.getcwd())

from src.core import register, YAMLConfig, BaseConfig

def verify_config():
    config_path = 'configs/rtdetr/rtdetr_visdrone_eb2.yml'
    try:
        cfg = YAMLConfig(config_path, resume='')
        print("Config parsed successfully.")
    except Exception as e:
        print(f"Error parsing config: {e}")
        return

    try:
        model = cfg.model
        print("Model instantiated successfully.")
        
        # 1. Test forward pass with dummy input AND dummy targets
        # RT-DETR during training needs targets for contrastive denoising
        dummy_input = torch.randn(2, 3, 640, 640) # Batch size 2
        
        # Dummy targets for batch size 2
        dummy_targets = [
            {
                'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32), 
                'labels': torch.tensor([1], dtype=torch.int64)
            },
            {
                'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32), 
                'labels': torch.tensor([1], dtype=torch.int64)
            }
        ]

        if torch.cuda.is_available():
            model = model.cuda()
            dummy_input = dummy_input.cuda()
            dummy_targets = [{k: v.cuda() for k, v in t.items()} for t in dummy_targets]
            
        # Set to train mode to test denoising logic
        model.train()
        output = model(dummy_input, dummy_targets)
        print("Forward pass (Training Mode) successful.")
        
        if isinstance(output, dict):
            for k, v in output.items():
                if isinstance(v, torch.Tensor):
                    print(f"Output {k}: {v.shape}")
                else:
                    print(f"Output {k}: {v}")
                    
    except Exception as e:
        print(f"Error during model instantiation or forward pass: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    verify_config()
