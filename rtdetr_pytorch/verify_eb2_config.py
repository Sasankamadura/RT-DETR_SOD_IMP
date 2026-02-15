
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn 

from src.core import register, YAMLConfig
from src.nn.backbone.efficientnet import EfficientNet

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
        print(model)
        
        # Test forward pass with dummy input
        dummy_input = torch.randn(1, 3, 640, 640)
        if torch.cuda.is_available():
            model = model.cuda()
            dummy_input = dummy_input.cuda()
            
        output = model(dummy_input)
        print("Forward pass successful.")
        
        if isinstance(output, dict):
            for k, v in output.items():
                print(f"Output {k}: {v.shape}")
        else:
            print(f"Output shape: {output.shape}")
            
    except Exception as e:
        print(f"Error during model instantiation or forward pass: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    verify_config()
