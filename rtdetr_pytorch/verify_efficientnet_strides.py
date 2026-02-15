import torch
import torchvision

def verify_strides():
    model = torchvision.models.efficientnet_b2(pretrained=False)
    x = torch.randn(1, 3, 640, 640)
    
    print("Features architecture:")
    current_stride = 1
    for i, layer in enumerate(model.features):
        x = layer(x)
        print(f"Layer {i}: Output shape {x.shape}")

if __name__ == "__main__":
    verify_strides()
