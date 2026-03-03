import torch
import torchvision.models as models
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor

m = models.efficientnet_b2(weights=None)
nodes, _ = get_graph_node_names(m)

# Find layers for stride 4, 8, 16, 32
x = torch.randn(1, 3, 640, 640)
for n in nodes:
    if 'features' in n and n.endswith('.add') == False:
        try:
            fe = create_feature_extractor(m, return_nodes={n: 'out'})
            out = fe(x)['out']
            print(f"Node: {n}, shape: {out.shape}")
        except:
            pass
