
import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseP2Selector(nn.Module):
    def __init__(self, d_model=256, k_sparse=300):
        super().__init__()
        self.k_sparse = k_sparse
        # A simple saliency predictor: 1x1 conv on P3 (which projects to 1 channel)
        # We assume P3 input is 'd_model' channels.
        self.saliency_conv = nn.Sequential(
            nn.Conv2d(d_model, d_model // 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(d_model // 4, 1, 1)
        )
        
        # P2 projection to match d_model
        # Assuming P2 typically has fewer channels (e.g. 256 or 64), but we'll project to d_model
        # We'll allow the user to specify input dimension if needed, but for now specific to logical flow
        self.p2_proj = nn.Conv2d(d_model, d_model, 1) 

    def get_pos_embed(self, coords, d_model):
        """
        Generate sine-cosine positional embeddings for selected coordinates.
        coords: (B, K, 2) normalized [0,1] (y, x)
        """
        B, K, _ = coords.shape
        # This is a simplified pos embedding for demo; normally use sine/cosine
        # Adapt from standard DETR sinusoidal embedding
        
        num_pos_feats = d_model // 2
        temperature = 10000
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=coords.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

        y_embed = coords[:, :, 0].unsqueeze(-1) / dim_t
        x_embed = coords[:, :, 1].unsqueeze(-1) / dim_t
        
        pos_embed = torch.cat((y_embed.sin(), y_embed.cos(), x_embed.sin(), x_embed.cos()), dim=3).flatten(2)
        return pos_embed

    def forward(self, p2_feat, p3_feat):
        """
        p2_feat: (B, C, H2, W2) - High Res Detail
        p3_feat: (B, C, H3, W3) - Semantic Context (used for selection)
        """
        B, C, H2, W2 = p2_feat.shape
        
        # 1. Predict Saliency from P3
        # We implicitly learn "where are small objects?"
        saliency_logits = self.saliency_conv(p3_feat) # (B, 1, H3, W3)
        
        # 2. Upsample Saliency to P2 Resolution
        saliency_map = F.interpolate(saliency_logits, size=(H2, W2), mode='bilinear', align_corners=False) # (B, 1, H2, W2)
        saliency_probs = torch.sigmoid(saliency_map.flatten(2)) # (B, H2*W2)

        # 3. Select Top-K Indices
        k = min(self.k_sparse, saliency_probs.shape[1])
        topk_vals, topk_indices = torch.topk(saliency_probs, k, dim=1) # (B, K)
        
        # 4. Gather P2 Features
        # flat_p2: (B, C, H2*W2) -> (B, H2*W2, C)
        flat_p2 = self.p2_proj(p2_feat).flatten(2).permute(0, 2, 1) 
        
        # Gather: (B, K, C)
        # We need to expand indices to (B, K, C)
        idx_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, C)
        sparse_p2 = torch.gather(flat_p2, 1, idx_expanded)
        
        # 5. Calculate Coords for Pos Embed
        # indices are flat indices in (H2, W2)
        y = (topk_indices // W2).float() / H2
        x = (topk_indices % W2).float() / W2
        coords = torch.stack([y, x], dim=-1) # (B, K, 2)
        
        # sparse_pos = self.get_pos_embed(coords, C) # Implement proper pos embed if needed, or pass coords
        
        return sparse_p2, coords
