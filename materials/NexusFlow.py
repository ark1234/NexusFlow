# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

# You'll need this for the Deformable Attention layer.
# Install it via: pip install deformable-attention-pytorch
try:
    from deformable_attention_pytorch import DeformableAttention2D
except ImportError:
    print("DeformableAttention2D not found. Please run: pip install deformable-attention-pytorch")
    # A simple fallback so the script doesn't crash if the lib is missing.
    class DeformableAttention2D(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.dim = kwargs.get("dim", 256)
            print("WARNING: Using a dummy DeformableAttention2D. Install the real library for this to work properly.")
        def forward(self, x):
            return torch.randn(x.shape[0], self.dim, device=x.device)


class CouplingLayer(nn.Module):
    """
    A simple RealNVP-style coupling layer for invertible transformations.
    
    The core idea is to split a feature vector in half. One half is used to
    predict a scale and shift that gets applied to the other half. It's simple,
    fast, and information-preserving.
    """
    def __init__(self, dim: int, hidden_dim: int = 512):
        super().__init__()
        # A simple MLP to predict the scale (s) and translation (t) params.
        # It takes one half of the vector to predict params for the other half.
        self.s_t_net = nn.Sequential(
            nn.Linear(dim // 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim) # Output is `dim` because we need `dim/2` for scale and `dim/2` for shift.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split feature in half. One half learns to transform the other.
        x1, x2 = x.chunk(2, dim=1)

        # Predict scale and shift from the first half.
        s_t_output = self.s_t_net(x1)
        s, t = s_t_output.chunk(2, dim=1)
        s = torch.tanh(s) # Stabilize training by constraining the scale.

        # Scale and shift x2. This is the core invertible operation.
        z2 = x2 * torch.exp(s) + t

        # Re-concatenate to form the output. Note that x1 remains unchanged.
        return torch.cat([x1, z2], dim=1)


class SurrogateNetwork(nn.Module):
    """
    A single branch of the NexusFlow module. It takes the big BEV feature map,
    squashes it down to a vector, and then transforms it through a coupling layer.
    """
    def __init__(self, bev_channels: int = 256, reduced_dim: int = 256, device: str = 'cpu'):
        super().__init__()
        
        # 1. Feature Reducer: Squashes the (B, C, H, W) feature map to (B, C).
        # We use Deformable Attention because it's good at focusing on important
        # parts of a large feature map without huge computational cost.
        self.reducer = DeformableAttention2D(
            dim=bev_channels,
            dim_head=64,
            heads=4,
            dropout=0.1,
            downsample_factor=16,
            offset_kernel_size=5
        ).to(device)

        # 2. Invertible Transformation
        assert reduced_dim % 2 == 0, "CouplingLayer needs an even feature dimension."
        self.coupling_layer = CouplingLayer(dim=reduced_dim).to(device)

    def forward(self, bev_feature_map: torch.Tensor) -> torch.Tensor:
        # Input shape: (B, C, H, W)
        compact_feature = self.reducer(bev_feature_map) # -> (B, C)
        aligned_feature = self.coupling_layer(compact_feature) # -> (B, C)
        return aligned_feature


class NexusFlow(nn.Module):
    """
    The complete NexusFlow module. It's basically just two of the SurrogateNetworks
    running in parallel, one for each task's "perspective".
    """
    def __init__(self, bev_channels: int = 256, reduced_dim: int = 256, device: str = 'cpu'):
        super().__init__()
        self.g_map = SurrogateNetwork(bev_channels, reduced_dim, device)
        self.g_track = SurrogateNetwork(bev_channels, reduced_dim, device)

    def forward(self, bev_feature_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # The exact same feature map is processed by both branches.
        z_map = self.g_map(bev_feature_map)
        z_track = self.g_track(bev_feature_map)
        return z_map, z_track

# =================================================================
#  DEMO SCRIPT
# =================================================================
if __name__ == "__main__":
    # --- 1. Config ---
    BATCH_SIZE = 4
    BEV_CHANNELS = 256
    BEV_H, BEV_W = 200, 200
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"--> Running NexusFlow Demo on {DEVICE}")

    # --- 2. Model Init ---
    nexus_flow_module = NexusFlow(
        bev_channels=BEV_CHANNELS,
        reduced_dim=BEV_CHANNELS, # Reducer output dim must match coupling layer input dim
        device=DEVICE
    )
    print("--> NexusFlow module created.")
    
    # --- 3. Mock Data ---
    # This simulates the shared feature map from your main model's BEV encoder.
    mock_bev_feature = torch.randn(BATCH_SIZE, BEV_CHANNELS, BEV_H, BEV_W).to(DEVICE)
    print(f"--> Mock BEV feature created with shape: {mock_bev_feature.shape}")

    # --- 4. Forward Pass ---
    # Get the two transformed features for alignment.
    z_map, z_track = nexus_flow_module(mock_bev_feature)
    print(f"--> Output z_map shape: {z_map.shape}")
    print(f"--> Output z_track shape: {z_track.shape}")

    # --- 5. Loss Calculation ---
    # The alignment loss is just the MSE between the two outputs.
    # Simple, but effective.
    # TODO: Explore other loss functions like L1 or Cosine Similarity.
    alignment_loss_fn = nn.MSELoss()
    l_align = alignment_loss_fn(z_map, z_track)
    print(f"--> Calculated Alignment Loss (L_align): {l_align.item():.4f}")

    # --- 6. Backpropagation ---
    # This shows that gradients can flow through the module.
    l_align.backward()
    print("--> Backward pass successful.")
    
    grad_check = nexus_flow_module.g_map.coupling_layer.s_t_net[0].weight.grad
    if grad_check is not None and grad_check.abs().sum() > 0:
        print("--> Gradients computed successfully! Module is ready for training.")
    else:
        print("--> Error: Gradients not computed.")
