"""
Kolmogorov-Arnold Network (KAN) classifiers for Temporal NetFlow Benchmark v3.
Implements:
1. KANStaticClassifier: takes the LAST flow (no history) from the 10-flow window.
2. KANWindowClassifier: takes the full flattened 10-flow window (10 x D features).
Both use learnable B-spline activation functions on edges.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class KANLinear(nn.Module):
    """Efficient KAN Linear layer using B-spline basis functions."""
    def __init__(self, in_features, out_features, grid_size=5, spline_order=2,
                 scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
                 base_activation=nn.SiLU, grid_eps=0.02, grid_range=[-1, 1]):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        
        # Grid definition
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)
        
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features * (grid_size + spline_order)))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (torch.rand(self.grid_size + self.spline_order, self.in_features, self.out_features) - 1/2) * self.scale_noise / self.grid_size
            self.spline_weight.data.copy_((self.scale_spline * torch.ones_like(noise) + noise).permute(2, 1, 0).contiguous().view(self.out_features, -1))
            
    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid = self.grid
        x = x.unsqueeze(-1)  # (batch, in, 1)
        
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, :-(k+1)]) / (grid[:, k:-1] - grid[:, :-(k+1)] + 1e-8) * bases[:, :, :-1] +
                (grid[:, k+1:] - x) / (grid[:, k+1:] - grid[:, 1:-k] + 1e-8) * bases[:, :, 1:]
            )
        return bases.contiguous().view(x.size(0), -1)

    def forward(self, x):
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(self.b_splines(x), self.spline_weight)
        return base_output + spline_output


class KANStaticClassifier(nn.Module):
    """
    KAN-static: Takes only the LAST flow's features (no temporal history).
    Architecture matches the original benchmark anchor model.
    """
    def __init__(self, input_dim: int, num_classes: int, hidden_dims=[64, 32],
                 grid_size=5, spline_order=2):
        super().__init__()
        self.input_dim = input_dim
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        dims = [input_dim] + hidden_dims
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            self.layers.append(KANLinear(in_dim, out_dim, grid_size=grid_size, spline_order=spline_order))
            self.norms.append(nn.LayerNorm(out_dim))
            
        self.output_layer = KANLinear(dims[-1], num_classes, grid_size=grid_size, spline_order=spline_order)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If input is a window (batch, seq_len, num_features), extract ONLY the last flow
        if x.dim() == 3:
            x = x[:, -1, :]
            
        for layer, norm in zip(self.layers, self.norms):
            x = layer(x)
            x = norm(x)
        return self.output_layer(x)


class KANWindowClassifier(nn.Module):
    """
    KAN-window: Takes the flattened 10-flow window (10 x num_features).
    Provides KAN access to the exact same temporal history the sequence models receive.
    """
    def __init__(self, input_dim: int, num_classes: int, window_size: int = 10,
                 hidden_dims=[32, 16], grid_size=3, spline_order=2):
        super().__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        flattened_dim = input_dim * window_size
        
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        dims = [flattened_dim] + hidden_dims
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            self.layers.append(KANLinear(in_dim, out_dim, grid_size=grid_size, spline_order=spline_order))
            self.norms.append(nn.LayerNorm(out_dim))
            
        self.output_layer = KANLinear(dims[-1], num_classes, grid_size=grid_size, spline_order=spline_order)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If input is (batch, seq_len, num_features), flatten to (batch, seq_len * num_features)
        if x.dim() == 3:
            x = x.contiguous().view(x.size(0), -1)
            
        for layer, norm in zip(self.layers, self.norms):
            x = layer(x)
            x = norm(x)
        return self.output_layer(x)
