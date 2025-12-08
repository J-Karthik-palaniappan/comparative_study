import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

def closest_square_factors(N):
    """
    Return a pair (r, c) with r*c = N and |r-c| minimized.
    """
    s = int(math.isqrt(N))   
    for i in range(s, 0, -1):
        if N % i == 0:
            return i, N // i

class ExpLinearAfterThreshold(nn.Module):
    def __init__(self, max_val=5.0):
        super(ExpLinearAfterThreshold, self).__init__()
        self.max_val = max_val

    def forward(self, x):
        return torch.exp(x.clamp(max=self.max_val)) + torch.relu(x - self.max_val)

class PnPNystraAttention(nn.Module):
    r""" Proposed approximation of window based multi-head self attention (W-MSA) module.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, num_landmarks, iters, dim, window_size, num_heads, attn_drop=0., 
                 *args, **kwargs):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads

        self.attn_drop = nn.Dropout(attn_drop)
        self.activ = ExpLinearAfterThreshold()

        self.num_landmarks = num_landmarks
        self.iters = iters

    def forward(self, q, k, v, x_shape, **kwargs):

        B_, N, C = x_shape
        window_size = self.window_size[0]
        h, w = closest_square_factors(self.num_landmarks)

        q_m = F.adaptive_avg_pool2d(q.reshape(B_*self.num_heads, window_size, window_size, self.dim//self.num_heads).permute(0, 3, 1, 2), output_size = (h, w)).permute(0, 2, 3, 1).reshape(B_, self.num_heads, self.num_landmarks, self.dim//self.num_heads)
        k_m = F.adaptive_avg_pool2d(k.reshape(B_*self.num_heads, window_size, window_size, self.dim//self.num_heads).permute(0, 3, 1, 2), output_size = (h, w)).permute(0, 2, 3, 1).reshape(B_, self.num_heads, self.num_landmarks, self.dim//self.num_heads)

        temp = self.activ(q_m @ k_m.transpose(-2, -1)) 
        
        pseudo_inv = self.moore_penrose_iter_pinv(temp, self.iters)
        prod = (self.activ(q @ k_m.transpose(-2, -1)) @ pseudo_inv) @ (self.activ(q_m @ k.transpose(-2,-1)) @ torch.cat([v, torch.ones_like(v[..., :1])], dim=-1))

        x = (prod[..., :-1] / (prod[..., -1].unsqueeze(-1) + 1e-12))
        return x
    
    def moore_penrose_iter_pinv(self, x, iters = 6):
        device = x.device

        abs_x = torch.abs(x)
        col = abs_x.sum(dim = -1)
        row = abs_x.sum(dim = -2)
        z = rearrange(x, '... i j -> ... j i') / (torch.max(col) * torch.max(row) + 1e-15)

        I = torch.eye(x.shape[-1], device = device)
        I = rearrange(I, 'i j -> () i j')

        for _ in range(iters):
            xz = x @ z
            z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

        return z

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'