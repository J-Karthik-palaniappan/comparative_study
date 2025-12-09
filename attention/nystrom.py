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

class NystromAttention(nn.Module):
    def __init__(self, num_landmarks, iters, dim, window_size, num_heads,
                 init_option="exact", *args, **kwargs):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        
        self.num_landmarks = num_landmarks
        self.iters = 6 #iters hardcoding iters
        self.init_option = init_option
        # ======================================== debug ========================================
        self.capture = {}
        # ======================================== debug ========================================

    def forward(self, q, k, v, x_shape, **kwargs):

        B_, N, C = x_shape
        window_size = self.window_size[0]
        h, w = closest_square_factors(self.num_landmarks)

        # q_m = q.reshape(-1, self.num_heads, self.num_landmarks, N // self.num_landmarks, self.dim//self.num_heads).mean(dim = -2)
        # k_m = k.reshape(-1, self.num_heads, self.num_landmarks, N // self.num_landmarks, self.dim//self.num_heads).mean(dim = -2)
        
        q_m = F.adaptive_avg_pool2d(q.reshape(B_*self.num_heads, window_size, window_size, self.dim//self.num_heads).permute(0, 3, 1, 2), output_size = (h, w)).permute(0, 2, 3, 1).reshape(B_, self.num_heads, self.num_landmarks, self.dim//self.num_heads)
        k_m = F.adaptive_avg_pool2d(k.reshape(B_*self.num_heads, window_size, window_size, self.dim//self.num_heads).permute(0, 3, 1, 2), output_size = (h, w)).permute(0, 2, 3, 1).reshape(B_, self.num_heads, self.num_landmarks, self.dim//self.num_heads)
        # print(q.shape," -> ",q_m.shape)
        kernel_1 = F.softmax(q @ k_m.transpose(-1, -2), dim=-1)
        kernel_2 = F.softmax(q_m @ k_m.transpose(-1, -2), dim=-1)
        kernel_3 = F.softmax(q_m @ k.transpose(-1, -2), dim=-1)

        # x = (kernel_1 @ self.iterative_inv(kernel_2, n_iter=self.iters)) @ (kernel_3 @ v)
        approx_attn = kernel_1 @ self.iterative_inv(kernel_2, n_iter=self.iters) @ kernel_3
        # approx_attn = F.softmax(q @ k.transpose(-2, -1), dim=-1)
        x = approx_attn @ v
        # ======================================== debug ========================================
        self.capture['approx'] = approx_attn
        self.capture['softmax'] = F.softmax(q @ k.transpose(-2, -1), dim=-1)
        # ======================================== debug ========================================
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

    def iterative_inv(self, mat, n_iter = 6):
        # works only for softmax matrix -> all positive, row stochastic
        I = torch.eye(mat.size(-1), device = mat.device)
        K = mat
        
        # The entries of K are positive and ||K||_{\infty} = 1 due to softmax
        if self.init_option == "original":
            # This original implementation is more conservative to compute coefficient of Z_0. 
            V = 1 / torch.max(torch.sum(K, dim = -2)) * K.transpose(-1, -2)
        else:
            # This is the exact coefficient computation, 1 / ||K||_1, of initialization of Z_0, leading to faster convergence. 
            V = 1 / torch.max(torch.sum(K, dim = -2), dim = -1).values[:, :, None, None] * K.transpose(-1, -2)
            
        for _ in range(n_iter):
            KV = torch.matmul(K, V)
            V = torch.matmul(0.25 * V, 13 * I - torch.matmul(KV, 15 * I - torch.matmul(KV, 7 * I - KV)))
        return V

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'