import torch
import torch.nn as nn
import math

class LinformerAttention(nn.Module):
    projection_matrix = None

    def __init__(self, dim, num_heads, num_landmarks, max_seq_len, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.num_head = num_heads
        self.head_dim = dim // num_heads
        self.linformer_k = num_landmarks
        self.seq_len = max_seq_len

        if LinformerAttention.projection_matrix is not None:
            self.E = LinformerAttention.projection_matrix
        else:
            LinformerAttention.projection_matrix = nn.Parameter(torch.Tensor(self.num_head, self.linformer_k, self.seq_len))
            torch.nn.init.normal_(LinformerAttention.projection_matrix, std = 0.02)
            self.E = LinformerAttention.projection_matrix

        self.qkv_self = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv_self(x).reshape(B_, N, 3, self.num_head, C // self.num_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C
        x_out = self.attention(q, k, v, (B_, N, C))

        # projection
        x = self.attn_drop(x_out)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def attention(self, Q, K, V, x_shape):
        B_, seq_len, C = x_shape
        K = self.E @ K
        V = self.E @ V

        dot = Q @ torch.transpose(K, -2, -1)
        dot = dot / math.sqrt(self.head_dim)

        attn = nn.functional.softmax(dot, dim = -1)

        X = (attn @ V).reshape(B_, seq_len, C)

        return X

    def attention_mask(self, Q, K, V, mask):
        K = torch.matmul(self.E, K * mask[:, None, :, None])
        V = torch.matmul(self.E, V * mask[:, None, :, None])

        dot = torch.matmul(Q, torch.transpose(K, -2, -1))
        dot = dot / math.sqrt(self.head_dim)

        attn = nn.functional.softmax(dot, dim = -1)

        X = torch.matmul(attn, V)

        return X