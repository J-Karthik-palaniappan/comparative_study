import torch
from attention import PnPNystraAttention, NystromAttention, LinformerAttention
from models import swinir

B = 9                 # batch size
N = 1024               # sequence length (must be close to square factorable)
C = 180                # embedding dimension
num_heads = 6
num_landmarks = 16
iters = 2
window_size = [32]

# Create random input
x = torch.randn(B, N, C)

# Create attention module
attn = NystromAttention(
    num_landmarks=num_landmarks,
    iters=iters,
    dim=C,
    num_heads=num_heads,
    window_size = window_size,
    # use_conv=False
)

# attn = LinformerAttention(
#     dim=C,
#     num_heads=num_heads,
#     linformer_k = num_landmarks,
#     max_seq_len = N,
# )

# Forward pass
out = attn(x)

print("Input shape :", x.shape)
print("Output shape:", out.shape)