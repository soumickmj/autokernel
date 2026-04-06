"""
Reference implementations -- PyTorch-only ground truth for correctness verification.
DO NOT MODIFY. These are the oracles that the benchmark harness checks against.
"""

import torch
import torch.nn.functional as F

# Matrix Multiplication
def matmul_ref(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Standard matrix multiplication. A @ B."""
    return torch.matmul(A, B)

# Softmax
def softmax_ref(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Standard softmax along dim."""
    return F.softmax(x, dim=dim)

# Layer Normalization
def layernorm_ref(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Layer normalization over last dimension."""
    normalized_shape = x.shape[-1:]
    return F.layer_norm(x, normalized_shape, weight, bias, eps)

# RMS Normalization
def rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMS normalization."""
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return (x / rms) * weight

# Flash Attention
def flash_attention_ref(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, causal: bool = True, sm_scale: float = None) -> torch.Tensor:
    """Standard scaled dot-product attention."""
    if sm_scale is None:
        sm_scale = Q.shape[-1] ** -0.5
    attn = torch.matmul(Q, K.transpose(-2, -1)) * sm_scale
    if causal:
        seq_len_q, seq_len_k = Q.shape[-2], K.shape[-2]
        mask = torch.triu(torch.ones(seq_len_q, seq_len_k, device=Q.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float('-inf'))
    attn = F.softmax(attn, dim=-1)
    return torch.matmul(attn, V)

# Fused MLP (SwiGLU-style)
def fused_mlp_ref(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor, activation: str = "silu") -> torch.Tensor:
    """SwiGLU-style fused MLP: down(activation(gate(x)) * up(x))."""
    gate = x @ w_gate.T
    up = x @ w_up.T
    if activation == "silu":
        gate = F.silu(gate)
    elif activation == "gelu":
        gate = F.gelu(gate)
    elif activation == "relu2":
        gate = F.relu(gate) ** 2
    return (gate * up) @ w_down.T

# Cross Entropy Loss
def cross_entropy_ref(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Standard cross entropy loss."""
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

# Rotary Position Embedding
def rotary_embedding_ref(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    return torch.stack([rx1, rx2], dim=-1).flatten(-2)

# Parallel Reductions
def reduce_sum_ref(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sum reduction."""
    return x.sum(dim=dim)

def reduce_max_ref(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Max reduction."""
    return x.max(dim=dim).values

# 2D Convolution
def conv2d_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int = 1,
    padding: int = 0,
) -> torch.Tensor:
    """Standard 2D convolution."""
    return F.conv2d(x, weight, bias, stride=stride, padding=padding)

# 3D Convolution
def conv3d_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int = 1,
    padding: int = 0,
) -> torch.Tensor:
    """Standard 3D convolution."""
    return F.conv3d(x, weight, bias, stride=stride, padding=padding)

# Batch Normalization (2D)
def batchnorm2d_ref(
    x: torch.Tensor,
    running_mean: torch.Tensor | None,
    running_var: torch.Tensor | None,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Standard 2D batch normalization."""
    return F.batch_norm(x, running_mean, running_var, weight, bias, training, momentum, eps)
