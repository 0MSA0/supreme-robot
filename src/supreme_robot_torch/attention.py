import torch.nn as nn
import torch.nn.functional as F


class SupremeMultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_k: int,
        context_length: int,  # kept for API compatibility; no mask buffer needed anymore
        temperature: float,
        dropout: float,
        num_heads: int,
        qkv_bias=False,
    ):
        super().__init__()
        assert d_k % num_heads == 0, "d_k must be divisible by num_heads"

        self.d_out = d_out
        self.d_k = d_k
        self.temperature = temperature
        self.num_heads = num_heads
        self.head_dim = d_k // num_heads
        self.dropout_p = dropout

        # Same scaling as before: softmax(QK^T / (sqrt(head_dim) * temperature))
        self.scale = 1.0 / (self.head_dim**0.5 * temperature)

        self.W_key = nn.Linear(d_in, d_k, bias=qkv_bias)
        self.W_query = nn.Linear(d_in, d_k, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_k, bias=qkv_bias)
        self.out_proj = nn.Linear(d_k, d_out)

    def forward(self, x):
        b, num_tokens, _ = x.shape
        h, hd = self.num_heads, self.head_dim

        # (b, T, d_k) -> (b, h, T, hd)
        q = self.W_query(x).view(b, num_tokens, h, hd).transpose(1, 2)
        k = self.W_key(x).view(b, num_tokens, h, hd).transpose(1, 2)
        v = self.W_value(x).view(b, num_tokens, h, hd).transpose(1, 2)

        # Fused kernel: FlashAttention for fp16/bf16 on CUDA (falls back
        # automatically to mem-efficient / math kernels otherwise).
        # is_causal=True replaces the explicit mask (no mask tensor -> flash-eligible).
        context_vec = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
            scale=self.scale,
        )

        # (b, h, T, hd) -> (b, T, d_k)
        context_vec = context_vec.transpose(1, 2).reshape(b, num_tokens, self.d_k)
        return self.out_proj(context_vec)
