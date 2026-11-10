import torch
import torch.nn as nn


class SupremeMultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_k: int,
        context_length: int,
        temperature: float,
        dropout: float,
        num_heads: int,
        qkv_bias=False,
    ):
        super().__init__()
        # Ensure d_k is divisible by num_heads instead of d_out
        assert d_k % num_heads == 0, "d_k must be divisible by num_heads"

        # d_k and temperature as parameters
        self.d_out = d_out
        self.d_k = d_k
        self.temperature = temperature
        self.num_heads = num_heads

        # Calculate head_dim based on d_k
        self.head_dim = d_k // num_heads

        # Project key, query and value to d_k instead of d_out
        self.W_key = nn.Linear(d_in, d_k, bias=qkv_bias)
        self.W_query = nn.Linear(d_in, d_k, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_k, bias=qkv_bias)

        # Pretty much the same as above
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

        # Layer for projecting d_k back to output dimension d_out
        self.out_proj = nn.Linear(d_k, d_out)

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # Pretty much the same as above
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Pretty much the same as above
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -6.9e127)

        # Softmax with temperature
        attn_weights = torch.softmax(
            attn_scores / (self.head_dim**0.5 * self.temperature), dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        # Calculate the context vectors
        context_vec = attn_weights @ values

        # Bring 'num_heads' back to index 2 (b, num_tokens, num_heads, head_dim)
        # and flatten the head dimensions back to d_k (b, num_tokens, d_k).
        context_vec = context_vec.transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_k)

        # Project back to d_out
        context_vec = self.out_proj(context_vec)
        return context_vec
