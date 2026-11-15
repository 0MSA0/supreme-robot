import torch.nn as nn

from .attention import SupremeMultiHeadAttention
from .feed_forward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.att = SupremeMultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            d_k=cfg["emb_dim"],
            context_length=cfg["context_length"],
            temperature=cfg["temperature"],
            dropout=cfg["drop_rate"],
            num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"],
        )

        self.ff = FeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg["emb_dim"], eps=1e-5)
        self.norm2 = nn.LayerNorm(cfg["emb_dim"], eps=1e-5)
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x
