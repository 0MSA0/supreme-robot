import torch.nn as nn
from supreme_robot import SupremeMultiHeadAttention
from supreme_robot.feed_forward import FeedForward
from supreme_robot.layer_norm import LayerNorm


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
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
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
