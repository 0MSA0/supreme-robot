from pathlib import Path
import textwrap
import urllib.request

import tiktoken
import torch

from .attention import SupremeMultiHeadAttention
from .data_loader import SupremeDataset
from .feed_forward import FeedForward
from .gelu import GELU
from .gpt_model import SupremeGPTModel
from .layer_norm import LayerNorm
from .transformer import TransformerBlock

__all__ = [
    "FeedForward",
    "GELU",
    "LayerNorm",
    "SupremeDataset",
    "SupremeGPTModel",
    "SupremeMultiHeadAttention",
    "TransformerBlock",
    "main",
    "test_generate_text",
    "test_text",
]


def main() -> None:
    print("Hello from supreme-robot!")

    start_context = "Hello, I am"
    test_text(start_context)


def test_text(start_context: str) -> None:
    # Tokenizer to be used
    tokenizer = tiktoken.get_encoding("gpt2")

    gpt_config_124_m = {
        "vocab_size": tokenizer.n_vocab,  # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,  # Embedding dimension
        "n_heads": 12,  # Number of attention heads
        "n_layers": 12,  # Number of layers
        "temperature": 1.0,  # Temperature
        "drop_rate": 0.1,  # Dropout rate
        "qkv_bias": True,  # Query-Key-Value bias
    }

    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    untrained_model = SupremeGPTModel(gpt_config_124_m)
    untrained_model.eval()

    out = test_generate_text(
        model=untrained_model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=gpt_config_124_m["context_length"],
        top_k=50,
    )

    decoded_text_untrained = tokenizer.decode(out.squeeze(0).tolist())
    print("Input:  ", start_context)
    print("Output: ", decoded_text_untrained)

    weights_url = "https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/gpt2-small-124M.pth"
    weights_path = Path(__file__).resolve().parent / "weights" / "gpt2-small-124M.pth"

    if not weights_path.exists():
        print(f"Downloading weights from {weights_url}...")
        urllib.request.urlretrieve(weights_url, weights_path)

    state_dict = torch.load(weights_path, map_location="cpu")

    trained_model = SupremeGPTModel(gpt_config_124_m)
    trained_model.load_state_dict(state_dict)
    trained_model.eval()

    out_trained = test_generate_text(
        model=trained_model,
        idx=encoded_tensor,
        max_new_tokens=50,
        context_size=gpt_config_124_m["context_length"],
        top_k=50,
    )

    decoded_text_trained = tokenizer.decode(out_trained.squeeze(0).tolist())
    print("Input: ", start_context, "\n")
    print("Output:\n" + "\n".join(textwrap.wrap(decoded_text_trained, width=80)))


def test_generate_text(model, idx, max_new_tokens, context_size, top_k=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k, dim=-1)
            min_val = top_logits[:, [-1]]
            logits = torch.where(
                logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits
            )

        probas = torch.softmax(logits, dim=-1)

        if top_k is not None:
            idx_next = torch.multinomial(probas, num_samples=1)
        else:
            idx_next = torch.argmax(probas, dim=-1, keepdim=True)

        idx = torch.cat((idx, idx_next), dim=1)

    return idx
