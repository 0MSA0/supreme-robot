from pathlib import Path
import textwrap
#import urllib.request

import tiktoken
import torch

from datasets import load_dataset

from supreme_robot_torch.attention import SupremeMultiHeadAttention
from supreme_robot_torch.feed_forward import FeedForward
from supreme_robot_torch.trainer import train_model
from supreme_robot_torch.transformer import TransformerBlock
from supreme_robot_torch.gpt_model import SupremeGPTModel
from supreme_robot_torch.streaming_data_loader import StreamingSupremeDataset, create_streaming_dataloader

__all__ = [
    "SupremeGPTModel",
    "SupremeMultiHeadAttention",
    "TransformerBlock",
    "FeedForward",
    "StreamingSupremeDataset",
    "create_streaming_dataloader",
    "train_model"
]

def main() -> None:
    print("Hello from supreme-robot-torch!")
    
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Training settings
    config = {
        "epochs": 1,
        "lr": 1.5e-4,
        "batch_size": 8,
        "weight_decay": 0.1,
        "data_loader_num_workers": 2,
    }

    # Model settings
    gpt_config_124_m = {
        "vocab_size": tokenizer.n_vocab,  # Vocabulary size
        "context_length": 1024,           # Context length
        "emb_dim": 768,                   # Embedding dimension
        "n_heads": 12,                    # Number of attention heads
        "n_layers": 12,                   # Number of layers
        "temperature": 1.0,               # Temperature
        "drop_rate": 0.1,                 # Dropout rate
        "qkv_bias": True,                 # Query-Key-Value bias
    }

    wandb_project_name = "supreme-robot-torch"

    save_path = Path(__file__).resolve().parents[2] / "weights" / "gpt2-small-124M.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model = SupremeGPTModel(gpt_config_124_m)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    lambda1 = lambda epoch: 0.65 ** epoch

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda1
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    data_stream = load_dataset(
        "Salesforce/wikitext",
        name="wikitext-103-raw-v1",
        split="train",
        streaming=True
    ).shuffle(seed=69420, buffer_size=10000)

    dataloader = create_streaming_dataloader(
        hf_stream=data_stream,
        tokenizer=tokenizer,
        batch_size=config["batch_size"],
        max_length=gpt_config_124_m["context_length"],
        stride=gpt_config_124_m["context_length"],
        shuffle=False,
        drop_last=True,
        num_workers=config["data_loader_num_workers"]
    )

    train_model(model,
                optimizer,
                lr_scheduler,
                loss_fn,
                dataloader,
                config["epochs"],
                wandb_project_name,
                config,
                save_path,
                use_amp=True,
                compile_model=True
    )
    
    test_text("Hello I am")
    

def test_text(start_context: str) -> None:
    # Tokenizer to be used
    tokenizer = tiktoken.get_encoding("gpt2")
    device = get_device()
    print(f"Using {device} device.")

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
    encoded_tensor = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)

    untrained_model = SupremeGPTModel(gpt_config_124_m).to(device)
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

    #weights_url = "https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/gpt2-small-124M.pth"
    weights_path = Path(__file__).resolve().parent / "weights" / "gpt2-small-124M.pth"

    if not weights_path.exists():
        #print(f"Downloading weights from {weights_url}...")
        #weights_path.parent.mkdir(parents=True, exist_ok=True)
        #urllib.request.urlretrieve(weights_url, weights_path)
        print(f"Error: Weights file not found at {weights_path}. Please ensure the weights file is present.")
        exit(1)

    state_dict = normalize_state_dict(torch.load(weights_path, map_location=device))

    trained_model = SupremeGPTModel(gpt_config_124_m).to(device)
    trained_model.load_state_dict(state_dict, strict=True)
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
    device = next(model.parameters()).device
    idx = idx.to(device=device, dtype=torch.long)

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


def normalize_state_dict(state_dict):
    normalized = {}

    for key, value in state_dict.items():
        if key.endswith(".att.mask"):
            continue

        key = key.replace(".norm1.scale", ".norm1.weight")
        key = key.replace(".norm1.shift", ".norm1.bias")
        key = key.replace(".norm2.scale", ".norm2.weight")
        key = key.replace(".norm2.shift", ".norm2.bias")

        if key.startswith("final_norm.scale"):
            key = key.replace("final_norm.scale", "final_norm.weight")
        elif key.startswith("final_norm.shift"):
            key = key.replace("final_norm.shift", "final_norm.bias")

        normalized[key] = value

    return normalized


def get_device():
    # Get device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        # Use PyTorch 2.9 or newer for stable mps results
        major, minor = map(int, torch.__version__.split(".")[:2])
        if (major, minor) >= (2, 9):
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    return device
