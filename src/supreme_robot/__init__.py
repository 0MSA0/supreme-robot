import torch.nn as nn
import torch

from .attention import SupremeMultiHeadAttention


def main() -> None:
    print("Hello from supreme-robot!")

    inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89, 0.42],  # (x^1)
            [0.55, 0.87, 0.66, 0.69],  # (x^2)
            [0.57, 0.85, 0.64, 0.91],  # (x^3)
            [0.22, 0.58, 0.33, 0.22],  # (x^4)
            [0.77, 0.25, 0.10, 0.79],  # (x^5)
            [0.05, 0.80, 0.55, 0.19],
        ]  # (x^6)
    )

    d_in = inputs.shape[1]  # Input Embedding size
    d_out = d_in  # Output embedding size

    # 2 inputs with 6 tokens of dimension 4
    batch = torch.stack((inputs, inputs), dim=0)

    batch_size, context_length, d_in = batch.shape

    torch.manual_seed(420)
    d_k = 8
    temperature = 1e-8

    mha = SupremeMultiHeadAttention(
        d_in,
        d_out,
        d_k,
        context_length,
        temperature,
        0.0,
        num_heads=2
    )

    context_vecs = mha(batch)

    print("d_k: ", d_k, "temp: ", temperature)
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape, "\n")

    torch.manual_seed(420)
    d_k = 8
    temperature = 0.5
    mha = SupremeMultiHeadAttention(
        d_in,
        d_out,
        d_k,
        context_length,
        temperature,
        0.0,
        num_heads=2
    )

    context_vecs = mha(batch)

    print("d_k: ", d_k, "temp: ", temperature)
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape, "\n")

    torch.manual_seed(420)
    d_k = 4
    temperature = 0.5
    mha = SupremeMultiHeadAttention(
        d_in,
        d_out,
        d_k,
        context_length,
        temperature,
        0.0,
        num_heads=2
    )

    context_vecs = mha(batch)

    print("d_k: ", d_k, "temp: ", temperature)
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)
