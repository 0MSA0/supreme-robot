import torch
from torch.utils.data import IterableDataset, DataLoader

class StreamingSupremeDataset(IterableDataset):
    def __init__(self, hf_stream, tokenizer, max_length, stride):
        self.hf_stream = hf_stream
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride

        self.eos_id = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

    def __iter__(self):
        token_buffer = []


        for item in self.hf_stream:

            tokens = self.tokenizer.encode(item["text"], allowed_special={"<|endoftext|>"})
            tokens.append(self.eos_id)
            token_buffer.extend(tokens)

            # Sobald der Buffer genug Tokens hat, Chunks ausgeben
            while len(token_buffer) >= self.max_length + 1:
                input_chunk = token_buffer[:self.max_length]
                target_chunk = token_buffer[1:self.max_length + 1]

                yield torch.tensor(input_chunk), torch.tensor(target_chunk)

                # Buffer um 'stride' weiterrücken
                token_buffer = token_buffer[self.stride:]