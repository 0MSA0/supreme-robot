import torch
from torch.utils.data import Dataset, DataLoader


class SupremeDataset(Dataset):
    def __init__(self, input_text: str, tokenizer, max_length, stride) -> None:
        self.input_ids = []
        self.target_ids = []

        encoded_words = tokenizer.encode(input_text, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the input text
        # into overlapping sequences of max_length
        for i in range(0, len(encoded_words) - max_length, stride):
            input_chunk = encoded_words[i : i + max_length]
            target_chunk = encoded_words[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]

def create_supreme_dataloader(
    txt,
    tokenizer,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):

    # Create dataset
    dataset = SupremeDataset(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader