import pandas as pd
import torch
from torch.utils.data import Dataset


class SMILESDataset(Dataset):
    """
    PyTorch Dataset mapping SMILES strings to Transformer input encodings.
    Handles dynamic tokenization.
    """

    def __init__(self, df: pd.DataFrame, target_cols: list, tokenizer, max_length: int = 128):
        # We assume standard CSV structure where "smiles" goes first
        self.smiles = df["smiles"].tolist()
        self.labels = df[target_cols].values
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        text = self.smiles[idx]

        # Tokenize single text string
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Squeeze out the batch dimension added by return_tensors='pt'
        item = {key: val.squeeze(0) for key, val in encoding.items()}

        # Labels are floats for both classification (BCEWithLogitsLoss) and regression
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item
