"""
PyTorch Dataset that tokenizes SMILES strings for VAE training.
Pads sequences to a fixed max_len and returns input_ids tensors.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from molprop.data.smiles_vocab import PAD_IDX, SmilesVocab


class VAEDataset(Dataset):
    """Tokenised SMILES dataset for the SMILES VAE.

    Args:
        smiles_list: Raw SMILES strings.
        vocab:       SmilesVocab instance.
        max_len:     Maximum padded sequence length.
    """

    def __init__(self, smiles_list: list[str], vocab: SmilesVocab, max_len: int = 120):
        self.vocab = vocab
        self.max_len = max_len
        self.data: list[torch.Tensor] = []

        for smi in smiles_list:
            ids = vocab.encode(smi, max_len=max_len)
            # Pad to max_len
            if len(ids) < max_len:
                ids += [PAD_IDX] * (max_len - len(ids))
            self.data.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]
