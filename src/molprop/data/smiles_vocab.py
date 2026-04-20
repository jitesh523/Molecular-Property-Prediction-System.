"""
SMILES character-level tokenizer and vocabulary builder.

Provides a minimal but correct tokenizer that handles:
- Multi-character tokens: Cl, Br, Si, Se, etc.
- Special ring/bond notations: %, @, #, =, +, -, etc.
- Special tokens: <pad>, <sos>, <eos>, <unk>
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# Regex matching multi-char and single-char SMILES tokens (order matters)
_SMILES_PATTERN = re.compile(r"Cl|Br|Si|Se|@@|%\d{2}|[A-Za-z]|[\[\]()=#@+\-:\/\\\.%0-9]")

# Special token definitions
PAD = "<pad>"
SOS = "<sos>"
EOS = "<eos>"
UNK = "<unk>"
SPECIAL = [PAD, SOS, EOS, UNK]

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


def tokenize(smiles: str) -> list[str]:
    """Split a SMILES string into character-level tokens."""
    return _SMILES_PATTERN.findall(smiles)


class SmilesVocab:
    """Character-level SMILES vocabulary."""

    def __init__(self, token2idx: dict[str, int]):
        self.token2idx: dict[str, int] = token2idx
        self.idx2token: dict[int, str] = {v: k for k, v in token2idx.items()}

    def __len__(self) -> int:
        return len(self.token2idx)

    # ── Encoding ───────────────────────────────────────────────────────────────
    def encode(self, smiles: str, max_len: int | None = None) -> list[int]:
        """Encode a SMILES string → int list (with SOS/EOS).

        Args:
            smiles:  Input SMILES string.
            max_len: If given, truncates before EOS so total length ≤ max_len.

        Returns:
            List of token indices [SOS, t1, t2, …, EOS].
        """
        toks = tokenize(smiles)
        ids = [SOS_IDX]
        for t in toks:
            ids.append(self.token2idx.get(t, UNK_IDX))
        ids.append(EOS_IDX)
        if max_len is not None and len(ids) > max_len:
            ids = ids[: max_len - 1] + [EOS_IDX]
        return ids

    def decode(self, ids: list[int], strip_special: bool = True) -> str:
        """Decode int list → SMILES string."""
        tokens = []
        for idx in ids:
            tok = self.idx2token.get(idx, UNK)
            if strip_special and tok in SPECIAL:
                if tok == EOS:
                    break  # stop at end-of-sequence
                continue
            tokens.append(tok)
        return "".join(tokens)

    # ── Persistence ────────────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.token2idx, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "SmilesVocab":
        token2idx = json.loads(Path(path).read_text())
        return cls(token2idx)

    # ── Factory ────────────────────────────────────────────────────────────────
    @classmethod
    def from_smiles(cls, smiles_list: list[str]) -> "SmilesVocab":
        """Build vocabulary from a list of SMILES strings."""
        all_tokens: set[str] = set()
        for s in smiles_list:
            all_tokens.update(tokenize(s))

        # Build index: specials first, then sorted corpus tokens
        token2idx: dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL)}
        for tok in sorted(all_tokens):
            if tok not in token2idx:
                token2idx[tok] = len(token2idx)
        return cls(token2idx)
