"""SMILES tokenization must preserve dummy atoms and attachment labels."""

import pytest
from rdkit import Chem

from molprop.data.smiles_vocab import UNK_IDX, SmilesVocab, tokenize


@pytest.mark.parametrize(
    "smiles, expected",
    [
        ("*", ["*"]),
        ("*CC", ["*", "C", "C"]),
        ("C(*)O", ["C", "(", "*", ")", "O"]),
        ("[1*]CC[2*]", ["[", "1", "*", "]", "C", "C", "[", "2", "*", "]"]),
        ("[*:1]O", ["[", "*", ":", "1", "]", "O"]),
    ],
)
def test_tokenize_preserves_wildcard_atoms(smiles, expected):
    assert tokenize(smiles) == expected


@pytest.mark.parametrize("smiles", ["*", "*CC", "C(*)O", "[1*]CC[2*]", "[*:1]O"])
def test_vocabulary_round_trip_preserves_molecular_structure(smiles):
    vocab = SmilesVocab.from_smiles([smiles])
    decoded = vocab.decode(vocab.encode(smiles))
    assert decoded == smiles

    original = Chem.MolFromSmiles(smiles)
    recovered = Chem.MolFromSmiles(decoded)
    assert original is not None
    assert recovered is not None
    assert Chem.MolToSmiles(recovered) == Chem.MolToSmiles(original)
    assert recovered.GetNumAtoms() == original.GetNumAtoms()
    assert sum(atom.GetAtomicNum() == 0 for atom in recovered.GetAtoms()) == smiles.count("*")


def test_unknown_wildcard_uses_unknown_token_in_existing_vocabulary():
    vocab = SmilesVocab.from_smiles(["CO"])
    assert vocab.encode("*C")[1:-1] == [UNK_IDX, vocab.token2idx["C"]]


def test_saved_vocabulary_preserves_labeled_attachment_points(tmp_path):
    smiles = "[1*]CC[2*]"
    vocab = SmilesVocab.from_smiles([smiles])
    path = tmp_path / "vocab.json"
    vocab.save(path)
    restored = SmilesVocab.load(path)
    assert restored.token2idx == vocab.token2idx
    assert restored.decode(restored.encode(smiles)) == smiles


@pytest.mark.parametrize("smiles", ["ClCBr", "C1=CC=CC=C1", "C[C@@H](O)F"])
def test_existing_multi_character_and_stereo_tokens_round_trip(smiles):
    vocab = SmilesVocab.from_smiles([smiles])
    assert vocab.decode(vocab.encode(smiles)) == smiles
