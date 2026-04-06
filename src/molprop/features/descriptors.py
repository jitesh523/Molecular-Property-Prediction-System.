from typing import List, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors


def smiles_to_descriptors(smiles: str) -> Optional[np.ndarray]:
    """
    Extract a set of common physical descriptors from SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Common set of descriptors
    desc_vals = [
        Descriptors.MolLogP(mol),
        Descriptors.MolWt(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.MaxAbsPartialCharge(mol),
        Descriptors.MinAbsPartialCharge(mol),
        Descriptors.HeavyAtomCount(mol),
    ]

    return np.array(desc_vals, dtype=np.float32)


def batch_smiles_to_descriptors(smiles_list: List[str]) -> np.ndarray:
    """
    Batch conversion of SMILES to physical descriptors.
    """
    all_descs = []
    for s in smiles_list:
        d = smiles_to_descriptors(s)
        if d is None:
            raise ValueError(f"Invalid SMILES encountered in descriptor generation: {s}")
        all_descs.append(d)
    return np.stack(all_descs)


def get_descriptor_names() -> List[str]:
    """
    The names of the descriptors returned by smiles_to_descriptors.
    """
    return [
        "MolLogP",
        "MolWt",
        "TPSA",
        "NumHDonors",
        "NumHAcceptors",
        "NumRotatableBonds",
        "MaxAbsPartialCharge",
        "MinAbsPartialCharge",
        "HeavyAtomCount",
    ]
