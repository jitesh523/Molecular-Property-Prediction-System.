import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import List, Optional


def smiles_to_morgan(
    smiles: str, radius: int = 2, n_bits: int = 2048, use_chirality: bool = True
) -> Optional[np.ndarray]:
    """
    Convert SMILES to Morgan Fingerprint (ECFP-equivalent) as a numpy bit vector.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # GetMorganFingerprintAsBitVect returns a ExplicitBitVect
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius, nBits=n_bits, useChirality=use_chirality
    )

    # Convert to numpy array
    arr = np.zeros((1,), dtype=np.int8)
    # Using the documented way to convert BitVect to numpy
    # In newer RDKit, DataStructs.ConvertToNumpyArray(fp, arr) is common
    from rdkit import DataStructs
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def batch_smiles_to_morgan(
    smiles_list: List[str], radius: int = 2, n_bits: int = 2048, use_chirality: bool = True
) -> np.ndarray:
    """
    Convert a list of SMILES to a matrix of Morgan Fingerprints.
    """
    fps = []
    for s in smiles_list:
        fp = smiles_to_morgan(s, radius, n_bits, use_chirality)
        if fp is None:
            # For ML batches, we usually need consistent shapes. 
            # If a SMILES is invalid here, it's a critical error because
            # it should have been caught in the preprocessing step.
            raise ValueError(f"Invalid SMILES encountered in featurization: {s}")
        fps.append(fp)
    return np.stack(fps)
