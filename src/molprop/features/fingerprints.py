from typing import List, Optional

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys


def smiles_to_morgan(
    smiles: str, radius: int = 2, n_bits: int = 2048, use_chirality: bool = True
) -> Optional[np.ndarray]:
    """
    Convert SMILES to Morgan Fingerprint (ECFP-equivalent) as a numpy bit vector.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Use the modern MorganGenerator if available (RDKit 2023.09+)
    try:
        from rdkit.Chem import rdFingerprintGenerator

        gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits, includeChirality=use_chirality
        )
        fp = gen.GetFingerprint(mol)
    except (ImportError, AttributeError):
        # Fallback for older versions
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


def smiles_to_maccs(smiles: str) -> Optional[np.ndarray]:
    """
    Convert a SMILES string to a 167-bit MACCS keys fingerprint.

    MACCS keys encode the presence or absence of 166 defined structural
    fragments and are widely used in virtual screening and SAR analysis.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros((167,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def batch_smiles_to_maccs(smiles_list: List[str]) -> np.ndarray:
    """
    Convert a list of SMILES to a matrix of MACCS keys fingerprints.
    """
    fps = []
    for s in smiles_list:
        fp = smiles_to_maccs(s)
        if fp is None:
            raise ValueError(f"Invalid SMILES encountered in MACCS featurization: {s}")
        fps.append(fp)
    return np.stack(fps)
