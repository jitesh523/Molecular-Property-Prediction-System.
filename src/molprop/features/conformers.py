"""
3: 3D Conformer generation for skeletal molecules.

Uses RDKit's ETKDGv3 algorithm for initial embedding followed by
MMFF94 force field optimization to produce high-quality 3D geometries.
"""

import logging
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem

log = logging.getLogger(__name__)

def generate_3d_conformer(smiles: str, num_attempts: int = 10) -> Optional[Chem.Mol]:
    """
    Generate a single, optimized 3D conformer for a SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Add hydrogens (critical for 3D physics)
    mol = Chem.AddHs(mol)

    # Embed molecule in 3D space using ETKDG (Experimental-Torsion Knowledge Distance Geometry)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    
    embed_res = AllChem.EmbedMolecule(mol, params)
    if embed_res == -1:
        log.warning(f"Could not embed molecule: {smiles}")
        return None

    # Optimize geometry using MMFF (Merck Molecular Force Field)
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception as e:
        log.warning(f"MMFF optimization failed for {smiles}: {e}")
        # Return the unoptimized conformer as fallback
        pass

    return mol

def mol_to_xyz(mol: Chem.Mol) -> str:
    """Helper to convert RDKit molecule to XYZ format string."""
    return Chem.MolToXYZBlock(mol)

def mol_to_pdb(mol: Chem.Mol) -> str:
    """Helper to convert RDKit molecule to PDB/PDBQT format for 3Dmol.js."""
    return Chem.MolToPDBBlock(mol)
