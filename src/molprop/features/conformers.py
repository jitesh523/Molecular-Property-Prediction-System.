"""
3D Conformer generation for skeletal molecules.

Uses RDKit's ETKDGv3 algorithm for initial embedding followed by
MMFF94 force field optimization to produce high-quality 3D geometries.
Also supports multi-conformer ensemble generation with RMSD-based pruning.
"""

import logging
from typing import List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

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


def generate_multiple_conformers(
    smiles: str,
    n_confs: int = 10,
    rmsd_prune_threshold: float = 0.5,
    max_attempts: int = 3,
) -> Optional[Chem.Mol]:
    """
    Generate an ensemble of diverse, energy-minimized 3D conformers.

    Uses ETKDGv3 for embedding followed by MMFF94 optimization and
    RMSD-based pruning to keep only geometrically distinct conformers.

    Args:
        smiles: Input SMILES string.
        n_confs: Target number of conformers to generate before pruning.
        rmsd_prune_threshold: Minimum RMSD (Å) between retained conformers.
        max_attempts: Maximum embedding attempts per conformer.

    Returns:
        RDKit Mol with multiple embedded conformers, or None if embedding fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 0
    params.pruneRmsThresh = rmsd_prune_threshold

    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    if len(conf_ids) == 0:
        log.warning(f"Could not embed any conformers for: {smiles}")
        return None

    # Optimize each conformer with MMFF94
    results = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0)
    failed = [i for i, (converged, energy) in enumerate(results) if converged == -1]
    if failed:
        log.debug(f"{len(failed)} conformers failed MMFF optimization — kept as-is")

    log.info(f"Generated {mol.GetNumConformers()} conformers for {smiles}")
    return mol


def get_conformer_rmsd(mol: Chem.Mol) -> List[Tuple[int, int, float]]:
    """
    Compute pairwise RMSD between all conformers in a molecule.

    Args:
        mol: RDKit Mol with multiple embedded conformers.

    Returns:
        List of (conf_i, conf_j, rmsd) tuples for all pairs i < j.
    """
    n = mol.GetNumConformers()
    if n < 2:
        return []

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            rmsd = rdMolAlign.GetBestRMS(mol, mol, i, j)
            pairs.append((i, j, round(rmsd, 4)))
    return pairs
