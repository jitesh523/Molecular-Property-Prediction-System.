import logging
from typing import Optional

from rdkit import Chem
from rdkit.Chem import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize

log = logging.getLogger(__name__)

# Pre-initialize heavy RDKit objects to avoid recreating per molecule
salt_remover = SaltRemover.SaltRemover()
normalizer = rdMolStandardize.Normalizer()
reionizer = rdMolStandardize.Reionizer()

def standardize_smiles(smiles: str, keep_chirality: bool = True) -> Optional[str]:
    """
    Standardize a SMILES string using RDKit standard protocol.

    Protocol:
      1. Parse SMILES → Mol
      2. Keep largest sub-fragment (handles e.g. "A.B" salt mixtures)
      3. Strip remaining salts using default SaltRemover
      4. Normalize functional groups
      5. Reionize
      6. Canonicalize SMILES (with optional stereochemistry)
      
    Args:
        smiles (str): Input SMILES string
        keep_chirality (bool): If True, preserves stereochemical information in output
        
    Returns:
        Optional[str]: Clean canonical SMILES, or None if parsing fails
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            log.debug(f"Failed to parse SMILES: {smiles}")
            return None

        # Keep largest fragment (handles "A.B" mixtures)
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        if not frags:
            return None
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())

        # Clean steps
        mol = salt_remover.StripMol(mol, dontRemoveEverything=True)
        mol = normalizer.normalize(mol)
        mol = reionizer.reionize(mol)

        # Canonicalize
        # Note: useChiral in Python is expected as an int by RDKit
        can = Chem.CanonSmiles(Chem.MolToSmiles(mol), useChiral=int(keep_chirality))
        return can

    except Exception as e:
        log.debug(f"Standardization exception for {smiles}: {e}")
        return None
