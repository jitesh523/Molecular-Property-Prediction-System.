import logging
from typing import Dict, Optional

from rdkit import Chem
from rdkit.Chem import Descriptors, SaltRemover
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


def passes_lipinski_ro5(
    smiles: str,
    mw_limit: float = 500.0,
    logp_limit: float = 5.0,
    hbd_limit: int = 5,
    hba_limit: int = 10,
) -> Optional[Dict[str, object]]:
    """
    Check whether a molecule passes Lipinski's Rule of Five.

    The Ro5 is a widely used heuristic for oral bioavailability: a drug
    candidate is likely orally active if it satisfies *at least 3 of 4*
    criteria (allowing one violation as per the original paper).

    Args:
        smiles: Input SMILES string (will be standardized first).
        mw_limit: Molecular weight upper bound (default 500 Da).
        logp_limit: LogP upper bound (default 5).
        hbd_limit: H-bond donor count upper bound (default 5).
        hba_limit: H-bond acceptor count upper bound (default 10).

    Returns:
        Dict with keys 'passes' (bool), 'violations' (list[str]),
        and individual property values, or None if SMILES is invalid.
    """
    std = standardize_smiles(smiles)
    if std is None:
        return None

    mol = Chem.MolFromSmiles(std)
    if mol is None:
        return None

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)

    violations = []
    if mw > mw_limit:
        violations.append(f"MW={mw:.1f} > {mw_limit}")
    if logp > logp_limit:
        violations.append(f"LogP={logp:.2f} > {logp_limit}")
    if hbd > hbd_limit:
        violations.append(f"HBD={hbd} > {hbd_limit}")
    if hba > hba_limit:
        violations.append(f"HBA={hba} > {hba_limit}")

    return {
        "passes": len(violations) <= 1,
        "violations": violations,
        "MW": round(mw, 2),
        "LogP": round(logp, 3),
        "HBD": hbd,
        "HBA": hba,
    }


def veber_filter(
    smiles: str,
    rot_bonds_limit: int = 10,
    tpsa_limit: float = 140.0,
) -> Optional[Dict]:
    """
    Veber's oral bioavailability filter.

    A molecule passes if **both** conditions hold:
    - Rotatable bonds ≤ ``rot_bonds_limit`` (default 10)
    - TPSA ≤ ``tpsa_limit`` Å² (default 140)

    Reference: Veber et al., J. Med. Chem. 2002, 45, 2615–2623.

    Args:
        smiles: Input SMILES string.
        rot_bonds_limit: Maximum allowed rotatable bonds.
        tpsa_limit: Maximum allowed TPSA (Å²).

    Returns:
        Dict with 'passes' (bool), 'violations' (list), 'RotatableBonds' and
        'TPSA' values, or None if the SMILES is invalid.
    """
    std = standardize_smiles(smiles)
    if std is None:
        return None
    mol = Chem.MolFromSmiles(std)
    if mol is None:
        return None

    rot_bonds = Descriptors.NumRotatableBonds(mol)
    tpsa = Descriptors.TPSA(mol)

    violations = []
    if rot_bonds > rot_bonds_limit:
        violations.append(f"RotatableBonds={rot_bonds} > {rot_bonds_limit}")
    if tpsa > tpsa_limit:
        violations.append(f"TPSA={tpsa:.1f} > {tpsa_limit}")

    return {
        "passes": len(violations) == 0,
        "violations": violations,
        "RotatableBonds": rot_bonds,
        "TPSA": round(tpsa, 2),
    }


def ghose_filter(smiles: str) -> Optional[Dict]:
    """
    Ghose drug-likeness filter.

    A molecule passes if **all four** property ranges are satisfied:
    - −0.4 ≤ LogP ≤ 5.6
    - 160 ≤ MW ≤ 480 Da
    - 40 ≤ Molar Refractivity ≤ 130
    - 20 ≤ NumAtoms ≤ 70

    Reference: Ghose et al., J. Comb. Chem. 1999, 1, 55–68.

    Args:
        smiles: Input SMILES string.

    Returns:
        Dict with 'passes' (bool), 'violations' (list), and individual
        property values (LogP, MW, MR, NumAtoms), or None for invalid SMILES.
    """
    std = standardize_smiles(smiles)
    if std is None:
        return None
    mol = Chem.MolFromSmiles(std)
    if mol is None:
        return None

    logp = Descriptors.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    mr = Descriptors.MolMR(mol)
    num_atoms = mol.GetNumAtoms()

    violations = []
    if not (-0.4 <= logp <= 5.6):
        violations.append(f"LogP={logp:.2f} outside [-0.4, 5.6]")
    if not (160 <= mw <= 480):
        violations.append(f"MW={mw:.1f} outside [160, 480]")
    if not (40 <= mr <= 130):
        violations.append(f"MR={mr:.1f} outside [40, 130]")
    if not (20 <= num_atoms <= 70):
        violations.append(f"NumAtoms={num_atoms} outside [20, 70]")

    return {
        "passes": len(violations) == 0,
        "violations": violations,
        "LogP": round(logp, 3),
        "MW": round(mw, 2),
        "MR": round(mr, 2),
        "NumAtoms": num_atoms,
    }
