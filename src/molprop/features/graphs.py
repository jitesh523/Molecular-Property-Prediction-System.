from typing import List, Optional

import torch
from rdkit import Chem
from torch_geometric.data import Data

# Vocabulary sets for one-hot encoding
_ATOM_TYPES = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]  # H C N O F P S Cl Br I + other
_DEGREE = [0, 1, 2, 3, 4, 5]
_HYBRIDIZATION = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
_NUM_HS = [0, 1, 2, 3, 4]
_FORMAL_CHARGE = [-2, -1, 0, 1, 2]

# Total feature dimension:
# atom_type: 11 (10 known + other)
# degree: 7 (0-5 + other)
# hybridization: 6 (5 known + other)
# num_hs: 6 (0-4 + other)
# formal_charge: 6 (5 known + other)
# is_aromatic: 1
# in_ring: 1
# num_radical_electrons (normalized): 1
# chiral_tag: 1
# Total: 40
ATOM_FEATURE_DIM = 40


def _one_hot(value, vocab: list) -> List[float]:
    """One-hot encode value against vocab; last position is the 'other' bucket."""
    encoding = [0.0] * (len(vocab) + 1)
    if value in vocab:
        encoding[vocab.index(value)] = 1.0
    else:
        encoding[-1] = 1.0
    return encoding


def atom_to_features(atom: Chem.Atom) -> List[float]:
    """
    Convert an RDKit atom into a normalized/one-hot feature vector (dim=40).

    Uses one-hot encoding for discrete properties (atom type, degree,
    hybridization, H count, formal charge) and scalar flags for binary/
    low-range properties, avoiding the scale mismatch that arises from mixing
    raw atomic numbers (0–118) with binary flags.
    """
    features: List[float] = []
    features += _one_hot(atom.GetAtomicNum(), _ATOM_TYPES)       # 11
    features += _one_hot(atom.GetTotalDegree(), _DEGREE)          # 7
    features += _one_hot(atom.GetHybridization(), _HYBRIDIZATION) # 6
    features += _one_hot(atom.GetTotalNumHs(), _NUM_HS)           # 6
    features += _one_hot(atom.GetFormalCharge(), _FORMAL_CHARGE)  # 6
    features.append(float(atom.GetIsAromatic()))                   # 1
    features.append(float(atom.IsInRing()))                        # 1
    features.append(min(atom.GetNumRadicalElectrons(), 4) / 4.0)  # 1  (normalized 0-1)
    features.append(float(int(atom.GetChiralTag())) / 3.0)        # 1  (CW/CCW/other, normalized)
    return features  # total: 40


def bond_to_features(bond: Chem.Bond) -> List[float]:
    """
    Convert an RDKit bond into a 3-dimensional feature vector.
    """
    bond_type = bond.GetBondType()
    if bond_type == Chem.rdchem.BondType.SINGLE:
        bt = 1.0
    elif bond_type == Chem.rdchem.BondType.DOUBLE:
        bt = 2.0
    elif bond_type == Chem.rdchem.BondType.TRIPLE:
        bt = 3.0
    elif bond_type == Chem.rdchem.BondType.AROMATIC:
        bt = 4.0
    else:
        bt = 0.0

    features = [
        bt,
        float(int(bond.GetStereo())),
        float(int(bond.IsInRing())),
    ]
    return features


def smiles_to_graph(smiles: str, y: Optional[float] = None) -> Optional[Data]:
    """
    Convert a SMILES string to a PyTorch Geometric Data object.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(atom_to_features(atom))
    x = torch.tensor(atom_features, dtype=torch.float)

    # Edge features and connectivity
    edge_index_list = []
    edge_attr_list = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        attr = bond_to_features(bond)

        # Add edges in both directions (undirected graph)
        edge_index_list.append([i, j])
        edge_index_list.append([j, i])
        edge_attr_list.append(attr)
        edge_attr_list.append(attr)

    if not edge_index_list:
        # Handling isolated atoms/empty graphs
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

    # Label
    y_tensor = None
    if y is not None:
        y_tensor = torch.tensor([y], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y_tensor, smiles=smiles)


def batch_smiles_to_graphs(
    smiles_list: List[str],
    labels: Optional[List[float]] = None,
) -> List[Optional[Data]]:
    """
    Batch-convert a list of SMILES strings to PyTorch Geometric Data objects.

    Args:
        smiles_list: List of SMILES strings.
        labels: Optional per-molecule target values aligned with ``smiles_list``.
            When provided, each valid Data object will carry a ``y`` tensor.

    Returns:
        List of Data objects; ``None`` at position ``i`` means SMILES ``i``
        was invalid or could not be featurized.
    """
    if labels is None:
        labels = [None] * len(smiles_list)
    return [smiles_to_graph(s, y=y) for s, y in zip(smiles_list, labels, strict=False)]
