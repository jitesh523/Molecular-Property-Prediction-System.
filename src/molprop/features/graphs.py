from typing import List, Optional

import torch
from rdkit import Chem
from torch_geometric.data import Data


def atom_to_features(atom: Chem.Atom) -> List[float]:
    """
    Convert an RDKit atom into a 9-dimensional continuous/integer feature vector.
    """
    features = [
        float(atom.GetAtomicNum()),
        float(atom.GetTotalDegree()),
        float(atom.GetFormalCharge()),
        float(int(atom.GetHybridization())),
        float(int(atom.GetIsAromatic())),
        float(int(atom.IsInRing())),
        float(atom.GetTotalNumHs()),
        float(atom.GetNumRadicalElectrons()),
        float(int(atom.GetChiralTag())),
    ]
    return features


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
