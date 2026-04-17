from typing import List, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from torch_geometric.explain import Explanation


def explanation_to_highlights(
    explanation: Explanation, threshold: float = 0.1, top_k: int = 5
) -> Tuple[List[int], List[int]]:
    """
    Extracts atom and bond indices to highlight from a PyG Explanation object.
    """
    # Node attribution (aggregate across features if attributes mask is present)
    if hasattr(explanation, "node_mask"):
        node_scores = explanation.node_mask.abs().sum(dim=-1).numpy()
    else:
        node_scores = np.zeros(explanation.num_nodes)

    # Bond attribution
    if hasattr(explanation, "edge_mask"):
        edge_scores = explanation.edge_mask.numpy()
    else:
        edge_scores = np.zeros(explanation.num_edges)

    # Normalize scores to [0, 1]
    if node_scores.max() > 0:
        node_scores /= node_scores.max()
    if edge_scores.max() > 0:
        edge_scores /= edge_scores.max()

    # Get top-K atoms
    top_atoms = np.argsort(node_scores)[-top_k:].tolist()
    # Or atoms above threshold
    threshold_atoms = np.where(node_scores > threshold)[0].tolist()

    highlight_atoms = list(set(top_atoms + threshold_atoms))

    # For bonds, we need to map edge_index back to RDKit bond indices
    # However, highlighting bonds is trickier. Let's start with atoms only
    # as they provide the clearest "fragment" visual.

    return highlight_atoms, []


def draw_highlighted_mol(
    smiles: str, highlight_atoms: List[int], size: Tuple[int, int] = (400, 400)
) -> str:
    """
    Generates an SVG string of the molecule with highlighted atoms.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""

    # Generate coordinates if not present
    Chem.rdDepictor.Compute2DCoords(mol)

    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    options = drawer.drawOptions()
    options.useBWAtomPalette()  # Cleaner look for highlights

    # We can also add colors here
    highlight_colors = {i: (0.2, 0.8, 0.2) for i in highlight_atoms}  # Green for importance

    drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms, highlightAtomColors=highlight_colors)
    drawer.FinishDrawing()

    return drawer.GetDrawingText()


def get_explanation_image(
    smiles: str, explanation: Explanation, size: Tuple[int, int] = (400, 400)
) -> str:
    """
    High-level utility to go from explanation to highlighted SVG.
    """
    atoms, _ = explanation_to_highlights(explanation)
    return draw_highlighted_mol(smiles, atoms, size=size)
