"""
SHAP explainability for tree-based baseline models (RF, XGBoost).

Uses TreeExplainer for exact Shapley values and maps important Morgan
fingerprint bits back to atom substructures via RDKit's bitInfo dictionary.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import shap
from rdkit import Chem
from rdkit.Chem import AllChem

log = logging.getLogger(__name__)


def get_shap_explanation(
    model: Any,
    x_test: np.ndarray,
    max_samples: int = 100,
) -> shap.Explanation:
    """
    Compute SHAP values for a tree-based model using TreeExplainer.

    Args:
        model: A fitted sklearn RF or XGBoost model.
        x_test: Test feature matrix (Morgan fingerprints).
        max_samples: Max number of samples to explain (for speed).

    Returns:
        shap.Explanation object containing Shapley values.
    """
    explainer = shap.TreeExplainer(model)
    x_subset = x_test[:max_samples]
    shap_values = explainer(x_subset)
    return shap_values


def get_top_important_bits(
    shap_values: shap.Explanation,
    top_k: int = 20,
) -> List[int]:
    """
    Identify the top-k most important fingerprint bits by mean |SHAP value|.
    """
    # shap_values.values may be 3D for classification (samples, features, classes)
    vals = shap_values.values
    if vals.ndim == 3:
        vals = vals[:, :, 1]  # take positive class for binary classification
    mean_abs = np.abs(vals).mean(axis=0)
    top_indices = np.argsort(mean_abs)[::-1][:top_k]
    return top_indices.tolist()


def map_bits_to_fragments(
    smiles_list: List[str],
    important_bits: List[int],
    radius: int = 2,
    n_bits: int = 2048,
) -> Dict[int, Dict[str, Any]]:
    """
    Map Morgan fingerprint bit indices back to atom environments.

    For each important bit, finds molecules that activate that bit and
    extracts the atom environment (substructure) responsible.

    Returns:
        Dict mapping bit_index -> {
            "smarts": SMARTS pattern of the substructure,
            "example_smiles": a molecule that activates this bit,
            "atoms": list of atom indices in the environment,
            "radius": the radius at which this bit was set,
        }
    """
    bit_info_map: Dict[int, Dict[str, Any]] = {}

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        bit_info: Dict[int, list] = {}
        AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits, bitInfo=bit_info)

        for bit_idx in important_bits:
            if bit_idx in bit_info and bit_idx not in bit_info_map:
                # bit_info[bit_idx] is a list of (atom_center, radius) tuples
                center_atom, bit_radius = bit_info[bit_idx][0]
                # Get the atom environment
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, bit_radius, center_atom)
                atoms = set()
                for bond_idx in env:
                    bond = mol.GetBondWithIdx(bond_idx)
                    atoms.add(bond.GetBeginAtomIdx())
                    atoms.add(bond.GetEndAtomIdx())
                atoms.add(center_atom)

                # Generate SMARTS for the substructure
                try:
                    submol = Chem.PathToSubmol(mol, env)
                    smarts = Chem.MolToSmarts(submol)
                except Exception:
                    smarts = "N/A"

                bit_info_map[bit_idx] = {
                    "smarts": smarts,
                    "example_smiles": smiles,
                    "atoms": sorted(atoms),
                    "radius": bit_radius,
                }

        # Stop early if we've mapped all bits
        if len(bit_info_map) == len(important_bits):
            break

    return bit_info_map


def save_shap_report(
    shap_values: shap.Explanation,
    smiles_list: List[str],
    output_dir: Path,
    dataset_name: str,
    model_name: str,
    top_k: int = 20,
    radius: int = 2,
    n_bits: int = 2048,
) -> Path:
    """
    Generate and save a complete SHAP explanation report.

    Saves:
      - shap_summary.json: top bits, their SHAP importance, and fragment mappings
      - shap_global_importance.png: bar plot of global feature importance
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Get top important bits
    top_bits = get_top_important_bits(shap_values, top_k=top_k)

    # 2. Map bits to fragments
    fragment_map = map_bits_to_fragments(smiles_list, top_bits, radius=radius, n_bits=n_bits)

    # 3. Build importance values
    vals = shap_values.values
    if vals.ndim == 3:
        vals = vals[:, :, 1]
    mean_abs = np.abs(vals).mean(axis=0)

    # 4. Build JSON report
    report = {
        "dataset": dataset_name,
        "model": model_name,
        "top_k": top_k,
        "important_bits": [],
    }
    for bit_idx in top_bits:
        entry = {
            "bit_index": int(bit_idx),
            "mean_abs_shap": float(mean_abs[bit_idx]),
        }
        if bit_idx in fragment_map:
            entry.update(fragment_map[bit_idx])
        report["important_bits"].append(entry)

    json_path = output_dir / f"shap_{model_name}_{dataset_name}.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    log.info(f"SHAP report saved to {json_path}")

    # 5. Save global importance bar plot
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        bit_labels = [f"bit_{b}" for b in top_bits]
        importances = [float(mean_abs[b]) for b in top_bits]
        ax.barh(range(len(top_bits)), importances, color="#4ECDC4")
        ax.set_yticks(range(len(top_bits)))
        ax.set_yticklabels(bit_labels, fontsize=8)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"Top-{top_k} Morgan Bit Importance ({model_name} on {dataset_name})")
        ax.invert_yaxis()
        plt.tight_layout()
        png_path = output_dir / f"shap_{model_name}_{dataset_name}.png"
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        log.info(f"SHAP plot saved to {png_path}")
    except ImportError:
        log.warning("matplotlib not available; skipping SHAP plot generation.")

    return json_path
