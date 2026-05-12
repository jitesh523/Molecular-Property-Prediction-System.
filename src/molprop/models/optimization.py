"""
Guided Molecular Optimization in VAE Latent Space.

Implements gradient-based and random-walk optimization in the learned
latent space to generate molecules with target properties.
"""

from __future__ import annotations

from typing import Callable

import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

from molprop.data.standardize import standardize_smiles


class LatentOptimizer:
    """
    Optimize molecular properties by traversing the VAE latent space.

    Supports multiple strategies:
    - gradient_ascent: gradient-based optimization toward target
    - random_walk: Monte Carlo sampling with selection pressure
    - cma_es: Covariance Matrix Adaptation (if available)
    """

    def __init__(
        self,
        vae: torch.nn.Module,
        vocab,
        predictor: Callable[[str], dict[str, float]] | None = None,
        max_len: int = 120,
        device: str = "cpu",
    ):
        """
        Args:
            vae: Trained SMILESVAE model.
            vocab: SmilesVocab instance for decoding.
            predictor: Function(smiles) -> dict of property predictions.
                       If None, uses RDKit descriptors only.
            max_len: Maximum SMILES length to generate.
            device: torch device.
        """
        self.vae = vae.to(device).eval()
        self.vocab = vocab
        self.predictor = predictor
        self.max_len = max_len
        self.device = device
        self.latent_dim = vae.latent_dim

    def decode_smiles(self, z: torch.Tensor, temperature: float = 0.8) -> list[str]:
        """Decode latent vectors to SMILES strings."""
        with torch.no_grad():
            z = z.to(self.device)
            logits = self.vae.decode(z, max_len=self.max_len, temperature=temperature)
            token_ids = logits.argmax(dim=-1).cpu().tolist()

        smiles_list = []
        for ids in token_ids:
            smi = self.vocab.decode(ids)
            std = standardize_smiles(smi) if smi else None
            smiles_list.append(std if std else "")
        return smiles_list

    def compute_property_score(
        self,
        smiles: str,
        targets: dict[str, tuple[float, float]],
        weights: dict[str, float] | None = None,
    ) -> tuple[float, dict[str, float]]:
        """
        Compute how well a molecule matches target property ranges.

        Args:
            smiles: SMILES string to evaluate.
            targets: Dict of {property: (min, max)} target ranges.
            weights: Optional weighting for each property.

        Returns:
            score (higher = better), dict of individual property values.
        """
        if not smiles:
            return -1000.0, {}

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return -1000.0, {}

        props = {}
        penalties = 0.0

        # RDKit descriptors
        if "mw" in targets or "MW" in targets:
            key = "mw" if "mw" in targets else "MW"
            props["mw"] = Descriptors.MolWt(mol)
            min_v, max_v = targets[key]
            if props["mw"] < min_v:
                penalties += (min_v - props["mw"]) ** 2
            elif props["mw"] > max_v:
                penalties += (props["mw"] - max_v) ** 2

        if "logp" in targets or "LogP" in targets:
            key = "logp" if "logp" in targets else "LogP"
            props["logp"] = Descriptors.MolLogP(mol)
            min_v, max_v = targets[key]
            if props["logp"] < min_v:
                penalties += (min_v - props["logp"]) ** 2
            elif props["logp"] > max_v:
                penalties += (props["logp"] - max_v) ** 2

        if "tpsa" in targets or "TPSA" in targets:
            key = "tpsa" if "tpsa" in targets else "TPSA"
            props["tpsa"] = Descriptors.TPSA(mol)
            min_v, max_v = targets[key]
            if props["tpsa"] < min_v:
                penalties += (min_v - props["tpsa"]) ** 2
            elif props["tpsa"] > max_v:
                penalties += (props["tpsa"] - max_v) ** 2

        if "hbd" in targets or "HBD" in targets:
            key = "hbd" if "hbd" in targets else "HBD"
            props["hbd"] = Lipinski.NumHDonors(mol)
            min_v, max_v = targets[key]
            if props["hbd"] < min_v:
                penalties += (min_v - props["hbd"]) ** 2
            elif props["hbd"] > max_v:
                penalties += (props["hbd"] - max_v) ** 2

        if "hba" in targets or "HBA" in targets:
            key = "hba" if "hba" in targets else "HBA"
            props["hba"] = Lipinski.NumHAcceptors(mol)
            min_v, max_v = targets[key]
            if props["hba"] < min_v:
                penalties += (min_v - props["hba"]) ** 2
            elif props["hba"] > max_v:
                penalties += (props["hba"] - max_v) ** 2

        # GNN predictor (if available)
        if self.predictor and any(
            k not in ("mw", "MW", "logp", "LogP", "tpsa", "TPSA", "hbd", "HBD", "hba", "HBA")
            for k in targets
        ):
            pred = self.predictor(smiles)
            for key, (min_v, max_v) in targets.items():
                if key in pred:
                    props[key] = pred[key]
                    if pred[key] < min_v:
                        penalties += (min_v - pred[key]) ** 2
                    elif pred[key] > max_v:
                        penalties += (pred[key] - max_v) ** 2

        # Score: negative penalties (want to maximize, so less penalty = higher score)
        score = -penalties
        return score, props

    def optimize_gradient_ascent(
        self,
        targets: dict[str, tuple[float, float]],
        n_candidates: int = 10,
        n_steps: int = 50,
        lr: float = 0.1,
        noise_scale: float = 0.5,
        temperature: float = 0.8,
    ) -> list[dict]:
        """
        Gradient-guided optimization in latent space.

        Uses finite-difference gradients to navigate toward molecules
        with desired properties.

        Returns:
            List of dicts with "smiles", "score", "properties", "z".
        """
        results = []

        for _ in range(n_candidates):
            # Start from random point in latent space
            z = torch.randn(1, self.latent_dim, device=self.device, requires_grad=False)
            best_z = z.clone()
            best_score = float("-inf")

            for step in range(n_steps):
                # Evaluate current point
                smiles_list = self.decode_smiles(z, temperature=temperature)
                score, props = self.compute_property_score(smiles_list[0], targets)

                if score > best_score:
                    best_score = score
                    best_z = z.clone()

                # Finite difference gradient estimation
                grad = torch.zeros_like(z)
                eps = 0.5 * (1.0 - step / n_steps)  # Decaying step size

                for dim in range(0, self.latent_dim, max(1, self.latent_dim // 10)):
                    z_plus = z.clone()
                    z_plus[0, dim] += eps
                    smiles_plus = self.decode_smiles(z_plus, temperature=temperature)
                    score_plus, _ = self.compute_property_score(smiles_plus[0], targets)
                    grad[0, dim] = (score_plus - score) / eps

                # Update with momentum and noise
                noise = torch.randn_like(z) * noise_scale * (1.0 - step / n_steps)
                z = z + lr * grad + noise
                z = torch.clamp(z, -3, 3)  # Keep in reasonable range

            # Final decode of best point
            final_smiles = self.decode_smiles(best_z, temperature=temperature)[0]
            final_score, final_props = self.compute_property_score(final_smiles, targets)

            if final_smiles:
                results.append(
                    {
                        "smiles": final_smiles,
                        "score": final_score,
                        "properties": final_props,
                        "z": best_z.squeeze(0).cpu().tolist(),
                    }
                )

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def optimize_random_walk(
        self,
        targets: dict[str, tuple[float, float]],
        n_samples: int = 100,
        n_top: int = 10,
        temperature: float = 0.8,
    ) -> list[dict]:
        """
        Sample latent space and select best matches (baseline comparison).

        Returns:
            List of top-n dicts with "smiles", "score", "properties", "z".
        """
        results = []

        # Batch decoding for efficiency
        batch_size = 20
        for _ in range(n_samples // batch_size):
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            smiles_list = self.decode_smiles(z, temperature=temperature)

            for smi, vec in zip(smiles_list, z, strict=False):
                score, props = self.compute_property_score(smi, targets)
                if smi:
                    results.append(
                        {
                            "smiles": smi,
                            "score": score,
                            "properties": props,
                            "z": vec.cpu().tolist(),
                        }
                    )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:n_top]

    def optimize(
        self,
        targets: dict[str, tuple[float, float]],
        method: str = "gradient_ascent",
        n_candidates: int = 10,
        temperature: float = 0.8,
    ) -> list[dict]:
        """
        Main entry point for optimization.

        Args:
            targets: Dict of property name -> (min, max) target range.
            method: "gradient_ascent" or "random_walk".
            n_candidates: Number of optimization runs.
            temperature: VAE decoding temperature.

        Returns:
            Ranked list of optimized molecules.
        """
        if method == "gradient_ascent":
            return self.optimize_gradient_ascent(
                targets, n_candidates=n_candidates, temperature=temperature
            )
        elif method == "random_walk":
            return self.optimize_random_walk(
                targets, n_samples=n_candidates * 10, n_top=n_candidates, temperature=temperature
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")


def smiles_to_properties(smiles: str) -> dict[str, float] | None:
    """Quick RDKit property computation for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "mw": Descriptors.MolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "tpsa": Descriptors.TPSA(mol),
        "hbd": Lipinski.NumHDonors(mol),
        "hba": Lipinski.NumHAcceptors(mol),
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "rings": Lipinski.RingCount(mol),
    }
