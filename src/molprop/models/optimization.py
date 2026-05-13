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


def compute_qed(smiles: str) -> float | None:
    """Compute QED (drug-likeness) score using RDKit."""
    from rdkit.Chem import QED

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return QED.qed(mol)


def compute_sas(smiles: str) -> float | None:
    """Compute synthetic accessibility score (SAS) using RDKit."""

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # SAScore based on fragment contributions - simplified version
    # Full implementation requires SA_Score module, fallback to ring complexity
    n_rings = Lipinski.RingCount(mol)
    n_rot = Lipinski.NumRotatableBonds(mol)
    n_stereo = len(Chem.FindMolChiralCenters(mol))

    # Simple heuristic: more rings + rotatable bonds + chiral centers = harder to synthesize
    sas = 1.0 + 0.2 * n_rings + 0.1 * n_rot + 0.3 * n_stereo
    return min(sas, 10.0)  # Cap at 10


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

        # Drug-likeness (QED) - 0.0 to 1.0, higher is better
        if "qed" in targets or "QED" in targets:
            key = "qed" if "qed" in targets else "QED"
            props["qed"] = compute_qed(smiles) or 0.0
            min_v, max_v = targets[key]
            if props["qed"] < min_v:
                penalties += (min_v - props["qed"]) ** 2 * 100  # Weight heavily
            elif props["qed"] > max_v:
                penalties += (props["qed"] - max_v) ** 2 * 100

        # Synthetic Accessibility Score (SAS) - 1.0 to 10.0, lower is better (easier to make)
        if "sas" in targets or "SAS" in targets:
            key = "sas" if "sas" in targets else "SAS"
            props["sas"] = compute_sas(smiles) or 10.0
            min_v, max_v = targets[key]
            if props["sas"] < min_v:
                penalties += (min_v - props["sas"]) ** 2
            elif props["sas"] > max_v:
                penalties += (props["sas"] - max_v) ** 2 * 2  # Penalize high SAS more

        # GNN predictor (if available)
        builtin_props = (
            "mw",
            "MW",
            "logp",
            "LogP",
            "tpsa",
            "TPSA",
            "hbd",
            "HBD",
            "hba",
            "HBA",
            "qed",
            "QED",
            "sas",
            "SAS",
        )
        if self.predictor and any(k not in builtin_props for k in targets):
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

    def encode_seed(self, smiles: str) -> torch.Tensor | None:
        """Encode a seed SMILES string to a latent vector using the VAE."""

        std_smiles = standardize_smiles(smiles)
        if not std_smiles:
            return None

        try:
            tokens = self.vocab.encode(std_smiles, max_len=self.max_len)
            x = torch.tensor([tokens], dtype=torch.long, device=self.device)
            with torch.no_grad():
                mu, logvar = self.vae.encode(x)
                # Use mean (mu) as the latent representation
                z = self.vae.reparameterize(mu, logvar)
            return z
        except Exception:
            return None

    def optimize_gradient_ascent(
        self,
        targets: dict[str, tuple[float, float]],
        n_candidates: int = 10,
        n_steps: int = 50,
        lr: float = 0.1,
        noise_scale: float = 0.5,
        temperature: float = 0.8,
        seed_smiles: str | None = None,
    ) -> list[dict]:
        """
        Gradient-guided optimization in latent space.

        Uses finite-difference gradients to navigate toward molecules
        with desired properties.

        Args:
            seed_smiles: Optional seed molecule to start optimization from.
                        If None, starts from random latent vectors.

        Returns:
            List of dicts with "smiles", "score", "properties", "z".
        """
        results = []

        # Encode seed molecule if provided
        seed_z = None
        if seed_smiles:
            seed_z = self.encode_seed(seed_smiles)

        for _ in range(n_candidates):
            # Start from seed or random point in latent space
            if seed_z is not None:
                # Add small noise to seed for diversity
                z = seed_z + torch.randn_like(seed_z) * 0.5
            else:
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
        seed_smiles: str | None = None,
    ) -> list[dict]:
        """
        Sample latent space and select best matches (baseline comparison).

        Args:
            seed_smiles: Optional seed molecule to bias sampling toward.

        Returns:
            List of top-n dicts with "smiles", "score", "properties", "z".
        """
        results = []

        # Encode seed if provided
        seed_z = None
        if seed_smiles:
            seed_z = self.encode_seed(seed_smiles)

        # Batch decoding for efficiency
        batch_size = 20
        for _ in range(n_samples // batch_size):
            if seed_z is not None:
                # Sample around seed with increasing variance
                z = (
                    seed_z.repeat(batch_size, 1)
                    + torch.randn(batch_size, self.latent_dim, device=self.device) * 1.5
                )
            else:
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
        seed_smiles: str | None = None,
    ) -> list[dict]:
        """
        Main entry point for optimization.

        Args:
            targets: Dict of property name -> (min, max) target range.
            method: "gradient_ascent" or "random_walk".
            n_candidates: Number of optimization runs.
            temperature: VAE decoding temperature.
            seed_smiles: Optional seed molecule to start from.

        Returns:
            Ranked list of optimized molecules.
        """
        if method == "gradient_ascent":
            return self.optimize_gradient_ascent(
                targets, n_candidates=n_candidates, temperature=temperature, seed_smiles=seed_smiles
            )
        elif method == "random_walk":
            return self.optimize_random_walk(
                targets,
                n_samples=n_candidates * 10,
                n_top=n_candidates,
                temperature=temperature,
                seed_smiles=seed_smiles,
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
