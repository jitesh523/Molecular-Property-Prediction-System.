"""
Multi-Property Pareto Optimization for Molecular Design.

Finds molecules that are simultaneously optimal across multiple properties
without any single objective dominating (NSGA-II inspired Pareto fronts).
"""

from __future__ import annotations

from typing import Any

import torch

from molprop.models.optimization import (
    LatentOptimizer,
    compute_qed,
    compute_sas,
    smiles_to_properties,
)


def dominates(a: dict[str, float], b: dict[str, float], objectives: list[str]) -> bool:
    """
    Return True if solution `a` Pareto-dominates `b`.

    `a` dominates `b` if:
    - `a` is no worse than `b` on all objectives
    - `a` is strictly better than `b` on at least one objective
    """
    at_least_one_better = False
    for obj in objectives:
        va = a.get(obj, 0.0)
        vb = b.get(obj, 0.0)
        if va < vb:
            return False  # a is worse on this objective
        if va > vb:
            at_least_one_better = True
    return at_least_one_better


def compute_pareto_front(
    candidates: list[dict[str, Any]], objectives: list[str]
) -> list[dict[str, Any]]:
    """
    Identify the Pareto-optimal subset from a list of candidates.

    Args:
        candidates: List of dicts, each must have an "objectives" sub-dict.
        objectives: List of objective names to maximize.

    Returns:
        Subset of candidates that form the Pareto front (rank 0).
    """
    pareto = []
    for i, cand_i in enumerate(candidates):
        dominated = False
        for j, cand_j in enumerate(candidates):
            if i == j:
                continue
            if dominates(cand_j["objectives"], cand_i["objectives"], objectives):
                dominated = True
                break
        if not dominated:
            pareto.append(cand_i)
    return pareto


def compute_crowding_distance(
    pareto_front: list[dict[str, Any]], objectives: list[str]
) -> list[dict[str, Any]]:
    """
    Assign crowding distance to Pareto-front solutions for diversity preservation.

    Solutions with larger crowding distance are more isolated and preferred
    for diversity.
    """
    n = len(pareto_front)
    if n <= 2:
        for c in pareto_front:
            c["crowding_distance"] = float("inf")
        return pareto_front

    for c in pareto_front:
        c["crowding_distance"] = 0.0

    for obj in objectives:
        sorted_front = sorted(pareto_front, key=lambda x: x["objectives"].get(obj, 0.0))
        sorted_front[0]["crowding_distance"] = float("inf")
        sorted_front[-1]["crowding_distance"] = float("inf")

        obj_min = sorted_front[0]["objectives"].get(obj, 0.0)
        obj_max = sorted_front[-1]["objectives"].get(obj, 0.0)
        obj_range = obj_max - obj_min if obj_max != obj_min else 1e-9

        for k in range(1, n - 1):
            dist = (
                sorted_front[k + 1]["objectives"].get(obj, 0.0)
                - sorted_front[k - 1]["objectives"].get(obj, 0.0)
            ) / obj_range
            sorted_front[k]["crowding_distance"] += dist

    return pareto_front


class ParetoOptimizer:
    """
    Multi-objective Pareto optimization over VAE latent space.

    Generates a diverse pool of molecules and extracts those forming
    the Pareto front across user-specified objectives.
    """

    # Objectives to MAXIMIZE (we invert SAS since lower=better)
    SUPPORTED_OBJECTIVES = {
        "qed",
        "neg_sas",
        "logp_norm",
        "mw_norm",
        "tpsa_norm",
        "hbd_norm",
        "hba_norm",
    }

    def __init__(self, latent_optimizer: LatentOptimizer):
        """
        Args:
            latent_optimizer: A configured LatentOptimizer instance.
        """
        self.optimizer = latent_optimizer

    def _score_objectives(self, smiles: str, objective_names: list[str]) -> dict[str, float] | None:
        """
        Compute objective scores for a molecule (all as maximize).

        Objective names:
            qed       -> QED score (0-1), higher = better
            neg_sas   -> -SAS (1-10 inverted), higher = easier to synthesize
            logp_norm -> normalized LogP in [0,5] range
            mw_norm   -> normalized MW in [200,600] range
            tpsa_norm -> normalized TPSA in [20,140] range
        """
        props = smiles_to_properties(smiles)
        if props is None:
            return None

        scores: dict[str, float] = {}

        if "qed" in objective_names:
            scores["qed"] = compute_qed(smiles) or 0.0

        if "neg_sas" in objective_names:
            sas = compute_sas(smiles) or 10.0
            scores["neg_sas"] = -sas  # invert: higher = easier to synthesize

        if "logp_norm" in objective_names:
            # Optimal LogP is 0-5; score = 1 - (distance from center 2.5) / 2.5
            logp = props.get("logp", 0.0)
            scores["logp_norm"] = max(0.0, 1.0 - abs(logp - 2.5) / 2.5)

        if "mw_norm" in objective_names:
            # Optimal MW is 200-500; score = 1 - normalized distance from 350
            mw = props.get("mw", 0.0)
            scores["mw_norm"] = max(0.0, 1.0 - abs(mw - 350.0) / 350.0)

        if "tpsa_norm" in objective_names:
            # Optimal TPSA is 20-140; score = 1 - normalized distance from 80
            tpsa = props.get("tpsa", 0.0)
            scores["tpsa_norm"] = max(0.0, 1.0 - abs(tpsa - 80.0) / 80.0)

        if "hbd_norm" in objective_names:
            hbd = props.get("hbd", 0)
            scores["hbd_norm"] = max(0.0, 1.0 - max(0, hbd - 5) / 5.0)

        if "hba_norm" in objective_names:
            hba = props.get("hba", 0)
            scores["hba_norm"] = max(0.0, 1.0 - max(0, hba - 10) / 10.0)

        return scores

    def optimize_pareto(
        self,
        objectives: list[str],
        n_samples: int = 200,
        temperature: float = 0.8,
        seed_smiles: str | None = None,
    ) -> dict[str, Any]:
        """
        Sample the latent space and extract the Pareto front.

        Args:
            objectives: List of objective names to optimize simultaneously.
                        Supported: qed, neg_sas, logp_norm, mw_norm, tpsa_norm
            n_samples: Total molecules to sample before Pareto filtering.
            temperature: VAE decoding temperature.
            seed_smiles: Optional seed molecule for neighborhood exploration.

        Returns:
            Dict with "pareto_front", "all_candidates", "dominated" counts.
        """
        invalid_objs = [o for o in objectives if o not in self.SUPPORTED_OBJECTIVES]
        if invalid_objs:
            raise ValueError(
                f"Unsupported objectives: {invalid_objs}. Supported: {sorted(self.SUPPORTED_OBJECTIVES)}"
            )

        vae = self.optimizer.vae
        device = self.optimizer.device

        # Encode seed if provided
        seed_z = None
        if seed_smiles:
            seed_z = self.optimizer.encode_seed(seed_smiles)

        all_candidates = []
        batch_size = 20

        for _ in range(n_samples // batch_size):
            if seed_z is not None:
                z = (
                    seed_z.repeat(batch_size, 1)
                    + torch.randn(batch_size, vae.latent_dim, device=device) * 1.5
                )
            else:
                z = torch.randn(batch_size, vae.latent_dim, device=device)

            smiles_list = self.optimizer.decode_smiles(z, temperature=temperature)

            for smi, vec in zip(smiles_list, z, strict=False):
                if not smi:
                    continue
                obj_scores = self._score_objectives(smi, objectives)
                if obj_scores is None:
                    continue
                props = smiles_to_properties(smi) or {}
                qed_val = compute_qed(smi)
                sas_val = compute_sas(smi)
                if qed_val:
                    props["qed"] = qed_val
                if sas_val:
                    props["sas"] = sas_val

                all_candidates.append(
                    {
                        "smiles": smi,
                        "objectives": obj_scores,
                        "properties": props,
                        "z": vec.cpu().tolist(),
                    }
                )

        # Extract Pareto front
        pareto_front = compute_pareto_front(all_candidates, objectives)
        pareto_front = compute_crowding_distance(pareto_front, objectives)

        # Sort by crowding distance (diversity first), then by QED if available
        pareto_front.sort(key=lambda x: -x["crowding_distance"])

        return {
            "pareto_front": pareto_front,
            "total_sampled": len(all_candidates),
            "pareto_count": len(pareto_front),
            "dominated_count": len(all_candidates) - len(pareto_front),
            "objectives": objectives,
        }
