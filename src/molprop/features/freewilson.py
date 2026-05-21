"""
Free-Wilson SAR analysis.

Given a series of analogues that share a common core (R-group decomposed)
plus a measured property (e.g. pIC50, logP, potency), fit a least-squares
linear model where each R-group occupant becomes a one-hot feature.

The resulting model decomposes the property into:

    y_pred(mol) = intercept + Σᵢ contribution(R_i, occupant_i)

The contributions are interpretable additive R-group "deltas" — a classic
SAR/QSAR primitive (Free, Wilson 1964).

This implementation is dependency-free (uses ``numpy`` only) and uses the
Moore–Penrose pseudo-inverse for the least-squares fit so it gracefully
handles rank-deficient designs (small series, missing combinations).

Returned schema is JSON-friendly so it can be served directly from FastAPI
without extra serialisation logic.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from molprop.features.rgroups import decompose_rgroups


@dataclass
class FreeWilsonContribution:
    position: str
    occupant: str
    contribution: float
    n_occurrences: int

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class FreeWilsonResult:
    n_input: int
    n_matched: int
    n_used: int
    intercept: float
    r_squared: float
    rmse: float
    rgroup_labels: list[str]
    contributions: list[FreeWilsonContribution] = field(default_factory=list)
    predictions: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_input": self.n_input,
            "n_matched": self.n_matched,
            "n_used": self.n_used,
            "intercept": self.intercept,
            "r_squared": self.r_squared,
            "rmse": self.rmse,
            "rgroup_labels": list(self.rgroup_labels),
            "contributions": [c.to_dict() for c in self.contributions],
            "predictions": list(self.predictions),
        }


def free_wilson(
    core: str,
    smiles_list: list[str],
    activities: list[float],
    min_occurrences: int = 1,
) -> Optional[FreeWilsonResult]:
    """
    Fit a Free-Wilson additive R-group model.

    Parameters
    ----------
    core
        Core scaffold (SMARTS or SMILES) — same convention as
        :func:`molprop.features.rgroups.decompose_rgroups`.
    smiles_list
        Analogues to decompose around the core.
    activities
        Parallel list of measured property values (e.g. pIC50). Same length
        as ``smiles_list``.
    min_occurrences
        Drop R-group occupants that appear in fewer than this many molecules
        (helps avoid one-hot features that are perfectly co-linear with a
        single observation). Default 1 — keep everything.

    Returns
    -------
    FreeWilsonResult, or ``None`` if the core SMARTS cannot be parsed.
    """
    if len(activities) != len(smiles_list):
        raise ValueError("`activities` must be the same length as `smiles_list`")

    rg = decompose_rgroups(core, smiles_list)
    if rg is None:
        return None

    rgroup_labels = list(rg.rgroup_labels)
    if not rgroup_labels:
        return FreeWilsonResult(
            n_input=len(smiles_list),
            n_matched=rg.n_matched,
            n_used=0,
            intercept=float(np.mean(activities)) if activities else 0.0,
            r_squared=0.0,
            rmse=0.0,
            rgroup_labels=[],
        )

    # Index matched rows + their activities. ``rg.rows`` items are
    # RGroupRow dataclasses (see molprop.features.rgroups).
    matched_rows: list[tuple[dict, float, str]] = []
    for row, y in zip(rg.rows, activities, strict=True):
        if not getattr(row, "matched", False):
            continue
        r_groups = dict(getattr(row, "r_groups", {}) or {})
        matched_rows.append((r_groups, float(y), getattr(row, "smiles", "")))

    if not matched_rows:
        return FreeWilsonResult(
            n_input=len(smiles_list),
            n_matched=rg.n_matched,
            n_used=0,
            intercept=0.0,
            r_squared=0.0,
            rmse=0.0,
            rgroup_labels=rgroup_labels,
        )

    # Tally occupant frequencies per position
    counts: dict[str, dict[str, int]] = {pos: defaultdict(int) for pos in rgroup_labels}
    for r_groups, _y, _smi in matched_rows:
        for pos in rgroup_labels:
            occ = r_groups.get(pos, "")
            counts[pos][occ] += 1

    # Build feature lookup, filtered by min_occurrences. Reserve the most
    # frequent occupant per position as the implicit reference (its column
    # is dropped to avoid a dummy-variable trap and to keep the matrix
    # full-rank when possible).
    feature_index: dict[tuple[str, str], int] = {}
    references: dict[str, str] = {}
    for pos in rgroup_labels:
        sorted_occ = sorted(counts[pos].items(), key=lambda kv: (-kv[1], kv[0]))
        if not sorted_occ:
            continue
        ref_occ = sorted_occ[0][0]
        references[pos] = ref_occ
        for occ, n in sorted_occ:
            if occ == ref_occ:
                continue
            if n < min_occurrences:
                continue
            feature_index[(pos, occ)] = len(feature_index)

    n_features = len(feature_index)
    n_rows = len(matched_rows)

    # Design matrix [1 | one-hots]
    X = np.zeros((n_rows, 1 + n_features), dtype=float)
    X[:, 0] = 1.0
    y = np.zeros(n_rows, dtype=float)
    for i, (r_groups, yi, _smi) in enumerate(matched_rows):
        y[i] = yi
        for pos, occ in r_groups.items():
            j = feature_index.get((pos, occ))
            if j is not None:
                X[i, 1 + j] = 1.0

    # Pseudo-inverse least-squares (rank-safe)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    intercept = float(coef[0])
    y_pred = X @ coef
    residuals = y - y_pred
    rmse = float(np.sqrt(np.mean(residuals**2))) if n_rows else 0.0
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) if n_rows else 0.0
    r_squared = float(1.0 - np.sum(residuals**2) / ss_tot) if ss_tot > 1e-12 else 1.0

    # Build contribution list. Reference occupants get contribution 0.0
    # by construction; we still emit them so users see every occupant.
    contributions: list[FreeWilsonContribution] = []
    for pos in rgroup_labels:
        ref_occ = references.get(pos, "")
        for occ, n in counts[pos].items():
            if occ == ref_occ:
                contribution = 0.0
            else:
                idx = feature_index.get((pos, occ))
                contribution = float(coef[1 + idx]) if idx is not None else 0.0
            contributions.append(
                FreeWilsonContribution(
                    position=pos,
                    occupant=occ,
                    contribution=contribution,
                    n_occurrences=int(n),
                )
            )

    # Per-molecule prediction list
    predictions = [
        {
            "smiles": smi,
            "y_observed": float(y[i]),
            "y_predicted": float(y_pred[i]),
            "residual": float(residuals[i]),
        }
        for i, (_rg, _y, smi) in enumerate(matched_rows)
    ]

    return FreeWilsonResult(
        n_input=len(smiles_list),
        n_matched=rg.n_matched,
        n_used=n_rows,
        intercept=intercept,
        r_squared=r_squared,
        rmse=rmse,
        rgroup_labels=rgroup_labels,
        contributions=contributions,
        predictions=predictions,
    )
