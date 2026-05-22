"""
Bemis–Murcko scaffold clustering.

Group a list of SMILES by their Bemis–Murcko scaffold (and optionally by the
generic Murcko framework, in which all atom labels collapse to carbon and
bond orders collapse to single). This is the canonical first step of a
"scaffold-based" SAR / library-curation workflow.

The function is dependency-light (RDKit only) and JSON-friendly so it can be
served directly through FastAPI.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


@dataclass
class ScaffoldCluster:
    scaffold_smiles: str
    size: int
    members: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "scaffold_smiles": self.scaffold_smiles,
            "size": self.size,
            "members": list(self.members),
        }


@dataclass
class ScaffoldClusterResult:
    n_input: int
    n_valid: int
    n_clusters: int
    generic: bool
    clusters: list[ScaffoldCluster] = field(default_factory=list)
    unmatched: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_input": self.n_input,
            "n_valid": self.n_valid,
            "n_clusters": self.n_clusters,
            "generic": self.generic,
            "clusters": [c.to_dict() for c in self.clusters],
            "unmatched": list(self.unmatched),
        }


def _scaffold_smiles(mol: Chem.Mol, generic: bool) -> str:
    """Return Bemis–Murcko (or generic Murcko) scaffold SMILES for ``mol``."""
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if generic:
            scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return ""


def cluster_by_scaffold(
    smiles_list: list[str],
    generic: bool = False,
    min_cluster_size: int = 1,
) -> ScaffoldClusterResult:
    """
    Cluster ``smiles_list`` by Bemis–Murcko scaffold.

    Parameters
    ----------
    smiles_list
        Input SMILES — invalid ones are reported under ``unmatched``.
    generic
        If ``True``, use the generic Murcko framework (all atoms → C, all
        bonds → single). Useful for grouping isosteres / bioisosteres.
    min_cluster_size
        Drop scaffolds with fewer than this many members (still counted in
        ``n_valid`` but excluded from ``clusters``). Default 1 keeps all.

    Returns
    -------
    ScaffoldClusterResult with clusters sorted by descending size, then by
    scaffold SMILES (deterministic).
    """
    buckets: dict[str, list[str]] = defaultdict(list)
    unmatched: list[str] = []
    n_valid = 0

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            unmatched.append(smi)
            continue
        scaffold = _scaffold_smiles(mol, generic)
        if not scaffold:
            unmatched.append(smi)
            continue
        n_valid += 1
        buckets[scaffold].append(smi)

    clusters: list[ScaffoldCluster] = [
        ScaffoldCluster(scaffold_smiles=k, size=len(v), members=v)
        for k, v in buckets.items()
        if len(v) >= min_cluster_size
    ]
    clusters.sort(key=lambda c: (-c.size, c.scaffold_smiles))

    return ScaffoldClusterResult(
        n_input=len(smiles_list),
        n_valid=n_valid,
        n_clusters=len(clusters),
        generic=generic,
        clusters=clusters,
        unmatched=unmatched,
    )
