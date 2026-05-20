"""
Matched Molecular Pairs (MMP) analysis.

Given a list of molecules, fragment each one along every single, non-ring,
acyclic bond (single-cut MMP) and group molecules that share a common
"context" fragment. Pairs sharing a context with *different* substituents
form Matched Molecular Pairs.

This implementation is a lightweight single-cut MMP (the original Hussain &
Rea 2010 scheme allows multi-cut fragmentations; we restrict to one cut for
speed). It is well-suited to medicinal-chemistry SAR exploration on series
of ~10–1000 analogues.

References
----------
Hussain, J.; Rea, C. "Computationally Efficient Algorithm to Identify
Matched Molecular Pairs (MMPs) in Large Data Sets." JCIM 2010, 50, 339–348.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from rdkit import Chem


@dataclass
class MMPair:
    smiles_a: str
    smiles_b: str
    name_a: Optional[str]
    name_b: Optional[str]
    context: str  # the shared scaffold/context SMILES with [*:1] attachment
    r_a: str  # the substituent from molecule A
    r_b: str  # the substituent from molecule B
    delta_heavy_atoms: int

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class MMPResult:
    n_input: int
    n_valid: int
    n_pairs: int
    n_contexts: int
    pairs: list[MMPair] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_input": self.n_input,
            "n_valid": self.n_valid,
            "n_pairs": self.n_pairs,
            "n_contexts": self.n_contexts,
            "pairs": [p.to_dict() for p in self.pairs],
        }


def _single_cut_fragments(mol: Chem.Mol) -> list[tuple[str, str]]:
    """
    Return a list of (context_smiles, substituent_smiles) for every
    single-bond cut of a non-ring, acyclic, single bond.

    Both fragments carry a ``[*:1]`` dummy atom at the cleaved bond.
    """
    fragments: list[tuple[str, str]] = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        if bond.IsInRing():
            continue
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        try:
            frag_mol = Chem.FragmentOnBonds(mol, [bond.GetIdx()], dummyLabels=[(1, 1)])
        except Exception:  # noqa: S112
            continue
        smi = Chem.MolToSmiles(frag_mol)
        parts = smi.split(".")
        if len(parts) != 2:
            continue
        # Choose the larger half as the context (more heavy atoms)
        m0 = Chem.MolFromSmiles(parts[0])
        m1 = Chem.MolFromSmiles(parts[1])
        if m0 is None or m1 is None:
            continue
        if m0.GetNumHeavyAtoms() >= m1.GetNumHeavyAtoms():
            context, substituent = parts[0], parts[1]
        else:
            context, substituent = parts[1], parts[0]
        fragments.append((context, substituent))
        # silence unused warnings on edge cases
        _ = (a1, a2)
    return fragments


def find_mmp(
    smiles_list: list[str],
    names: Optional[list[str]] = None,
    max_substituent_atoms: int = 10,
    max_pairs: int = 500,
) -> MMPResult:
    """
    Compute single-cut Matched Molecular Pairs from a list of analogues.

    Parameters
    ----------
    smiles_list
        Molecules to analyse.
    names
        Optional parallel list of names (same length as ``smiles_list``).
    max_substituent_atoms
        Discard cuts whose substituent has more heavy atoms than this. The
        Hussain & Rea convention is ≤ 10; bigger cuts are usually
        scaffold-swaps, not single-point substitutions.
    max_pairs
        Cap the number of returned pairs (sorted by descending |Δ heavy
        atoms| then by context).
    """
    if names is None:
        names = [None] * len(smiles_list)
    if len(names) != len(smiles_list):
        raise ValueError("`names` must have the same length as `smiles_list`")

    # context -> list of (mol_idx, substituent_smiles)
    by_context: dict[str, list[tuple[int, str]]] = defaultdict(list)
    n_valid = 0
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        n_valid += 1
        seen_for_this_mol: set[tuple[str, str]] = set()
        for ctx, sub in _single_cut_fragments(mol):
            sub_mol = Chem.MolFromSmiles(sub)
            if sub_mol is None:
                continue
            if sub_mol.GetNumHeavyAtoms() - 1 > max_substituent_atoms:
                continue
            if (ctx, sub) in seen_for_this_mol:
                continue
            seen_for_this_mol.add((ctx, sub))
            by_context[ctx].append((i, sub))

    pairs: list[MMPair] = []
    for ctx, entries in by_context.items():
        if len(entries) < 2:
            continue
        # Generate all unordered pairs with different substituents
        for ai in range(len(entries)):
            for bi in range(ai + 1, len(entries)):
                idx_a, sub_a = entries[ai]
                idx_b, sub_b = entries[bi]
                if sub_a == sub_b:
                    continue
                m_a = Chem.MolFromSmiles(smiles_list[idx_a])
                m_b = Chem.MolFromSmiles(smiles_list[idx_b])
                if m_a is None or m_b is None:
                    continue
                pairs.append(
                    MMPair(
                        smiles_a=smiles_list[idx_a],
                        smiles_b=smiles_list[idx_b],
                        name_a=names[idx_a],
                        name_b=names[idx_b],
                        context=ctx,
                        r_a=sub_a,
                        r_b=sub_b,
                        delta_heavy_atoms=(m_b.GetNumHeavyAtoms() - m_a.GetNumHeavyAtoms()),
                    )
                )

    # Stable sort: by |Δ heavy atoms| ascending, then by context
    pairs.sort(key=lambda p: (abs(p.delta_heavy_atoms), p.context))
    pairs = pairs[:max_pairs]

    return MMPResult(
        n_input=len(smiles_list),
        n_valid=n_valid,
        n_pairs=len(pairs),
        n_contexts=sum(1 for v in by_context.values() if len(v) >= 2),
        pairs=pairs,
    )
