"""
Apply a reaction SMARTS to one or more substrate SMILES.

Wraps RDKit's `AllChem.ReactionFromSmarts` + `RunReactants`. Returns the
unique canonical product SMILES for every set of reactant matches.

Includes a small built-in library of named reactions for convenience
(amide_coupling, ester_hydrolysis, suzuki_coupling, etc.) which users can
reference by name instead of pasting a SMARTS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem

# ── Named reaction library ───────────────────────────────────────────────────
# Format: name -> SMARTS (single-reactant or multi-reactant separated by '.')

NAMED_REACTIONS: dict[str, dict[str, str]] = {
    "amide_coupling": {
        "smarts": "[C:1](=[O:2])[OH].[NH2:3]>>[C:1](=[O:2])[NH:3]",
        "n_reactants": 2,
        "description": "Carboxylic acid + primary amine → amide",
    },
    "ester_hydrolysis": {
        "smarts": "[C:1](=[O:2])[O:3][C:4]>>[C:1](=[O:2])[OH].[OH:3][C:4]",
        "n_reactants": 1,
        "description": "Ester → carboxylic acid + alcohol",
    },
    "n_methylation": {
        "smarts": "[NH2:1]>>[NH:1]C",
        "n_reactants": 1,
        "description": "Primary amine → N-methylated amine",
    },
    "alcohol_to_aldehyde": {
        "smarts": "[CH2:1][OH]>>[CH:1]=O",
        "n_reactants": 1,
        "description": "Primary alcohol → aldehyde (oxidation)",
    },
    "suzuki_coupling": {
        "smarts": "[c:1][Br].[c:2][B]([OH])[OH]>>[c:1][c:2]",
        "n_reactants": 2,
        "description": "Aryl bromide + arylboronic acid → biaryl",
    },
    "nitro_reduction": {
        "smarts": "[N+:1](=O)[O-]>>[NH2:1]",
        "n_reactants": 1,
        "description": "Nitro → primary amine (reduction)",
    },
}


@dataclass
class ReactionResult:
    smarts: str
    n_reactants: int
    n_input_sets: int
    n_product_sets: int
    products: list[list[str]]  # one list of product SMILES per match
    unique_products: list[str]

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def run_reaction(
    smarts: str,
    substrates: list[list[str]],
) -> Optional[ReactionResult]:
    """
    Apply a reaction SMARTS to a list of substrate tuples.

    ``substrates`` is a list of reactant-tuples. For a single-reactant
    reaction pass ``[[smi_a], [smi_b], ...]``; for a 2-reactant reaction
    pass ``[[smi_a, smi_b], [smi_a, smi_c], ...]``.

    Returns ``None`` if the SMARTS is invalid.
    """
    try:
        rxn = AllChem.ReactionFromSmarts(smarts)
    except Exception:
        return None
    if rxn is None:
        return None

    n_reactants = rxn.GetNumReactantTemplates()
    products: list[list[str]] = []
    seen: set[str] = set()

    for tup in substrates:
        if len(tup) != n_reactants:
            continue
        mols = []
        ok = True
        for smi in tup:
            m = Chem.MolFromSmiles(smi)
            if m is None:
                ok = False
                break
            mols.append(m)
        if not ok:
            continue
        try:
            outs = rxn.RunReactants(tuple(mols))
        except Exception:  # noqa: S112
            continue
        # outs: tuple of tuples of products
        local_set: list[str] = []
        for prod_tuple in outs:
            for p in prod_tuple:
                try:
                    Chem.SanitizeMol(p)
                except Exception:  # noqa: S112
                    continue
                smi = Chem.MolToSmiles(p)
                if smi and smi not in set(local_set):
                    local_set.append(smi)
                seen.add(smi)
        if local_set:
            products.append(local_set)

    return ReactionResult(
        smarts=smarts,
        n_reactants=n_reactants,
        n_input_sets=len(substrates),
        n_product_sets=len(products),
        products=products,
        unique_products=sorted(seen),
    )


def list_named_reactions() -> list[dict]:
    return [
        {
            "name": name,
            "smarts": cfg["smarts"],
            "n_reactants": cfg["n_reactants"],
            "description": cfg["description"],
        }
        for name, cfg in NAMED_REACTIONS.items()
    ]
