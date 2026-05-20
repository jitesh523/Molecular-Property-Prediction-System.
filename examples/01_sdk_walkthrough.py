"""
End-to-end walkthrough of the molprop Python SDK.

Runs every major cheminformatics endpoint against a small panel of common
drug-like molecules and prints a concise summary suitable for copy-pasting
into a notebook or README.

Run with a live API server::

    uvicorn molprop.serving.api:app --reload &
    python examples/01_sdk_walkthrough.py
"""

from __future__ import annotations

import json
import sys
from typing import Any

from molprop.client import MolpropAPIError, MolpropClient

# A tiny panel of well-known molecules
PANEL = {
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "acetaminophen": "CC(=O)NC1=CC=C(O)C=C1",
    "ibuprofen": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "ethanol": "CCO",
}


def _section(title: str) -> None:
    print(f"\n{'─' * 70}\n {title}\n{'─' * 70}")


def _short(obj: Any, max_len: int = 240) -> str:
    s = json.dumps(obj, default=str)
    return s if len(s) <= max_len else s[:max_len] + "…"


def main() -> int:
    client = MolpropClient("http://localhost:8000", timeout=30)

    # 1. Health check + version (also acts as connectivity probe)
    _section("1. Health & version")
    try:
        print(" health  →", _short(client.health()))
        print(" version →", _short(client.version()))
    except MolpropAPIError as e:
        print(f"  Server not reachable: {e}", file=sys.stderr)
        print("  Start it with: uvicorn molprop.serving.api:app --reload", file=sys.stderr)
        return 1

    # 2. Scaffold + SAScore for every panel member
    _section("2. Bemis–Murcko scaffolds + SAScore")
    for name, smi in PANEL.items():
        try:
            s = client.scaffold(smi)
            print(
                f"  {name:<14} murcko={s.get('murcko_smiles') or '(acyclic)'}  "
                f"SA={s.get('sa_score'):.2f}/10"
            )
        except MolpropAPIError as e:
            print(f"  {name}: {e.detail}")

    # 3. Functional groups
    _section("3. Functional groups (aspirin)")
    fg = client.functional_groups(PANEL["aspirin"])
    names = sorted({g["name"] for g in fg.get("groups", [])})
    print("  detected:", ", ".join(names))

    # 4. Compare two analogues + MCS
    _section("4. Compare ethanol vs methanol  +  MCS aspirin vs acetaminophen")
    cmp_ = client.compare("CCO", "CO")
    print(f"  ethanol↔methanol  MACCS Tanimoto = {cmp_.get('tanimoto_maccs'):.3f}")
    m = client.mcs(PANEL["aspirin"], PANEL["acetaminophen"])
    print(
        f"  MCS atoms={m['n_atoms']}  bonds={m['n_bonds']}  "
        f"covA={m['a_coverage'] * 100:.1f}%  covB={m['b_coverage'] * 100:.1f}%"
    )

    # 5. R-group decomposition over a phenol mini-series
    _section("5. R-group decomposition (benzene core)")
    rg = client.rgroups("c1ccccc1", ["c1ccc(O)cc1", "c1ccc(N)cc1", "c1ccc(C)cc1"])
    print(f"  matched {rg['n_matched']}/{rg['n_input']}  R-positions: {rg['rgroup_labels']}")
    for row in rg["rows"]:
        if row["matched"]:
            print(f"    {row['smiles']:<15} {row['r_groups']}")

    # 6. Reaction SMARTS — apply amide coupling to two acid+amine pairs
    _section("6. Reaction: amide coupling (named) on two acid+amine pairs")
    rx = client.react(
        substrates=[
            ["CC(=O)O", "NCC"],  # acetic acid + ethylamine
            ["c1ccccc1C(=O)O", "NCC"],  # benzoic acid + ethylamine
        ],
        named="amide_coupling",
    )
    print(f"  product sets: {rx['n_product_sets']}/{rx['n_input_sets']}")
    print("  unique products:", rx["unique_products"][:6])

    # 7. Structural alerts on every panel member
    _section("7. PAINS / Brenk / NIH structural alerts")
    for name, smi in PANEL.items():
        a = client.alerts(smi, catalogs=["PAINS", "BRENK", "NIH"])
        status = "✅ clean" if a["is_clean"] else f"⚠️  {a['n_alerts']} alerts"
        print(f"  {name:<14} {status}")

    # 8. Library round-trip
    _section("8. Compound library round-trip")
    project = "sdk_walkthrough_demo"
    saved_ids: list[int] = []
    for name, smi in PANEL.items():
        rec = client.library_save(smi, name=name, project=project, tags=["panel"])
        saved_ids.append(rec["id"])
    listed = client.library_list(project=project)
    print(
        f"  saved {len(saved_ids)} compounds — library now lists {len(listed)} in project '{project}'"
    )
    for cid in saved_ids:
        client.library_delete(cid)
    print(f"  cleaned up {len(saved_ids)} compounds.")

    # 9. Markdown report (printed truncated)
    _section("9. One-click Markdown report (aspirin, first 12 lines)")
    md = client.report(
        PANEL["aspirin"],
        include_admet=False,
        include_descriptors=False,
    )["markdown"]
    for line in md.splitlines()[:12]:
        print(" ", line)

    print("\nDone.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
