"""
ADMET Property Prediction using RDKit descriptors and rule-based filters.

Covers:
- Absorption: Lipinski Ro5, TPSA, LogP, Bioavailability score
- Distribution: BBB permeability (BBBP), PSA
- Metabolism: CYP450 inhibition risk flags
- Excretion: LogP-based renal clearance estimate
- Toxicity: PAINS alerts, hERG risk, Ames mutagenicity flag, structural alerts
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams


@dataclass
class ADMETResult:
    smiles: str
    absorption: dict[str, Any]
    distribution: dict[str, Any]
    metabolism: dict[str, Any]
    excretion: dict[str, Any]
    toxicity: dict[str, Any]
    overall_score: float  # 0-100 composite drug-likeness safety score
    alerts: list[str]  # human-readable warning strings
    pass_admet: bool  # simple overall pass/fail

    def to_dict(self) -> dict[str, Any]:
        return {
            "smiles": self.smiles,
            "absorption": self.absorption,
            "distribution": self.distribution,
            "metabolism": self.metabolism,
            "excretion": self.excretion,
            "toxicity": self.toxicity,
            "overall_score": self.overall_score,
            "alerts": self.alerts,
            "pass_admet": self.pass_admet,
        }


def _pains_alerts(mol: Chem.Mol) -> list[str]:
    """Check for PAINS (Pan Assay Interference Compounds) substructure alerts."""
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog(params)
    alerts = []
    entries = catalog.GetMatches(mol)
    for entry in entries:
        alerts.append(f"PAINS: {entry.GetDescription()}")
    return alerts


def _brenk_alerts(mol: Chem.Mol) -> list[str]:
    """Check for Brenk structural alert fragments (reactive/toxic groups)."""
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    catalog = FilterCatalog(params)
    alerts = []
    entries = catalog.GetMatches(mol)
    for entry in entries:
        alerts.append(f"Brenk alert: {entry.GetDescription()}")
    return alerts


def _herg_risk(mol: Chem.Mol, mw: float, logp: float) -> str:
    """
    Rule-based hERG cardiotoxicity risk estimate.

    High risk if: LogP > 3.7 AND MW > 300 AND basic nitrogen present.
    """
    has_basic_n = any(
        atom.GetAtomicNum() == 7 and atom.GetTotalNumHs() > 0 for atom in mol.GetAtoms()
    )
    if logp > 3.7 and mw > 300 and has_basic_n:
        return "High"
    if logp > 2.5 and mw > 250 and has_basic_n:
        return "Moderate"
    return "Low"


def _bbb_permeability(tpsa: float, mw: float, logp: float) -> str:
    """
    Rule-based BBB permeability prediction (Clark model approximation).

    BBB+ if: TPSA < 90, MW < 450, LogP 1-4.
    """
    if tpsa < 90 and mw < 450 and 1.0 <= logp <= 4.0:
        return "High"
    if tpsa < 120 and mw < 500:
        return "Moderate"
    return "Low"


def _ames_risk(mol: Chem.Mol) -> str:
    """
    Flag potential Ames mutagenicity based on presence of known mutagenic SMARTS.

    Only a heuristic — covers nitroaromatics, primary aromatic amines, etc.
    """
    MUTAGENIC_SMARTS = [
        "[N+](=O)[O-]",  # nitro group
        "c1ccc(N)cc1",  # primary aromatic amine
        "[NH2]c1ccccc1",  # aniline
        "C(=O)Cl",  # acid chloride
        "C(=O)Br",  # acid bromide
        "[N;H1]N",  # hydrazine
        "N=N",  # azo
        "O=C1NC(=O)c2ccccc21",  # N-substituted phthalimide
    ]
    for smarts in MUTAGENIC_SMARTS:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            return "Possible"
    return "Unlikely"


def _cyp_risk(mol: Chem.Mol, logp: float) -> dict[str, str]:
    """
    Simple rule-based CYP450 inhibition risk flags.

    Uses LogP and presence of known CYP-interacting moieties.
    """
    # Imidazole/triazole -> CYP3A4 risk
    azole_pattern = Chem.MolFromSmarts("[n;r5]")
    has_azole = mol.HasSubstructMatch(azole_pattern) if azole_pattern else False

    # Quinoline/isoquinoline -> CYP1A2
    quinoline = Chem.MolFromSmarts("c1ccnc2ccccc12")
    has_quinoline = mol.HasSubstructMatch(quinoline) if quinoline else False

    cyp3a4 = "Possible" if (has_azole or logp > 4) else "Unlikely"
    cyp1a2 = "Possible" if has_quinoline else "Unlikely"
    cyp2d6 = "Possible" if logp > 3 else "Unlikely"

    return {"CYP3A4": cyp3a4, "CYP1A2": cyp1a2, "CYP2D6": cyp2d6}


def compute_admet(smiles: str) -> ADMETResult | None:
    """
    Compute comprehensive ADMET properties for a molecule.

    Args:
        smiles: Input SMILES string.

    Returns:
        ADMETResult or None if SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)
    aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    frac_csp3 = rdMolDescriptors.CalcFractionCSP3(mol)

    alerts: list[str] = []

    # ── Absorption ────────────────────────────────────────────────────────────
    lipinski_pass = mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10
    if not lipinski_pass:
        alerts.append("Fails Lipinski Ro5 — poor oral absorption likely")

    # Veber rules: TPSA ≤ 140, rotatable bonds ≤ 10
    veber_pass = tpsa <= 140 and rotatable <= 10
    if not veber_pass:
        alerts.append("Fails Veber rules — poor oral bioavailability")

    # Bioavailability score (simplified: 0-6 Lipinski violations)
    ro5_violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    bioavailability = (
        "High" if ro5_violations == 0 else "Moderate" if ro5_violations == 1 else "Low"
    )

    absorption = {
        "mw": round(mw, 2),
        "logp": round(logp, 3),
        "hbd": hbd,
        "hba": hba,
        "tpsa": round(tpsa, 2),
        "rotatable_bonds": rotatable,
        "lipinski_pass": lipinski_pass,
        "veber_pass": veber_pass,
        "oral_bioavailability": bioavailability,
        "ro5_violations": ro5_violations,
    }

    # ── Distribution ─────────────────────────────────────────────────────────
    bbb = _bbb_permeability(tpsa, mw, logp)
    if bbb == "Low":
        alerts.append("Low BBB permeability — CNS activity unlikely")

    distribution = {
        "bbb_permeability": bbb,
        "tpsa": round(tpsa, 2),
        "frac_csp3": round(frac_csp3, 3),
        "aromatic_rings": aromatic_rings,
    }

    # ── Metabolism ───────────────────────────────────────────────────────────
    cyp_risks = _cyp_risk(mol, logp)
    if any(v == "Possible" for v in cyp_risks.values()):
        flagged = [k for k, v in cyp_risks.items() if v == "Possible"]
        alerts.append(f"CYP450 inhibition risk: {', '.join(flagged)}")

    metabolism = {
        "cyp_inhibition": cyp_risks,
        "rings": rings,
        "aromatic_rings": aromatic_rings,
    }

    # ── Excretion ────────────────────────────────────────────────────────────
    # Simple renal clearance proxy: low logP + low MW → faster renal excretion
    if logp < 0:
        clearance = "High (rapid renal excretion)"
    elif logp < 2:
        clearance = "Moderate"
    else:
        clearance = "Low (hepatic metabolism likely)"

    excretion = {
        "renal_clearance_estimate": clearance,
        "logp": round(logp, 3),
    }

    # ── Toxicity ─────────────────────────────────────────────────────────────
    herg = _herg_risk(mol, mw, logp)
    if herg in ("High", "Moderate"):
        alerts.append(f"hERG cardiotoxicity risk: {herg}")

    ames = _ames_risk(mol)
    if ames == "Possible":
        alerts.append("Possible Ames mutagenicity alert")

    pains = _pains_alerts(mol)
    alerts.extend(pains)

    brenk = _brenk_alerts(mol)
    alerts.extend(brenk[:3])  # cap at 3 most critical Brenk alerts

    toxicity = {
        "herg_risk": herg,
        "ames_mutagenicity": ames,
        "pains_alerts": len(pains),
        "brenk_alerts": len(brenk),
        "structural_alert_count": len(pains) + len(brenk),
    }

    # ── Overall Score (0-100) ─────────────────────────────────────────────────
    score = 100.0
    score -= ro5_violations * 10
    if not veber_pass:
        score -= 10
    if herg == "High":
        score -= 25
    elif herg == "Moderate":
        score -= 10
    if ames == "Possible":
        score -= 15
    score -= len(pains) * 8
    score -= min(len(brenk), 3) * 5
    score = max(0.0, min(100.0, score))

    pass_admet = score >= 60 and len(pains) == 0 and herg != "High" and ames != "Possible"

    return ADMETResult(
        smiles=smiles,
        absorption=absorption,
        distribution=distribution,
        metabolism=metabolism,
        excretion=excretion,
        toxicity=toxicity,
        overall_score=round(score, 1),
        alerts=alerts,
        pass_admet=pass_admet,
    )
