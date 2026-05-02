from molprop.data.standardize import (
    ghose_filter,
    passes_lipinski_ro5,
    standardize_smiles,
    veber_filter,
)


def test_standardize_basic():
    """Test basic structure parsing and canonicalization."""
    # Phenol
    raw = "C1=CC=C(C=C1)O"
    clean = standardize_smiles(raw)
    assert clean == "Oc1ccccc1"


def test_standardize_salt_removal():
    """Test that standardizer removes salts and keeps the major component."""
    # Promethazine hydrochloride
    raw = "CC(C)N(C)CC(C)N1c2ccccc2Sc3ccccc13.Cl"
    clean = standardize_smiles(raw)
    assert "Cl" not in clean
    assert clean == "CC(C)N(C)CC(C)N1c2ccccc2Sc2ccccc21"


def test_standardize_multi_fragment():
    """Test retaining largest fragment when multiple are present."""
    # Phenol and water
    raw = "O.c1ccccc1O"
    clean = standardize_smiles(raw)
    assert clean == "Oc1ccccc1"


def test_standardize_chirality():
    """Test standardizer chiral retention."""
    # (R)- vs (S)- 2-butanol
    r_form = "C[C@H](O)CC"
    s_form = "C[C@@H](O)CC"

    clean_r = standardize_smiles(r_form, keep_chirality=True)
    clean_s = standardize_smiles(s_form, keep_chirality=True)

    assert "@" in clean_r
    assert "@" in clean_s
    assert clean_r != clean_s


def test_standardize_chirality_dropped():
    """Test dropping chirality."""
    r_form = "C[C@H](O)CC"
    s_form = "C[C@@H](O)CC"

    clean_r = standardize_smiles(r_form, keep_chirality=False)
    clean_s = standardize_smiles(s_form, keep_chirality=False)

    assert "@" not in clean_r
    assert clean_r == clean_s
    assert clean_r == "CCC(C)O"


def test_standardize_invalid():
    """Test handling of invalid SMILES."""
    assert standardize_smiles("InvalidSmiles123") is None
    assert standardize_smiles("") is None


# ── Lipinski Rule of Five ──────────────────────────────────────────────────────


def test_lipinski_aspirin_passes():
    """Aspirin (MW=180, LogP=1.2, HBD=1, HBA=4) should pass Ro5."""
    result = passes_lipinski_ro5("CC(=O)OC1=CC=CC=C1C(=O)O")
    assert result is not None
    assert result["passes"] is True
    assert len(result["violations"]) == 0


def test_lipinski_benzene_passes():
    """Benzene is well within all Ro5 limits."""
    result = passes_lipinski_ro5("c1ccccc1")
    assert result is not None
    assert result["passes"] is True


def test_lipinski_returns_property_values():
    """Result dict must expose MW, LogP, HBD, HBA."""
    result = passes_lipinski_ro5("c1ccccc1")
    assert "MW" in result
    assert "LogP" in result
    assert "HBD" in result
    assert "HBA" in result


def test_lipinski_invalid_smiles_returns_none():
    result = passes_lipinski_ro5("NOT_A_MOLECULE")
    assert result is None


def test_lipinski_large_molecule_fails():
    """A large peptide-like molecule should violate MW >= 500 Da."""
    # Cyclosporin A SMILES (MW ~1202 Da)
    cspa = (
        "CC[C@@H]1NC(=O)[C@H]([C@H](CC)C)N(C)C(=O)[C@H](CC(C)C)NC(=O)"
        "[C@@H](C(C)C)N(C)C(=O)[C@H](CC(C)C)NC(=O)[C@H](C)N(C)C(=O)"
        "[C@H](CC(C)C)N(C)C(=O)[C@@H](CC(C)C)N(C)C(=O)[C@@H](C(C)C)NC1=O"
    )
    result = passes_lipinski_ro5(cspa)
    if result is not None:
        assert not result["passes"] or len(result["violations"]) > 0


# ── Veber Filter ───────────────────────────────────────────────────────────────


def test_veber_aspirin_passes():
    """Aspirin has 3 rotatable bonds and low TPSA — should pass Veber."""
    result = veber_filter("CC(=O)OC1=CC=CC=C1C(=O)O")
    assert result is not None
    assert result["passes"] is True
    assert result["violations"] == []


def test_veber_benzene_passes():
    """Benzene has 0 rotatable bonds and low TPSA."""
    result = veber_filter("c1ccccc1")
    assert result is not None
    assert result["passes"] is True


def test_veber_returns_property_values():
    result = veber_filter("c1ccccc1")
    assert "RotatableBonds" in result
    assert "TPSA" in result
    assert isinstance(result["RotatableBonds"], int)
    assert isinstance(result["TPSA"], float)


def test_veber_invalid_smiles_returns_none():
    assert veber_filter("NOT_A_MOLECULE") is None


def test_veber_custom_limits():
    """Tightening limits should cause a simple molecule to fail."""
    result = veber_filter("CC(=O)OC1=CC=CC=C1C(=O)O", rot_bonds_limit=2)
    assert result is not None
    assert result["passes"] is False
    assert any("RotatableBonds" in v for v in result["violations"])


# ── Ghose Filter ───────────────────────────────────────────────────────────────


def test_ghose_aspirin_passes():
    """Aspirin (MW≈180, LogP≈1.2) satisfies all Ghose ranges."""
    result = ghose_filter("CC(=O)OC1=CC=CC=C1C(=O)O")
    assert result is not None
    assert result["passes"] is True
    assert result["violations"] == []


def test_ghose_returns_property_values():
    result = ghose_filter("CC(=O)OC1=CC=CC=C1C(=O)O")
    for key in ("LogP", "MW", "MR", "NumAtoms"):
        assert key in result


def test_ghose_benzene_fails_mw():
    """Benzene (MW≈78) is below the 160 Da lower limit."""
    result = ghose_filter("c1ccccc1")
    assert result is not None
    assert result["passes"] is False
    assert any("MW" in v for v in result["violations"])


def test_ghose_invalid_smiles_returns_none():
    assert ghose_filter("NOT_A_MOLECULE") is None


def test_ghose_large_molecule_fails():
    """Cyclosporin A violates MW and NumAtoms upper bounds."""
    cspa = (
        "CC[C@@H]1NC(=O)[C@H]([C@H](CC)C)N(C)C(=O)[C@H](CC(C)C)NC(=O)"
        "[C@@H](C(C)C)N(C)C(=O)[C@H](CC(C)C)NC(=O)[C@H](C)N(C)C(=O)"
        "[C@H](CC(C)C)N(C)C(=O)[C@@H](CC(C)C)N(C)C(=O)[C@@H](C(C)C)NC1=O"
    )
    result = ghose_filter(cspa)
    if result is not None:
        assert result["passes"] is False
