from molprop.data.standardize import standardize_smiles


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
