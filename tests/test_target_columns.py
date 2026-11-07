"""Training labels must not include SMILES, names or numeric identifiers."""

import pandas as pd
import pytest

from molprop.data.targets import MOLNET_TARGETS, target_columns


@pytest.mark.parametrize("dataset, targets", list(MOLNET_TARGETS.items()))
def test_registered_assay_is_selected_after_traceability_columns(dataset, targets):
    frame = pd.DataFrame({
        "std_smiles": ["CCO", "CCN"], "smiles": ["CCO", "CCN"],
        "num": [1, 2], "name": ["ethanol", "ethylamine"],
        "unrelated_numeric": [4.0, 5.0], targets[0]: [0.0, 1.0],
    })
    assert target_columns(frame, dataset) == targets


def test_missing_registered_assay_fails_instead_of_training_on_metadata():
    with pytest.raises(ValueError, match="Missing target columns"):
        target_columns(pd.DataFrame({"smiles": ["CCO"], "num": [1]}), "bbbp")


def test_custom_data_uses_numeric_labels_without_identifier_columns():
    frame = pd.DataFrame({"std_smiles": ["CCO"], "num": [3], "activity": [1.2]})
    assert target_columns(frame, "custom") == ["activity"]
