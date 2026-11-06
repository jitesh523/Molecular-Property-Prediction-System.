"""Select assay labels while excluding processed-data traceability columns."""

import pandas as pd

MOLNET_TARGETS = {
    "bbbp": ["p_np"],
    "delaney": ["measured log solubility in mols per litre"],
    "freesolv": ["expt"],
    "lipo": ["exp"],
}


def target_columns(df: pd.DataFrame, dataset_name: str) -> list[str]:
    """Use registered labels, or numeric non-metadata columns for custom data."""
    if dataset_name in MOLNET_TARGETS:
        targets = MOLNET_TARGETS[dataset_name]
        missing = [name for name in targets if name not in df.columns]
        if missing:
            raise ValueError(f"Missing target columns for {dataset_name}: {missing}")
        return list(targets)

    metadata = {"std_smiles", "smiles", "mol", "name", "id", "num", "index", "compound_id"}
    targets = [
        name for name in df.columns
        if name.lower() not in metadata and pd.api.types.is_numeric_dtype(df[name])
    ]
    if not targets:
        raise ValueError(f"No numeric target columns found for {dataset_name}")
    return targets
