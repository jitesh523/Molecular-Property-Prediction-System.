import logging
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

from molprop.data.standardize import standardize_smiles

log = logging.getLogger(__name__)


class DatasetProcessor:
    """
    Handles loading raw datasets, applying chemical standardization,
    and saving processed artifacts.
    """

    def __init__(self, raw_dir: Path, processed_dir: Path):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def process_dataset(
        self,
        name: str,
        smiles_col: str = "smiles",
        keep_chirality: bool = True,
        force: bool = False,
    ) -> Optional[pd.DataFrame]:
        """Processes a single dataset: Load -> Standardize -> Drop Invalid -> Save.
        
        Args:
            name: Dataset name (alphanumeric + underscore/hyphen only). Used to find
                  raw data in {raw_dir}/{name}/full.csv and save to {processed_dir}/{name}/processed.csv
            smiles_col: Column name in raw CSV containing SMILES strings (default 'smiles')
            keep_chirality: Preserve stereochemistry information in standardized SMILES
            force: If True, reprocess even if processed.csv already exists
            
        Returns:
            Processed DataFrame on success, None if inputs invalid or processing fails
            
        Raises:
            ValueError: If dataset name contains invalid characters (path traversal protection)
        """
        # Validate dataset name - prevent path traversal attacks
        if not re.match(r'^[a-zA-Z0-9_\-]+$', name):
            raise ValueError(
                f"Invalid dataset name '{name}'. "
                "Must contain only alphanumeric characters, underscores, and hyphens."
            )
        
        raw_path = self.raw_dir / name / "full.csv"
        out_dir = self.processed_dir / name
        out_path = out_dir / "processed.csv"

        if not raw_path.exists():
            log.error(f"Raw data not found for {name} at {raw_path}")
            return None

        if out_path.exists() and not force:
            log.info(f"  ⏭  {name}: already processed – skipping")
            return pd.read_csv(out_path)

        log.info(f"  ⚙️  Processing {name}...")
        try:
            df = pd.read_csv(raw_path, encoding='utf-8')
        except UnicodeDecodeError:
            log.error(f"Failed to decode {name}; file is not valid UTF-8")
            return None
        except Exception as e:
            log.error(f"Failed to read {name}: {e}")
            return None

        if smiles_col not in df.columns:
            available = ', '.join(df.columns.tolist())
            raise ValueError(
                f"SMILES column '{smiles_col}' not found in {name}. "
                f"Available columns: {available}"
            )

        # Apply standardization
        tqdm.pandas(desc=f"Standardizing {name}")
        df["std_smiles"] = df[smiles_col].progress_apply(
            lambda s: standardize_smiles(s, keep_chirality=keep_chirality)
        )

        # Record original count
        orig_count = len(df)

        # Drop invalid SMILES (where standardize_smiles returned None)
        df = df.dropna(subset=["std_smiles"])
        valid_count = len(df)

        # Deduplicate by standardized SMILES
        # Keep the first occurrence (or could aggregate if needed, but for MolNet first is usually fine)
        df = df.drop_duplicates(subset=["std_smiles"])
        final_count = len(df)

        log.info(
            f"     ✓ {name}: {orig_count} total -> {valid_count} valid -> {final_count} unique"
        )

        # Save processed data — keep original SMILES for traceability
        out_dir.mkdir(parents=True, exist_ok=True)
        # Reorder: std_smiles first, original SMILES second, then remaining columns
        other_cols = [c for c in df.columns if c not in ["std_smiles", smiles_col]]
        cols = ["std_smiles", smiles_col] + other_cols
        df[cols].to_csv(out_path, index=False)

        # Save simplified metadata
        with open(out_dir / "processing_summary.txt", "w") as f:
            f.write(f"Original Row Count: {orig_count}\n")
            f.write(f"Valid SMILES Count: {valid_count}\n")
            f.write(f"Unique SMILES Count: {final_count}\n")
            f.write(f"Dropped: {orig_count - final_count}\n")

        return df[cols]  # type: ignore[return-value]


def process_all_benchmark_datasets(raw_dir: Path, processed_dir: Path, datasets: List[str] = None):
    """Utility to batch process datasets."""
    processor = DatasetProcessor(raw_dir, processed_dir)
    if datasets is None:
        # Get all subdirs in raw_dir
        datasets = [d.name for d in raw_dir.iterdir() if d.is_dir()]

    for name in datasets:
        try:
            processor.process_dataset(name)
        except Exception as e:
            log.error(f"  ✗ Failed to process {name}: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ROOT = Path(__file__).resolve().parent.parent.parent.parent
    raw_path = ROOT / "data" / "raw"
    processed_path = ROOT / "data" / "processed"

    # For initial run, focus on the core benchmarks mentioned in the roadmap
    priority_datasets = ["delaney", "bbbp", "freesolv", "lipo"]
    process_all_benchmark_datasets(raw_path, processed_path, priority_datasets)
