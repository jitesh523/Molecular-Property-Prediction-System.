import logging
from pathlib import Path
import pandas as pd
from chembl_webresource_client.new_client import new_client
from tqdm import tqdm
import math

log = logging.getLogger(__name__)

def fetch_chembl_target(target_gene: str = "EGFR", out_dir: Path = None):
    """
    Fetches bioactivities for a specific target from ChEMBL.
    """
    if out_dir is None:
        ROOT = Path(__file__).resolve().parent.parent.parent.parent
        out_dir = ROOT / "data" / "raw" / f"chembl_{target_gene.lower()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "full.csv"

    log.info(f"Fetching Target ID for {target_gene}...")
    target = new_client.target
    target_query = target.search(target_gene)
    targets = pd.DataFrame.from_dict(target_query)
    
    if targets.empty:
        log.error(f"No target found for {target_gene}")
        return None
        
    target_id = targets.iloc[0]['target_chembl_id']
    log.info(f"Resolved {target_gene} to ChEMBL ID: {target_id}")
    
    # Fetch activities
    log.info(f"Fetching IC50 activities for {target_id}...")
    activity = new_client.activity
    res = activity.filter(target_chembl_id=target_id).filter(standard_type="IC50")
    
    records = []
    # We use tqdm to show progress
    for r in tqdm(res, desc=f"Downloading {target_gene} activities"):
        smiles = r.get('canonical_smiles')
        val = r.get('standard_value')
        units = r.get('standard_units')
        pchembl = r.get('pchembl_value')
        
        if smiles and val is not None:
            records.append({
                'smiles': smiles,
                'standard_value': val,
                'standard_units': units,
                'pchembl_value': pchembl,
                'chembl_id': r.get('molecule_chembl_id')
            })
            
    df = pd.DataFrame(records)
    log.info(f"Downloaded {len(df)} initial records.")
    if df.empty:
        return df
        
    # Filter valid measurements
    df = df.dropna(subset=['standard_value'])
    df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')
    df = df.dropna(subset=['standard_value'])
    
    # Convert nM to pIC50 if pchembl is missing, or just use pchembl
    def calculate_pIC50(row):
        if pd.notna(row['pchembl_value']):
            return float(row['pchembl_value'])
        if row['standard_units'] == 'nM' and row['standard_value'] > 0:
            return -math.log10(row['standard_value'] * 1e-9)
        return None
        
    df['target'] = df.apply(calculate_pIC50, axis=1)
    df = df.dropna(subset=['target'])
    
    df.to_csv(out_path, index=False)
    log.info(f"Saved {len(df)} curated records to {out_path}")
    return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # For CI and demonstrations, we limit the dataset pull or just show the structure
    # Testing it manually with EGFR
    fetch_chembl_target("EGFR")
