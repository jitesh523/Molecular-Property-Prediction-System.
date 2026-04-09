import logging
from pathlib import Path
import pandas as pd
import requests

log = logging.getLogger(__name__)

def fetch_pubchem_assay(aid: str = "260895", out_dir: Path = None):
    """
    Fetches logical table for a specific PubChem BioAssay using PUG-REST.
    """
    if out_dir is None:
        ROOT = Path(__file__).resolve().parent.parent.parent.parent
        out_dir = ROOT / "data" / "raw" / f"pubchem_{aid}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "full.csv"

    log.info(f"Fetching PubChem BioAssay AID {aid}...")
    
    # PUG-REST endpoint for concise CSV
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/concise/CSV"
    
    response = requests.get(url, timeout=60)
    if response.status_code != 200:
        log.error(f"Failed to fetch assay {aid}: HTTP {response.status_code}")
        return None
        
    with open(out_path, "wb") as f:
        f.write(response.content)
        
    log.info(f"Downloaded concise assay data to {out_path}")
    
    # Sanitize and prep for pipeline
    df = pd.read_csv(out_path)
    
    # Uniform smiles mapping
    # Concise CSV only has CIDs
    if "CID" not in df.columns:
        log.error("No CID column found in assay data.")
        return None
        
    cids = df["CID"].dropna().unique().astype(int).tolist()
    log.info(f"Fetching SMILES for {len(cids)} CIDs in batches...")
    
    smiles_mapping = {}
    batch_size = 200
    for i in range(0, len(cids), batch_size):
        batch_cids = cids[i:i+batch_size]
        cid_str = ",".join(map(str, batch_cids))
        prop_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid_str}/property/IsomericSMILES,CanonicalSMILES/JSON"
        prop_resp = requests.get(prop_url, timeout=60)
        if prop_resp.status_code == 200:
            data = prop_resp.json()
            for item in data.get("PropertyTable", {}).get("Properties", []):
                # Prefer Isomeric over Canonical, or whatever is available
                smi = item.get("IsomericSMILES") or item.get("CanonicalSMILES") or item.get("ConnectivitySMILES")
                smiles_mapping[item["CID"]] = smi
                
    df["smiles"] = df["CID"].map(smiles_mapping)
    
    # Supervised label mapping
    if "Activity Value [uM]" in df.columns:
        # Convert uM to pIC50 if possible, or just raw
        import math
        def to_pic50(val):
            try:
                v = float(val)
                if v > 0 and not math.isnan(v):
                    return -math.log10(v * 1e-6)
            except (ValueError, TypeError):
                pass
            return None
        df["target"] = df["Activity Value [uM]"].apply(to_pic50)
    elif "Activity Outcome" in df.columns:
        df["target"] = df["Activity Outcome"].apply(lambda x: 1 if str(x).lower() == 'active' else 0)
        
    # Drop where no smiles or target
    df = df.dropna(subset=["smiles", "target"])
        
    df.to_csv(out_path, index=False)
    log.info(f"Validated {len(df)} records for AID {aid}.")
    return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    fetch_pubchem_assay("260895")
