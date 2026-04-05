"""
Download MolNet benchmark datasets directly from their source URLs.
No TensorFlow or heavy ML dependencies required.

Datasets are saved to:  data/raw/<dataset_name>/full.csv

Classification : BACE, BBBP, ClinTox, HIV, MUV, SIDER, Tox21, ToxCast
Regression     : Delaney (ESOL), FreeSolv, Lipophilicity, SAMPL, QM7, QM8, QM9
"""

import csv
import io
import logging
import time
import urllib.request
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Base URL for DeepChem's dataset mirror
DC_BASE = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets"

# ── Dataset registry ──────────────────────────────────────────────────────────
# Each entry: url, task_type, description, smiles_col, task targets

DATASETS = {
    # ── Classification ────────────────────────────────────────────────────────
    "bace_classification": {
        "url": f"{DC_BASE}/bace.csv",
        "task": "classification",
        "desc": "BACE-1 inhibition (binary classification)",
        "smiles_col": "mol",
        "targets": ["Class"],
    },
    "bbbp": {
        "url": f"{DC_BASE}/BBBP.csv",
        "task": "classification",
        "desc": "Blood-Brain Barrier Permeability",
        "smiles_col": "smiles",
        "targets": ["p_np"],
    },
    "clintox": {
        "url": f"{DC_BASE}/clintox.csv.gz",
        "task": "classification",
        "desc": "Clinical Toxicity (FDA approval & CT toxicity)",
        "smiles_col": "smiles",
        "targets": ["FDA_APPROVED", "CT_TOX"],
    },
    "hiv": {
        "url": f"{DC_BASE}/HIV.csv",
        "task": "classification",
        "desc": "HIV replication inhibition",
        "smiles_col": "smiles",
        "targets": ["HIV_active"],
    },
    "muv": {
        "url": f"{DC_BASE}/muv.csv.gz",
        "task": "classification",
        "desc": "Maximum Unbiased Validation (17 targets)",
        "smiles_col": "smiles",
        "targets": [
            "MUV-466",
            "MUV-548",
            "MUV-600",
            "MUV-644",
            "MUV-652",
            "MUV-689",
            "MUV-692",
            "MUV-712",
            "MUV-713",
            "MUV-733",
            "MUV-737",
            "MUV-810",
            "MUV-832",
            "MUV-846",
            "MUV-852",
            "MUV-858",
            "MUV-859",
        ],
    },
    "sider": {
        "url": f"{DC_BASE}/sider.csv.gz",
        "task": "classification",
        "desc": "Side Effect Resource (27 side-effect types)",
        "smiles_col": "smiles",
        "targets": [
            "Hepatobiliary disorders",
            "Metabolism & nutrition disorders",
            "Product issues",
            "Eye disorders",
            "Investigations",
            "Musculoskeletal & connective tissue disorders",
            "Gastrointestinal disorders",
            "Social circumstances",
            "Immune system disorders",
            "Reproductive system & breast disorders",
            "Neoplasms benign malignant & unspecified (incl cysts & polyps)",
            "General disorders & administration site conditions",
            "Endocrine disorders",
            "Surgical & medical procedures",
            "Vascular disorders",
            "Blood & lymphatic system disorders",
            "Skin & subcutaneous tissue disorders",
            "Congenital familial & genetic disorders",
            "Infections & infestations",
            "Respiratory thoracic & mediastinal disorders",
            "Psychiatric disorders",
            "Renal & urinary disorders",
            "Pregnancy puerperium & perinatal conditions",
            "Ear & labyrinth disorders",
            "Cardiac disorders",
            "Nervous system disorders",
            "Injury poisoning & procedural complications",
        ],
    },
    "tox21": {
        "url": f"{DC_BASE}/tox21.csv.gz",
        "task": "classification",
        "desc": "Tox21 Challenge (12 toxicity tasks)",
        "smiles_col": "smiles",
        "targets": [
            "NR-AR",
            "NR-AR-LBD",
            "NR-AhR",
            "NR-Aromatase",
            "NR-ER",
            "NR-ER-LBD",
            "NR-PPAR-gamma",
            "SR-ARE",
            "SR-ATAD5",
            "SR-HSE",
            "SR-MMP",
            "SR-p53",
        ],
    },
    "toxcast": {
        "url": f"{DC_BASE}/toxcast_data.csv.gz",
        "task": "classification",
        "desc": "ToxCast (617 assay endpoints)",
        "smiles_col": "smiles",
        "targets": [],  # All remaining columns after smiles
    },
    # ── Regression ────────────────────────────────────────────────────────────
    "delaney": {
        "url": f"{DC_BASE}/delaney-processed.csv",
        "task": "regression",
        "desc": "Aqueous Solubility / ESOL (Delaney)",
        "smiles_col": "smiles",
        "targets": ["measured log solubility in mols per litre"],
    },
    "freesolv": {
        "url": f"{DC_BASE}/SAMPL.csv",
        "task": "regression",
        "desc": "Hydration Free Energy (FreeSolv / SAMPL)",
        "smiles_col": "smiles",
        "targets": ["expt"],
    },
    "lipo": {
        "url": f"{DC_BASE}/Lipophilicity.csv",
        "task": "regression",
        "desc": "Lipophilicity (AstraZeneca)",
        "smiles_col": "smiles",
        "targets": ["exp"],
    },
    "bace_regression": {
        "url": f"{DC_BASE}/bace.csv",
        "task": "regression",
        "desc": "BACE-1 binding affinity (pIC50)",
        "smiles_col": "mol",
        "targets": ["pIC50"],
    },
    "qm7": {
        "url": f"{DC_BASE}/qm7.csv",
        "task": "regression",
        "desc": "QM7 – atomisation energies (DFT, eV)",
        "smiles_col": "smiles",
        "targets": ["u0_atom"],
    },
    "qm8": {
        "url": f"{DC_BASE}/qm8.csv",
        "task": "regression",
        "desc": "QM8 – electronic spectra (12 properties)",
        "smiles_col": "smiles",
        "targets": [
            "E1-CC2",
            "E2-CC2",
            "f1-CC2",
            "f2-CC2",
            "E1-PBE0",
            "E2-PBE0",
            "f1-PBE0",
            "f2-PBE0",
            "E1-CAM",
            "E2-CAM",
            "f1-CAM",
            "f2-CAM",
        ],
    },
    "qm9": {
        "url": f"{DC_BASE}/qm9.csv",
        "task": "regression",
        "desc": "QM9 – 12 quantum-chemical properties",
        "smiles_col": "smiles",
        "targets": [
            "mu",
            "alpha",
            "homo",
            "lumo",
            "gap",
            "r2",
            "zpve",
            "u0",
            "u298",
            "h298",
            "g298",
            "cv",
        ],
    },
}

# ── Download helpers ───────────────────────────────────────────────────────────


def download_file(url: str, dest: Path, retries: int = 3) -> bool:
    """Download url → dest, with simple retry logic."""
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            dest.write_bytes(data)
            return True
        except Exception as exc:
            log.warning(f"     attempt {attempt}/{retries} failed: {exc}")
            if attempt < retries:
                time.sleep(2**attempt)
    return False


def read_csv_bytes(raw_bytes: bytes, url: str) -> list[dict]:
    """Parse raw bytes (handles .gz automatically) into list-of-dicts."""
    if url.endswith(".gz"):
        import gzip

        raw_bytes = gzip.decompress(raw_bytes)
    text = raw_bytes.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    return list(reader)


def download_dataset(name: str, info: dict) -> bool:
    out_dir = DATA_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    dest_raw = out_dir / ("full.csv.gz" if info["url"].endswith(".gz") else "full.csv")
    dest_csv = out_dir / "full.csv"
    meta_path = out_dir / "metadata.txt"

    # Skip if already done
    if dest_csv.exists() and meta_path.exists():
        log.info(f"  ⏭  {name}: already exists – skipping")
        return True

    log.info(f"  ⬇  {name}: {info['desc']}")
    log.info(f"     URL: {info['url']}")

    # Download raw file
    if not download_file(info["url"], dest_raw):
        log.error(f"  ✗ {name}: download failed after retries")
        return False

    # Parse and normalise to a clean CSV
    raw_bytes = dest_raw.read_bytes()
    rows = read_csv_bytes(raw_bytes, info["url"])

    if not rows:
        log.error(f"  ✗ {name}: no rows parsed")
        return False

    smiles_col = info["smiles_col"]

    # Determine target columns
    if info["targets"]:
        # Use columns listed in registry (intersect with what's actually present)
        available = set(rows[0].keys())
        targets = [t for t in info["targets"] if t in available]
    else:
        # toxcast: all columns except smiles
        targets = [c for c in rows[0].keys() if c != smiles_col]

    # Write clean CSV  (smiles + targets only)
    with open(dest_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["smiles"] + targets, extrasaction="ignore")
        writer.writeheader()
        valid_rows = 0
        for row in rows:
            smiles = row.get(smiles_col, "").strip()
            if not smiles:
                continue
            out_row = {"smiles": smiles}
            out_row.update({t: row.get(t, "") for t in targets})
            writer.writerow(out_row)
            valid_rows += 1

    # Remove raw .gz if different from dest_csv
    if dest_raw != dest_csv and dest_raw.exists():
        dest_raw.unlink()

    # Write metadata
    with open(meta_path, "w") as fh:
        fh.write(f"Dataset   : {name}\n")
        fh.write(f"Task type : {info['task']}\n")
        fh.write(f"Source    : {info['url']}\n")
        fh.write(f"Targets   : {', '.join(targets)}\n")
        fh.write(f"Molecules : {valid_rows}\n")

    log.info(
        f"     ✓ {valid_rows} molecules  |  {len(targets)} target(s)  →  {dest_csv.relative_to(ROOT)}"
    )
    return True


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    log.info("=" * 64)
    log.info("  MolNet Dataset Downloader")
    log.info(f"  Output: {DATA_DIR}")
    log.info("=" * 64)

    ok, failed = [], []

    for name, info in DATASETS.items():
        try:
            success = download_dataset(name, info)
            (ok if success else failed).append(name)
        except Exception as exc:
            log.error(f"  ✗ {name}: {exc}")
            failed.append(name)

    log.info("")
    log.info("=" * 64)
    log.info(f"  ✅  Succeeded : {len(ok)}")
    if ok:
        for n in ok:
            log.info(f"       • {n}")
    if failed:
        log.warning(f"  ❌  Failed    : {len(failed)}")
        for n in failed:
            log.warning(f"       • {n}")
    log.info("=" * 64)


if __name__ == "__main__":
    main()
