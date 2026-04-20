"""
Train the SMILES Variational Autoencoder (VAE) for generative molecular design.

Usage:
    python scripts/train_vae.py
    python scripts/train_vae.py --dataset bbbp --epochs 50 --latent-dim 128
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import mlflow
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from molprop.data.smiles_vocab import SmilesVocab
from molprop.data.vae_dataset import VAEDataset
from molprop.models.vae import SMILESVAE

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SMILES VAE")
    p.add_argument("--dataset", default="bbbp", help="Processed dataset name")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--max-len", type=int, default=120)
    p.add_argument("--kl-weight", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Device ─────────────────────────────────────────────────────────────────
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    log.info(f"Device: {device}")

    # ── Load SMILES ─────────────────────────────────────────────────────────────
    data_path = ROOT / "data" / "processed" / args.dataset / "processed.csv"
    if not data_path.exists():
        log.error(f"Data not found: {data_path}")
        return

    df = pd.read_csv(data_path)
    smiles_list = df["std_smiles"].dropna().tolist()
    log.info(f"Loaded {len(smiles_list)} SMILES from {args.dataset}")

    # ── Build / save vocab ─────────────────────────────────────────────────────
    vocab_path = ROOT / "data" / "processed" / args.dataset / "smiles_vocab.json"
    if vocab_path.exists():
        vocab = SmilesVocab.load(vocab_path)
        log.info(f"Loaded existing vocab: {len(vocab)} tokens")
    else:
        vocab = SmilesVocab.from_smiles(smiles_list)
        vocab.save(vocab_path)
        log.info(f"Built vocab: {len(vocab)} tokens → {vocab_path}")

    # ── Dataset & DataLoaders ──────────────────────────────────────────────────
    dataset = VAEDataset(smiles_list, vocab, max_len=args.max_len)
    n_val = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # ── Model ──────────────────────────────────────────────────────────────────
    model = SMILESVAE(
        vocab_size=len(vocab),
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        pad_idx=0,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    log.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ── MLflow ─────────────────────────────────────────────────────────────────
    tracking_uri = f"file://{ROOT}/mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("smiles-vae")

    best_val = float("inf")
    patience_ctr = 0
    best_path = ROOT / f"best_model_vae_{args.dataset}.pt"

    with mlflow.start_run(run_name=f"vae_{args.dataset}"):
        mlflow.log_params(vars(args))

        for epoch in range(1, args.epochs + 1):
            # ── Train ──────────────────────────────────────────────────────────
            model.train()
            train_losses: list[dict] = []
            for batch in tqdm(train_loader, leave=False, desc=f"Epoch {epoch:03d}"):
                batch = batch.to(device)
                optimizer.zero_grad()
                logits, mu, logvar = model(batch)
                loss, info = SMILESVAE.loss(logits, batch, mu, logvar, kl_weight=args.kl_weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(info)

            train_total = sum(d["total"] for d in train_losses) / len(train_losses)
            train_recon = sum(d["recon"] for d in train_losses) / len(train_losses)
            train_kl = sum(d["kl"] for d in train_losses) / len(train_losses)

            # ── Validate ───────────────────────────────────────────────────────
            model.eval()
            val_losses: list[dict] = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    logits, mu, logvar = model(batch)
                    _, info = SMILESVAE.loss(logits, batch, mu, logvar, kl_weight=args.kl_weight)
                    val_losses.append(info)
            val_total = sum(d["total"] for d in val_losses) / len(val_losses)

            scheduler.step(val_total)

            # Log metrics
            mlflow.log_metrics(
                {
                    "train_total": train_total,
                    "train_recon": train_recon,
                    "train_kl": train_kl,
                    "val_total": val_total,
                },
                step=epoch,
            )
            log.info(
                f"Epoch {epoch:03d} | Train {train_total:.4f} "
                f"(recon {train_recon:.4f} / kl {train_kl:.4f}) | Val {val_total:.4f}"
            )

            # Sample a molecule every 5 epochs to track learning
            if epoch % 5 == 0:
                sample = _sample_smiles(model, vocab, device, n=3, temperature=0.8)
                for i, smi in enumerate(sample):
                    log.info(f"  🧪 Sample {i + 1}: {smi}")
                    mlflow.log_text(smi, f"epoch_{epoch}_sample_{i}.txt")

            # Early stopping
            if val_total < best_val:
                best_val = val_total
                patience_ctr = 0
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "vocab_size": len(vocab),
                        "latent_dim": args.latent_dim,
                        "hidden_dim": args.hidden_dim,
                        "max_len": args.max_len,
                        "dataset": args.dataset,
                    },
                    best_path,
                )
                mlflow.log_artifact(str(best_path))
            else:
                patience_ctr += 1
                if patience_ctr >= args.patience:
                    log.info("Early stopping triggered.")
                    break

    log.info(f"Training complete. Best val loss: {best_val:.4f}")
    log.info(f"Checkpoint saved → {best_path}")

    # Save vocab path alongside the model for the API to load
    vocab_ref = ROOT / f"vocab_vae_{args.dataset}.json"
    vocab.save(vocab_ref)
    log.info(f"Vocab saved → {vocab_ref}")


def _sample_smiles(
    model: SMILESVAE,
    vocab: SmilesVocab,
    device: torch.device,
    n: int = 5,
    temperature: float = 1.0,
) -> list[str]:
    """Sample n novel SMILES from the prior N(0, I)."""
    model.eval()
    with torch.no_grad():
        z = torch.randn(n, model.latent_dim, device=device)
        logits = model.decode(z, max_len=model.latent_dim, temperature=temperature)
        token_ids = logits.argmax(dim=-1).cpu().tolist()
    return [vocab.decode(ids) for ids in token_ids]


if __name__ == "__main__":
    main()
