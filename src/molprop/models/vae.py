"""
SMILES Variational Autoencoder (VAE) for generative molecular design.

Architecture: GRU Encoder → (μ, log σ²) → Reparameterize → z → GRU Decoder
The decoder auto-regressively reconstructs SMILES tokens from the latent vector.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SMILESVAE(nn.Module):
    """
    Sequence-to-sequence Variational Autoencoder on SMILES strings.

    Args:
        vocab_size:    Number of unique SMILES tokens (including PAD/SOS/EOS).
        embedding_dim: Dimension of the token embedding table.
        hidden_dim:    GRU hidden state dimension.
        latent_dim:    Bottleneck latent space dimension z.
        pad_idx:       Index of the <pad> token (for embedding mask).
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 512,
        latent_dim: int = 128,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.pad_idx = pad_idx

        # Shared token embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # ── Encoder ────────────────────────────────────────────────────────────
        self.encoder_gru = nn.GRU(
            embedding_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # ── Decoder ────────────────────────────────────────────────────────────
        # Project z back to hidden state
        self.fc_z2h = nn.Linear(latent_dim, hidden_dim)
        # GRU takes [token_emb | z] at each step so it always has z in context
        self.decoder_gru = nn.GRU(
            embedding_dim + latent_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1
        )
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    # ── Encoder ────────────────────────────────────────────────────────────────
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len) token indices
        Returns:
            mu, logvar: each (batch, latent_dim)
        """
        emb = self.embedding(x)  # (B, L, E)
        _, h = self.encoder_gru(emb)  # h: (2, B, H) – take last layer
        h_top = h[-1]  # (B, H)
        return self.fc_mu(h_top), self.fc_logvar(h_top)

    # ── Reparameterisation ─────────────────────────────────────────────────────
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu  # deterministic at inference

    # ── Decoder ────────────────────────────────────────────────────────────────
    def decode(
        self,
        z: torch.Tensor,
        target: torch.Tensor | None = None,
        max_len: int = 100,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Teacher-forced decoding during training; greedy/sampled at inference.

        Args:
            z:           (batch, latent_dim) latent vector.
            target:      (batch, seq_len) ground-truth tokens for teacher forcing.
                         Pass None for autoregressive inference.
            max_len:     Maximum generation length (ignored when target is given).
            temperature: Sampling temperature (1.0 = argmax rescaled).

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size = z.size(0)
        # Initialise decoder hidden state from z
        h = torch.tanh(self.fc_z2h(z)).unsqueeze(0).repeat(2, 1, 1)  # (2, B, H)

        # SOS token index is 1 by convention; embed it as the first input
        sos_tok = torch.ones(batch_size, 1, dtype=torch.long, device=z.device)
        emb_sos = self.embedding(sos_tok)  # (B, 1, E)

        if target is not None:
            # Teacher-forced: feed entire target (shifted right)
            emb_tgt = self.embedding(target)  # (B, L, E)
            emb_in = torch.cat([emb_sos, emb_tgt[:, :-1]], dim=1)  # (B, L, E)
            z_expand = z.unsqueeze(1).repeat(1, emb_in.size(1), 1)  # (B, L, latent)
            gru_in = torch.cat([emb_in, z_expand], dim=-1)  # (B, L, E+latent)
            out, _ = self.decoder_gru(gru_in, h)
            return self.fc_out(out)  # (B, L, vocab)
        else:
            # Autoregressive greedy decoding
            logits_all = []
            tok = sos_tok
            for _ in range(max_len):
                emb_t = self.embedding(tok)  # (B, 1, E)
                z_t = z.unsqueeze(1)  # (B, 1, latent)
                gru_in = torch.cat([emb_t, z_t], dim=-1)  # (B, 1, E+latent)
                out, h = self.decoder_gru(gru_in, h)
                logit = self.fc_out(out)  # (B, 1, vocab)
                logits_all.append(logit)
                # Sample next token
                probs = F.softmax(logit.squeeze(1) / temperature, dim=-1)
                tok = torch.multinomial(probs, 1)  # (B, 1)
            return torch.cat(logits_all, dim=1)  # (B, max_len, vocab)

    # ── Forward pass ───────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len) token indices
        Returns:
            logits: (batch, seq_len, vocab_size)
            mu:     (batch, latent_dim)
            logvar: (batch, latent_dim)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, target=x)
        return logits, mu, logvar

    # ── Loss ───────────────────────────────────────────────────────────────────
    @staticmethod
    def loss(
        logits: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        kl_weight: float = 0.05,
        pad_idx: int = 0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Combined reconstruction (CE) + KL divergence loss.

        Args:
            logits:    (batch, seq_len, vocab)
            target:    (batch, seq_len) ground-truth token indices
            mu, logvar: latent distribution parameters
            kl_weight: β coefficient for KL annealing
            pad_idx:   padding token index (ignored in CE)

        Returns:
            total_loss, {"recon": ..., "kl": ..., "total": ...}
        """
        # Reconstruction loss
        B, L, V = logits.shape
        recon = F.cross_entropy(
            logits.reshape(B * L, V),
            target.reshape(B * L),
            ignore_index=pad_idx,
        )
        # KL divergence
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total = recon + kl_weight * kl
        return total, {"recon": recon.item(), "kl": kl.item(), "total": total.item()}
