import torch
import torch.nn as nn
import torch.nn.functional as F

class SMILESVAE(nn.Module):
    """
    SMILES-based Variational Autoencoder with GRU Encoder/Decoder.
    Used for latent space sampling and generative design.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 256, latent_dim: int = 64):
        super(SMILESVAE, self).__init__()
        self.latent_dim = latent_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Encoder
        self.encoder_gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_gru = nn.GRU(embedding_dim + latent_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, teacher_forcing_ratio: float = 0.5):
        # x shape: (batch, seq_len)
        embedded = self.embedding(x)
        
        # Encode
        _, h = self.encoder_gru(embedded) # h: (1, batch, hidden)
        h = h.squeeze(0)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar) # (batch, latent_dim)
        
        # Decode
        # For simplicity in this dummy version, we'll just return mu/logvar/z
        # A full decoder would iterate through time steps.
        return mu, logvar, z

    def decode(self, z, max_len: int = 100):
        """Generates SMILES sequence from latent vector z."""
        # This would be used during inference
        pass
