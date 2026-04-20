import torch.nn as nn


class EmbeddingAutoencoder(nn.Module):
    def __init__(self, input_dim=1152, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, input_dim),
        )
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(latent_dim),
            nn.Linear(latent_dim, 1),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def classify(self, z):
        return self.classifier(z).squeeze(-1)

    def forward(self, x):
        z = self.encode(x)
        reconstructed = self.decode(z)
        logit = self.classify(z)
        return reconstructed, logit, z
