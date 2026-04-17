import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.autoencoder_model import EmbeddingAutoencoder


class EmbeddingsEncoder:
    def __init__(self, checkpoint_path, input_dim=1152, latent_dim=128):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ae = EmbeddingAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
        ae.load_state_dict(
            torch.load(checkpoint_path, weights_only=True, map_location=self.device)
        )
        self.encoder = ae.encoder.to(self.device).eval()

    @torch.no_grad()
    def encode(self, embeddings):
        """Reduce embedding dimensionality.

        embeddings: ndarray (N, input_dim).
        Returns: ndarray (N, latent_dim).
        """
        x = torch.from_numpy(np.asarray(embeddings, dtype=np.float32)).to(self.device)
        return self.encoder(x).cpu().numpy()
