from pathlib import Path

import numpy as np
import torch

from models.autoencoder_model import EmbeddingAutoencoder
from models.mlp_model import MLPModel


def _find_fold_mlp(mlp_dir, k):
    """Accept both training layout (fold_K/best_model.pt) and
    tuning layout (fold_K.pt)."""
    candidates = [
        mlp_dir / f"fold_{k}" / "best_model.pt",
        mlp_dir / f"fold_{k}.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


class EnsembleModel:
    """Plain-PyTorch k-fold ensemble of MLP classifiers (+ optional per-fold encoders).

    Loads checkpoints from disk and returns mean/variance of sigmoid probs.
    """

    def __init__(self, mlp_dir, encoder_dir=None, latent_dim=128, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlp_dir = Path(mlp_dir)
        encoder_dir = Path(encoder_dir) if encoder_dir else None

        self.encoders = []
        self.models = []

        k = 1
        while True:
            cls_path = _find_fold_mlp(mlp_dir, k)
            if cls_path is None:
                break

            cls_state = torch.load(cls_path, weights_only=True, map_location=self.device)
            mlp_in = cls_state["network.0.running_mean"].shape[0]

            encoder = None
            if encoder_dir is not None:
                ae_path = encoder_dir / f"fold_{k}" / "best_autoencoder.pt"
                if not ae_path.exists():
                    raise FileNotFoundError(f"Autoencoder checkpoint not found: {ae_path}")
                ae_state = torch.load(ae_path, weights_only=True, map_location=self.device)
                raw_dim = ae_state["encoder.0.running_mean"].shape[0]
                ae = EmbeddingAutoencoder(input_dim=raw_dim, latent_dim=latent_dim)
                ae.load_state_dict(ae_state)
                encoder = ae.encoder.to(self.device).eval()

            model = MLPModel(input_dim=mlp_in).to(self.device)
            model.load_state_dict(cls_state)
            model.eval()

            self.encoders.append(encoder)
            self.models.append(model)
            k += 1

        if not self.models:
            raise FileNotFoundError(
                f"No fold checkpoints found in {mlp_dir} "
                "(expected fold_K/best_model.pt or fold_K.pt)"
            )

    @torch.no_grad()
    def predict(self, X):
        """X: array-like [N, D]. Returns [N, 2] = (mean_prob, var_prob)."""
        x = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self.device)

        stacked = torch.stack(
            [
                torch.sigmoid(m(enc(x) if enc is not None else x))
                for enc, m in zip(self.encoders, self.models)
            ],
            dim=0,
        )

        mean_probs = stacked.mean(dim=0).cpu().numpy()
        var_probs = (
            stacked.var(dim=0).cpu().numpy()
            if stacked.shape[0] > 1
            else np.zeros_like(mean_probs)
        )
        return np.column_stack([mean_probs, var_probs])
