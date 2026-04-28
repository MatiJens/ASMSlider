import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.mlp_model import MLPModel
from models.autoencoder_model import EmbeddingAutoencoder


def resolve_checkpoints(checkpoint_path, pattern="*.pt"):
    """Accept a .pt file, a directory, or a list. Return list of checkpoint Paths."""
    if isinstance(checkpoint_path, (list, tuple)):
        return [Path(p) for p in checkpoint_path]
    p = Path(checkpoint_path)
    if p.is_dir():
        files = sorted(p.rglob(pattern))
        if not files:
            raise ValueError(f"No checkpoints matching '{pattern}' found in {p}")
        return files
    return [p]


class EmbeddingsClassifier:
    """Ensemble of MLP classifiers, optionally paired with per-fold autoencoders.

    When encoder_path is provided, each raw embedding is encoded by AE_k before
    being passed to MLP_k. Pairs are matched by sorted order, so the directory
    layouts should be parallel (e.g. fold_1/, fold_2/, ...).
    """

    def __init__(
        self,
        checkpoint_path,
        encoder_path=None,
        input_dim=1152,
        latent_dim=128,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        mlp_paths = resolve_checkpoints(checkpoint_path)

        self.encoders = []
        if encoder_path is not None:
            enc_paths = resolve_checkpoints(encoder_path)
            if len(enc_paths) != len(mlp_paths):
                raise ValueError(
                    f"Encoder/classifier count mismatch: {len(enc_paths)} encoders "
                    f"vs {len(mlp_paths)} classifiers. They must be paired 1:1."
                )
            for ep in enc_paths:
                ae = EmbeddingAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
                ae.load_state_dict(
                    torch.load(ep, weights_only=True, map_location=self.device)
                )
                self.encoders.append(ae.encoder.to(self.device).eval())

        self.models = []
        for mp in mlp_paths:
            state_dict = torch.load(mp, weights_only=True, map_location=self.device)
            mlp_in = state_dict["network.0.running_mean"].shape[0]
            model = MLPModel(input_dim=mlp_in).to(self.device)
            model.load_state_dict(state_dict)
            model.eval()
            self.models.append(model)

        if self.encoders:
            print(f"Loaded {len(self.encoders)} (encoder, classifier) pair(s).")
        else:
            print(f"Loaded {len(self.models)} classifier checkpoint(s).")

    @torch.no_grad()
    def predict(self, embeddings):
        """Predict probabilities, averaged over all loaded (encoder, classifier) pairs
        (or over classifiers alone if no encoders were provided)."""
        x = torch.from_numpy(np.asarray(embeddings, dtype=np.float32)).to(self.device)
        if self.encoders:
            probs = torch.stack(
                [
                    torch.sigmoid(mlp(enc(x)))
                    for enc, mlp in zip(self.encoders, self.models)
                ],
                dim=0,
            ).mean(dim=0)
        else:
            probs = torch.stack(
                [torch.sigmoid(m(x)) for m in self.models], dim=0
            ).mean(dim=0)
        return probs.cpu().numpy()
