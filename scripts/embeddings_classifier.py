import numpy as np
import torch

from cnn_model import CNNModel
from mlp_model import MLPModel

_MODEL_REGISTRY = {"mlp": MLPModel, "cnn": CNNModel}


def _pad_sequences(arrays):
    """Pad list of (seq_len_i, emb_dim) arrays to (N, max_len, emb_dim) float32."""
    max_len = max(a.shape[0] for a in arrays)
    emb_dim = arrays[0].shape[1]
    out = np.zeros((len(arrays), max_len, emb_dim), dtype=np.float32)
    for i, a in enumerate(arrays):
        out[i, : a.shape[0]] = a.astype(np.float32)
    return out


class EmbeddingsClassifier:
    def __init__(self, checkpoint_path, model_type="mlp"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _MODEL_REGISTRY[model_type]().to(self.device)
        self.model.load_state_dict(
            torch.load(checkpoint_path, weights_only=True, map_location=self.device)
        )
        self.model.eval()

    @torch.no_grad()
    def predict(self, embeddings):
        """Predict probabilities.

        embeddings: ndarray (N, emb_dim), ndarray (N, seq_len, emb_dim),
                    or list of (seq_len_i, emb_dim) arrays (padded automatically).
        """
        if isinstance(embeddings, list):
            embeddings = _pad_sequences(embeddings)
        x = torch.from_numpy(np.asarray(embeddings, dtype=np.float32)).to(self.device)
        return torch.sigmoid(self.model(x)).cpu().numpy()
