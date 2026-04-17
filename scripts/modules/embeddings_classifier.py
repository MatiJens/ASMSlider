import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.mlp_model import MLPModel


class EmbeddingsClassifier:
    def __init__(self, checkpoint_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(checkpoint_path, weights_only=True, map_location=self.device)
        input_dim = state_dict["network.0.running_mean"].shape[0]
        self.model = MLPModel(input_dim=input_dim).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.no_grad()
    def predict(self, embeddings):
        """Predict probabilities."""
        x = torch.from_numpy(np.asarray(embeddings, dtype=np.float32)).to(self.device)
        return torch.sigmoid(self.model(x)).cpu().numpy()
