import numpy as np
import torch


class EmbeddingsClassifier:
    def __init__(self, model, checkpoint_path):
        self.model = model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(
            torch.load(checkpoint_path, weights_only=True, map_location=device)
        )
        self.model.eval()

    @torch.no_grad()
    def predict(self, embeddings):
        """Predict probability for a single embedding. Returns float."""
        logits = self.model(torch.from_numpy(embeddings))
        return torch.sigmoid(logits).numpy()
