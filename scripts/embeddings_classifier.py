import torch

from mlp_model import MLPModel


class EmbeddingsClassifier:
    def __init__(self, checkpoint_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MLPModel().to(self.device)
        self.model.load_state_dict(
            torch.load(checkpoint_path, weights_only=True, map_location=self.device)
        )
        self.model.eval()

    @torch.no_grad()
    def predict(self, embeddings):
        """Predict probabilities for embeddings. Returns numpy array of probabilities."""
        x = torch.from_numpy(embeddings).to(self.device)
        logits = self.model(x)
        return torch.sigmoid(logits).cpu().numpy()
