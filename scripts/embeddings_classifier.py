import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch

from data_loader import SequenceLoader
from mlp_model import MLPModel

logger = logging.getLogger(__name__)


class EmbeddingsClassifier:
    def __init__(self, checkpoint_path):
        self.model = MLPModel.load_from_checkpoint(checkpoint_path, map_location="cpu")
        self.model.eval()
        logger.info(f"MLP classifier loaded: {checkpoint_path}")

    @torch.no_grad()
    def predict(self, embedding):
        """Predict probability for a single embedding. Returns float."""
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        logits = self.model(torch.from_numpy(embedding.astype(np.float32)))
        return float(torch.sigmoid(logits)[0])

    @torch.no_grad()
    def predict_batch(self, embeddings):
        """Predict probabilities for a batch. Returns np.ndarray of shape (n,)."""
        logits = self.model(torch.from_numpy(embeddings.astype(np.float32)))
        return torch.sigmoid(logits).numpy()

    def predict_from_file(self, input_file, output_file):
        """Load embeddings from .npz, predict and save results as JSON."""
        keys, X = SequenceLoader.load_embeddings(input_file)
        proba = self.predict_batch(X)
        results = {name: float(p) for name, p in zip(keys, proba)}

        os.makedirs(Path(output_file).parent, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Predictions saved to {output_file}")


def create_parser():
    parser = argparse.ArgumentParser(
        description="Classify embeddings using a trained MLP model."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to .npz file with embeddings.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        required=True,
        help="Output .json file path.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to Lightning .ckpt file.",
    )
    return parser


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = create_parser()
    args = parser.parse_args()

    classifier = EmbeddingsClassifier(checkpoint_path=args.checkpoint_path)
    classifier.predict_from_file(args.input_path, args.output_path)


if __name__ == "__main__":
    main()
