import json
import logging
import h5py
import sys
import pickle
import numpy as np
import os

from pathlib import Path

import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


class EmbeddingsClassifier:
    def __init__(self, model_path):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        logger.info(f"Classifier model loaded: {model_path}")

    def _load_emb(self, filepath):
        with h5py.File(filepath, "r") as f:
            keys = list(f.keys())
            embeddings = np.stack([f[k][:] for k in keys], axis=0).astype(np.float32)
        return keys, embeddings

    def _save_results(self, results, output_file):
        os.makedirs(Path(output_file).parent, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Predictions saved to {output_file}")
        return results

    def predict(self, embedding):
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        proba = self.model.predict_proba(embedding.astype(np.float32))
        return float(proba[0, 1])

    def predict_from_file(self, path, output_file):
        keys, X = self._load_emb(path)
        proba = self.model.predict_proba(X)
        results = {name: float(p) for name, p in zip(keys, proba[:, 1])}
        self._save_results(results, output_file)

    def predict_batch(self, embeddings):
        return self.model.predict_proba(embeddings.astype(np.float32))[:, 1]
