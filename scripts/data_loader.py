import json
import logging
import os
from pathlib import Path

import numpy as np
from Bio import SeqIO

logger = logging.getLogger(__name__)


class DataLoader:
    @staticmethod
    def parse_config(config_path):
        """Load and validate JSON config. Returns dict."""
        with open(config_path, "r") as f:
            config = json.load(f)

        if not isinstance(config, dict):
            raise ValueError(f"Config must be a JSON dict, got {type(config).__name__}")

        for name, classes in config.items():
            if not isinstance(classes, dict) or not classes:
                raise ValueError(f"Entry '{name}' must be a non-empty dict")
            for class_name, filepath in classes.items():
                if not os.path.isfile(filepath):
                    raise FileNotFoundError(
                        f"[{name}/{class_name}] File not found: {filepath}"
                    )
        return config

    @staticmethod
    def iter_entries(config):
        """Yield (entry_name, class_name, filepath) for every file in config."""
        for entry_name, classes in config.items():
            for class_name, filepath in classes.items():
                yield entry_name, class_name, filepath

    @staticmethod
    def load_fasta(filepath):
        """Load FASTA file. Returns list of {"id": str, "seq": str}, sorted by length desc."""
        sequences = [
            {"id": record.id, "seq": str(record.seq)}
            for record in SeqIO.parse(filepath, "fasta")
        ]
        if not sequences:
            raise ValueError(f"No sequences found in {filepath}")
        sequences.sort(key=lambda x: len(x["seq"]), reverse=True)
        logger.info(f"Loaded {len(sequences)} sequences from {Path(filepath).name}")
        return sequences

    @staticmethod
    def load_embeddings(filepath):
        """Load .npz embeddings. Returns (keys: list[str], embeddings: np.ndarray float32)."""
        data = np.load(filepath)
        keys = data["keys"].tolist()
        embeddings = data["embeddings"].astype(np.float32)
        logger.info(
            f"Loaded {len(keys)} embeddings {embeddings.shape} from {Path(filepath).name}"
        )
        return keys, embeddings

    @staticmethod
    def save_embeddings(keys, embeddings, filepath):
        """Save embeddings to .npz."""
        os.makedirs(Path(filepath).parent, exist_ok=True)
        np.savez(filepath, keys=np.array(keys), embeddings=embeddings)
        logger.info(f"Saved {len(keys)} embeddings to {filepath}")

    @staticmethod
    def _get_fold_indices(config):
        return sorted({int(k.split("_")[1]) for k in config if k.startswith("train_")})

    @staticmethod
    def _load_labeled_pair(positive_file, negative_file):
        """Load positive + negative embeddings, return (X, y)."""
        _, pos = DataLoader.load_embeddings(positive_file)
        _, neg = DataLoader.load_embeddings(negative_file)
        X = np.concatenate([pos, neg], axis=0)
        y = np.array([1] * len(pos) + [0] * len(neg))
        return X, y

    @staticmethod
    def load_split(config, split_name):
        """Load a single split by name (e.g. 'train_1', 'test'). Returns (X, y)."""
        entry = config[split_name]
        return DataLoader._load_labeled_pair(entry["positive"], entry["negative"])

    @staticmethod
    def iter_folds(config):
        """Yield (fold_idx, X_train, y_train, X_val, y_val) for each k-fold split."""
        for fold_idx in DataLoader._get_fold_indices(config):
            X_train, y_train = DataLoader.load_split(config, f"train_{fold_idx}")
            X_val, y_val = DataLoader.load_split(config, f"val_{fold_idx}")
            yield fold_idx, X_train, y_train, X_val, y_val

    @staticmethod
    def load_all_folds_and_test(config):
        """Load all folds + test set. Returns (folds_list, X_test, y_test)."""
        folds = list(DataLoader.iter_folds(config))
        X_test, y_test = DataLoader.load_split(config, "test")
        return folds, X_test, y_test
