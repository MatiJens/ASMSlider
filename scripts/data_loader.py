import logging
import os
from pathlib import Path

import numpy as np
from Bio import SeqIO

logger = logging.getLogger(__name__)


class SequenceLoader:
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
