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
    def save_embeddings(embeddings, filepath):
        """Save embeddings to .npy."""
        os.makedirs(Path(filepath).parent, exist_ok=True)
        np.save(Path(filepath).with_suffix(".npy"), embeddings)
        logger.info(f"Saved {embeddings.shape[0]} embeddings to {filepath}")

    @staticmethod
    def load_embeddings(filepath):
        """Load .npy embeddings. Returns np.ndarray float32."""
        embeddings = np.load(Path(filepath).with_suffix(".npy")).astype(np.float32)
        logger.info(f"Loaded {embeddings.shape[0]} embeddings {embeddings.shape}")
        return embeddings
