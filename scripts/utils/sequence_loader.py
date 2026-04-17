import os
from pathlib import Path

import numpy as np
from Bio import SeqIO


def load_fasta(filepath):
    """Load FASTA file. Returns list of {"id": str, "seq": str}, sorted by length desc."""
    sequences = [
        {"id": record.id, "seq": str(record.seq)}
        for record in SeqIO.parse(filepath, "fasta")
    ]
    if not sequences:
        raise ValueError(f"No sequences found in {filepath}")
    sequences.sort(key=lambda x: len(x["seq"]), reverse=True)
    print(f"Loaded {len(sequences)} sequences from {Path(filepath).name}")
    return sequences


def save_embeddings(embeddings, filepath):
    """Save embeddings to .npy."""
    os.makedirs(Path(filepath).parent, exist_ok=True)
    np.save(Path(filepath).with_suffix(".npy"), embeddings)
    print(f"Saved {embeddings.shape[0]} embeddings to {filepath}")


def load_embeddings(filepath):
    """Load .npy embeddings. Returns np.ndarray float32."""
    embeddings = np.load(Path(filepath).with_suffix(".npy")).astype(np.float32)
    print(f"Loaded {embeddings.shape[0]} embeddings {embeddings.shape}")
    return embeddings


def load_embeddings_dir(dir_path):
    """Load and concatenate all .npy files from a directory. Returns np.ndarray float32."""
    arrays = [
        load_embeddings(f)
        for f in sorted(Path(dir_path).glob("*.npy"))
    ]
    if not arrays:
        raise ValueError(f"No .npy files found in {dir_path}")
    return np.concatenate(arrays)
