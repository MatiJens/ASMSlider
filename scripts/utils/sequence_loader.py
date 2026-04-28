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
    return np.load(Path(filepath).with_suffix(".npy")).astype(np.float32)


def load_embeddings_dir(dir_path):
    """Load and concatenate all .npy files from a directory."""
    files = sorted(Path(dir_path).glob("*.npy"))
    if not files:
        raise ValueError(f"No .npy files found in {dir_path}")
    return np.concatenate([load_embeddings(f) for f in files])


def load_pos_neg(base_path, subdir, pattern="*.npy"):
    """Load positive/<subdir>/<pattern> + negative/<subdir>/<pattern>. Returns (X, y)."""
    p = Path(base_path)
    pos_files = sorted((p / "positive" / subdir).glob(pattern))
    neg_files = sorted((p / "negative" / subdir).glob(pattern))
    if not pos_files:
        raise ValueError(f"No .npy files matching '{pattern}' in {p / 'positive' / subdir}")
    if not neg_files:
        raise ValueError(f"No .npy files matching '{pattern}' in {p / 'negative' / subdir}")
    pos = np.concatenate([load_embeddings(f) for f in pos_files])
    neg = np.concatenate([load_embeddings(f) for f in neg_files])
    X = np.concatenate([pos, neg])
    y = np.array([1] * len(pos) + [0] * len(neg), dtype=np.float32)
    return X, y
