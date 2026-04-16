import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from umap import UMAP

logger = logging.getLogger(__name__)


def _pool_per_residue(arr):
    """Mean-pool per-residue object array → (N, emb_dim) float32."""
    return np.stack([emb.astype(np.float32).mean(axis=0) for emb in arr])


def _load_npy(path):
    """Load a single .npy, mean-pooling per-residue arrays automatically."""
    raw = np.load(path, allow_pickle=True)
    if raw.dtype == object:
        return _pool_per_residue(raw)
    return raw.astype(np.float32)


def load_npy_dir(dir_path):
    arrays = [_load_npy(f) for f in sorted(Path(dir_path).glob("*.npy"))]
    if not arrays:
        raise ValueError(f"No .npy files found in {dir_path}")
    return np.concatenate(arrays)


def create_parser():
    parser = argparse.ArgumentParser(
        description="Generate a 2D UMAP plot from embedding .npy files."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Directories or .npy files to plot. Each entry becomes a separate group.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Labels for each input (must match --inputs length). Defaults to file/dir names.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="umap_2d.png",
        help="Output image path (default: umap_2d.png).",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors (default: 15).",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist (default: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=5.0,
        help="Scatter point size (default: 5).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Scatter point alpha (default: 0.6).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output DPI (default: 200).",
    )
    return parser


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    args = create_parser().parse_args()

    groups = []
    for path_str in args.inputs:
        p = Path(path_str)
        if p.is_dir():
            groups.append(load_npy_dir(p))
        elif p.suffix == ".npy":
            groups.append(_load_npy(p))
        else:
            raise ValueError(f"Input must be a directory or .npy file: {p}")
        logger.info(f"Loaded {groups[-1].shape[0]} embeddings from {p}")

    if args.labels:
        if len(args.labels) != len(args.inputs):
            raise ValueError("--labels must have the same length as --inputs")
        labels = args.labels
    else:
        labels = [Path(p).stem for p in args.inputs]

    all_emb = np.concatenate(groups)
    group_ids = np.concatenate(
        [np.full(len(g), i) for i, g in enumerate(groups)]
    )

    logger.info(
        f"Running UMAP on {all_emb.shape[0]} embeddings (dim={all_emb.shape[1]})"
    )
    reducer = UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.seed,
    )
    coords = reducer.fit_transform(all_emb)
    logger.info("UMAP done")

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, label in enumerate(labels):
        mask = group_ids == i
        if i == 0:
            color, alpha, zorder = "lightgrey", 0.25, 0
        else:
            color, alpha, zorder = None, args.alpha, 1
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=args.point_size,
            alpha=alpha,
            color=color,
            zorder=zorder,
            label=f"{label} (n={mask.sum()})",
        )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=3)
    ax.set_title("UMAP 2D")
    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    logger.info(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
