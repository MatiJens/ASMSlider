import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
from umap import UMAP

from models.autoencoder_model import EmbeddingAutoencoder
from utils.sequence_loader import load_embeddings, load_embeddings_dir


@torch.no_grad()
def encode_with_ae(embeddings, ae_checkpoint, latent_dim, batch_size=1024):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae = EmbeddingAutoencoder(input_dim=embeddings.shape[1], latent_dim=latent_dim)
    ae.load_state_dict(torch.load(ae_checkpoint, weights_only=True, map_location=device))
    encoder = ae.encoder.to(device).eval()

    x = torch.from_numpy(embeddings.astype(np.float32))
    out = [encoder(x[i : i + batch_size].to(device)).cpu()
           for i in range(0, len(x), batch_size)]
    return torch.cat(out).numpy()


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
        "--n-neighbors", type=int, default=15, help="UMAP n_neighbors (default: 15)."
    )
    parser.add_argument(
        "--min-dist", type=float, default=0.1, help="UMAP min_dist (default: 0.1)."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)."
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=2.5,
        help="Scatter point size (default: 2.5).",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.6, help="Scatter point alpha (default: 0.6)."
    )
    parser.add_argument(
        "--dpi", type=int, default=200, help="Output DPI (default: 200)."
    )
    parser.add_argument("--title", type=str, default="UMAP 2D", help="Plot title.")
    parser.add_argument(
        "--encoder", type=str, default=None,
        help="Path to an autoencoder .pt checkpoint. If given, embeddings are encoded "
             "(1152 -> latent_dim) before UMAP.",
    )
    parser.add_argument(
        "--latent-dim", type=int, default=128,
        help="AE latent dim (default: 128). Used only with --encoder.",
    )
    return parser


def main():
    args = create_parser().parse_args()

    groups = []
    for path_str in args.inputs:
        p = Path(path_str)
        if p.is_dir():
            groups.append(load_embeddings_dir(p))
        elif p.suffix == ".npy":
            groups.append(load_embeddings(p))
        else:
            raise ValueError(f"Input must be a directory or .npy file: {p}")
        print(f"Loaded {groups[-1].shape[0]} embeddings from {p}")

    if args.labels:
        if len(args.labels) != len(args.inputs):
            raise ValueError("--labels must have the same length as --inputs")
        labels = args.labels
    else:
        labels = [Path(p).stem for p in args.inputs]

    all_emb = np.concatenate(groups)
    group_ids = np.concatenate([np.full(len(g), i) for i, g in enumerate(groups)])

    if args.encoder:
        print(f"Encoding with {args.encoder} ({all_emb.shape[1]} -> {args.latent_dim})")
        all_emb = encode_with_ae(all_emb, args.encoder, args.latent_dim)

    reducer = UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.seed,
    )
    coords = reducer.fit_transform(all_emb)
    print("UMAP done")

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
            label=label,
        )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=3)
    ax.set_title(args.title)
    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
