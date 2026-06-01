import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve

from models.autoencoder_model import EmbeddingAutoencoder
from models.mlp_model import MLPModel
from training.train_classifier import load_dataset, make_loader, encode_dataset, predict_probs


def load_model(mlp_dir, ae_dir, fold, device):
    mlp_path = Path(mlp_dir) / f"fold_{fold}" / "best_model.pt"
    cls_state = torch.load(mlp_path, weights_only=True, map_location=device)
    mlp_input_dim = cls_state["network.0.running_mean"].shape[0]
    mlp = MLPModel(input_dim=mlp_input_dim)
    mlp.load_state_dict(cls_state)
    mlp.to(device).eval()

    encoder = None
    if ae_dir:
        ae_path = Path(ae_dir) / f"fold_{fold}" / "best_autoencoder.pt"
        ae_state = torch.load(ae_path, weights_only=True, map_location=device)
        ae_input_dim = ae_state["encoder.0.running_mean"].shape[0]
        ae = EmbeddingAutoencoder(input_dim=ae_input_dim, latent_dim=mlp_input_dim)
        ae.load_state_dict(ae_state)
        encoder = ae.encoder.to(device).eval()

    return mlp, encoder


def create_parser():
    parser = argparse.ArgumentParser(
        description="Compare models via PR curves on validation data."
    )
    parser.add_argument(
        "--model",
        action="append",
        nargs="+",
        required=True,
        metavar="ARG",
        help="NAME INPUT_PATH MLP_DIR [AE_DIR]. Pass once per model.",
    )
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output-dir", default=".")
    return parser


def main():
    args = create_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    results = []

    for entry in args.model:
        if len(entry) == 4:
            name, input_path, mlp_dir, ae_dir = entry
        elif len(entry) == 3:
            name, input_path, mlp_dir = entry
            ae_dir = None
        else:
            raise ValueError(f"--model expects 3 or 4 args, got {len(entry)}: {entry}")

        print(f"\n[{name}] mlp={mlp_dir}, ae={ae_dir or 'none'}")
        mlp, encoder = load_model(mlp_dir, ae_dir, args.fold, device)

        val_ds = load_dataset(input_path, "val", f"*val{args.fold}.npy")
        if encoder is not None:
            val_ds = encode_dataset(val_ds, encoder, device)
        loader = make_loader(val_ds, args.batch_size)
        probs, targets, _ = predict_probs(mlp, loader, device)

        ap = average_precision_score(targets, probs)
        precision, recall, _ = precision_recall_curve(targets, probs)
        ax.plot(recall, precision, label=f"{name} AP={ap:.3f}")
        results.append((name, ap, len(targets), int(targets.sum())))
        print(f"[{name}] AP={ap:.6f}")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision-Recall comparison")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    plot_path = out_dir / "pr_comparison.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved to {plot_path}")

    summary_path = out_dir / "pr_comparison.txt"
    with open(summary_path, "w") as f:
        f.write("model\tap\tn\tpos\n")
        for name, ap, n, pos in results:
            f.write(f"{name}\t{ap:.6f}\t{n}\t{pos}\n")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
