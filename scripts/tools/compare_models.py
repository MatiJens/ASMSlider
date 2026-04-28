import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve

from models.autoencoder_model import EmbeddingAutoencoder
from models.mlp_model import MLPModel
from utils.sequence_loader import load_pos_neg


def load_mlp(ckpt_path, device):
    state_dict = torch.load(ckpt_path, weights_only=True, map_location=device)
    input_dim = state_dict["network.0.running_mean"].shape[0]
    model = MLPModel(input_dim=input_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_encoder(ae_path, input_dim, latent_dim, device):
    ae = EmbeddingAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
    ae.load_state_dict(torch.load(ae_path, weights_only=True, map_location=device))
    return ae.encoder.to(device).eval()


@torch.no_grad()
def batched_forward(x_np, module, device, batch_size=1024):
    x = torch.from_numpy(x_np)
    out = [
        module(x[i : i + batch_size].to(device)).cpu()
        for i in range(0, len(x), batch_size)
    ]
    return torch.cat(out).numpy()


def oof_predict(input_path, mlp_dir, encoder_dir, latent_dim, folds, device):
    """Per-fold val predictions concatenated into one OOF vector."""
    all_probs, all_targets = [], []
    for k in range(1, folds + 1):
        X, y = load_pos_neg(input_path, "val", f"*val{k}.npy")
        if encoder_dir is not None:
            ae_path = Path(encoder_dir) / f"fold_{k}" / "best_autoencoder.pt"
            X = batched_forward(
                X, load_encoder(ae_path, X.shape[1], latent_dim, device), device
            )
        mlp_path = Path(mlp_dir) / f"fold_{k}" / "best_model.pt"
        logits = batched_forward(X, load_mlp(mlp_path, device), device)
        probs = 1.0 / (1.0 + np.exp(-logits))
        all_probs.append(probs)
        all_targets.append(y)
        print(
            f"  fold {k}: {len(y)} samples, fold-AUPRC={average_precision_score(y, probs):.4f}"
        )
    return np.concatenate(all_probs), np.concatenate(all_targets)


def create_parser():
    parser = argparse.ArgumentParser(
        description="Compare MLP variants via out-of-fold validation PR curves."
    )
    parser.add_argument(
        "--model",
        action="append",
        nargs="+",
        required=True,
        metavar="ARG",
        help="Variant: NAME INPUT_PATH CHECKPOINT_DIR [ENCODER_DIR]. Pass once per model.",
    )
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--folds", type=int, default=6)
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
        if len(entry) not in (3, 4):
            raise ValueError(
                f"--model expects 3 or 4 values (NAME INPUT_PATH CHECKPOINT [ENCODER]), got {entry}"
            )
        name, input_path, checkpoint = entry[0], entry[1], entry[2]
        encoder = entry[3] if len(entry) == 4 else None

        print(
            f"\n[{name}] OOF prediction over {args.folds} val folds "
            f"(encoder={'yes' if encoder else 'no'})"
        )
        probs, targets = oof_predict(
            input_path, checkpoint, encoder, args.latent_dim, args.folds, device
        )
        ap = average_precision_score(targets, probs)
        precision, recall, _ = precision_recall_curve(targets, probs)
        ax.plot(recall, precision, label=f"{name} (AUPRC={ap:.4f})")
        results.append((name, ap, len(targets), int(targets.sum())))
        print(
            f"[{name}] OOF AUPRC = {ap:.6f}  (n={len(targets)}, pos={int(targets.sum())})"
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision-Recall (out-of-fold validation)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    plot_path = out_dir / "pr_comparison.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved to {plot_path}")

    summary_path = out_dir / "pr_comparison.txt"
    with open(summary_path, "w") as f:
        f.write("model\tauprc\tn\tpos\n")
        for name, ap, n, pos in results:
            f.write(f"{name}\t{ap:.6f}\t{n}\t{pos}\n")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
