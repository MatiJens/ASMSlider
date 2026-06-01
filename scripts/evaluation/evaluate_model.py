import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    precision_recall_curve,
)

from models.ensemble import EnsembleModel
from utils.metrics import best_f1, tpr_at_fpr
from utils.sequence_loader import load_pos_neg


def batched_predict(model, X, batch_size):
    all_probs = []
    for i in range(0, len(X), batch_size):
        result = model.predict(X[i : i + batch_size])
        all_probs.append(result[:, 0])
    return np.concatenate(all_probs)


def plot_pr_curve(y_true, y_score, ap, output_path):
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"AP={ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision-Recall curve")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True,
                        help="Directory with positive/test/ and negative/test/.")
    parser.add_argument("--checkpoint-dir", required=True,
                        help="Directory with per-fold MLP checkpoints "
                             "(fold_K/best_model.pt or fold_K.pt).")
    parser.add_argument("--encoder-checkpoint-dir", default=None,
                        help="Optional directory with per-fold AE checkpoints "
                             "(fold_K/best_autoencoder.pt).")
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--batch-size", type=int, default=512)
    return parser


def main():
    args = create_parser().parse_args()

    X, y_true = load_pos_neg(args.input_path, "test")
    model = EnsembleModel(
        mlp_dir=args.checkpoint_dir,
        encoder_dir=args.encoder_checkpoint_dir,
        latent_dim=args.latent_dim,
    )
    y_score = batched_predict(model, X, args.batch_size)

    ap = average_precision_score(y_true, y_score)
    thr, f1, _, _ = best_f1(y_true, y_score)
    mcc = matthews_corrcoef(y_true, (y_score >= thr).astype(int))
    fpr_targets = [1e-3, 1e-4, 1e-5, 1e-6]
    tpr_results = {fpr: tpr_at_fpr(y_true, y_score, fpr) for fpr in fpr_targets}

    print(f"AP:                 {ap:.6f}")
    print(f"Best F1:            {f1:.6f}  (threshold={thr:.4f})")
    print(f"MCC:                {mcc:.6f}")
    for fpr, tpr in tpr_results.items():
        print(f"TPR @ FPR={fpr:.0e}:   {tpr:.6f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {"ap": ap, "f1": f1, "f1_threshold": thr, "mcc": mcc}
    for fpr, tpr in tpr_results.items():
        metrics[f"tpr_at_fpr_{fpr:.0e}"] = tpr

    metrics_path = out_dir / "metrics.txt"
    with open(metrics_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    print(f"Metrics saved to {metrics_path}")

    pr_path = out_dir / "pr_curve.png"
    plot_pr_curve(y_true, y_score, ap, pr_path)
    print(f"PR curve saved to {pr_path}")


if __name__ == "__main__":
    main()
