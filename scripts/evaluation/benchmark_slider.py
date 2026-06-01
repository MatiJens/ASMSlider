import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from Bio import SeqIO
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
)

from models.ensemble import EnsembleModel
from modules.embeddings_generator import EmbeddingsGenerator
from utils.metrics import best_f1, tpr_at_fpr


def score_sequences(sequences, model, pooling, window_size, stride, batch_size):
    """For each sequence, slide a window and return the max probability."""
    scores = []
    for i, seq in enumerate(sequences):
        if window_size > len(seq):
            scores.append(0.0)
            continue
        starts = range(0, len(seq) - window_size + 1, stride)
        fragments = [seq[s : s + window_size] for s in starts]
        embeddings = EmbeddingsGenerator.generate(fragments, pooling, batch_size)
        result = model.predict(embeddings)
        probas = result[:, 0]
        scores.append(float(probas.max()))
        if (i + 1) % 50 == 0:
            print(f"  scored {i + 1}/{len(sequences)} sequences")
    return np.array(scores)


def compute_metrics(labels, scores):
    ap = average_precision_score(labels, scores)
    fpr_targets = [1e-3, 1e-4, 1e-5]
    tprs = {f"TPR@FPR={fpr:.0e}": tpr_at_fpr(labels, scores, fpr) for fpr in fpr_targets}
    thresh, f1, prec, rec = best_f1(labels, scores)
    return {
        "AP": ap,
        **tprs,
        "best_F1": f1,
        "best_F1_threshold": thresh,
        "best_F1_precision": prec,
        "best_F1_recall": rec,
    }


def plot_pr_curves(results, output_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for ws, metrics, labels, scores in results:
        precision, recall, _ = precision_recall_curve(labels, scores)
        ax.plot(recall, precision,
                label=f"w={ws} (AP={metrics['AP']:.3f})", linewidth=1.5)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Sequence-level Precision-Recall (sliding window)")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"PR plot saved to {output_path}")


def create_parser():
    p = argparse.ArgumentParser(
        description="Benchmark ASMSlider on positive/negative FASTA: "
                    "slide windows, take max score per sequence, compute AP and TPR@FPR."
    )
    p.add_argument("--positive-fasta", required=True,
                   help="FASTA with ASM-containing sequences.")
    p.add_argument("--negative-fasta", required=True,
                   help="FASTA with non-ASM sequences.")
    p.add_argument("--checkpoint-dir", required=True,
                   help="Directory with per-fold MLP checkpoints "
                        "(fold_K/best_model.pt or fold_K.pt).")
    p.add_argument("--encoder-checkpoint-dir", default=None,
                   help="Optional directory with per-fold AE checkpoints "
                        "(fold_K/best_autoencoder.pt).")
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--pooling", default="max", choices=["mean", "max"])
    p.add_argument("--window-sizes", type=int, nargs="+", default=[30],
                   help="Window sizes to benchmark (e.g. 20 30 40 50).")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--outdir", required=True)
    return p


def main():
    args = create_parser().parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model = EnsembleModel(
        mlp_dir=args.checkpoint_dir,
        encoder_dir=args.encoder_checkpoint_dir,
        latent_dim=args.latent_dim,
    )

    pos_seqs = [str(r.seq) for r in SeqIO.parse(args.positive_fasta, "fasta")]
    neg_seqs = [str(r.seq) for r in SeqIO.parse(args.negative_fasta, "fasta")]
    print(f"Positive: {len(pos_seqs)} sequences")
    print(f"Negative: {len(neg_seqs)} sequences")

    labels = np.array([1.0] * len(pos_seqs) + [0.0] * len(neg_seqs))
    all_seqs = pos_seqs + neg_seqs

    results = []
    all_metrics = {}

    for ws in args.window_sizes:
        print(f"\n--- Window size = {ws} ---")
        scores = score_sequences(all_seqs, model, args.pooling, ws, args.stride, args.batch_size)
        metrics = compute_metrics(labels, scores)
        results.append((ws, metrics, labels, scores))
        all_metrics[ws] = metrics
        for k, v in metrics.items():
            print(f"  {k}: {v:.6f}")

    plot_pr_curves(results, outdir / "pr_curves.png")

    sep = "=" * 85
    print(f"\n{sep}")
    header = (f"{'window':>8}  {'AP':>8}"
              f"  {'TPR@1e-3':>10}  {'TPR@1e-4':>10}  {'TPR@1e-5':>10}"
              f"  {'best_F1':>8}  {'thresh':>8}")
    print(header)
    print("-" * 85)
    for ws in args.window_sizes:
        m = all_metrics[ws]
        row = (f"{ws:>8}  {m['AP']:>8.4f}"
               f"  {m['TPR@FPR=1e-03']:>10.4f}  {m['TPR@FPR=1e-04']:>10.4f}  {m['TPR@FPR=1e-05']:>10.4f}"
               f"  {m['best_F1']:>8.4f}  {m['best_F1_threshold']:>8.4f}")
        print(row)
    print(sep)

    with open(outdir / "metrics.json", "w") as f:
        json.dump({str(ws): m for ws, m in all_metrics.items()}, f, indent=2)
    print(f"\nMetrics saved to {outdir / 'metrics.json'}")

    with open(outdir / "summary.txt", "w") as f:
        f.write("=== Benchmark parameters ===\n")
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
        f.write(f"\nPositive sequences: {len(pos_seqs)}\n")
        f.write(f"Negative sequences: {len(neg_seqs)}\n")
        f.write(f"\n=== Results ===\n")
        for ws in args.window_sizes:
            m = all_metrics[ws]
            f.write(f"\nwindow_size={ws}:\n")
            for k, v in m.items():
                f.write(f"  {k}: {v:.6f}\n")
    print(f"Summary saved to {outdir / 'summary.txt'}")


if __name__ == "__main__":
    main()
