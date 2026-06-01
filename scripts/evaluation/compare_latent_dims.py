import argparse
import shutil
import sys
from datetime import datetime
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, precision_recall_curve
from umap import UMAP

from models.autoencoder_model import EmbeddingAutoencoder
from models.mlp_model import MLPModel
from training.train_autoencoder import (
    evaluate as eval_ae,
    train_one_epoch as train_ae_epoch,
)
from training.train_classifier import (
    encode_dataset,
    load_dataset,
    make_loader,
    predict_probs,
    train_one_epoch as train_cls_epoch,
)
from training.training_utils import train_loop
from utils.focal_loss import FocalLoss


def eval_cls(model, loader, *, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            total_loss += criterion(model(x), y).item()
            n += 1
    return {"loss": total_loss / n}


def train_pipeline_for_dim(latent_dim, args, device, run_dir):
    dim_dir = run_dir / f"dim_{latent_dim}"
    dim_dir.mkdir(parents=True, exist_ok=True)

    focal = FocalLoss(alpha=args.alpha, gamma=args.gamma)
    encoders = []
    cls_ckpts = []

    for k in range(1, args.folds + 1):
        print(f"\n  Fold {k}/{args.folds}")
        fold_dir = dim_dir / f"fold_{k}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_ds = load_dataset(args.input_path, "train", f"*trn{k}.npy")
        val_ds = load_dataset(args.input_path, "val", f"*val{k}.npy")
        input_dim = train_ds.tensors[0].shape[1]

        ae = EmbeddingAutoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
        ae_opt = torch.optim.AdamW(ae.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        ae_ckpt = fold_dir / "best_autoencoder.pt"

        ae_kwargs = dict(
            rec_criterion=nn.MSELoss(),
            cls_criterion=focal,
            lambda_rec=args.lambda_rec, lambda_cls=args.lambda_cls, device=device,
        )
        train_loop(
            ae, make_loader(train_ds, args.batch_size, shuffle=True),
            make_loader(val_ds, args.batch_size),
            partial(train_ae_epoch, **ae_kwargs),
            partial(eval_ae, **ae_kwargs),
            ae_opt, args.ae_epochs, args.patience, ae_ckpt,
        )
        ae.load_state_dict(torch.load(ae_ckpt, weights_only=True, map_location=device))
        encoder = ae.encoder.eval()
        encoders.append(encoder)

        train_enc = encode_dataset(train_ds, encoder, device)
        val_enc = encode_dataset(val_ds, encoder, device)

        mlp = MLPModel(input_dim=latent_dim).to(device)
        cls_opt = torch.optim.AdamW(mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        cls_ckpt = fold_dir / "best_model.pt"

        cls_kwargs = dict(criterion=focal, device=device)
        train_loop(
            mlp, make_loader(train_enc, args.batch_size, shuffle=True),
            make_loader(val_enc, args.batch_size),
            partial(train_cls_epoch, **cls_kwargs),
            partial(eval_cls, **cls_kwargs),
            cls_opt, args.cls_epochs, args.patience, cls_ckpt,
        )
        cls_ckpts.append(cls_ckpt)

    return encoders, cls_ckpts, dim_dir


def per_fold_val_ap(args, device, latent_dim, encoders, cls_ckpts):
    fold_aps = []
    for k, (encoder, ckpt) in enumerate(zip(encoders, cls_ckpts), 1):
        val_ds = load_dataset(args.input_path, "val", f"*val{k}.npy")
        val_enc = encode_dataset(val_ds, encoder, device)
        loader = make_loader(val_enc, args.batch_size)
        mlp = MLPModel(input_dim=latent_dim).to(device)
        mlp.load_state_dict(torch.load(ckpt, weights_only=True, map_location=device))
        probs, targets, _ = predict_probs(mlp, loader, device)
        fold_aps.append(average_precision_score(targets, probs))
    return fold_aps


def ensemble_val_predictions(args, device, latent_dim, encoders, cls_ckpts):
    all_probs, all_targets = [], []
    for k, (encoder, ckpt) in enumerate(zip(encoders, cls_ckpts), 1):
        val_ds = load_dataset(args.input_path, "val", f"*val{k}.npy")
        val_enc = encode_dataset(val_ds, encoder, device)
        loader = make_loader(val_enc, args.batch_size)
        mlp = MLPModel(input_dim=latent_dim).to(device)
        mlp.load_state_dict(torch.load(ckpt, weights_only=True, map_location=device))
        probs, targets, _ = predict_probs(mlp, loader, device)
        all_probs.append(probs)
        all_targets.append(targets)
    return np.concatenate(all_probs), np.concatenate(all_targets)


@torch.no_grad()
def get_latent_embeddings(args, device, encoders):
    all_emb, all_labels = [], []
    for k, encoder in enumerate(encoders, 1):
        val_ds = load_dataset(args.input_path, "val", f"*val{k}.npy")
        X, y = val_ds.tensors
        encoded = [
            encoder(X[i : i + 1024].to(device)).cpu()
            for i in range(0, len(X), 1024)
        ]
        all_emb.append(torch.cat(encoded).numpy())
        all_labels.append(y.numpy())
    return np.concatenate(all_emb), np.concatenate(all_labels)


def plot_pr_curves(results, output_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for dim, probs, targets, mean_ap, var_ap in results:
        precision, recall, _ = precision_recall_curve(targets, probs)
        ensemble_ap = average_precision_score(targets, probs)
        ax.plot(recall, precision,
                label=f"dim={dim} AP={ensemble_ap:.3f}",
                linewidth=1.5)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall: latent dim comparison")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"PR curve saved to {output_path}")


def plot_umaps(umap_data, output_path, seed=42):
    n = len(umap_data)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (dim, embeddings, labels) in zip(axes, umap_data):
        reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=seed)
        coords = reducer.fit_transform(embeddings)
        neg_mask = labels == 0
        pos_mask = labels == 1
        ax.scatter(coords[neg_mask, 0], coords[neg_mask, 1],
                   s=2, alpha=0.25, color="lightgrey", label="negative", zorder=0)
        ax.scatter(coords[pos_mask, 0], coords[pos_mask, 1],
                   s=2, alpha=0.6, label="positive", zorder=1)
        ax.set_title(f"dim={dim}")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.legend(markerscale=3, fontsize=8)

    fig.suptitle("UMAP: latent dim comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"UMAP plot saved to {output_path}")


def create_parser():
    p = argparse.ArgumentParser(
        description="Compare autoencoder latent dimensions: train AE+classifier per dim, "
                    "then plot PR curves and UMAPs."
    )
    p.add_argument("--input-path", required=True,
                   help="Directory with positive/{train,val} and negative/{train,val}.")
    p.add_argument("--latent-dims", type=int, nargs="+", required=True,
                   help="Latent dimensions to compare (e.g. 64 128 256).")
    p.add_argument("--folds", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--ae-epochs", type=int, default=200)
    p.add_argument("--cls-epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--lambda-rec", type=float, default=1.0)
    p.add_argument("--lambda-cls", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--outdir", default="results/compare_dims")
    return p


def main():
    args = create_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.set_float32_matmul_precision("high")

    run_dir = Path(args.outdir) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    pr_results = []
    umap_data = []
    all_stats = {}

    for dim in args.latent_dims:
        print(f"\n{'='*60}\nLatent dim = {dim}\n{'='*60}")
        encoders, cls_ckpts, dim_dir = train_pipeline_for_dim(
            dim, args, device, run_dir
        )

        val_aps = per_fold_val_ap(args, device, dim, encoders, cls_ckpts)
        val_mean = float(np.mean(val_aps))
        val_var = float(np.var(val_aps))
        print(f"dim={dim} per-fold AP: {[f'{a:.4f}' for a in val_aps]}")
        print(f"dim={dim} mean_AP={val_mean:.4f}  var_AP={val_var:.6f}")

        probs, targets = ensemble_val_predictions(args, device, dim, encoders, cls_ckpts)
        ensemble_ap = average_precision_score(targets, probs)
        print(f"dim={dim} ensemble_AP={ensemble_ap:.4f}")

        pr_results.append((dim, probs, targets, val_mean, val_var))
        all_stats[dim] = {
            "fold_aps": val_aps,
            "mean_ap": val_mean,
            "var_ap": val_var,
            "ensemble_ap": ensemble_ap,
        }

        latent_emb, labels = get_latent_embeddings(args, device, encoders)
        umap_data.append((dim, latent_emb, labels))

    plot_pr_curves(pr_results, run_dir / "pr_curves.png")
    plot_umaps(umap_data, run_dir / "umap_comparison.png")

    best_dim = max(all_stats, key=lambda d: all_stats[d]["mean_ap"])
    best_dir = run_dir / "best"
    shutil.copytree(run_dir / f"dim_{best_dim}", best_dir)
    print(f"\nBest dim={best_dim} (mean_AP={all_stats[best_dim]['mean_ap']:.4f}) "
          f"copied to {best_dir}")

    with open(run_dir / "summary.txt", "w") as f:
        f.write("=== Latent dim comparison ===\n")
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
        f.write("\n=== Per-dimension results (validation) ===\n")
        for dim in args.latent_dims:
            s = all_stats[dim]
            f.write(f"\ndim={dim}:\n")
            for i, ap in enumerate(s["fold_aps"], 1):
                f.write(f"  fold_{i} AP: {ap:.4f}\n")
            f.write(f"  mean_AP:     {s['mean_ap']:.4f}\n")
            f.write(f"  var_AP:      {s['var_ap']:.6f}\n")
            f.write(f"  std_AP:      {np.sqrt(s['var_ap']):.4f}\n")
            f.write(f"  ensemble_AP: {s['ensemble_ap']:.4f}\n")
        f.write(f"\n=== Best dimension ===\n")
        f.write(f"dim={best_dim} (mean_AP={all_stats[best_dim]['mean_ap']:.4f})\n")
        f.write(f"Checkpoints copied to: {best_dir}\n")

    print(f"\nAll results saved to {run_dir}")


if __name__ == "__main__":
    main()
