import argparse
import sys
from datetime import datetime
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    recall_score,
)

from models.autoencoder_model import EmbeddingAutoencoder
from models.mlp_model import MLPModel
from training.training_utils import train_loop
from utils.focal_loss import FocalLoss
from utils.sequence_loader import load_pos_neg


def load_dataset(input_path, subdir, pattern="*.npy"):
    X, y = load_pos_neg(input_path, subdir, pattern)
    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))


def make_loader(ds, batch_size, shuffle=False):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True)


def load_fold_encoder(ae_dir, fold_idx, input_dim, latent_dim, device):
    ae_path = Path(ae_dir) / f"fold_{fold_idx}" / "best_autoencoder.pt"
    if not ae_path.exists():
        raise FileNotFoundError(f"Autoencoder checkpoint not found: {ae_path}")
    ae = EmbeddingAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
    ae.load_state_dict(torch.load(ae_path, weights_only=True, map_location=device))
    return ae.encoder.to(device).eval()


@torch.no_grad()
def encode_dataset(ds, encoder, device, batch_size=1024):
    X, y = ds.tensors
    out = [encoder(X[i : i + batch_size].to(device)).cpu()
           for i in range(0, len(X), batch_size)]
    return TensorDataset(torch.cat(out), y)


def train_one_epoch(model, loader, optimizer, *, criterion, device):
    model.train()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / n


@torch.no_grad()
def predict_probs(model, loader, device, criterion=None):
    model.eval()
    total_loss, n = 0.0, 0
    probs, targets = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if criterion is not None:
            total_loss += criterion(logits, y).item()
            n += 1
        probs.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(y.cpu().numpy())
    mean_loss = total_loss / n if criterion is not None else None
    return np.concatenate(probs), np.concatenate(targets), mean_loss


def metrics_from_probs(probs, targets, threshold=0.7):
    preds = (probs >= threshold).astype(int)
    return {
        "mcc": matthews_corrcoef(targets, preds),
        "auprc": average_precision_score(targets, probs),
        "recall": recall_score(targets, preds, zero_division=0),
        "f1": f1_score(targets, preds, zero_division=0),
    }


def evaluate(model, loader, *, criterion, device):
    probs, targets, loss = predict_probs(model, loader, device, criterion)
    return {"loss": loss, **metrics_from_probs(probs, targets)}


def plot_loss_curves(history, out_path, title):
    epochs = range(1, len(history["train_losses"]) + 1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, history["train_losses"], label="train loss")
    ax.plot(epochs, history["val_losses"], label="val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def create_parser():
    parser = argparse.ArgumentParser(description="Train MLP classifier on embeddings with k-fold CV.")
    parser.add_argument("--input-path", required=True,
                        help="Directory with positive/{train,val,test} and negative/{train,val,test}. "
                             "Per-fold files matched by '*trn{k}.npy' / '*val{k}.npy'.")
    parser.add_argument("--ae-checkpoint-dir", default=None,
                        help="Per-fold AE checkpoints (fold_{k}/best_autoencoder.pt). "
                             "If set, each fold's embeddings are encoded by its paired AE.")
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    return parser


def train_single_fold(fold_idx, args, device, run_dir):
    print(f"\n{'='*60}\nFold {fold_idx}/{args.folds}\n{'='*60}")

    train_ds = load_dataset(args.input_path, "train", f"*trn{fold_idx}.npy")
    val_ds = load_dataset(args.input_path, "val", f"*val{fold_idx}.npy")

    encoder = None
    if args.ae_checkpoint_dir:
        raw_dim = train_ds.tensors[0].shape[1]
        encoder = load_fold_encoder(args.ae_checkpoint_dir, fold_idx, raw_dim, args.latent_dim, device)
        train_ds = encode_dataset(train_ds, encoder, device)
        val_ds = encode_dataset(val_ds, encoder, device)
        print(f"Encoded: {raw_dim} -> {train_ds.tensors[0].shape[1]}")

    train_loader = make_loader(train_ds, args.batch_size, shuffle=True)
    val_loader = make_loader(val_ds, args.batch_size)

    input_dim = train_ds.tensors[0].shape[1]
    model = MLPModel(input_dim=input_dim).to(device)
    criterion = FocalLoss(alpha=args.alpha, gamma=args.gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    fold_dir = run_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = fold_dir / "best_model.pt"

    eval_fn = partial(evaluate, criterion=criterion, device=device)
    history = train_loop(
        model, train_loader, val_loader,
        partial(train_one_epoch, criterion=criterion, device=device),
        eval_fn, optimizer, args.max_epochs, args.patience, ckpt_path,
    )

    model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location=device))
    val_metrics = eval_fn(model, val_loader)
    plot_loss_curves(history, fold_dir / "loss_curves.png", title=f"Fold {fold_idx}")

    return history, val_metrics, ckpt_path, encoder


def ensemble_test_eval(args, device, ckpt_paths, encoders):
    """Average sigmoid probs across folds. Each fold encodes raw test data with its paired AE."""
    test_ds_raw = load_dataset(args.input_path, "test")
    probs_sum, targets_ref = None, None

    for k, ckpt in enumerate(ckpt_paths, start=1):
        enc = encoders[k - 1] if encoders else None
        test_ds = encode_dataset(test_ds_raw, enc, device) if enc is not None else test_ds_raw
        test_loader = make_loader(test_ds, args.batch_size)

        model = MLPModel(input_dim=test_ds.tensors[0].shape[1]).to(device)
        model.load_state_dict(torch.load(ckpt, weights_only=True, map_location=device))
        probs, targets, _ = predict_probs(model, test_loader, device)
        probs_sum = probs if probs_sum is None else probs_sum + probs
        targets_ref = targets

    return metrics_from_probs(probs_sum / len(ckpt_paths), targets_ref)


def save_cv_run(run_dir, args, per_fold, ensemble_metrics):
    metric_keys = [k for k in per_fold[0]["val_metrics"].keys() if k != "loss"]
    agg = {k: np.array([f["val_metrics"][k] for f in per_fold]) for k in metric_keys}

    with open(run_dir / "summary.txt", "w") as f:
        f.write("=== Hyperparameters ===\n")
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

        f.write("\n=== Per-fold validation metrics ===\n")
        f.write("fold  " + "  ".join(f"{k:>10}" for k in metric_keys) + "\n")
        for fold in per_fold:
            row = f"{fold['fold']:>4}  " + "  ".join(
                f"{fold['val_metrics'][k]:>10.4f}" for k in metric_keys
            )
            f.write(row + "\n")

        f.write("\n=== Aggregate (mean +/- std across folds) ===\n")
        for k in metric_keys:
            f.write(f"{k}: {agg[k].mean():.4f} +/- {agg[k].std():.4f}\n")

        f.write("\n=== Ensemble (mean of fold probs) on test set ===\n")
        for k, v in ensemble_metrics.items():
            f.write(f"{k}: {v:.6f}\n")

        f.write("\n=== Per-fold epochs trained ===\n")
        for fold in per_fold:
            f.write(f"fold {fold['fold']}: epochs_run={len(fold['history']['train_losses'])}, "
                    f"best_val_loss={fold['history']['best_val_loss']:.6f}\n")
    print(f"CV run saved to {run_dir}")


def main():
    args = create_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.set_float32_matmul_precision("high")

    run_dir = Path(args.checkpoint_dir) / f"train_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    per_fold, ckpt_paths, encoders = [], [], []
    for k in range(1, args.folds + 1):
        history, val_metrics, ckpt_path, encoder = train_single_fold(k, args, device, run_dir)
        per_fold.append({"fold": k, "val_metrics": val_metrics, "history": history})
        ckpt_paths.append(ckpt_path)
        encoders.append(encoder)
        parts = " | ".join(f"{kk}: {vv:.4f}" for kk, vv in val_metrics.items())
        print(f"Fold {k} val: {parts}")

    print("\nEvaluating ensemble on test set...")
    ensemble_metrics = ensemble_test_eval(
        args, device, ckpt_paths, encoders if args.ae_checkpoint_dir else None
    )
    parts = " | ".join(f"{k}: {v:.4f}" for k, v in ensemble_metrics.items())
    print(f"Ensemble (mean of {args.folds} models) test: {parts}")

    save_cv_run(run_dir, args, per_fold, ensemble_metrics)


if __name__ == "__main__":
    main()
