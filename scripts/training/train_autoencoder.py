import argparse
import sys
from datetime import datetime
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import average_precision_score, matthews_corrcoef

from models.autoencoder_model import EmbeddingAutoencoder
from training.training_utils import train_loop
from utils.focal_loss import FocalLoss
from utils.sequence_loader import load_pos_neg


def fold_loader(input_path, subdir, pattern, batch_size, shuffle=False):
    X, y = load_pos_neg(input_path, subdir, pattern)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True)


def train_one_epoch(model, loader, optimizer,
                    *, rec_criterion, cls_criterion, lambda_rec, lambda_cls, device):
    model.train()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        reconstructed, logit, _ = model(x)
        loss = lambda_rec * rec_criterion(reconstructed, x) + lambda_cls * cls_criterion(logit, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader,
             *, rec_criterion, cls_criterion, lambda_rec, lambda_cls, device):
    model.eval()
    total_loss = total_rec = total_cls = 0.0
    n = 0
    all_probs, all_targets = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        reconstructed, logit, _ = model(x)
        rec = rec_criterion(reconstructed, x)
        cls = cls_criterion(logit, y)
        total_loss += (lambda_rec * rec + lambda_cls * cls).item()
        total_rec += rec.item()
        total_cls += cls.item()
        n += 1
        all_probs.append(torch.sigmoid(logit).cpu())
        all_targets.append(y.cpu())

    probs = torch.cat(all_probs).numpy()
    targets = torch.cat(all_targets).numpy()
    preds = (probs >= 0.5).astype(int)
    return {
        "loss": total_loss / n,
        "rec": total_rec / n,
        "cls": total_cls / n,
        "mcc": matthews_corrcoef(targets, preds),
        "auprc": average_precision_score(targets, probs),
    }


def create_parser():
    parser = argparse.ArgumentParser(
        description="Train per-fold supervised autoencoders (reconstruction + classification)."
    )
    parser.add_argument("--input-path", required=True,
                        help="Directory with positive/{train,val} and negative/{train,val}. "
                             "Per-fold files matched by '*trn{k}.npy' / '*val{k}.npy'.")
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lambda-rec", type=float, default=1.0)
    parser.add_argument("--lambda-cls", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    return parser


def train_single_fold(fold_idx, args, device, run_dir):
    print(f"\n{'='*60}\nFold {fold_idx}/{args.folds} AE\n{'='*60}")

    train_loader = fold_loader(args.input_path, "train", f"*trn{fold_idx}.npy",
                               args.batch_size, shuffle=True)
    val_loader = fold_loader(args.input_path, "val", f"*val{fold_idx}.npy", args.batch_size)

    input_dim = train_loader.dataset.tensors[0].shape[1]
    model = EmbeddingAutoencoder(input_dim=input_dim, latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    fold_dir = run_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = fold_dir / "best_autoencoder.pt"

    loss_kwargs = dict(
        rec_criterion=nn.MSELoss(),
        cls_criterion=FocalLoss(alpha=args.alpha, gamma=args.gamma),
        lambda_rec=args.lambda_rec, lambda_cls=args.lambda_cls, device=device,
    )
    return train_loop(
        model, train_loader, val_loader,
        partial(train_one_epoch, **loss_kwargs),
        partial(evaluate, **loss_kwargs),
        optimizer, args.max_epochs, args.patience, ckpt_path,
    )


def main():
    args = create_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.set_float32_matmul_precision("high")

    run_dir = Path(args.checkpoint_dir) / f"autoencoder_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    histories = []
    for k in range(1, args.folds + 1):
        history = train_single_fold(k, args, device, run_dir)
        histories.append(history)
        print(f"Fold {k} AE best_val_loss: {history['best_val_loss']:.6f}")

    with open(run_dir / "summary.txt", "w") as f:
        f.write("=== Hyperparameters ===\n")
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
        f.write("\n=== Per-fold autoencoder training ===\n")
        for i, h in enumerate(histories, start=1):
            f.write(f"fold {i}: epochs_run={len(h['train_losses'])}, "
                    f"best_val_loss={h['best_val_loss']:.6f}\n")
    print(f"AE run saved to {run_dir}")


if __name__ == "__main__":
    main()
