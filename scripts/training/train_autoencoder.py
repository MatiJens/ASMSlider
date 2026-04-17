import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
from functools import partial
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import matthews_corrcoef, average_precision_score

from utils.focal_loss import FocalLoss
from utils.sequence_loader import load_embeddings_dir
from models.autoencoder_model import EmbeddingAutoencoder
from training.training_utils import train_loop


def load_split(base_path, split):
    p = Path(base_path)
    pos = load_embeddings_dir(p / "positive" / split)
    neg = load_embeddings_dir(p / "negative" / split)
    X = np.concatenate([pos, neg])
    y = np.array([1] * len(pos) + [0] * len(neg), dtype=np.float32)
    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))


def make_loader(input_path, split, batch_size, shuffle=False):
    ds = load_split(input_path, split)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True)


def train_one_epoch(
    model,
    loader,
    optimizer,
    *,
    rec_criterion,
    cls_criterion,
    lambda_rec,
    lambda_cls,
    device,
):
    model.train()
    total_loss = 0.0
    total_rec = 0.0
    total_cls = 0.0
    n_batches = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        reconstructed, logit, _ = model(x)
        rec_loss = rec_criterion(reconstructed, x)
        cls_loss = cls_criterion(logit, y)
        loss = lambda_rec * rec_loss + lambda_cls * cls_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_rec += rec_loss.item()
        total_cls += cls_loss.item()
        n_batches += 1
    return total_loss / n_batches


@torch.no_grad()
def evaluate(
    model, loader, *, rec_criterion, cls_criterion, lambda_rec, lambda_cls, device
):
    model.eval()
    total_loss = 0.0
    total_rec = 0.0
    total_cls = 0.0
    n_batches = 0
    all_probs = []
    all_targets = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        reconstructed, logit, _ = model(x)
        rec_loss = rec_criterion(reconstructed, x)
        cls_loss = cls_criterion(logit, y)
        loss = lambda_rec * rec_loss + lambda_cls * cls_loss
        total_loss += loss.item()
        total_rec += rec_loss.item()
        total_cls += cls_loss.item()
        n_batches += 1
        all_probs.append(torch.sigmoid(logit).cpu())
        all_targets.append(y.cpu())

    probs = torch.cat(all_probs).numpy()
    targets = torch.cat(all_targets).numpy()
    preds = (probs >= 0.5).astype(int)

    return {
        "loss": total_loss / n_batches,
        "rec": total_rec / n_batches,
        "cls": total_cls / n_batches,
        "mcc": matthews_corrcoef(targets, preds),
        "auprc": average_precision_score(targets, probs),
    }


def create_parser():
    parser = argparse.ArgumentParser(
        description="Train supervised autoencoder (reconstruction + classification)."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Directory with positive/negative embedding dirs.",
    )
    parser.add_argument("--input-dim", type=int, default=1152)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument(
        "--lambda-rec",
        type=float,
        default=1.0,
        help="Weight for reconstruction loss (default: 1.0).",
    )
    parser.add_argument(
        "--lambda-cls",
        type=float,
        default=1.0,
        help="Weight for classification loss (default: 1.0).",
    )
    parser.add_argument("--alpha", type=float, default=0.25, help="Focal loss alpha.")
    parser.add_argument("--gamma", type=float, default=2.0, help="Focal loss gamma.")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    return parser


def main():
    args = create_parser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.set_float32_matmul_precision("high")

    train_loader = make_loader(args.input_path, "train", args.batch_size, shuffle=True)
    val_loader = make_loader(args.input_path, "val", args.batch_size)

    model = EmbeddingAutoencoder(
        input_dim=args.input_dim, latent_dim=args.latent_dim
    ).to(device)
    rec_criterion = nn.MSELoss()
    cls_criterion = FocalLoss(alpha=args.alpha, gamma=args.gamma)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    loss_kwargs = dict(
        rec_criterion=rec_criterion,
        cls_criterion=cls_criterion,
        lambda_rec=args.lambda_rec,
        lambda_cls=args.lambda_cls,
        device=device,
    )
    train_fn = partial(train_one_epoch, **loss_kwargs)
    eval_fn = partial(evaluate, **loss_kwargs)

    best_val_loss = train_loop(
        model,
        train_loader,
        val_loader,
        train_fn,
        eval_fn,
        optimizer,
        args.max_epochs,
        args.patience,
        ckpt_dir / "best_autoencoder.pt",
    )
    print(f"Best val_loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
