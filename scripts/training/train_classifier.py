import argparse
import sys
import numpy as np
import torch
from functools import partial
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import (
    matthews_corrcoef,
    average_precision_score,
    recall_score,
    f1_score,
)

from utils.focal_loss import FocalLoss
from models.mlp_model import MLPModel
from utils.sequence_loader import load_embeddings_dir
from training.training_utils import train_loop


def load_split(base_path, split):
    """Load positive/<split>/ and negative/<split>/, return TensorDataset."""
    p = Path(base_path)
    pos = load_embeddings_dir(p / "positive" / split)
    neg = load_embeddings_dir(p / "negative" / split)
    X = np.concatenate([pos, neg])
    y = np.array([1] * len(pos) + [0] * len(neg), dtype=np.float32)
    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))


def make_loader(input_path, split, batch_size, shuffle=False):
    ds = load_split(input_path, split)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True)


def train_one_epoch(model, loader, optimizer, *, criterion, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, loader, *, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_probs = []
    all_targets = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item()
        n_batches += 1
        all_probs.append(torch.sigmoid(logits).cpu())
        all_targets.append(y.cpu())

    probs = torch.cat(all_probs).numpy()
    targets = torch.cat(all_targets).numpy()
    preds = (probs >= 0.5).astype(int)

    return {
        "loss": total_loss / n_batches,
        "mcc": matthews_corrcoef(targets, preds),
        "auprc": average_precision_score(targets, probs),
        "recall": recall_score(targets, preds, zero_division=0),
        "f1": f1_score(targets, preds, zero_division=0),
    }


def create_parser():
    parser = argparse.ArgumentParser(description="Train MLP classifier on embeddings.")
    parser.add_argument(
        "--input-path", type=str, required=True,
        help="Directory with {positive,negative}/{train,val,test} embedding dirs.",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3).")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (default: 1e-4).")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64).")
    parser.add_argument("--max-epochs", type=int, default=100, help="Max epochs (default: 100).")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (default: 15).")
    parser.add_argument("--alpha", type=float, default=0.25, help="Focal loss alpha (default: 0.25).")
    parser.add_argument("--gamma", type=float, default=2.0, help="Focal loss gamma (default: 2.0).")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory for saving checkpoints.")
    return parser


def main():
    args = create_parser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.set_float32_matmul_precision("high")

    train_loader = make_loader(args.input_path, "train", args.batch_size, shuffle=True)
    val_loader = make_loader(args.input_path, "val", args.batch_size)

    input_dim = train_loader.dataset.tensors[0].shape[1]
    model = MLPModel(input_dim=input_dim).to(device)
    criterion = FocalLoss(alpha=args.alpha, gamma=args.gamma)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best_model.pt"

    train_fn = partial(train_one_epoch, criterion=criterion, device=device)
    eval_fn = partial(evaluate, criterion=criterion, device=device)

    train_loop(
        model, train_loader, val_loader, train_fn, eval_fn, optimizer,
        args.max_epochs, args.patience, ckpt_path,
    )

    # Final evaluation on test set
    model.load_state_dict(
        torch.load(ckpt_path, weights_only=True, map_location=device)
    )
    test_loader = make_loader(args.input_path, "test", args.batch_size)
    m = eval_fn(model, test_loader)
    parts = " | ".join(f"{k}: {v:.4f}" for k, v in m.items())
    print(f"Test set: {parts}")

    torch.save(model.state_dict(), ckpt_dir / f"best_model_{m['mcc']:.4f}.pt")


if __name__ == "__main__":
    main()
