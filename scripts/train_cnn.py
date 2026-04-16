import argparse
import numpy as np
import logging
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from sklearn.metrics import (
    matthews_corrcoef,
    average_precision_score,
    recall_score,
    f1_score,
)

from cnn_model import CNNModel
from embeddings_classifier import _pad_sequences
from focal_loss import FocalLoss

logger = logging.getLogger(__name__)


def load_npy_dir(dir_path):
    """Load and concatenate all .npy files from a directory (per-residue, object arrays)."""
    arrays = [
        np.load(f, allow_pickle=True) for f in sorted(Path(dir_path).glob("*.npy"))
    ]
    if not arrays:
        raise ValueError(f"No .npy files found in {dir_path}")
    return np.concatenate(arrays)


def load_split(base_path, split):
    """Load positive/<split>/ and negative/<split>/. Returns (X, y)."""
    p = Path(base_path)
    pos = load_npy_dir(p / "positive" / split)
    neg = load_npy_dir(p / "negative" / split)
    logger.info(f"[{split}] positive: {len(pos)}, negative: {len(neg)}")
    X = np.concatenate([pos, neg])
    y = np.array([1] * len(pos) + [0] * len(neg), dtype=np.float32)
    return X, y


class _PerResidueDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx].astype(np.float32), self.labels[idx]


def _collate_per_residue(batch):
    seqs, labels = zip(*batch)
    padded = torch.from_numpy(_pad_sequences(list(seqs)))
    return padded, torch.stack(list(labels))


def make_loader(input_path, split, batch_size, shuffle=False):
    X, y = load_split(input_path, split)
    ds = _PerResidueDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=True, collate_fn=_collate_per_residue)


def train_one_epoch(model, loader, criterion, optimizer, device):
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
def evaluate(model, loader, criterion, device):
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
    parser = argparse.ArgumentParser(description="Train CNN classifier on per-residue embeddings.")
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Directory with positive/{train,val}/ and negative/{train,val}/ subdirs containing per-residue .npy files.",
    )
    parser.add_argument(
        "--lr", type=float, default=5e-4, help="Learning rate (default: 5e-4)."
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)."
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size (default: 32)."
    )
    parser.add_argument(
        "--max-epochs", type=int, default=100, help="Max epochs (default: 100)."
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (default: 20).",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.85, help="Focal loss alpha (default: 0.85)."
    )
    parser.add_argument(
        "--gamma", type=float, default=3.0, help="Focal loss gamma (default: 3.0)."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for saving checkpoints.",
    )
    return parser


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    args = create_parser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    torch.set_float32_matmul_precision("high")

    train_loader = make_loader(args.input_path, "train", args.batch_size, shuffle=True)
    val_loader = make_loader(args.input_path, "val", args.batch_size)

    model = CNNModel().to(device)
    criterion = FocalLoss(alpha=args.alpha, gamma=args.gamma)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        m = evaluate(model, val_loader, criterion, device)
        logger.info(
            f"Epoch {epoch:3d} | train_loss: {train_loss:.6f} | "
            f"val_loss: {m['loss']:.6f} | mcc: {m['mcc']:.6f} | "
            f"F1: {m['f1']:.6f} | auprc: {m['auprc']:.6f} |  recall: {m['recall']:.6f}"
        )

        if m["loss"] < best_val_loss:
            best_val_loss = m["loss"]
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_dir / "best_cnn_model.pt")
            logger.info(f"New best model saved (val_loss: {m['loss']:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(
                    f"Early stopping after {epoch} epochs (patience={args.patience})"
                )
                break

    model.load_state_dict(
        torch.load(ckpt_dir / "best_cnn_model.pt", weights_only=True, map_location=device)
    )
    m = evaluate(model, val_loader, criterion, device)
    logger.info(
        "Best model:"
        f"test_loss: {m['loss']:.4f} | mcc: {m['mcc']:.4f} | "
        f"F1: {m['f1']:.4f} | auprc: {m['auprc']:.4f} |  recall: {m['recall']:.4f}"
    )

    torch.save(model.state_dict(), ckpt_dir / f"best_cnn_model_{m['mcc']:.4f}.pt")


if __name__ == "__main__":
    main()
