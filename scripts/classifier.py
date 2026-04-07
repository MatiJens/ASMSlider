import argparse
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.metrics import (
    matthews_corrcoef,
    average_precision_score,
    recall_score,
    f1_score,
)

logger = logging.getLogger(__name__)


class BinaryFocalLoss(nn.Module):
    def __init__(
        self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class MLPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.BatchNorm1d(1152),
            nn.Linear(1152, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


def load_data(pos_file, neg_file):
    pos_emb = np.load(pos_file).astype(np.float32)
    neg_emb = np.load(neg_file).astype(np.float32)
    X = np.concatenate([pos_emb, neg_emb])
    y = np.array([1] * len(pos_emb) + [0] * len(neg_emb))
    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y).float())


def make_loader(input_path, split, batch_size, shuffle=False):
    p = Path(input_path)
    ds = load_data(p / f"positive_{split}.npy", p / f"negative_{split}.npy")
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True)


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
    parser = argparse.ArgumentParser(description="Train MLP classifier on embeddings.")
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Directory with {positive,negative}_{train,val,test}.npz files.",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)."
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size (default: 64)."
    )
    parser.add_argument(
        "--max-epochs", type=int, default=100, help="Max epochs (default: 100)."
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience (default: 15).",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.25, help="Focal loss alpha (default: 0.25)."
    )
    parser.add_argument(
        "--gamma", type=float, default=2.0, help="Focal loss gamma (default: 2.0)."
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
    test_loader = make_loader(args.input_path, "test", args.batch_size)

    model = MLPModel().to(device)
    criterion = BinaryFocalLoss(alpha=args.alpha, gamma=args.gamma)
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
            f"Epoch {epoch:3d} | train_loss: {train_loss:.4f} | "
            f"val_loss: {m['loss']:.4f} | mcc: {m['mcc']:.4f} | "
            f"F1: {m['f1']:.4f} | auprc: {m['auprc']:.4f} |  recall: {m['recall']:.4f}"
        )

        if m["loss"] < best_val_loss:
            best_val_loss = m["loss"]
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
            logger.info(f"New best model saved (val_loss: {m['loss']:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(
                    f"Early stopping after {epoch} epochs (patience={args.patience})"
                )
                break

    model.load_state_dict(
        torch.load(ckpt_dir / "best_model.pt", weights_only=True, map_location=device)
    )
    m = evaluate(model, val_loader, criterion, device)
    logger.info(
        f"Epoch {epoch:3d} | train_loss: {train_loss:.4f} | "
        f"val_loss: {m['loss']:.4f} | mcc: {m['mcc']:.4f} | "
        f"F1: {m['f1']:.4f} | auprc: {m['auprc']:.4f} |  recall: {m['recall']:.4f}"
    )

    torch.save(model.state_dict(), ckpt_dir / f"best_model_{m['mcc']:.4f}.pt")


if __name__ == "__main__":
    main()
