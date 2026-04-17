import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
from functools import partial
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.sequence_loader import load_embeddings_dir
from models.autoencoder_model import EmbeddingAutoencoder
from training.training_utils import train_loop


def load_split(base_path, split):
    p = Path(base_path)
    pos = load_embeddings_dir(p / "positive" / split)
    neg = load_embeddings_dir(p / "negative" / split)
    X = np.concatenate([pos, neg])
    return TensorDataset(torch.from_numpy(X))


def make_loader(input_path, split, batch_size, shuffle=False):
    ds = load_split(input_path, split)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True)


def train_one_epoch(model, loader, optimizer, *, criterion, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for (x,) in loader:
        x = x.to(device)
        optimizer.zero_grad()
        reconstructed, _ = model(x)
        loss = criterion(reconstructed, x)
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
    for (x,) in loader:
        x = x.to(device)
        reconstructed, _ = model(x)
        total_loss += criterion(reconstructed, x).item()
        n_batches += 1
    return {"loss": total_loss / n_batches}


def create_parser():
    parser = argparse.ArgumentParser(
        description="Train autoencoder for embedding dimensionality reduction."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Directory with positive/negative embedding dirs.",
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=1152,
        help="Input embedding dimension (default: 1152).",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=128,
        help="Latent (reduced) dimension (default: 128).",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)."
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)."
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Batch size (default: 256)."
    )
    parser.add_argument(
        "--max-epochs", type=int, default=200, help="Max epochs (default: 200)."
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (default: 20).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for saving checkpoints.",
    )
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
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_fn = partial(train_one_epoch, criterion=criterion, device=device)
    eval_fn = partial(evaluate, criterion=criterion, device=device)

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
