import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import h5py
import logging
import json
from sklearn.metrics import matthews_corrcoef

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


class CreateMLP(nn.Module):
    def __init__(
        self,
        input_path: str,
        output_path: str,
        input_dim: int = 1152,
        hidden_dims: list = [512, 256],
        dropout: float = 0.3,
        lr: float = 1e-4,
        weight_decay: float = 1e-3,
        batch_size: int = 128,
        epochs: int = 50,
        patience: int = 8,
    ):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], 1),
        )

    def _load_files(self, positive_file, negative_file):
        with h5py.File(positive_file, "r") as f:
            pos_emb = []
            for key in f.keys():
                pos_emb.append(f[key][:])

        with h5py.File(negative_file, "r") as f:
            neg_emb = []
            for key in f.keys():
                neg_emb.append(f[key][:])

        X = np.stack(pos_emb + neg_emb, axis=0).astype(np.float32)
        y = np.array([1] * len(pos_emb) + [0] * len(neg_emb))
        return X, y

    def _create_splits(self):
        with open(self.input_path) as f:
            splits = json.load(f)

            self.X_train, self.y_train = self._load_files(
                splits["train"]["positive"], splits["train"]["negative"]
            )
            self.X_val, self.y_val = self._load_files(
                splits["val"]["positive"], splits["val"]["negative"]
            )
            self.X_test, self.y_test = self._load_files(
                splits["test"]["positive"], splits["test"]["negative"]
            )

    @torch.no_grad()
    def _evaluate(self, loader, device):
        self.eval()
        all_logits, all_labels = [], []

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = self.network(X_batch).squeeze(-1)
            all_logits.append(logits.cpu())
            all_labels.append(y_batch)

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        preds = (torch.sigmoid(all_logits) >= 0.5).int().numpy()
        labels = all_labels.int().numpy()

        return matthews_corrcoef(labels, preds)

    def _save_encoder(self):
        encoder = nn.Sequential(*list(self.network.children())[:-2])
        encoder_path = os.path.join(self.output_path, "encoder.pt")
        torch.save(encoder.state_dict(), encoder_path)
        logger.info(f"Encoder saved to {encoder_path}")

    def train_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self._create_splits()

        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(self.X_train), torch.from_numpy(self.y_train).float()
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(self.X_val), torch.from_numpy(self.y_val).float()
            ),
            batch_size=self.batch_size,
        )

        n_pos = self.y_train.sum()
        n_neg = len(self.y_train) - n_pos
        pos_weight = torch.tensor([n_neg / n_pos], device=device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=3,
        )

        best_val_mcc = -1.0
        best_state = None
        epochs_no_improve = 0

        for epoch in range(1, self.epochs + 1):
            self.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                logits = self.network(X_batch).squeeze(-1)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(y_batch)
            train_loss /= len(train_loader.dataset)

            val_mcc = self._evaluate(val_loader, device)
            scheduler.step(val_mcc)

            logger.info(
                f"Epoch {epoch:3d}/{self.epochs} │ "
                f"train_loss: {train_loss:.4f} │ val_mcc: {val_mcc:.4f}"
            )

            if val_mcc > best_val_mcc:
                best_val_mcc = val_mcc
                best_state = self.state_dict().copy()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}.")
                    break

        self.load_state_dict(best_state)
        logger.info(f"Best val_mcc: {best_val_mcc:.4f})")
        self._save_encoder()
