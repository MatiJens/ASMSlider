import numpy as np
import logging
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
from data_loader import SequenceLoader
from torch_focalloss import BinaryFocalLoss

from pathlib import Path

logger = logging.getLogger(__name__)


class _MLPModel(nn.Module):
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


class MLPTrainer:
    def __init__(
        self,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        epochs: int = 100,
        patience: int = 15,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.alpha = alpha
        self.gamma = gamma

    def _load_data(self, positive_path, negative_path):
        _, pos_emb = SequenceLoader.load_embeddings(positive_path)
        _, neg_emb = SequenceLoader.load_embeddings(negative_path)
        X = np.concatenate([pos_emb, neg_emb])
        y = np.array([1] * len(pos_emb) + [0] * len(neg_emb))

        return X, y

    def _make_loader(self, X, y, shuffle=True):
        return DataLoader(
            TensorDataset(torch.from_numpy(X), torch.from_numpy(y).float()),
            batch_size=self.batch_size,
            shuffle=shuffle,
        )

    def run_training(self, input_path, output_path):
        input_path = Path(input_path)
        pos_trn = Path(input_path, "positive_train.npz")
        neg_trn = Path(input_path, "negative_train.npz")
        X_train, y_train = self._load_data(pos_trn, neg_trn)
        train_loader = self._make_loader(X_train, y_train)

        pos_val = Path(input_path, "positive_val.npz")
        neg_val = Path(input_path, "negative_val.npz")
        X_val, y_val = self._load_data(pos_val, neg_val)
        val_loader = self._make_loader(X_val, y_val)

        pos_test = Path(input_path, "positive_test.npz")
        neg_test = Path(input_path, "negative_test.npz")
        X_test, y_test = self._load_data(pos_test, neg_test)
        test_loader = self._make_loader(X_test, y_test)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = _MLPModel()
        criterion = BinaryFocalLoss(alpha=self.alpha, gamma=self.gamma)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
