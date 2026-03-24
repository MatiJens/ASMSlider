import argparse
import copy
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
    format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


import copy
import json
import logging
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import matthews_corrcoef
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


class MLPTrainer(nn.Module):
    def __init__(
        self,
        input_dim: int = 1152,
        hidden_dims: list = [256, 64],
        dropout: float = 0.3,
        lr: float = 1e-4,
        weight_decay: float = 1e-3,
        batch_size: int = 128,
        epochs: int = 50,
        patience: int = 8,
        seed: int = 67,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.seed = seed

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.encoder = self._build_encoder(input_dim, hidden_dims, dropout)
        self.classifier = nn.Linear(hidden_dims[-1], 2)

    def _build_encoder(self, input_dim, hidden_dims, dropout):
        layers = []
        prev_dim = input_dim
        for i, dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout))
            prev_dim = dim
        return nn.Sequential(*layers)

    def _iter_json_config(self, input_path):
        try:
            with open(input_path, "r") as f:
                config = json.load(f)
        except FileNotFoundError:
            logger.error(f"JSON config file not found: {input_path}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {input_path}: {e}")
            sys.exit(1)

        if not isinstance(config, dict) or not all(
            isinstance(v, dict) for v in config.values()
        ):
            logger.error(
                "JSON config must be a dict of dicts (split_name -> {class: filepath})."
            )
            sys.exit(1)

        for split_name, classes in config.items():
            for class_name, filepath in classes.items():
                if not os.path.isfile(filepath):
                    logger.error(
                        f"[{split_name}/{class_name}] File not found: {filepath}"
                    )
                    sys.exit(1)
                yield split_name, class_name, filepath

    def _load_h5(self, filepath):
        with h5py.File(filepath, "r") as f:
            return np.stack([f[k][:] for k in f.keys()], axis=0).astype(np.float32)

    def _load_fold(self, config, fold_idx):
        X_pos = self._load_h5(config[f"train_{fold_idx}"]["positive"])
        X_neg = self._load_h5(config[f"train_{fold_idx}"]["negative"])
        X_train = np.concatenate([X_pos, X_neg], axis=0)
        y_train = np.array([1] * len(X_pos) + [0] * len(X_neg))

        X_pos = self._load_h5(config[f"val_{fold_idx}"]["positive"])
        X_neg = self._load_h5(config[f"val_{fold_idx}"]["negative"])
        X_val = np.concatenate([X_pos, X_neg], axis=0)
        y_val = np.array([1] * len(X_pos) + [0] * len(X_neg))

        return X_train, y_train, X_val, y_val

    def _create_splits(self, input_path):
        try:
            with open(input_path) as f:
                config = json.load(f)
        except FileNotFoundError:
            logger.error(f"JSON config file not found: {input_path}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {input_path}: {e}")
            sys.exit(1)

        fold_nums = sorted(
            {int(k.split("_")[1]) for k in config if k.startswith("train_")}
        )

        folds = []
        for fold_idx in fold_nums:
            folds.append((fold_idx, *self._load_fold(config, fold_idx)))

        X_pos = self._load_h5(config["test"]["positive"])
        X_neg = self._load_h5(config["test"]["negative"])
        X_test = np.concatenate([X_pos, X_neg], axis=0)
        y_test = np.array([1] * len(X_pos) + [0] * len(X_neg))

        return folds, X_test, y_test

    def _make_loader(self, X, y, shuffle=False):
        return DataLoader(
            TensorDataset(torch.from_numpy(X), torch.from_numpy(y).long()),
            batch_size=self.batch_size,
            shuffle=shuffle,
        )

    def _train_epoch(self, train_loader, optimizer, criterion, device):
        super().train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = self.classifier(self.encoder(X_batch))
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y_batch)
        return train_loss / len(train_loader.dataset)

    @torch.no_grad()
    def _evaluate(self, loader, device):
        super().eval()
        all_logits, all_labels = [], []
        for X_batch, y_batch in loader:
            logits = self.classifier(self.encoder(X_batch.to(device)))
            all_logits.append(logits.cpu())
            all_labels.append(y_batch)

        preds = torch.cat(all_logits).argmax(dim=1).numpy()
        labels = torch.cat(all_labels).numpy()
        return matthews_corrcoef(labels, preds)

    @torch.no_grad()
    def _ensemble_evaluate(self, fold_states, test_loader, device):
        all_logits, all_labels = [], []
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            fold_logits = []
            for state in fold_states:
                self.load_state_dict(state)
                super().eval()
                fold_logits.append(self.classifier(self.encoder(X_batch)).cpu())
            all_logits.append(torch.stack(fold_logits).mean(dim=0))
            all_labels.append(y_batch)

        preds = torch.cat(all_logits).argmax(dim=1).numpy()
        labels = torch.cat(all_labels).numpy()
        return matthews_corrcoef(labels, preds)

    def _save_encoder(self, output_path, fold_idx):
        path = os.path.join(output_path, f"mlp_encoder_fold{fold_idx}.pt")
        torch.save(self.encoder.state_dict(), path)
        logger.info(f"Encoder saved under {path}")

    def _save_classifier(self, output_path, fold_idx):
        path = os.path.join(output_path, f"mlp_classifier_fold{fold_idx}.pt")
        torch.save(self.state_dict(), path)
        logger.info(f"Classifier saved under {path}")

    def _save_config(self, output_path):
        config = {
            "input_dim": self.encoder[0].in_features,
            "hidden_dims": [
                l.out_features for l in self.encoder if isinstance(l, nn.Linear)
            ],
            "dropout": next(
                (l.p for l in self.encoder if isinstance(l, nn.Dropout)), 0.0
            ),
        }
        path = os.path.join(output_path, "mlp_config.json")
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Config saved under {path}")

    def _train_fold(self, fold_idx, X_train, y_train, X_val, y_val, device):
        self.encoder = self._build_encoder(
            self.input_dim, self.hidden_dims, self.dropout
        ).to(device)
        self.classifier = nn.Linear(self.hidden_dims[-1], 2).to(device)

        logger.info(
            f"Fold {fold_idx} | Train: {len(y_train)} (pos: {y_train.sum()}) | "
            f"Val: {len(y_val)} (pos: {y_val.sum()})"
        )

        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        val_loader = self._make_loader(X_val, y_val)

        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        weights = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        best_val_mcc = -1.0
        best_state = None
        epochs_no_improve = 0

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(train_loader, optimizer, criterion, device)
            val_mcc = self._evaluate(val_loader, device)

            logger.info(
                f"Fold {fold_idx} | Epoch {epoch:3d}/{self.epochs} | "
                f"train_loss: {train_loss:.4f} | val_mcc: {val_mcc:.4f}"
            )

            if val_mcc > best_val_mcc:
                best_val_mcc = val_mcc
                best_state = copy.deepcopy(self.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    logger.info(f"Fold {fold_idx} | Early stopping at epoch {epoch}")
                    break

        self.load_state_dict(best_state)
        logger.info(f"Fold {fold_idx} | Best val_mcc: {best_val_mcc:.4f}")
        return best_state, best_val_mcc

    def run_training(self, input_path, output_path):
        """Train MLP with cross-validation, evaluate ensemble on test set."""
        os.makedirs(output_path, exist_ok=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        folds, X_test, y_test = self._create_splits(input_path)
        logger.info("Files loaded")

        test_loader = self._make_loader(X_test, y_test)

        fold_states = []
        val_mccs = []

        for fold_idx, X_train, y_train, X_val, y_val in folds:
            best_state, best_val_mcc = self._train_fold(
                fold_idx, X_train, y_train, X_val, y_val, device
            )
            fold_states.append(best_state)
            val_mccs.append(best_val_mcc)

            test_mcc = self._evaluate(test_loader, device)
            logger.info(f"Fold {fold_idx} | Test mcc: {test_mcc:.4f}")

            self._save_encoder(output_path, fold_idx)
            self._save_classifier(output_path, fold_idx)

        logger.info(
            f"All folds done | Val MCCs: {[f'{m:.4f}' for m in val_mccs]} | "
            f"Mean: {np.mean(val_mccs):.4f} +/- {np.std(val_mccs):.4f}"
        )

        ensemble_mcc = self._ensemble_evaluate(fold_states, test_loader, device)
        logger.info(f"Ensemble | Test mcc: {ensemble_mcc:.4f}")

        self._save_config(output_path)


def create_parser():
    parser = argparse.ArgumentParser(
        description="Train MLP encoder and classifier with cross-validation."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to JSON config with k-fold split paths.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        required=True,
        help="Directory where trained models and config will be saved.",
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=1152,
        help="Input embedding dimension.",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[256, 64],
        help="Hidden layer dimensions.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Max epochs per fold.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=67,
        help="Random seed.",
    )
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    trainer = MLPTrainer(
        input_dim=args.input_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
    )
    trainer.run_training(args.input_path, args.output_path)


if __name__ == "__main__":
    main()
