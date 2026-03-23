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
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


class MLPCreator(nn.Module):
    def __init__(
        self,
        input_path: str,
        output_path: str,
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
        self.input_path = input_path
        self.output_path = output_path
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

        os.makedirs(self.output_path, exist_ok=True)

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

        fold_nums = sorted(
            {int(k.split("_")[1]) for k in splits if k.startswith("train_")}
        )

        folds = []
        for fold_idx in fold_nums:
            X_train, y_train = self._load_files(
                splits[f"train_{fold_idx}"]["positive"],
                splits[f"train_{fold_idx}"]["negative"],
            )
            X_val, y_val = self._load_files(
                splits[f"val_{fold_idx}"]["positive"],
                splits[f"val_{fold_idx}"]["negative"],
            )
            folds.append((fold_idx, X_train, y_train, X_val, y_val))

        X_test, y_test = self._load_files(
            splits["test"]["positive"], splits["test"]["negative"]
        )

        return folds, X_test, y_test

    def _train_model(self, train_loader, optimizer, criterion, device):
        self.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = self.classifier(self.encoder(X_batch))
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y_batch)
        train_loss /= len(train_loader.dataset)
        return train_loss

    @torch.no_grad()
    def _evaluate_model(self, loader, device):
        self.eval()
        all_logits, all_labels = [], []

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = self.classifier(self.encoder(X_batch))
            all_logits.append(logits.cpu())
            all_labels.append(y_batch)

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        preds = all_logits.argmax(dim=1).numpy()
        labels = all_labels.numpy()

        return matthews_corrcoef(labels, preds)

    def _save_encoder(self, fold_idx):
        encoder_path = os.path.join(self.output_path, f"mlp_encoder_fold{fold_idx}.pt")
        torch.save(self.encoder.state_dict(), encoder_path)
        logger.info(f"Encoder saved under {encoder_path}.")

    def _save_classifier(self, fold_idx):
        classifier_path = os.path.join(
            self.output_path, f"mlp_classifier_fold{fold_idx}.pt"
        )
        torch.save(self.state_dict(), classifier_path)
        logger.info(f"Classifier saved under {classifier_path}.")

    def _save_config(self):
        config = {
            "input_dim": self.encoder[0].in_features,
            "hidden_dims": [
                layer.out_features
                for layer in self.encoder
                if isinstance(layer, nn.Linear)
            ],
            "dropout": next(
                (layer.p for layer in self.encoder if isinstance(layer, nn.Dropout)),
                0.0,
            ),
        }
        config_path = os.path.join(self.output_path, "mlp_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Config saved under {config_path}.")

    @torch.no_grad()
    def _ensemble_evaluate(self, fold_states, test_loader, device):
        all_logits, all_labels = [], []

        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            fold_logits = []
            for state in fold_states:
                self.load_state_dict(state)
                self.eval()
                fold_logits.append(self.classifier(self.encoder(X_batch)).cpu())
            mean_logits = torch.stack(fold_logits).mean(dim=0)
            all_logits.append(mean_logits)
            all_labels.append(y_batch)

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        preds = all_logits.argmax(dim=1).numpy()
        labels = all_labels.numpy()
        return matthews_corrcoef(labels, preds)

    def _train_fold(self, fold_idx, X_train, y_train, X_val, y_val, device):
        self.encoder = self._build_encoder(
            self.input_dim, self.hidden_dims, self.dropout
        ).to(device)
        self.classifier = nn.Linear(self.hidden_dims[-1], 2).to(device)

        logger.info(
            f"Fold {fold_idx} | "
            f"Starting learning model | "
            f"Train: {len(y_train)} (pos: {y_train.sum()}) | "
            f"Val: {len(y_val)} (pos: {y_val.sum()})"
        )

        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_train),
                torch.from_numpy(y_train).long(),
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_val),
                torch.from_numpy(y_val).long(),
            ),
            batch_size=self.batch_size,
        )

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
            train_loss = self._train_model(train_loader, optimizer, criterion, device)
            val_mcc = self._evaluate_model(val_loader, device)

            logger.info(
                f"Fold {fold_idx} │ Epoch {epoch:3d}/{self.epochs} │ "
                f"train_loss: {train_loss:.4f} │ val_mcc: {val_mcc:.4f}"
            )

            if val_mcc > best_val_mcc:
                best_val_mcc = val_mcc
                best_state = copy.deepcopy(self.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    logger.info(f"Fold {fold_idx} │ Early stopping at epoch {epoch}.")
                    break

        self.load_state_dict(best_state)
        logger.info(f"Fold {fold_idx} │ Best val_mcc: {best_val_mcc:.4f}")
        return best_state, best_val_mcc

    def run(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        folds, X_test, y_test = self._create_splits()
        logger.info(
            f"Detected {len(folds)} folds | Test: {len(y_test)} (pos: {y_test.sum()})"
        )

        test_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_test),
                torch.from_numpy(y_test).long(),
            ),
            batch_size=self.batch_size,
        )

        fold_states = []
        val_mccs = []

        for fold_idx, X_train, y_train, X_val, y_val in folds:
            best_state, best_val_mcc = self._train_fold(
                fold_idx, X_train, y_train, X_val, y_val, device
            )
            fold_states.append(best_state)
            val_mccs.append(best_val_mcc)

            test_mcc = self._evaluate_model(test_loader, device)
            logger.info(f"Fold {fold_idx} │ Test mcc: {test_mcc:.4f}")

            self._save_encoder(fold_idx)
            self._save_classifier(fold_idx)

        logger.info(
            f"All folds done | "
            f"Val MCCs: {[f'{m:.4f}' for m in val_mccs]} | "
            f"Mean: {np.mean(val_mccs):.4f} +/- {np.std(val_mccs):.4f}"
        )

        ensemble_mcc = self._ensemble_evaluate(fold_states, test_loader, device)
        logger.info(f"Ensemble │ Test mcc: {ensemble_mcc:.4f}")

        self._save_config()


def create_parser():
    parser = argparse.ArgumentParser(
        description="Script that create, teach and save MLP encoder and classifier."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="""Path to json file with k-fold split paths.""",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        required=True,
        help="Path where models should be saved.",
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=1152,
        help="Size of input layer. It should be equal to input embedding dim.",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[256, 64],
        help="List of hidden layer dimensions.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout MLP parameter.",
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
        help="Max number of learning epochs per fold.",
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
    try:
        mlp_creator = MLPCreator(
            input_path=args.input_path,
            output_path=args.output_path,
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
        mlp_creator.run()
    except Exception as e:
        logger.exception(f"Job failed: {e}")
        raise


if __name__ == "__main__":
    main()
