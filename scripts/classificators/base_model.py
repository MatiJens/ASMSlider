import os
import sys
import json
import logging
import pickle
from abc import ABC, abstractmethod

import h5py
import numpy as np
from sklearn.metrics import (
    matthews_corrcoef,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    confusion_matrix,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


class BaseModel(ABC):
    def __init__(self, ready_model_path=None):

        if ready_model_path:
            self.model = self.load_model(ready_model_path)
        else:
            self.model = None

    @abstractmethod
    def _build_model(self, **params): ...

    @abstractmethod
    def _get_default_params(self): ...

    @abstractmethod
    def _get_optimization_space(self, trial): ...

    @staticmethod
    def _load_emb(path):
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            embeddings = np.stack([f[k][:] for k in keys], axis=0).astype(np.float32)
        return keys, embeddings

    @staticmethod
    def _load_split_pair(positive_file, negative_file):
        _, pos_emb = BaseModel._load_emb(positive_file)
        _, neg_emb = BaseModel._load_emb(negative_file)

        X = np.concatenate([pos_emb, neg_emb], axis=0)
        y = np.array([1] * len(pos_emb) + [0] * len(neg_emb))
        return X, y

    @staticmethod
    def _parse_splits(splits_path):
        with open(splits_path) as f:
            splits = json.load(f)

        fold_nums = sorted(
            {int(k.split("_")[1]) for k in splits if k.startswith("train_")}
        )

        folds = []
        for fold_idx in fold_nums:
            X_train, y_train = BaseModel._load_split_pair(
                splits[f"train_{fold_idx}"]["positive"],
                splits[f"train_{fold_idx}"]["negative"],
            )
            X_val, y_val = BaseModel._load_split_pair(
                splits[f"val_{fold_idx}"]["positive"],
                splits[f"val_{fold_idx}"]["negative"],
            )
            folds.append((fold_idx, X_train, y_train, X_val, y_val))

        X_test, y_test = BaseModel._load_split_pair(
            splits["test"]["positive"], splits["test"]["negative"]
        )
        return folds, X_test, y_test

    @staticmethod
    def evaluate(y_true, y_pred, y_proba=None):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics = {
            "mcc": matthews_corrcoef(y_true, y_pred),
            "tpr": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        }
        if y_proba is not None:
            metrics["avg_precision"] = average_precision_score(y_true, y_proba)
        return metrics

    def _optimize(self, folds, n_trials=50):
        import optuna

        _, X_train, y_train, X_val, y_val = folds[0]

        def objective(trial):
            params = self._get_optimization_space(trial)
            model = self._build_model(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            return matthews_corrcoef(y_val, preds)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        logger.info(
            f"Optuna best MCC: {study.best_value:.4f} │ Params: {study.best_params}"
        )
        return study.best_params

    def train_model(
        self,
        splits_path,
        output_path,
        n_trials=50,
        optimize=True,
    ):
        os.makedirs(output_path, exist_ok=True)
        folds, X_test, y_test = self._parse_splits(splits_path)

        logger.info(
            f"{self.__class__.__name__} │ "
            f"{len(folds)} folds │ Test: {len(y_test)} (pos: {y_test.sum()})"
        )

        if optimize:
            logger.info(f"Starting Optuna optimization ({n_trials} trials).")
            best_params = self._optimize(folds, n_trials=n_trials)
            params = {**self._get_default_params(), **best_params}

            params_path = os.path.join(
                output_path, f"{self.__class__.__name__}_best_params.json"
            )
            with open(params_path, "w") as f:
                json.dump(best_params, f, indent=2)
            logger.info(
                f"Best params for model {self.__class__.__name__} saved under: {params_path}"
            )
        else:
            params = self._get_default_params()

        fold_models = []
        fold_metrics = []

        for fold_idx, X_train, y_train, X_val, y_val in folds:
            model = self._build_model(**params)
            model.fit(X_train, y_train)

            val_preds = model.predict(X_val)
            val_proba = model.predict_proba(X_val)[:, 1]
            val_metrics = self.evaluate(y_val, val_preds, val_proba)

            test_preds = model.predict(X_test)
            test_proba = model.predict_proba(X_test)[:, 1]
            test_metrics = self.evaluate(y_test, test_preds, test_proba)

            logger.info(
                f"Fold {fold_idx} │ "
                f"Val MCC: {val_metrics['mcc']:.4f} │ "
                f"Test MCC: {test_metrics['mcc']:.4f}"
            )

            fold_models.append(model)
            fold_metrics.append(
                {
                    "fold": fold_idx,
                    "val": val_metrics,
                    "test": test_metrics,
                }
            )

        best_idx = np.argmax([s["val"]["mcc"] for s in fold_metrics])
        best_fold = fold_metrics[best_idx]
        self.model = fold_models[best_idx]

        logger.info(
            f"Best fold: {best_fold['fold']} │ "
            f"Val  — MCC: {best_fold['val']['mcc']:.4f}, "
            f"TPR: {best_fold['val']['tpr']:.4f}, "
            f"FPR: {best_fold['val']['fpr']:.4f}, "
            f"AP: {best_fold['val']['avg_precision']:.4f} │ "
            f"Test — MCC: {best_fold['test']['mcc']:.4f}, "
            f"TPR: {best_fold['test']['tpr']:.4f}, "
            f"FPR: {best_fold['test']['fpr']:.4f}, "
            f"AP: {best_fold['test']['avg_precision']:.4f}"
        )

        model_path = os.path.join(output_path, f"{self.__class__.__name__}_final.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info(f"Model {self.__class__.__name__} saved under {model_path}")

        metrics_out = {
            "best_fold": best_fold,
            "all_folds": fold_metrics,
        }
        metrics_path = os.path.join(
            output_path, f"{self.__class__.__name__}_metrics.json"
        )
        with open(metrics_path, "w") as f:
            json.dump(metrics_out, f, indent=2)
        logger.info(f"Metrics saved under {metrics_path}")

    def load_model(self, path):
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        logger.info(f"Classificator model loaded: {path}")

    def predict(self, embedding):
        if self.model is None:
            logger.error("No model loaded. Call train_model() or load_model() first.")
            sys.exit(1)

        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        proba = self.model.predict_proba(embedding.astype(np.float32))
        return float(proba[0, 1])

    def predict_from_file(self, path):

        if self.model is None:
            raise RuntimeError(
                "No model loaded. Call train_model() or load_model() first."
            )

        keys, X = self._load_emb(path)
        proba = self.model.predict_proba(X)

        return {name: float(p) for name, p in zip(keys, proba[:, 1])}
