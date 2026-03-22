import os
import torch
import numpy as np
from abc import ABC, abstractmethod


class Model(ABC):
    def __init__(
        self,
        input_dir: list[str],
        result_path: str,
        model_path: str,
        n_optuna_trials: int = 60,
    ):
        self.input_dir = input_dir
        self.result_path = result_path
        self.model_path = model_path
        self.n_optuna_trials = n_optuna_trials

        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)

    def _load_pt(self, path):
        return torch.load(path, map_location="cpu", weights_only=False)

    def _make_dataset(self, positive_path, negative_path):
        pos = self._load_pt(positive_path)
        neg = self._load_pt(negative_path)
        X = (
            torch.cat(
                [torch.stack(list(pos.values())), torch.stack(list(neg.values()))]
            )
            .float()
            .numpy()
        )
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        return X, y

    @abstractmethod
    def train(self, X, y): ...

    @abstractmethod
    def predict(self, X, y): ...

    @abstractmethod
    def optimize(self, X, y): ...


class LogisticRegression(Model):
    def train(self, X, y):
        pass

    def predict(self, X, y):
        pass

    def optimize(self, X, y):
        pass


class RandomForest(Model):
    def train(self, X, y):
        pass

    def predict(self, X, y):
        pass

    def optimize(self, X, y):
        pass


class XGBoost(Model):
    def train(self, X, y):
        pass

    def predict(self, X, y):
        pass

    def optimize(self, X, y):
        pass
