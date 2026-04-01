from sklearn.ensemble import RandomForestClassifier

from .base_model import BaseModel


class RandomForestModel(BaseModel):
    def _get_default_params(self):
        return {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "n_jobs": -1,
            "random_state": 67,
        }

    def _build_model(self, **params):
        return RandomForestClassifier(**params)

    def _get_optimization_space(self, trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "class_weight": trial.suggest_categorical(
                "class_weight", ["balanced", "balanced_subsample"]
            ),
            "random_state": 67,
        }
