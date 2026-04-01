from sklearn.linear_model import LogisticRegression

from .base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    def _get_default_params(self):
        return {
            "C": 1.0,
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 1000,
            "class_weight": "balanced",
            "random_state": 67,
        }

    def _build_model(self, **params):
        return LogisticRegression(**params)

    def _get_optimization_space(self, trial):
        penalty = trial.suggest_categorical("penalty", ["l2", "elasticnet"])

        if penalty == "elasticnet":
            solver = "saga"
            l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
        else:
            solver = "lbfgs"
            l1_ratio = None

        params = {
            "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
            "penalty": penalty,
            "solver": solver,
            "max_iter": 2000,
            "class_weight": trial.suggest_categorical(
                "class_weight", ["balanced", None]
            ),
            "random_state": 67,
        }

        if l1_ratio is not None:
            params["l1_ratio"] = l1_ratio

        return params
