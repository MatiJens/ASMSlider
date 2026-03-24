from sklearn.linear_model import LogisticRegression

from base_model import BaseModel


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
        penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])

        params = {
            "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
            "penalty": penalty,
            "max_iter": 2000,
            "class_weight": trial.suggest_categorical(
                "class_weight", ["balanced", None]
            ),
            "random_state": 67,
        }

        if penalty == "elasticnet":
            params["solver"] = "saga"
            params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)
        elif penalty == "l1":
            params["solver"] = "saga"
        else:
            params["solver"] = "lbfgs"

        return params
