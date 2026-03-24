from xgboost import XGBClassifier

from base_model import BaseModel


class XGBoostModel(BaseModel):
    def _get_default_params(self):
        return {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "scale_pos_weight": 1.0,
            "eval_metric": "logloss",
            "early_stopping_rounds": 20,
            "n_jobs": -1,
            "random_state": 67,
        }

    def _build_model(self, **params):
        return XGBClassifier(**params)

    def _fit_model(self, model, X_train, y_train, X_val=None, y_val=None):
        fit_params = {}
        if X_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["verbose"] = False
        model.fit(X_train, y_train, **fit_params)
        return model

    def _get_optimization_space(self, trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 5.0),
            "eval_metric": "logloss",
            "early_stopping_rounds": 20,
            "n_jobs": -1,
            "random_state": 67,
        }
