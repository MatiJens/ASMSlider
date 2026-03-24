import argparse
import logging

from scripts.training.rf_model_trainer import RandomForestModel
from scripts.training.xgboost_model_trainer import XGBoostModel
from scripts.training.lr_model_trainer import LogisticRegressionModel

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    parser = argparse.ArgumentParser(
        description="Train RandomForest, XGBoost and LogisticRegression classifiers."
    )
    parser.add_argument(
        "--splits-path",
        type=str,
        required=True,
        help="Path to JSON file with k-fold split definitions.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Base directory where models will be saved.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials per model.",
    )
    args = parser.parse_args()

    models = {
        "logistic_regression": LogisticRegressionModel(),
        "xgboost": XGBoostModel(),
        "random_forest": RandomForestModel(),
    }

    for name, model in models.items():
        logger.info(f"{'=' * 60}")
        logger.info(f"Training: {name}")
        logger.info(f"{'=' * 60}")

        model.train_model(
            splits_path=args.splits_path,
            output_path=f"{args.output_path}/{name}",
            n_trials=args.n_trials,
            optimize=True,
        )


if __name__ == "__main__":
    main()
