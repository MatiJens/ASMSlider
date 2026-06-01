import argparse
import sys
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import optuna
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import mlflow
except ImportError:
    mlflow = None

from training.train_classifier import (
    encode_dataset,
    evaluate,
    load_dataset,
    load_fold_encoder,
    make_loader,
    train_one_epoch,
)
from models.mlp_model import MLPModel
from training.training_utils import train_loop
from utils.focal_loss import FocalLoss


def create_parser():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for MLP classifier using Optuna."
    )
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--ae-checkpoint-dir", default=None)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--study-name", default="classifier_tuning")
    parser.add_argument("--storage", default=None,
                        help="Optuna storage URL (e.g. sqlite:///optuna.db). "
                             "Default: in-memory.")
    return parser


def objective(trial, args, device, run_dir):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    alpha = trial.suggest_float("alpha", 0.1, 0.9)
    gamma = trial.suggest_float("gamma", 0.5, 5.0)

    trial_dir = run_dir / f"trial_{trial.number}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    if mlflow is not None:
        mlflow.start_run(run_name=f"trial_{trial.number}")
        mlflow.log_params(trial.params)

    try:
        fold_mccs = []
        for k in range(1, args.folds + 1):
            if mlflow is not None:
                mlflow.start_run(run_name=f"trial_{trial.number}_fold_{k}", nested=True)

            try:
                train_ds = load_dataset(args.input_path, "train", f"*trn{k}.npy")
                val_ds = load_dataset(args.input_path, "val", f"*val{k}.npy")

                if args.ae_checkpoint_dir:
                    raw_dim = train_ds.tensors[0].shape[1]
                    encoder = load_fold_encoder(
                        args.ae_checkpoint_dir, k, raw_dim, args.latent_dim, device
                    )
                    train_ds = encode_dataset(train_ds, encoder, device)
                    val_ds = encode_dataset(val_ds, encoder, device)

                train_loader = make_loader(train_ds, batch_size, shuffle=True)
                val_loader = make_loader(val_ds, batch_size)

                input_dim = train_ds.tensors[0].shape[1]
                model = MLPModel(input_dim=input_dim).to(device)
                criterion = FocalLoss(alpha=alpha, gamma=gamma)
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=lr, weight_decay=weight_decay
                )

                ckpt_path = trial_dir / f"fold_{k}.pt"
                eval_fn = partial(evaluate, criterion=criterion, device=device)

                history = train_loop(
                    model, train_loader, val_loader,
                    partial(train_one_epoch, criterion=criterion, device=device),
                    eval_fn, optimizer, args.max_epochs, args.patience, ckpt_path,
                )
            finally:
                if mlflow is not None and mlflow.active_run():
                    mlflow.end_run()

            if not ckpt_path.exists():
                raise optuna.TrialPruned(f"No checkpoint saved for fold {k} (model diverged)")

            model.load_state_dict(
                torch.load(ckpt_path, weights_only=True, map_location=device)
            )
            val_metrics = eval_fn(model, val_loader)
            fold_mccs.append(val_metrics["mcc"])

            trial.report(np.mean(fold_mccs), k)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_mcc = np.mean(fold_mccs)
        if mlflow is not None:
            mlflow.log_metric("mean_mcc", mean_mcc)
        return mean_mcc
    finally:
        if mlflow is not None:
            while mlflow.active_run():
                mlflow.end_run()


def main():
    args = create_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.set_float32_matmul_precision("high")

    run_dir = (
        Path(args.checkpoint_dir)
        / f"tuning_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    if mlflow is not None:
        mlflow.set_tracking_uri(f"sqlite:///{run_dir.resolve()}/mlflow.db")
        mlflow.set_experiment(args.study_name)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial, args, device, run_dir),
        n_trials=args.n_trials,
        catch=(Exception,),
    )

    print("\n" + "=" * 60)
    print("Best trial:")
    print(f"  Mean MCC: {study.best_trial.value:.4f}")
    print("  Params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")

    with open(run_dir / "best_params.txt", "w") as f:
        f.write(f"mean_mcc: {study.best_trial.value:.6f}\n")
        for k, v in study.best_trial.params.items():
            f.write(f"{k}: {v}\n")

    print(f"\nResults saved to {run_dir}")
    if mlflow is not None:
        print(f"MLflow UI: mlflow ui --backend-store-uri file://{run_dir.resolve()}/mlruns")


if __name__ == "__main__":
    main()
