import argparse
import logging

import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score, f1_score

from mlp_model import MLPModel

logger = logging.getLogger(__name__)


def load_npy_dir(dir_path):
    arrays = [
        np.load(f).astype(np.float32) for f in sorted(Path(dir_path).glob("*.npy"))
    ]
    if not arrays:
        raise ValueError(f"No .npy files found in {dir_path}")
    return np.concatenate(arrays)


def load_test_data(base_path):
    p = Path(base_path)
    pos = load_npy_dir(p / "positive" / "test")
    neg = load_npy_dir(p / "negative" / "test")
    logger.info(f"Test set: {pos.shape[0]} positive, {neg.shape[0]} negative")
    X = np.concatenate([pos, neg])
    y = np.array([1] * len(pos) + [0] * len(neg), dtype=np.float32)
    return X, y


@torch.no_grad()
def get_predictions(model, X, device, batch_size=512):
    model.eval()
    all_probs = []
    for i in range(0, len(X), batch_size):
        x = torch.from_numpy(X[i : i + batch_size]).to(device)
        logits = model(x)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(all_probs)


def tpr_at_fpr(y_true, y_score, target_fpr):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    idx = max(idx, 0)
    return tpr[idx]


def best_f1_threshold(y_true, y_score):
    thresholds = np.linspace(0, 1, 1001)
    best_f1, best_thr = 0.0, 0.5
    for thr in thresholds:
        preds = (y_score >= thr).astype(int)
        f = f1_score(y_true, preds, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_thr = thr
    return best_thr, best_f1


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Base directory with positive/test/ and negative/test/ subdirs containing .npy files.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory for output file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for inference (default: 512).",
    )
    return parser


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    args = create_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = MLPModel().to(device)
    model.load_state_dict(
        torch.load(args.checkpoint, weights_only=True, map_location=device)
    )

    X, y_true = load_test_data(args.input_path)
    y_score = get_predictions(model, X, device, args.batch_size)

    auroc = roc_auc_score(y_true, y_score)

    fpr_targets = [1e-3, 1e-4, 1e-5, 1e-6]
    tpr_results = {fpr: tpr_at_fpr(y_true, y_score, fpr) for fpr in fpr_targets}

    best_thr, best_f1 = best_f1_threshold(y_true, y_score)

    logger.info(f"AUROC:              {auroc:.6f}")
    for fpr, tpr in tpr_results.items():
        logger.info(f"TPR @ FPR={fpr:.0e}:   {tpr:.6f}")
    logger.info(f"Best threshold:     {best_thr:.4f}")
    logger.info(f"Best F1:            {best_f1:.6f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "auroc": auroc,
        "best_threshold": best_thr,
        "best_f1": best_f1,
    }
    for fpr, tpr in tpr_results.items():
        metrics[f"tpr_at_fpr_{fpr:.0e}"] = tpr

    metrics_path = out_dir / "metrics.txt"
    with open(metrics_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    logger.info(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
