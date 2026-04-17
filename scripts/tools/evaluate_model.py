import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_curve,
)

from models.mlp_model import MLPModel
from utils.sequence_loader import load_embeddings_dir


def load_test_data(base_path):
    p = Path(base_path)
    pos = load_embeddings_dir(p / "positive" / "test")
    neg = load_embeddings_dir(p / "negative" / "test")
    print(f"Test set: {len(pos)} positive, {len(neg)} negative")
    X = np.concatenate([pos, neg])
    y = np.array([1] * len(pos) + [0] * len(neg), dtype=np.float32)
    return X, y


@torch.no_grad()
def get_predictions(model, X, device, batch_size=512):
    model.eval()
    all_probs = []
    for i in range(0, len(X), batch_size):
        batch = X[i : i + batch_size]
        x = torch.from_numpy(batch).to(device)
        all_probs.append(torch.sigmoid(model(x)).cpu().numpy())
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
        "--input-path", type=str, required=True,
        help="Base directory with positive/test/ and negative/test/ subdirs containing .npy files.",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=".",
        help="Directory for output file.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=512,
        help="Batch size for inference (default: 512).",
    )
    return parser


def main():
    args = create_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X, y_true = load_test_data(args.input_path)

    model = MLPModel(input_dim=X.shape[1]).to(device)
    model.load_state_dict(
        torch.load(args.checkpoint, weights_only=True, map_location=device)
    )

    y_score = get_predictions(model, X, device, args.batch_size)

    ap = average_precision_score(y_true, y_score)

    best_thr, best_f1 = best_f1_threshold(y_true, y_score)

    y_pred = (y_score >= best_thr).astype(int)
    mcc = matthews_corrcoef(y_true, y_pred)

    fpr_targets = [1e-3, 1e-4, 1e-5, 1e-6]
    tpr_results = {fpr: tpr_at_fpr(y_true, y_score, fpr) for fpr in fpr_targets}

    print(f"AP:                 {ap:.6f}")
    print(f"Best F1:            {best_f1:.6f}  (threshold={best_thr:.4f})")
    print(f"MCC:                {mcc:.6f}")
    for fpr, tpr in tpr_results.items():
        print(f"TPR @ FPR={fpr:.0e}:   {tpr:.6f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "ap": ap,
        "f1": best_f1,
        "f1_threshold": best_thr,
        "mcc": mcc,
    }
    for fpr, tpr in tpr_results.items():
        metrics[f"tpr_at_fpr_{fpr:.0e}"] = tpr

    metrics_path = out_dir / "metrics.txt"
    with open(metrics_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
