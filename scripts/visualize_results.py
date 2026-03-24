"""
Ewaluacja predykcji ASM - per plik, parowanie po modelu tylko z PB40.
Wykresy: 3 modele obok siebie (LR | RF | XGB) per dataset.

Struktura:
    results/
      positive/   <- JSONy z pozytywami
      negative/   <- JSONy z negatywami (parowane sa tylko PB40_*)

Uzycie:
    python evaluate_asm.py results/
"""

import json, sys, csv, re
from collections import defaultdict
import numpy as np
from pathlib import Path
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt


def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_model(filename):
    m = re.search(r"_(lr|rf|xgb)_results", filename)
    return m.group(1) if m else None


def get_dataset(filename):
    """Usuwa suffix _lr/_rf/_xgb_results -> nazwa datasetu."""
    return re.sub(r"_(lr|rf|xgb)_results$", "", filename)


def evaluate(pos_scores, neg_scores):
    y_scores = np.array(pos_scores + neg_scores)
    y_true = np.array([1] * len(pos_scores) + [0] * len(neg_scores))

    ap = average_precision_score(y_true, y_scores)

    prec, rec, thr = precision_recall_curve(y_true, y_scores)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    best_f1 = f1[np.argmax(f1)]
    best_thr = thr[np.argmax(f1)]

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    tpr_at = {}
    for target in [1e-2, 1e-3]:
        idx = max(0, np.searchsorted(fpr, target, side="right") - 1)
        tpr_at[target] = tpr[idx]

    return ap, best_f1, best_thr, tpr_at


def plot_trio(files_by_model, dataset_name, label, out_path):
    """Rysuje 3 histogramy obok siebie: LR | RF | XGB."""
    models = ["lr", "rf", "xgb"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, model in zip(axes, models):
        if model in files_by_model:
            scores = list(load_json(files_by_model[model]).values())
            bins = np.linspace(0, 1, 50)
            ax.hist(
                scores,
                bins=bins,
                color="steelblue" if label == "neg" else "tomato",
                edgecolor="white",
                linewidth=0.3,
            )
            ax.set_title(f"{model.upper()} (n={len(scores)})")
        else:
            ax.set_title(f"{model.upper()} (brak)")
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")

    fig.suptitle(f"{dataset_name} [{label}]", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    pos_dir = root / "positive"
    neg_dir = root / "negative"

    if not pos_dir.exists() or not neg_dir.exists():
        print(f"Potrzebuje {pos_dir}/ i {neg_dir}/")
        sys.exit(1)

    out_dir = root / "eval"
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    pos_files = sorted(pos_dir.glob("*.json"))
    neg_files_all = sorted(neg_dir.glob("*.json"))

    # --- Wykresy: grupowanie po datasecie ---
    print("=== Wykresy rozkladu ===")

    for folder, label in [(pos_dir, "pos"), (neg_dir, "neg")]:
        grouped = defaultdict(dict)
        for f in sorted(folder.glob("*.json")):
            model = get_model(f.name)
            dataset = get_dataset(f.stem)
            if model:
                grouped[dataset][model] = f

        for dataset, files_by_model in sorted(grouped.items()):
            out_path = plots_dir / f"{dataset}.png"
            plot_trio(files_by_model, dataset, label, out_path)
            print(f"  {dataset}.png")

    # --- Metryki: parowanie po modelu, tylko PB40 ---
    print("\n=== Metryki (parowanie po modelu, tylko PB40) ===")
    pb40_files = [f for f in neg_files_all if f.name.startswith("PB40")]
    rows = []

    for pf in pos_files:
        p_model = get_model(pf.name)
        if not p_model:
            continue

        pos_scores = list(load_json(pf).values())

        for nf in pb40_files:
            if get_model(nf.name) != p_model:
                continue

            neg_scores = list(load_json(nf).values())
            ap, f1, thr, tpr_at = evaluate(pos_scores, neg_scores)

            name = f"{pf.stem}_vs_{nf.stem}"
            row = {
                "model": p_model,
                "positive": pf.stem,
                "negative": nf.stem,
                "n_pos": len(pos_scores),
                "n_neg": len(neg_scores),
                "AP": round(ap, 4),
                "best_F1": round(f1, 4),
                "F1_threshold": round(thr, 4),
                "TPR@FPR=1e-2": round(tpr_at[1e-2], 4),
                "TPR@FPR=1e-3": round(tpr_at[1e-3], 4),
            }
            rows.append(row)
            print(
                f"  [{p_model}] {name}: AP={ap:.4f} F1={f1:.4f} TPR@1e-3={tpr_at[1e-3]:.4f}"
            )

    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"\nMetryki: {csv_path}")
    print(f"Wykresy: {plots_dir}/")


if __name__ == "__main__":
    main()
