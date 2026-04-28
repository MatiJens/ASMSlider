import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

LOCATION_TOL = 15


def load_reference(path, pred_pids=None):
    """Load asm_by_tax.json. If pred_pids given, only keep tax groups that
    contain at least one predicted protein (= filter to relevant organism)."""
    with open(path) as f:
        data = json.load(f)

    if pred_pids is not None:
        relevant_tax = {
            tax_id for tax_id, proteins in data.items()
            if pred_pids & proteins.keys()
        }
    else:
        relevant_tax = set(data.keys())

    ref = {}
    for tax_id in relevant_tax:
        for pid, info in data[tax_id].items():
            beg, end = info["asm_beg"], info["asm_end"]
            asm_id = info.get("asm_id", "unknown")
            ref.setdefault(pid, []).append((beg, end, asm_id))
    return ref


def load_predictions(path):
    with open(path) as f:
        data = json.load(f)
    preds = {}
    for entry in data:
        pid = entry["protein"]
        beg, end = map(int, entry["location"].split("-"))
        prob = entry.get("probability", 0.0)
        preds.setdefault(pid, []).append((beg, end, prob))
    return preds


def location_match(ref_region, pred_region, tol):
    if ref_region[0] is None or ref_region[1] is None:
        return False
    return abs(pred_region[0] - ref_region[0]) <= tol and abs(pred_region[1] - ref_region[1]) <= tol


def evaluate(ref, preds, tol):
    found_protein = 0
    found_location = 0
    ref_proteins_with_pred = set()

    for pid, pred_regions in preds.items():
        if pid in ref:
            ref_proteins_with_pred.add(pid)
            found_protein += 1
            ref_regions = ref[pid]
            if any(location_match((r[0], r[1]), (p[0], p[1]), tol)
                   for r in ref_regions for p in pred_regions):
                found_location += 1

    not_found = len(ref) - len(ref_proteins_with_pred)
    false_positive = len(set(preds) - set(ref))

    n_ref = len(ref)
    n_pred = len(preds)
    return {
        "total_reference": n_ref,
        "total_predictions": n_pred,
        "found_protein": found_protein,
        "found_location_exact": found_location,
        "not_found": not_found,
        "false_positive": false_positive,
        "protein_recall": found_protein / n_ref if n_ref else 0.0,
        "protein_precision": found_protein / n_pred if n_pred else 0.0,
        "location_recall": found_location / n_ref if n_ref else 0.0,
        "location_precision": found_location / n_pred if n_pred else 0.0,
    }


def print_metrics(m, tol):
    print(f"Reference sequences:  {m['total_reference']}")
    print(f"Predicted sequences:  {m['total_predictions']}")
    print()
    print("--- Protein-level (ASM found in sequence) ---")
    print(f"  Found:       {m['found_protein']}")
    print(f"  Not found:   {m['not_found']}")
    print(f"  False pos:   {m['false_positive']}")
    print(f"  Precision:   {m['protein_precision']:.4f}")
    print(f"  Recall:      {m['protein_recall']:.4f}")
    print()
    print(f"--- Location-level (exact position ±{tol} aa) ---")
    print(f"  Exact match: {m['found_location_exact']}")
    print(f"  Precision:   {m['location_precision']:.4f}")
    print(f"  Recall:      {m['location_recall']:.4f}")


def build_family_colormap(ref):
    families = sorted({asm_id for regions in ref.values()
                       for _, _, asm_id in regions if _ is not None})
    n = max(len(families), 1)
    cmap_name = "gist_ncar" if n > 20 else "tab20"
    cmap = plt.colormaps.get_cmap(cmap_name).resampled(n)
    return {fam: cmap(i) for i, fam in enumerate(families)}


MAX_PLOT = 75


def plot_proteome(ref, preds, tol, output_path):
    ref_pids = set(ref)
    top_preds = {}
    for pid, regions in preds.items():
        top_preds[pid] = max(r[2] for r in regions)
    top_pids = sorted(top_preds, key=top_preds.get, reverse=True)[:MAX_PLOT]

    relevant = sorted(ref_pids | set(top_pids))
    if not relevant:
        print("No proteins to plot.")
        return

    if len(preds) > len(top_pids):
        print(f"Plot limited to {MAX_PLOT} top predictions (out of {len(preds)})")

    plot_ref = {pid: ref[pid] for pid in relevant if pid in ref}
    family_colors = build_family_colormap(plot_ref)

    bar_h = 0.35
    fig_height = max(4, len(relevant) * 0.4 + 1.5)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    for i, pid in enumerate(relevant):
        y = i
        ref_regions = ref.get(pid, [])
        pred_regions = preds.get(pid, [])

        max_pos = 0
        for beg, end, _ in ref_regions:
            if beg is not None and end is not None:
                max_pos = max(max_pos, end)
        for beg, end, _ in pred_regions:
            max_pos = max(max_pos, end)

        ax.barh(y, max_pos * 1.1, height=bar_h * 2, left=0, color="#f0f0f0",
                edgecolor="#cccccc", linewidth=0.5, zorder=1)

        for beg, end, asm_id in ref_regions:
            if beg is None or end is None:
                continue
            color = family_colors.get(asm_id, "grey")
            ax.barh(y + bar_h / 2, end - beg, height=bar_h, left=beg,
                    color=color, edgecolor="black", linewidth=0.6, zorder=2)
            mid = (beg + end) / 2
            ax.text(mid, y + bar_h / 2, asm_id, ha="center", va="center",
                    fontsize=5, fontweight="bold", zorder=4, color="white",
                    clip_on=True)

        for beg, end, prob in pred_regions:
            is_match = any(
                location_match((r[0], r[1]), (beg, end), tol) for r in ref_regions
            )
            edge_color = "#2ecc71" if is_match else "#e74c3c"
            ax.barh(y - bar_h / 2, end - beg, height=bar_h, left=beg,
                    color="none", edgecolor=edge_color, linewidth=1.8,
                    linestyle="--", zorder=3)
            mid = (beg + end) / 2
            ax.text(mid, y - bar_h / 2, f"{prob:.2f}", ha="center", va="center",
                    fontsize=5, color=edge_color, zorder=4, clip_on=True)

    ax.set_yticks(range(len(relevant)))
    ax.set_yticklabels(relevant, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("Position (aa)")
    ax.set_title("ASM Predictions vs Reference")

    legend_handles = []
    for fam, color in sorted(family_colors.items()):
        legend_handles.append(mpatches.Patch(facecolor=color, edgecolor="black",
                                             linewidth=0.6, label=fam))
    legend_handles.append(mpatches.Patch(facecolor="none", edgecolor="#2ecc71",
                                         linewidth=1.8, linestyle="--",
                                         label=f"Pred (match ±{tol})"))
    legend_handles.append(mpatches.Patch(facecolor="none", edgecolor="#e74c3c",
                                         linewidth=1.8, linestyle="--",
                                         label="Pred (no match)"))

    ax.legend(handles=legend_handles, loc="upper right", fontsize=6,
              ncol=2, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {output_path}")


def create_parser():
    p = argparse.ArgumentParser(
        description="Evaluate ASMSlider predictions against asm_by_tax.json reference."
    )
    p.add_argument("--predictions", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--tolerance", type=int, default=LOCATION_TOL)
    p.add_argument("--outdir", required=True)
    return p


def main():
    args = create_parser().parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    preds = load_predictions(args.predictions)
    ref = load_reference(args.reference, pred_pids=set(preds.keys()))
    metrics = evaluate(ref, preds, args.tolerance)
    print_metrics(metrics, args.tolerance)

    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {outdir / 'metrics.json'}")

    plot_proteome(ref, preds, args.tolerance, outdir / "plot.png")


if __name__ == "__main__":
    main()
