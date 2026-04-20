import argparse
import csv
import json
from pathlib import Path

CLOSE_TOL = 15


def load_reference(path):
    """Load reference ASM regions. Converts 1-indexed inclusive (biology) to
    0-indexed half-open to match slider output."""
    ref = {}
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            beg, end = row["asm_beg"].strip(), row["asm_end"].strip()
            region = (
                (None, None) if beg == "-" or end == "-" else (int(beg) - 1, int(end))
            )
            ref.setdefault(row["seq_id"], []).append(region)
    return ref


def load_predictions(path):
    with open(path) as f:
        data = json.load(f)
    return {
        sid: [(h["start"], h["end"]) for h in hits]
        for sid, hits in data.items()
        if hits
    }


def overlaps(r, p):
    return r[0] is None or (p[0] < r[1] and p[1] > r[0])


def close_match(r, p):
    return (
        r[0] is not None
        and abs(p[0] - r[0]) <= CLOSE_TOL
        and abs(p[1] - r[1]) <= CLOSE_TOL
    )


def evaluate(ref, preds):
    tp = fn = 0
    for sid, ref_regions in ref.items():
        pred_regions = preds.get(sid, [])
        if any(overlaps(r, p) for r in ref_regions for p in pred_regions):
            tp += 1
        else:
            fn += 1
    fp = len(set(preds) - set(ref))

    close_tp = close_fn = close_fp = close_skipped = 0
    for sid, ref_regions in ref.items():
        pred_regions = preds.get(sid, [])
        for r in ref_regions:
            if r[0] is None:
                continue
            if any(close_match(r, p) for p in pred_regions):
                close_tp += 1
            else:
                close_fn += 1
    for sid, pred_regions in preds.items():
        positional = [r for r in ref.get(sid, []) if r[0] is not None]
        if sid in ref and not positional:
            close_skipped += len(pred_regions)
            continue
        for p in pred_regions:
            if not any(close_match(r, p) for r in positional):
                close_fp += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    close_precision = close_tp / (close_tp + close_fp) if close_tp + close_fp else 0.0
    close_recall = close_tp / (close_tp + close_fn) if close_tp + close_fn else 0.0
    close_f1 = (
        2 * close_precision * close_recall / (close_precision + close_recall)
        if close_precision + close_recall
        else 0.0
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "close_tp": close_tp,
        "close_fp": close_fp,
        "close_fn": close_fn,
        "close_skipped": close_skipped,
        "close_precision": close_precision,
        "close_recall": close_recall,
        "close_f1": close_f1,
    }


def print_metrics(name, m):
    print(f"=== {name} ===")
    print(f"  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}")
    print(f"  Precision: {m['precision']:.4f}")
    print(f"  Recall:    {m['recall']:.4f}")
    print(f"  F1:        {m['f1']:.4f}")
    print(f"  --- Close matches (±{CLOSE_TOL} aa) ---")
    print(
        f"  TP={m['close_tp']}  FP={m['close_fp']}  FN={m['close_fn']}  skipped={m['close_skipped']}"
    )
    print(f"  Precision: {m['close_precision']:.4f}")
    print(f"  Recall:    {m['close_recall']:.4f}")
    print(f"  F1:        {m['close_f1']:.4f}")


def create_parser():
    p = argparse.ArgumentParser(
        description="Evaluate ASMSlider predictions against reference ASM annotations."
    )
    p.add_argument("--predictions", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--output", default=None)
    return p


def main():
    args = create_parser().parse_args()
    name = Path(args.reference).stem
    metrics = evaluate(
        load_reference(args.reference), load_predictions(args.predictions)
    )
    print_metrics(name, metrics)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
