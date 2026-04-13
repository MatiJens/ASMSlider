import argparse
import csv
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_reference(tsv_path):
    """Load reference ASM annotations from TSV.

    Returns dict: seq_id -> list of (beg, end) tuples.
    Entries with '-' as beg/end are treated as whole-sequence hits (no position info)
    and stored as (None, None).
    """
    ref = {}
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sid = row["seq_id"]
            beg = row["asm_beg"].strip()
            end = row["asm_end"].strip()
            if beg == "-" or end == "-":
                region = (None, None)
            else:
                region = (int(beg), int(end))
            ref.setdefault(sid, []).append(region)
    return ref


def load_predictions(json_path):
    """Load slider output JSON.

    Returns dict: seq_id -> list of (start, end) tuples.
    """
    with open(json_path) as f:
        data = json.load(f)
    preds = {}
    for sid, hits in data.items():
        if hits:
            preds[sid] = [(h["start"], h["end"]) for h in hits]
    return preds


def regions_overlap(ref_region, pred_region):
    """Check if a reference region and predicted region overlap.

    If reference has no position info (None, None), any prediction on that
    sequence counts as overlapping.
    """
    ref_beg, ref_end = ref_region
    pred_beg, pred_end = pred_region
    if ref_beg is None:
        return True
    return pred_beg < ref_end and pred_end > ref_beg


def overlap_length(ref_region, pred_region):
    """Compute overlap length between two regions. Returns 0 if no position info."""
    ref_beg, ref_end = ref_region
    pred_beg, pred_end = pred_region
    if ref_beg is None:
        return 0
    return max(0, min(ref_end, pred_end) - max(ref_beg, pred_beg))


def evaluate(ref, preds):
    """Compute evaluation metrics.

    Sequence-level metrics:
      - TP: reference sequence found in predictions (at least one overlapping hit)
      - FN: reference sequence not found or no overlapping hit
      - FP: predicted sequence not in reference at all

    Region-level metrics (only for references with position info):
      - Region recall: fraction of reference regions that overlap with at least one prediction
      - Mean IoU: average intersection-over-union for matched reference regions
    """
    ref_ids = set(ref.keys())
    pred_ids = set(preds.keys())

    # --- sequence-level ---
    seq_tp = 0
    seq_fn = 0
    for sid in ref_ids:
        if sid in pred_ids:
            ref_regions = ref[sid]
            pred_regions = preds[sid]
            has_overlap = any(
                regions_overlap(rr, pr)
                for rr in ref_regions
                for pr in pred_regions
            )
            if has_overlap:
                seq_tp += 1
            else:
                seq_fn += 1
        else:
            seq_fn += 1

    seq_fp = len(pred_ids - ref_ids)

    seq_precision = seq_tp / (seq_tp + seq_fp) if (seq_tp + seq_fp) > 0 else 0.0
    seq_recall = seq_tp / (seq_tp + seq_fn) if (seq_tp + seq_fn) > 0 else 0.0
    seq_f1 = (
        2 * seq_precision * seq_recall / (seq_precision + seq_recall)
        if (seq_precision + seq_recall) > 0
        else 0.0
    )

    # --- region-level (only where positions are known) ---
    region_total = 0
    region_found = 0
    ious = []

    for sid, ref_regions in ref.items():
        pred_regions = preds.get(sid, [])
        for rr in ref_regions:
            if rr[0] is None:
                continue
            region_total += 1
            best_iou = 0.0
            matched = False
            for pr in pred_regions:
                if regions_overlap(rr, pr):
                    matched = True
                    inter = overlap_length(rr, pr)
                    union = (rr[1] - rr[0]) + (pr[1] - pr[0]) - inter
                    iou = inter / union if union > 0 else 0.0
                    best_iou = max(best_iou, iou)
            if matched:
                region_found += 1
                ious.append(best_iou)

    region_recall = region_found / region_total if region_total > 0 else 0.0
    mean_iou = sum(ious) / len(ious) if ious else 0.0

    return {
        "seq_tp": seq_tp,
        "seq_fp": seq_fp,
        "seq_fn": seq_fn,
        "seq_precision": seq_precision,
        "seq_recall": seq_recall,
        "seq_f1": seq_f1,
        "region_total": region_total,
        "region_found": region_found,
        "region_recall": region_recall,
        "mean_iou": mean_iou,
    }


def print_metrics(name, metrics):
    logger.info(f"=== {name} ===")
    logger.info(f"  Sequence-level: TP={metrics['seq_tp']}  FP={metrics['seq_fp']}  FN={metrics['seq_fn']}")
    logger.info(f"  Seq Precision:  {metrics['seq_precision']:.4f}")
    logger.info(f"  Seq Recall:     {metrics['seq_recall']:.4f}")
    logger.info(f"  Seq F1:         {metrics['seq_f1']:.4f}")
    if metrics["region_total"] > 0:
        logger.info(f"  Region Recall:  {metrics['region_found']}/{metrics['region_total']} = {metrics['region_recall']:.4f}")
        logger.info(f"  Mean IoU:       {metrics['mean_iou']:.4f}")
    else:
        logger.info("  Region Recall:  N/A (no positional annotations)")


def print_missed(name, ref, preds):
    """Log reference sequences that were not detected."""
    missed = []
    for sid, ref_regions in ref.items():
        if sid not in preds:
            missed.append((sid, "not_found"))
        else:
            pred_regions = preds[sid]
            has_overlap = any(
                regions_overlap(rr, pr)
                for rr in ref_regions
                for pr in pred_regions
            )
            if not has_overlap:
                missed.append((sid, "no_overlap"))
    if missed:
        logger.info(f"  Missed in {name}:")
        for sid, reason in missed:
            logger.info(f"    {sid} ({reason})")


def create_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate ASMSlider predictions against reference ASM annotations."
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        nargs="+",
        help="Path(s) to slider output JSON file(s).",
    )
    parser.add_argument(
        "--references",
        type=str,
        required=True,
        nargs="+",
        help="Path(s) to reference TSV file(s). Matched to predictions by order.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save aggregated metrics as JSON.",
    )
    return parser


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    args = create_parser().parse_args()

    if len(args.predictions) != len(args.references):
        raise ValueError(
            f"Number of prediction files ({len(args.predictions)}) must match "
            f"number of reference files ({len(args.references)})."
        )

    all_metrics = {}
    agg_tp, agg_fp, agg_fn = 0, 0, 0
    agg_region_total, agg_region_found = 0, 0
    agg_ious = []

    for pred_path, ref_path in zip(args.predictions, args.references):
        name = Path(ref_path).stem
        ref = load_reference(ref_path)
        preds = load_predictions(pred_path)
        metrics = evaluate(ref, preds)

        print_metrics(name, metrics)
        print_missed(name, ref, preds)
        all_metrics[name] = metrics

        agg_tp += metrics["seq_tp"]
        agg_fp += metrics["seq_fp"]
        agg_fn += metrics["seq_fn"]
        agg_region_total += metrics["region_total"]
        agg_region_found += metrics["region_found"]

    # aggregated
    if len(args.predictions) > 1:
        agg_prec = agg_tp / (agg_tp + agg_fp) if (agg_tp + agg_fp) > 0 else 0.0
        agg_rec = agg_tp / (agg_tp + agg_fn) if (agg_tp + agg_fn) > 0 else 0.0
        agg_f1 = (
            2 * agg_prec * agg_rec / (agg_prec + agg_rec)
            if (agg_prec + agg_rec) > 0
            else 0.0
        )
        agg_region_rec = (
            agg_region_found / agg_region_total if agg_region_total > 0 else 0.0
        )

        logger.info("=== AGGREGATED ===")
        logger.info(f"  Sequence-level: TP={agg_tp}  FP={agg_fp}  FN={agg_fn}")
        logger.info(f"  Seq Precision:  {agg_prec:.4f}")
        logger.info(f"  Seq Recall:     {agg_rec:.4f}")
        logger.info(f"  Seq F1:         {agg_f1:.4f}")
        if agg_region_total > 0:
            logger.info(f"  Region Recall:  {agg_region_found}/{agg_region_total} = {agg_region_rec:.4f}")

        all_metrics["aggregated"] = {
            "seq_tp": agg_tp,
            "seq_fp": agg_fp,
            "seq_fn": agg_fn,
            "seq_precision": agg_prec,
            "seq_recall": agg_rec,
            "seq_f1": agg_f1,
            "region_total": agg_region_total,
            "region_found": agg_region_found,
            "region_recall": agg_region_rec,
        }

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_metrics, f, indent=2)
        logger.info(f"Metrics saved to {args.output}")


if __name__ == "__main__":
    main()
