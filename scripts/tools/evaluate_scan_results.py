#!/usr/bin/env python3
"""Per-threshold evaluation of scan_results against asm_references and pfam_references."""
import argparse
import csv
import json
from pathlib import Path


def load_refs(path):
    """Return dict[seq_id] -> list[(beg, end)] or [(None, None)] if positions missing."""
    refs = {}
    with open(path) as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            sid = row[0]
            beg_s = row[1].strip() if len(row) > 1 else ""
            end_s = row[2].strip() if len(row) > 2 else ""
            if beg_s in ("", "NA", "None", "-") or end_s in ("", "NA", "None", "-"):
                refs.setdefault(sid, []).append((None, None))
            else:
                refs.setdefault(sid, []).append((int(beg_s), int(end_s)))
    return refs


def load_scan(path):
    """Return list of (seq_id, beg, end)."""
    with open(path) as f:
        data = json.load(f)
    out = []
    for hit in data:
        beg_s, end_s = hit["location"].split("-")
        out.append((hit["protein"], int(beg_s), int(end_s)))
    return out


def overlap_len(a_beg, a_end, b_beg, b_end):
    return max(0, min(a_end, b_end) - max(a_beg, b_beg) + 1)


def evaluate(proteome_dir):
    proteome_dir = Path(proteome_dir)
    asm = load_refs(proteome_dir / "asm_reference.tsv")
    pfam = load_refs(proteome_dir / "pfam_references.tsv")
    scan_root = proteome_dir / "scan_results"

    total_asm = sum(len(v) for v in asm.values())
    total_pfam = sum(len(v) for v in pfam.values())

    rows = []
    for th_dir in sorted(scan_root.iterdir()):
        json_files = list(th_dir.glob("*.json"))
        if not json_files:
            continue
        hits = load_scan(json_files[0])

        # (1) total found
        total_found = len(hits)

        # (2) count found motifs matching asm_references:
        #   - asm_ref with positions: each found motif covering >=45% of asm region counts
        #   - asm_ref without positions: counts as 1 if any hit exists on that protein
        hits_by_sid = {}
        for sid, hb, he in hits:
            hits_by_sid.setdefault(sid, []).append((hb, he))
        matched_found = 0
        for sid, regions in asm.items():
            sid_hits = hits_by_sid.get(sid, [])
            for ab, ae in regions:
                if ab is None:
                    if sid_hits:
                        matched_found += 1
                    continue
                alen = ae - ab + 1
                for hb, he in sid_hits:
                    if alen > 0 and overlap_len(hb, he, ab, ae) / alen >= 0.45:
                        matched_found += 1
                        break

        # (3) pfam references with a found motif within 35 aa (overlap or adjacent)
        max_dist = 35
        matched_pfam = 0
        for sid, regions in pfam.items():
            sid_hits = [(b, e) for s, b, e in hits if s == sid]
            for pb, pe in regions:
                if any(
                    overlap_len(pb, pe, hb, he) > 0
                    or (hb > pe and hb - pe <= max_dist)
                    or (he < pb and pb - he <= max_dist)
                    for hb, he in sid_hits
                ):
                    matched_pfam += 1

        sensitivity = matched_found / total_asm if total_asm else 0.0

        rows.append({
            "threshold": th_dir.name,
            "total_found": total_found,
            "asm_refs_total": total_asm,
            "found_in_asm_refs_>=45%": matched_found,
            "pfam_refs_within_35aa": matched_pfam,
            "sensitivity": f"{sensitivity:.4f}",
        })

    headers = ["threshold", "total_found", "asm_refs_total",
               "found_in_asm_refs_>=45%", "pfam_refs_within_35aa", "sensitivity"]
    widths = {h: max(len(h), *(len(str(r[h])) for r in rows)) for h in headers}
    print(f"asm_references total: {total_asm} | pfam_references total: {total_pfam}\n")
    print("  ".join(h.ljust(widths[h]) for h in headers))
    print("  ".join("-" * widths[h] for h in headers))
    for r in rows:
        print("  ".join(str(r[h]).ljust(widths[h]) for h in headers))

    out_path = proteome_dir / "scan_evaluation.tsv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("proteome_dir",
                    help="folder containing asm_reference.tsv, pfam_references.tsv, scan_results/")
    args = ap.parse_args()
    evaluate(args.proteome_dir)
