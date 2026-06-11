#!/usr/bin/env python3
"""Per-threshold evaluation of scan_results against asm_references and pfam_references."""
import argparse
import csv
from pathlib import Path

from scan_common import (
    find_proteome_fasta,
    load_asm_refs,
    load_pfam_refs,
    load_protein_lengths,
    load_scan,
    overlap_len,
)

# A hit is in the N-terminus if it starts before this position, and in the
# C-terminus if it ends past (protein_length - this margin).
TERM_MARGIN = 50
# Minimum fraction of a true ASM region a hit must cover to count as a match.
ASM_OVERLAP_FRAC = 0.45

HEADERS = ["threshold", "total_found", "n_term_hits", "c_term_hits", "asm_refs_total",
           "found_in_asm_refs_>=45%", "pfam_refs_same_protein", "sensitivity"]


def evaluate_threshold(th_name, hits, asm, pfam, lengths, total_asm):
    hits_by_sid = {}
    for h in hits:
        hits_by_sid.setdefault(h["sid"], []).append((h["beg"], h["end"]))

    n_term_hits = sum(1 for h in hits if h["beg"] < TERM_MARGIN)
    c_term_hits = sum(
        1 for h in hits
        if lengths.get(h["sid"]) and h["end"] > lengths[h["sid"]] - TERM_MARGIN
    )

    # ASM references matched by a found motif:
    #   - positioned ref: a hit covering >= ASM_OVERLAP_FRAC of the region
    #   - ref without positions: any hit on that protein
    matched_asm = 0
    for sid, regions in asm.items():
        sid_hits = hits_by_sid.get(sid, [])
        for ab, ae in regions:
            if ab is None:
                if sid_hits:
                    matched_asm += 1
                continue
            alen = ae - ab + 1
            if alen > 0 and any(
                overlap_len(hb, he, ab, ae) / alen >= ASM_OVERLAP_FRAC
                for hb, he in sid_hits
            ):
                matched_asm += 1

    # PFAM references with a found motif anywhere in the same protein.
    matched_pfam = sum(
        len(regions) for sid, regions in pfam.items() if hits_by_sid.get(sid)
    )

    sensitivity = matched_asm / total_asm if total_asm else 0.0
    return {
        "threshold": th_name,
        "total_found": len(hits),
        "n_term_hits": n_term_hits,
        "c_term_hits": c_term_hits,
        "asm_refs_total": total_asm,
        "found_in_asm_refs_>=45%": matched_asm,
        "pfam_refs_same_protein": matched_pfam,
        "sensitivity": f"{sensitivity:.4f}",
    }


def evaluate(proteome_dir):
    proteome_dir = Path(proteome_dir)
    asm = load_asm_refs(proteome_dir / "asm_reference.tsv")
    pfam = load_pfam_refs(proteome_dir / "pfam_references.tsv")
    lengths = load_protein_lengths(find_proteome_fasta(proteome_dir))

    total_asm = sum(len(v) for v in asm.values())
    total_pfam = sum(len(v) for v in pfam.values())

    rows = []
    for th_dir in sorted((proteome_dir / "scan_results").iterdir()):
        json_files = list(th_dir.glob("*.json"))
        if not json_files:
            continue
        hits = load_scan(json_files[0])
        rows.append(evaluate_threshold(th_dir.name, hits, asm, pfam, lengths, total_asm))

    widths = {h: max(len(h), *(len(str(r[h])) for r in rows)) for h in HEADERS}
    print(f"asm_references total: {total_asm} | "
          f"pfam_references total (allowed accessions only): {total_pfam}\n")
    print("  ".join(h.ljust(widths[h]) for h in HEADERS))
    print("  ".join("-" * widths[h] for h in HEADERS))
    for r in rows:
        print("  ".join(str(r[h]).ljust(widths[h]) for h in HEADERS))

    out_path = proteome_dir / "scan_evaluation.tsv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADERS, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("proteome_dir",
                    help="folder containing asm_reference.tsv, pfam_references.tsv, scan_results/")
    args = ap.parse_args()
    evaluate(args.proteome_dir)
