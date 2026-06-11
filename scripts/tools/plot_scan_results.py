#!/usr/bin/env python3
"""Plot per-threshold ASM scan predictions vs reference ASM motifs and PFAM domains.

For each threshold under <proteome_dir>/scan_results/<thr>/, draws one figure showing
every protein with at least one predicted ASM motif. Each protein row contains:
  - a grey background bar spanning the protein length
  - true ASM motifs from asm_reference.tsv (filled box) -- skipped when positions are absent
  - PFAM domains from pfam_references.tsv (filled box)
  - predicted ASM motifs (dashed), coloured by how they relate to the references:
    green if they cover a true ASM (>=45%), blue if the protein carries a PFAM
    reference, red otherwise; the prediction probability is annotated
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from scan_common import (
    find_proteome_fasta,
    load_asm_refs,
    load_pfam_refs,
    load_protein_lengths,
    load_scan,
    overlap_len,
)

OVERLAP_FRAC = 0.45

ASM_COLOR = "#2a9d8f"
ASM_EDGE = "#13524a"
PFAM_COLOR = "#f4a261"
PFAM_EDGE = "#9c5a25"
PRED_PFAM_COLOR = "#1d6fb8"
PRED_OTHER_COLOR = "#e63946"


def predicted_matches_asm(hit, asm_regions):
    """True if hit covers >=OVERLAP_FRAC of any positioned ASM ref on the same protein."""
    for ab, ae in asm_regions:
        if ab is None:
            continue
        alen = ae - ab + 1
        if alen > 0 and overlap_len(hit["beg"], hit["end"], ab, ae) / alen >= OVERLAP_FRAC:
            return True
    return False


def predicted_in_pfam_protein(pfam_regions):
    """True if the protein carries any positioned PFAM reference (distance ignored)."""
    return any(pb is not None for pb, _pe, _acc in pfam_regions)


def plot_panel(sids, by_sid, asm, pfam, lengths, out_path, title):
    row_h = 0.5
    fig_h = max(3.0, 0.45 * len(sids) + 1.5)
    fig, ax = plt.subplots(figsize=(16, fig_h))

    max_x = 0
    for y, sid in enumerate(sids):
        plen = lengths.get(sid, 0)
        if plen <= 0:
            ends = [h["end"] for h in by_sid[sid]]
            ends += [e for _, e in asm.get(sid, []) if e is not None]
            ends += [e for _, e, _acc in pfam.get(sid, []) if e is not None]
            plen = max(ends) if ends else 1
        max_x = max(max_x, plen)

        ax.add_patch(Rectangle((0, y - row_h / 2), plen, row_h,
                               facecolor="#ececec", edgecolor="#bbbbbb", linewidth=0.5))

        # PFAM domains first so predicted motifs overlay them.
        for pb, pe, acc in pfam.get(sid, []):
            if pb is None:
                continue
            ax.add_patch(Rectangle((pb, y - row_h / 2), pe - pb + 1, row_h,
                                   facecolor=PFAM_COLOR, edgecolor=PFAM_EDGE,
                                   linewidth=0.6, alpha=0.85))
            ax.text((pb + pe) / 2, y, acc, ha="center", va="center",
                    fontsize=14, fontweight="bold", color="white")

        for ab, ae in asm.get(sid, []):
            if ab is None:
                continue
            ax.add_patch(Rectangle((ab, y - row_h / 2), ae - ab + 1, row_h,
                                   facecolor=ASM_COLOR, edgecolor=ASM_EDGE, linewidth=0.6))
            ax.text((ab + ae) / 2, y, "ASM", ha="center", va="center",
                    fontsize=10, color="white")

        asm_regions = asm.get(sid, [])
        pfam_regions = pfam.get(sid, [])
        for h in by_sid[sid]:
            if predicted_matches_asm(h, asm_regions):
                color = ASM_COLOR
            elif predicted_in_pfam_protein(pfam_regions):
                color = PRED_PFAM_COLOR
            else:
                color = PRED_OTHER_COLOR
            ax.add_patch(Rectangle((h["beg"], y - row_h / 2), h["end"] - h["beg"] + 1, row_h,
                                   facecolor="none", edgecolor=color,
                                   linewidth=1.4, linestyle="--"))
            ax.text((h["beg"] + h["end"]) / 2, y + row_h / 2 + 0.05,
                    f"{h['prob']:.2f}", ha="center", va="bottom", fontsize=9, color=color)

    ax.set_yticks(range(len(sids)))
    ax.set_yticklabels(sids, fontsize=11)
    ax.set_ylim(-0.8, len(sids) - 0.2)
    ax.invert_yaxis()
    ax.set_xlim(0, max_x * 1.02)
    ax.set_xlabel("Position (aa)", fontsize=14)
    ax.tick_params(axis="x", labelsize=12)
    ax.set_title(title, fontsize=16)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    legend_handles = [
        Rectangle((0, 0), 1, 1, facecolor=ASM_COLOR, edgecolor=ASM_EDGE, label="True ASM"),
        Rectangle((0, 0), 1, 1, facecolor=PFAM_COLOR, edgecolor=PFAM_EDGE,
                  label="PFAM (NLR / effector)"),
        Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=ASM_COLOR,
                  linestyle="--", linewidth=1.4,
                  label=f"Pred (ASM match >={int(OVERLAP_FRAC * 100)}%)"),
        Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=PRED_PFAM_COLOR,
                  linestyle="--", linewidth=1.4, label="Pred (same protein as PFAM)"),
        Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=PRED_OTHER_COLOR,
                  linestyle="--", linewidth=1.4, label="Pred (other)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=11, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_threshold(threshold, hits, asm, pfam, lengths, out_dir, proteome_name, page_size):
    by_sid = {}
    for h in hits:
        by_sid.setdefault(h["sid"], []).append(h)

    # Proteins carrying a reference are plotted first.
    sids = sorted(by_sid, key=lambda s: (s not in asm and s not in pfam, s))
    if not sids:
        print(f"[{threshold}] no predictions -- skipping")
        return

    pages = [sids[i:i + page_size] for i in range(0, len(sids), page_size)]
    for idx, page in enumerate(pages, 1):
        suffix = f"_p{idx:02d}" if len(pages) > 1 else ""
        out_path = out_dir / f"scan_plot_{threshold}{suffix}.png"
        page_note = f", page {idx}/{len(pages)}" if len(pages) > 1 else ""
        title = (f"ASM predictions vs reference -- {proteome_name} "
                 f"(threshold {threshold}{page_note})")
        plot_panel(page, by_sid, asm, pfam, lengths, out_path, title)
    print(f"[{threshold}] {len(sids)} proteins -> {len(pages)} page(s) in {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("proteome_dir",
                    help="folder containing asm_reference.tsv, pfam_references.tsv, "
                         "scan_results/, and the proteome fasta")
    ap.add_argument("--out-dir", default=None,
                    help="output directory (default: <proteome_dir>/scan_plots)")
    ap.add_argument("--page-size", type=int, default=80,
                    help="max proteins per figure; extra proteins paginate into _p02, _p03, ...")
    ap.add_argument("--only-thresholds", nargs="*", default=None,
                    help="restrict to these threshold subfolder names (e.g. 090 095)")
    args = ap.parse_args()

    pdir = Path(args.proteome_dir)
    asm = load_asm_refs(pdir / "asm_reference.tsv")
    pfam = load_pfam_refs(pdir / "pfam_references.tsv")
    lengths = load_protein_lengths(find_proteome_fasta(pdir))

    out_dir = Path(args.out_dir) if args.out_dir else (pdir / "scan_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    for th_dir in sorted((pdir / "scan_results").iterdir()):
        if not th_dir.is_dir():
            continue
        if args.only_thresholds and th_dir.name not in args.only_thresholds:
            continue
        jsons = list(th_dir.glob("*.json"))
        if not jsons:
            continue
        hits = load_scan(jsons[0])
        plot_threshold(th_dir.name, hits, asm, pfam, lengths,
                       out_dir, pdir.name, args.page_size)


if __name__ == "__main__":
    main()
