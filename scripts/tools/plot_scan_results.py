#!/usr/bin/env python3
"""Plot per-threshold ASM scan predictions vs reference ASM motifs and PFAM domains.

For each threshold under <proteome_dir>/scan_results/<thr>/, draws one figure showing
every protein with at least one predicted ASM motif. Each protein row contains:
  - a grey background bar spanning the protein length
  - predicted ASM motifs (dashed): green if they overlap a true ASM reference (>=45%),
    red otherwise; the prediction probability is annotated
  - true ASM motifs from asm_reference.tsv (filled box) -- skipped when positions are absent
  - PFAM domains from pfam_references.tsv (filled box)
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


OVERLAP_FRAC = 0.45
PFAM_NEAR_AA = 35


def load_refs(path):
    refs = {}
    if not path.exists():
        return refs
    with open(path) as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if not row:
                continue
            sid = row[0]
            beg_s = row[1].strip() if len(row) > 1 else ""
            end_s = row[2].strip() if len(row) > 2 else ""
            if beg_s in ("", "NA", "None", "-") or end_s in ("", "NA", "None", "-"):
                refs.setdefault(sid, []).append((None, None))
            else:
                refs.setdefault(sid, []).append((int(beg_s), int(end_s)))
    return refs


def load_scan(path):
    with open(path) as f:
        data = json.load(f)
    hits = []
    for h in data:
        b, e = h["location"].split("-")
        hits.append({
            "sid": h["protein"],
            "beg": int(b),
            "end": int(e),
            "prob": float(h.get("probability", 0.0)),
        })
    return hits


def load_protein_lengths(fasta_path):
    lengths = {}
    sid, n = None, 0
    with open(fasta_path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if sid is not None:
                    lengths[sid] = n
                sid = line[1:].split()[0]
                n = 0
            else:
                n += len(line)
        if sid is not None:
            lengths[sid] = n
    return lengths


def overlap_len(a_beg, a_end, b_beg, b_end):
    return max(0, min(a_end, b_end) - max(a_beg, b_beg) + 1)


def predicted_matches_asm(hit, asm_regions):
    """True if hit covers >=OVERLAP_FRAC of any positioned asm ref on the same protein."""
    for ab, ae in asm_regions:
        if ab is None:
            continue
        alen = ae - ab + 1
        if alen > 0 and overlap_len(hit["beg"], hit["end"], ab, ae) / alen >= OVERLAP_FRAC:
            return True
    return False


def predicted_near_pfam(hit, pfam_regions, window=PFAM_NEAR_AA):
    """True if hit lies within `window` AA of any positioned PFAM region (or overlaps it)."""
    for pb, pe in pfam_regions:
        if pb is None:
            continue
        gap = max(hit["beg"] - pe, pb - hit["end"], 0)
        if gap <= window:
            return True
    return False


def plot_panel(sids, by_sid, asm, pfam, lengths, out_path, title):
    row_h = 0.5
    fig_h = max(3.0, 0.45 * len(sids) + 1.5)
    fig, ax = plt.subplots(figsize=(16, fig_h))

    max_x = 0
    for i, sid in enumerate(sids):
        y = i
        plen = lengths.get(sid, 0)
        if plen <= 0:
            # fallback: span of features
            ends = [h["end"] for h in by_sid[sid]]
            ends += [e for _, e in asm.get(sid, []) if e is not None]
            ends += [e for _, e in pfam.get(sid, []) if e is not None]
            plen = max(ends) if ends else 1
        max_x = max(max_x, plen)

        # protein background bar
        ax.add_patch(Rectangle((0, y - row_h / 2), plen, row_h,
                               facecolor="#ececec", edgecolor="#bbbbbb", linewidth=0.5))

        # PFAM (drawn first so motifs overlay)
        for pb, pe in pfam.get(sid, []):
            if pb is None:
                continue
            ax.add_patch(Rectangle((pb, y - row_h / 2), pe - pb + 1, row_h,
                                   facecolor="#f4a261", edgecolor="#9c5a25",
                                   linewidth=0.6, alpha=0.85))
            ax.text((pb + pe) / 2, y, "PFAM", ha="center", va="center",
                    fontsize=7, color="white")

        # true ASM motifs (positioned only)
        for ab, ae in asm.get(sid, []):
            if ab is None:
                continue
            ax.add_patch(Rectangle((ab, y - row_h / 2), ae - ab + 1, row_h,
                                   facecolor="#2a9d8f", edgecolor="#13524a",
                                   linewidth=0.6))
            ax.text((ab + ae) / 2, y, "ASM", ha="center", va="center",
                    fontsize=7, color="white")

        # predicted motifs: green if overlaps a true ASM, blue if near PFAM, else red
        asm_regions = asm.get(sid, [])
        pfam_regions = pfam.get(sid, [])
        for h in by_sid[sid]:
            if predicted_matches_asm(h, asm_regions):
                color = "#2a9d8f"
            elif predicted_near_pfam(h, pfam_regions):
                color = "#1d6fb8"
            else:
                color = "#e63946"
            ax.add_patch(Rectangle((h["beg"], y - row_h / 2),
                                   h["end"] - h["beg"] + 1, row_h,
                                   facecolor="none", edgecolor=color,
                                   linewidth=1.4, linestyle="--"))
            ax.text((h["beg"] + h["end"]) / 2, y + row_h / 2 + 0.05,
                    f"{h['prob']:.2f}", ha="center", va="bottom",
                    fontsize=6, color=color)

    ax.set_yticks(range(len(sids)))
    ax.set_yticklabels(sids, fontsize=8)
    ax.set_ylim(-0.8, len(sids) - 0.2)
    ax.invert_yaxis()
    ax.set_xlim(0, max_x * 1.02)
    ax.set_xlabel("Position (aa)")
    ax.set_title(title)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    legend_handles = [
        Rectangle((0, 0), 1, 1, facecolor="#2a9d8f", edgecolor="#13524a", label="True ASM"),
        Rectangle((0, 0), 1, 1, facecolor="#f4a261", edgecolor="#9c5a25", label="PFAM"),
        Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="#2a9d8f",
                  linestyle="--", linewidth=1.4,
                  label=f"Pred (ASM match >={int(OVERLAP_FRAC*100)}%)"),
        Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="#1d6fb8",
                  linestyle="--", linewidth=1.4,
                  label=f"Pred (near PFAM +/-{PFAM_NEAR_AA} aa)"),
        Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="#e63946",
                  linestyle="--", linewidth=1.4, label="Pred (other)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_threshold(threshold, hits, asm, pfam, lengths, out_dir, proteome_name, page_size):
    by_sid = {}
    for h in hits:
        by_sid.setdefault(h["sid"], []).append(h)

    def has_ref(sid):
        return sid in asm or sid in pfam

    sids = sorted(by_sid.keys(), key=lambda s: (not has_ref(s), s))
    if not sids:
        print(f"[{threshold}] no predictions -- skipping")
        return

    pages = [sids[i:i + page_size] for i in range(0, len(sids), page_size)]
    for idx, page in enumerate(pages, 1):
        suffix = f"_p{idx:02d}" if len(pages) > 1 else ""
        out_path = out_dir / f"scan_plot_{threshold}{suffix}.png"
        title = (f"ASM predictions vs reference -- {proteome_name} "
                 f"(threshold {threshold}"
                 + (f", page {idx}/{len(pages)}" if len(pages) > 1 else "")
                 + ")")
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
    asm = load_refs(pdir / "asm_reference.tsv")
    pfam = load_refs(pdir / "pfam_references.tsv")

    fastas = list(pdir.glob("*.fasta"))
    if not fastas:
        raise SystemExit(f"No .fasta found in {pdir}")
    lengths = load_protein_lengths(fastas[0])

    out_dir = Path(args.out_dir) if args.out_dir else (pdir / "scan_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    scan_root = pdir / "scan_results"
    for th_dir in sorted(scan_root.iterdir()):
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
