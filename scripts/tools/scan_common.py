"""Shared loaders and constants for ASM scan-result evaluation and plotting.

A "proteome directory" is expected to contain:
  - asm_reference.tsv     true ASM motifs (seq_id, beg, end; positions optional)
  - pfam_references.tsv   PFAM domains (seq_id, beg, end, acc, [name])
  - the proteome <name>.fasta
  - scan_results/<threshold>/<...>.json  predicted motifs per probability threshold
"""
import csv
import json
from pathlib import Path


# PFAM accessions used as references. NLR proteins carry the ASM in the
# N-terminus; effector domains carry it in the C-terminus.
NLR_PFAM = {"PF00931", "PF05729"}
EFFECTOR_PFAM = {
    "PF14479", "PF17111", "PF17107", "PF05057", "PF07819", "PF12697",
    "PF01048", "PF17100", "PF17109", "PF06985", "PF01734", "PF00168",
    "PF04607", "PF12770", "PF01582", "PF13676", "PF00082", "PF00069",
    "PF07714",
}
ALLOWED_PFAM = NLR_PFAM | EFFECTOR_PFAM

_MISSING = {"", "NA", "None", "-"}


def load_asm_refs(path):
    """dict[seq_id] -> list[(beg, end)]; (None, None) when positions are missing."""
    refs = {}
    path = Path(path)
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
            if beg_s in _MISSING or end_s in _MISSING:
                refs.setdefault(sid, []).append((None, None))
            else:
                refs.setdefault(sid, []).append((int(beg_s), int(end_s)))
    return refs


def load_pfam_refs(path):
    """dict[seq_id] -> list[(beg, end, acc)] for ALLOWED_PFAM accessions only.

    Expects columns: seq_id, pfam_beg, pfam_end, pfam_acc, [pfam_name].
    """
    refs = {}
    path = Path(path)
    if not path.exists():
        return refs
    with open(path) as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if len(row) < 4:
                continue
            sid, beg_s, end_s, acc = row[0], row[1].strip(), row[2].strip(), row[3].strip()
            if acc not in ALLOWED_PFAM or beg_s in _MISSING or end_s in _MISSING:
                continue
            refs.setdefault(sid, []).append((int(beg_s), int(end_s), acc))
    return refs


def load_protein_lengths(fasta_path):
    """dict[seq_id] -> sequence length (aa)."""
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


def load_scan(path):
    """list of hit dicts: {sid, beg, end, prob}."""
    with open(path) as f:
        data = json.load(f)
    hits = []
    for h in data:
        beg_s, end_s = h["location"].split("-")
        hits.append({
            "sid": h["protein"],
            "beg": int(beg_s),
            "end": int(end_s),
            "prob": float(h.get("probability", 0.0)),
        })
    return hits


def overlap_len(a_beg, a_end, b_beg, b_end):
    """Length of the overlap between two inclusive intervals (0 if disjoint)."""
    return max(0, min(a_end, b_end) - max(a_beg, b_beg) + 1)


def find_proteome_fasta(proteome_dir):
    """Return the single proteome .fasta in a proteome directory."""
    fastas = list(Path(proteome_dir).glob("*.fasta"))
    if not fastas:
        raise SystemExit(f"No .fasta found in {proteome_dir}")
    return fastas[0]
