#!/usr/bin/env python3
"""Build per-organism folders under proteomes/ for the top-7 organisms by ASM
count in slider_data/journal.pcbi.1010787.s002.csv.

For each organism we write:
  - asm_reference.tsv : seq_id <tab> asm_beg <tab> asm_end  (protein coords from journal)
  - proteome_<folder>.fasta : protein FASTA downloaded from NCBI Datasets

The PFAM file (pfam_reference.tsv) is produced separately by running
InterProScan on the proteome FASTA (see scripts/tools/interproscan_job.slurm).
"""
from __future__ import annotations

import csv
import io
import sys
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SLIDER = ROOT / "slider_data"
JOURNAL = SLIDER / "journal.pcbi.1010787.s002.csv"
OUT = ROOT / "proteomes"

# (assembly_accession, folder_name, tax_name in journal)
TOP = [
    ("GCA_002287475.2", "Pyrrhoderma_noxium",            "Pyrrhoderma noxium"),
    ("GCA_000697645.1", "Galerina_marginata_CBS_339_88", "Galerina marginata CBS 339.88"),
    ("GCA_000827265.1", "Gymnopus_luxurians_FD-317_M1",  "Gymnopus luxurians FD-317 M1"),
    ("GCA_002938375.1", "Psilocybe_cyanescens",          "Psilocybe cyanescens"),
    ("GCA_000488995.1", "Moniliophthora_roreri_MCA_2997","Moniliophthora roreri MCA 2997"),
    ("GCA_000827195.1", "Laccaria_amethystina_LaAM-08-1","Laccaria amethystina LaAM-08-1"),
    ("GCA_000827485.1", "Amanita_muscaria_Koide_BX008",  "Amanita muscaria Koide BX008"),
]


def asms_by_tax() -> dict[str, list[tuple[str, str, str]]]:
    """seq_id, asm_beg, asm_end (protein coords) per tax_name. Rows where
    old_ntm == 'PFD-LIKE' or asm_beg != 'n/a' are considered ASM hits."""
    out: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    with JOURNAL.open() as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["old_ntm"] != "PFD-LIKE" and row["asm_beg"] == "n/a":
                continue
            beg = row["asm_beg"] if row["asm_beg"] != "n/a" else "-"
            end = row["asm_end"] if row["asm_end"] != "n/a" else "-"
            out[row["tax_name"]].append((row["seq_id"], beg, end))
    return out


def write_asm_tsv(path: Path, rows: list[tuple[str, str, str]]) -> None:
    with path.open("w") as fh:
        fh.write("seq_id\tasm_beg\tasm_end\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")


def download_proteome(gca: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  proteome exists: {dest.name}")
        return
    url = (
        f"https://api.ncbi.nlm.nih.gov/datasets/v2alpha/genome/accession/{gca}"
        "/download?include_annotation_type=PROT_FASTA"
    )
    print(f"  downloading {gca} -> {dest.name}")
    with urllib.request.urlopen(url, timeout=600) as resp:
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        fasta_member = next((n for n in zf.namelist() if n.endswith(".faa")), None)
        if fasta_member is None:
            raise RuntimeError(f"no protein FASTA in archive for {gca}: {zf.namelist()}")
        with zf.open(fasta_member) as src, dest.open("wb") as out:
            out.write(src.read())


def main() -> None:
    OUT.mkdir(exist_ok=True)
    asms = asms_by_tax()

    for gca, folder, tax in TOP:
        org_dir = OUT / folder
        org_dir.mkdir(exist_ok=True)
        print(f"\n[{gca}] {folder}  ({tax})")

        rows = asms.get(tax, [])
        write_asm_tsv(org_dir / "asm_reference.tsv", rows)
        print(f"  wrote asm_reference.tsv ({len(rows)} rows)")

        fasta_path = org_dir / f"proteome_{folder}.fasta"
        try:
            download_proteome(gca, fasta_path)
        except Exception as e:
            print(f"  WARN: proteome download failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
