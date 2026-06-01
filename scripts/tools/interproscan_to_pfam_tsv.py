#!/usr/bin/env python3
"""Convert an InterProScan TSV output file into our pfam_reference.tsv format.

InterProScan TSV columns (no header):
  1  protein_accession
  2  md5
  3  seq_length
  4  analysis            (e.g. Pfam, ProSite, SMART)
  5  signature_accession (e.g. PF00400)
  6  signature_desc      (e.g. WD40)
  7  start               (protein coord, 1-based)
  8  end                 (protein coord, 1-based)
  9  e-value
 10  status
 11  date
 ...

We keep only rows where analysis == 'Pfam' and emit:
  seq_id <tab> pfam_beg <tab> pfam_end <tab> pfam_acc <tab> pfam_name

Usage:  interproscan_to_pfam_tsv.py <ips.tsv> <out.tsv>
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("usage: interproscan_to_pfam_tsv.py <ips.tsv> <out.tsv>")
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    n = 0
    with src.open() as fh, dst.open("w") as out:
        out.write("seq_id\tpfam_beg\tpfam_end\tpfam_acc\tpfam_name\n")
        for row in csv.reader(fh, delimiter="\t"):
            if len(row) < 8 or row[3] != "Pfam":
                continue
            out.write(f"{row[0]}\t{row[6]}\t{row[7]}\t{row[4]}\t{row[5]}\n")
            n += 1
    print(f"wrote {n} Pfam rows to {dst}")


if __name__ == "__main__":
    main()
