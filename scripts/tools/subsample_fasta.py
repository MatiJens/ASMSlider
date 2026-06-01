#!/usr/bin/env python3
"""Subsample a FASTA file to n% of its sequences."""

import argparse
import random
from pathlib import Path

from Bio import SeqIO


def main():
    parser = argparse.ArgumentParser(
        description="Subsample a FASTA file to n%% of its sequences."
    )
    parser.add_argument("input", help="Input FASTA file")
    parser.add_argument("percent", type=float, help="Percentage of sequences to keep (0-100)")
    parser.add_argument("-o", "--output", help="Output file (default: <input>_<n>pct.fa)")
    parser.add_argument("-s", "--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    if not (0 < args.percent <= 100):
        parser.error("percent must be between 0 and 100 (exclusive 0)")

    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        pct_str = str(int(args.percent)) if args.percent == int(args.percent) else str(args.percent)
        output_path = input_path.with_stem(f"{input_path.stem}_{pct_str}pct")

    records = list(SeqIO.parse(args.input, "fasta"))
    total = len(records)
    k = max(1, round(total * args.percent / 100))

    rng = random.Random(args.seed)
    sampled = rng.sample(records, k)

    SeqIO.write(sampled, output_path, "fasta")
    print(f"{total} sequences in input -> {k} sampled ({args.percent}%) -> {output_path}")


if __name__ == "__main__":
    main()
