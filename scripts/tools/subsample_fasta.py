#!/usr/bin/env python3
"""Subsample a FASTA file to n% of its sequences."""

import argparse
import random
from pathlib import Path


def parse_fasta(path: str) -> list[tuple[str, str]]:
    sequences = []
    header = None
    seq_lines = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    sequences.append((header, "".join(seq_lines)))
                header = line
                seq_lines = []
            else:
                seq_lines.append(line)
        if header is not None:
            sequences.append((header, "".join(seq_lines)))
    return sequences


def write_fasta(sequences: list[tuple[str, str]], path: str) -> None:
    with open(path, "w") as f:
        for header, seq in sequences:
            f.write(f"{header}\n{seq}\n")


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

    sequences = parse_fasta(args.input)
    total = len(sequences)
    k = max(1, round(total * args.percent / 100))

    rng = random.Random(args.seed)
    sampled = rng.sample(sequences, k)

    write_fasta(sampled, output_path)
    print(f"{total} sequences in input -> {k} sampled ({args.percent}%) -> {output_path}")


if __name__ == "__main__":
    main()
