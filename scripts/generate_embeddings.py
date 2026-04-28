import argparse
import os
from pathlib import Path

from modules.embeddings_generator import EmbeddingsGenerator
from utils.sequence_loader import save_embeddings


def create_parser():
    parser = argparse.ArgumentParser(description="Generate ESM-C embeddings from a FASTA file.")
    parser.add_argument("--input-path", required=True, help="Input .fasta file.")
    parser.add_argument("--output-path", required=True, help="Output directory.")
    parser.add_argument("--pooling", default="mean", choices=["mean", "max"])
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser


def main():
    args = create_parser().parse_args()
    embeddings = EmbeddingsGenerator.generate_from_file(
        args.input_path, args.pooling, args.batch_size
    )
    save_embeddings(embeddings, os.path.join(args.output_path, Path(args.input_path).stem))


if __name__ == "__main__":
    main()
