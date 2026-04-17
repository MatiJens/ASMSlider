import argparse
import os
from pathlib import Path

from modules.embeddings_generator import EmbeddingsGenerator
from utils.sequence_loader import save_embeddings


def create_parser():
    parser = argparse.ArgumentParser(
        description="Generate embeddings from FASTA sequences."
    )
    parser.add_argument(
        "--input-path", required=True, help="Path to input .fasta file."
    )
    parser.add_argument(
        "--output-path", required=True, help="Directory where results will be saved."
    )
    parser.add_argument(
        "--pooling",
        default="mean",
        choices=["mean", "max"],
        help="Type of pooling: 'mean' or 'max' (default: mean).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1024, help="Sequences per batch."
    )
    parser.add_argument(
        "--encoder-checkpoint",
        default=None,
        help="Path to trained autoencoder checkpoint. If provided, embeddings are reduced via encoder.",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=128,
        help="Encoder latent dimension (default: 128). Only used with --encoder-checkpoint.",
    )
    return parser


def main():
    args = create_parser().parse_args()

    embeddings = EmbeddingsGenerator.generate_from_file(
        args.input_path, args.pooling, args.batch_size
    )

    if args.encoder_checkpoint:
        from modules.embeddings_encoder import EmbeddingsEncoder

        encoder = EmbeddingsEncoder(args.encoder_checkpoint, latent_dim=args.latent_dim)
        embeddings = encoder.encode(embeddings)

    output_file = os.path.join(args.output_path, Path(args.input_path).stem)
    save_embeddings(embeddings, output_file)


if __name__ == "__main__":
    main()
