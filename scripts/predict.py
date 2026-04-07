import argparse
import logging

from asmfinder import ASMFinder

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


def create_parser():
    parser = argparse.ArgumentParser(description="Predict ASM regions in a FASTA file.")
    parser.add_argument(
        "--input-fasta",
        type=str,
        required=True,
        help="Path to input .fasta file.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        help="Directory where results will be saved.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix for output file names.",
    )
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    logger.info(f"Predicting: {args.input_fasta}")
    ASMFinder.predict(args.input_fasta, args.output_dir, args.prefix)
    logger.info(f"Results saved under {args.output_dir}")


if __name__ == "__main__":
    main()
