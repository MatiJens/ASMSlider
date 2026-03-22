import argparse
import glob
import logging
import os
import sys
from pathlib import Path
import h5py
import numpy as np

import torch
from Bio import SeqIO
from esm.models.esmc import ESMC
from tqdm import tqdm


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


class EmbeddingsGenerator:
    def __init__(
        self,
        input_dir: list[str],
        output_path: str,
        batch_size: int = 2048,
        pooling_type: str = "per_residue",
    ):
        self.input_dir = input_dir
        self.output_path = output_path
        self.batch_size = batch_size
        self.pooling_type = pooling_type

        os.makedirs(self.output_path, exist_ok=True)

    def _load_sequence_from_dir(self, directory):
        seq_list = []
        files = glob.glob(os.path.join(directory, "*.fasta")) + glob.glob(
            os.path.join(directory, "*.fa")
        )

        for filepath in files:
            try:
                for record in SeqIO.parse(filepath, "fasta"):
                    seq_list.append(
                        {
                            "id": record.id,
                            "seq": str(record.seq),
                            "len": len(record.seq),
                        }
                    )
            except Exception as e:
                msg = f"Error while reading {filepath}: {e}"
                logger.error(msg)
                sys.exit(1)
        return seq_list

    def generate(self):
        """Main method that load sequences, generates embeddings and save them to *.h5 file."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "esmc_600m"
        max_length = 1024

        try:
            model = ESMC.from_pretrained(model_name, device=device)
        except Exception as e:
            msg = f"Error while loading model: {e}"
            logger.error(msg)
            sys.exit(1)

        for directory in self.input_dir:
            output_file = os.path.join(self.output_path, f"{Path(directory).stem}.h5")
            try:
                seq_list = self._load_sequence_from_dir(directory)
            except Exception as e:
                msg = f"Error while loading input files: {e}"
                logger.error(msg)
                sys.exit(1)

            if len(seq_list) == 0:
                msg = "Files with sequences are empty."
                logger.error(msg)
                sys.exit(1)

            seq_list.sort(key=lambda x: x["len"], reverse=True)

            logger.info(f"Found {len(seq_list)} sequences.")
            model.eval()

            with h5py.File(output_file, "w") as h5f:
                for i in tqdm(
                    range(0, len(seq_list), self.batch_size),
                    desc=Path(directory).stem,
                ):
                    batch = seq_list[i : i + self.batch_size]
                    seqs = [item["seq"][:max_length] for item in batch]
                    ids = [item["id"] for item in batch]

                    try:
                        with torch.no_grad():
                            input_ids = model._tokenize(seqs).to(device)
                            output = model(input_ids)
                            batch_embeddings = output.embeddings

                        for j, seq_id in enumerate(ids):
                            if seq_id in h5f:
                                logger.warning(f"Duplicate ID: {seq_id}, skipping.")
                                continue

                            pad_idx = model.tokenizer.pad_token_id
                            valid_mask = input_ids[j] != pad_idx
                            seq_embedding = batch_embeddings[j][valid_mask]
                            match self.pooling_type:
                                case "mean_pooling":
                                    embedding = (
                                        seq_embedding.mean(dim=0)
                                        .to(dtype=torch.float16)
                                        .cpu()
                                    )
                                case "max_pooling":
                                    embedding = (
                                        seq_embedding.max(dim=0)
                                        .values.to(dtype=torch.float16)
                                        .cpu()
                                    )
                                case _:
                                    embedding = seq_embedding.to(
                                        dtype=torch.float16
                                    ).cpu()
                            h5f.create_dataset(
                                seq_id,
                                data=embedding.numpy(),
                                dtype=np.float16,
                                compression="gzip",
                            )

                        del input_ids, output, batch_embeddings
                    except Exception as e:
                        msg = f"Error while generating embedding: {e}.\nTry reducing batch_size."
                        logger.error(msg)
                        sys.exit(1)

        logger.info(
            f"Done - all embedding generated and saved under {self.output_path}"
        )


def create_parser():
    parser = argparse.ArgumentParser(
        description="Script that generate embeddings by ESMC 600M model."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        nargs="+",
        help="Path to directories with *.fa or *.fasta files.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        required=True,
        help="Path where embeddings should be saved.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Max size of sequences per batch. For WCSS 2048 is okay, for smaller PCs you should set this to 128/256.",
    )
    parser.add_argument(
        "--pooling-type",
        type=str,
        default="per_residue",
        choices=["per_residue", "mean_pooling", "max_pooling"],
        help="Type of pooling. Options: per_residue, mean_pooling, max_pooling",
    )
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    generator = EmbeddingsGenerator(
        input_dir=args.input_dir,
        output_path=args.output_path,
        batch_size=args.batch_size,
        pooling_type=args.pooling_type,
    )
    generator.generate()


if __name__ == "__main__":
    main()
