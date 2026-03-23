import argparse
import json
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
        input_path: list[str],
        output_path: str,
        batch_size: int = 2048,
        pooling_type: str = "per_residue",
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.batch_size = batch_size
        self.pooling_type = pooling_type

        os.makedirs(self.output_path, exist_ok=True)

    def _load_json_config(self):
        try:
            with open(self.input_path, "r") as f:
                config = json.load(f)
        except FileNotFoundError:
            logger.error(f"JSON config file not found: {self.input_path}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {self.input_path}: {e}")
            sys.exit(1)

        if not isinstance(config, dict):
            logger.error("JSON config must be a top-level dictionary.")
            sys.exit(1)

        return config

    def _load_sequences_from_fasta(self, filepath):
        seq_list = []
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
            logger.error(f"Error while reading {filepath}: {e}")
            sys.exit(1)
        return seq_list

    def _process_batch(self, model, device, seq_list, max_length, h5f, filename):
        for i in tqdm(range(0, len(seq_list), self.batch_size), desc=filename):
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
                                seq_embedding.mean(dim=0).to(dtype=torch.float16).cpu()
                            )
                        case "max_pooling":
                            embedding = (
                                seq_embedding.max(dim=0)
                                .values.to(dtype=torch.float16)
                                .cpu()
                            )
                        case _:
                            embedding = seq_embedding.to(dtype=torch.float16).cpu()

                    h5f.create_dataset(
                        seq_id,
                        data=embedding.numpy(),
                        dtype=np.float16,
                        compression="gzip",
                        compression_opts=9,
                        shuffle=True,
                    )

                del input_ids, output, batch_embeddings
            except Exception as e:
                logger.error(
                    f"Error while generating embedding: {e}.\nTry reducing batch_size."
                )
                sys.exit(1)

    def generate(self):
        """Main method that load sequences, generates embeddings and save them to *.h5 file."""
        config = self._load_json_config()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "esmc_600m"
        max_length = 1024

        try:
            model = ESMC.from_pretrained(model_name, device=device)
        except Exception as e:
            logger.error(f"Error while loading model: {e}")
            sys.exit(1)

        model.eval()

        for split_name, classes in config.items():
            if not isinstance(classes, dict):
                logger.warning(
                    f"Skipping '{split_name}': expected a dict of class->filepath, "
                    f"got {type(classes).__name__}."
                )
                continue

            for class_name, filepath in classes.items():
                if not os.path.isfile(filepath):
                    logger.error(
                        f"[{split_name}/{class_name}] File not found: {filepath}"
                    )
                    sys.exit(1)

                class_dir = os.path.join(self.output_path, class_name)
                os.makedirs(class_dir, exist_ok=True)
                output_file = os.path.join(class_dir, f"{Path(filepath).stem}.h5")

                if os.path.exists(output_file):
                    logger.info(
                        f"[{split_name}/{class_name}] Output already exists, skipping: {output_file}"
                    )
                    continue

                logger.info(
                    f"[{split_name}/{class_name}] Processing {Path(filepath).stem}"
                )

                seq_list = self._load_sequences_from_fasta(filepath)

                if len(seq_list) == 0:
                    logger.error(
                        f"[{split_name}/{class_name}] No sequences found in {filepath}"
                    )
                    sys.exit(1)

                seq_list.sort(key=lambda x: x["len"], reverse=True)
                logger.info(f"  Found {len(seq_list)} sequences.")

                with h5py.File(output_file, "w") as h5f:
                    self._process_batch(
                        model,
                        device,
                        seq_list,
                        max_length,
                        h5f,
                        Path(filepath).stem,
                    )

                logger.info(f"  Saved to {output_file}")

        logger.info(
            f"Done - all embeddings generated and saved under {self.output_path}"
        )


def create_parser():
    parser = argparse.ArgumentParser(
        description="Script that generate embeddings by ESMC 600M model."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="""Path to json file with similar structure:
        {
            "train_1": {"positive": "/mnt/lustre/data/positive/bass_ctm_motif_trn1.fa", "negative": "/mnt/lustre/data/negative/PB40_1z20_clu50_trn1.fa"},
            "val_1":   {"positive": "/mnt/lustre/data/positive/bass_ctm_motif_val1.fa",   "negative": "/mnt/lustre/data/negative/PB40_1z20_clu50_val1.fa"},

            "train_2": {"positive": "/mnt/lustre/data/positive/bass_ctm_motif_trn2.fa", "negative": "/mnt/lustre/data/negative/PB40_1z20_clu50_trn2.fa"},
            "val_2":   {"positive": "/mnt/lustre/data/positive/bass_ctm_motif_val2.fa",   "negative": "/mnt/lustre/data/negative/PB40_1z20_clu50_val2.fa"},

            "train_3": {"positive": "/mnt/lustre/data/positive/bass_ctm_motif_trn3.fa", "negative": "/mnt/lustre/data/negative/PB40_1z20_clu50_trn3.fa"},
            "val_3":   {"positive": "/mnt/lustre/data/positive/bass_ctm_motif_val3.fa",   "negative": "/mnt/lustre/data/negative/PB40_1z20_clu50_val3.fa"},

            "train_4": {"positive": "/mnt/lustre/data/positive/bass_ctm_motif_trn4.fa", "negative": "/mnt/lustre/data/negative/PB40_1z20_clu50_trn4.fa"},
            "val_4":   {"positive": "/mnt/lustre/data/positive/bass_ctm_motif_val4.fa",   "negative": "/mnt/lustre/data/negative/PB40_1z20_clu50_val4.fa"},

            "train_5": {"positive": "/mnt/lustre/data/positive/bass_ctm_motif_trn5.fa", "negative": "/mnt/lustre/data/negative/PB40_1z20_clu50_trn5.fa"},
            "val_5":   {"positive": "/mnt/lustre/data/positive/bass_ctm_motif_val5.fa",   "negative": "/mnt/lustre/data/negative/PB40_1z20_clu50_val5.fa"},

            "train_6": {"positive": "/mnt/lustre/data/positive/bass_ctm_motif_trn6.fa", "negative": "/mnt/lustre/data/negative/PB40_1z20_clu50_trn6.fa"},
            "val_6":   {"positive": "/mnt/lustre/data/positive/bass_ctm_motif_val6.fa",   "negative": "/mnt/lustre/data/negative/PB40_1z20_clu50_val6.fa"},

            "test":  {"positive": "/mnt/lustre/data/positive/bass_ntm_motif_test.fa",  "negative": "/mnt/lustre/data/negative//PB40_1z20_clu50_test.fa"}
        }
        """,
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
        input_path=args.input_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
        pooling_type=args.pooling_type,
    )
    generator.generate()


if __name__ == "__main__":
    main()
