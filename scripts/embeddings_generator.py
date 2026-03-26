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
    format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


class EmbeddingsGenerator:
    def __init__(
        self,
        batch_size: int = 2048,
        pooling_type: str = "per_residue",
        max_length: int = 2048,
        model_name: str = "esmc_600m",
    ):
        self.batch_size = batch_size
        self.pooling_type = pooling_type
        if max_length > 2048:
            logger.error("Max length error cannot be bigger than 2048.")
            sys.exit(1)
        self.max_length = max_length
        if model_name not in (
            "esmc_300m",
            "esmc_600m",
        ):
            logger.error(
                f"{model_name} doesn't exist. Available esmc_300m or esmc_600m."
            )
            sys.exit(1)
        self.model_name = model_name

    def _iter_json_config(self, input_path):
        try:
            with open(input_path, "r") as f:
                config = json.load(f)
        except FileNotFoundError:
            logger.error(f"JSON config file not found: {input_path}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {input_path}: {e}")
            sys.exit(1)

        if not isinstance(config, dict) or not all(
            isinstance(v, dict) for v in config.values()
        ):
            logger.error(
                "JSON config must be a dict of dicts {split_name : {class: filepath}})."
            )
            sys.exit(1)

        for split_name, classes in config.items():
            for class_name, filepath in classes.items():
                if not os.path.isfile(filepath):
                    logger.error(
                        f"[{split_name}/{class_name}] File not found: {filepath}"
                    )
                    sys.exit(1)
                yield split_name, class_name, filepath

    def _load_sequences_from_fasta(self, filepath):
        seq_list = []
        try:
            for record in SeqIO.parse(filepath, "fasta"):
                seq_list.append(
                    {
                        "id": record.id,
                        "seq": str(record.seq),
                    }
                )
        except Exception as e:
            logger.error(f"Error while reading {filepath}: {e}")
            sys.exit(1)
        if len(seq_list) == 0:
            logger.error(f"No sequences found in {filepath}")
            sys.exit(1)

        seq_list.sort(key=lambda x: len(x["seq"]), reverse=True)
        logger.info(f"Found {len(seq_list)} sequences.")
        return seq_list

    def _generate_emb(self, model, device, sequences):
        try:
            with torch.no_grad():
                input_ids = model._tokenize(sequences).to(device)
                output = model(input_ids)
                pad_idx = model.tokenizer.pad_token_id
                mask = input_ids != pad_idx
                del input_ids
                return [emb[m] for emb, m in zip(output.embeddings, mask)]
        except Exception as e:
            logger.error(
                f"Error while generating embedding: {e}.\nTry reducing batch_size."
            )
            sys.exit(1)

    def _pooling(self, embedding):
        match self.pooling_type:
            case "mean_pooling":
                return embedding.mean(dim=0).to(dtype=torch.float16).cpu()
            case "max_pooling":
                return embedding.max(dim=0).values.to(dtype=torch.float16).cpu()
            case _:
                return embedding.to(dtype=torch.float16).cpu()

    def _process_batch(self, model, device, seq_list, h5f):
        for i in tqdm(range(0, len(seq_list), self.batch_size)):
            batch = seq_list[i : i + self.batch_size]
            seqs = [item["seq"][: self.max_length] for item in batch]
            ids = [item["id"] for item in batch]

            embeddings = self._generate_emb(model, device, seqs)

            for seq_id, emb in zip(ids, embeddings):
                if seq_id in h5f:
                    logger.warning(f"Duplicate ID: {seq_id}, skipping.")
                    continue

                emb = self._pooling(emb)
                h5f.create_dataset(
                    seq_id,
                    data=emb.numpy(),
                    dtype=np.float16,
                    compression="gzip",
                    compression_opts=9,
                    shuffle=True,
                )
            del embeddings

    def _load_model(self):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = ESMC.from_pretrained(self.model_name, device=device)
            model.eval()
            return device, model
        except Exception as e:
            logger.error(f"Error while loading model: {e}")
            sys.exit(1)

    def generate_from_json(self, input_path, output_path):
        """Method that load files from json sequences, generates embeddings and save them to corresponding *.h5 files."""
        device, model = self._load_model()

        for split_name, class_name, filepath in self._iter_json_config(input_path):
            logger.info(f"[{split_name}/{class_name}] Processing {Path(filepath).stem}")
            seq_list = self._load_sequences_from_fasta(filepath)

            output_file = os.path.join(
                output_path, class_name, Path(filepath).stem + ".h5"
            )
            os.makedirs(Path(output_file).parent, exist_ok=True)

            with h5py.File(output_file, "w") as h5f:
                self._process_batch(model, device, seq_list, h5f)
                logger.info(f"Saved to {output_file}")

        logger.info(f"All embeddings generated and saved under {output_path}")

    def generate_from_file(self, input_file, output_file):
        """Method that load one file, generate embeddings and save them to *.h5 file."""
        device, model = self._load_model()

        seq_list = self._load_sequences_from_fasta(input_file)
        os.makedirs(Path(output_file).parent, exist_ok=True)
        with h5py.File(output_file, "w") as h5f:
            self._process_batch(model, device, seq_list, h5f)
            logger.info(f"Saved to {output_file}")

    def generate_from_list(self, sequences):
        device, model = self._load_model()

        all_embeddings = []
        for i in range(0, len(sequences), self.batch_size):
            batch_seqs = [
                s[: self.max_length] for s in sequences[i : i + self.batch_size]
            ]
            raw_embeddings = self._generate_emb(model, device, batch_seqs)

            for emb in raw_embeddings:
                pooled = emb.mean(dim=0).cpu()
                all_embeddings.append(pooled)

            del raw_embeddings

        return torch.stack(all_embeddings).float().numpy()


def create_parser():
    parser = argparse.ArgumentParser(
        description="Generate embeddings using ESMC model."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to JSON config or single .fasta file.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        required=True,
        help="Output directory (for JSON config) or output .h5 file path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Sequences per batch. Default 2048, reduce to 128/256 for smaller GPUs.",
    )
    parser.add_argument(
        "--pooling-type",
        type=str,
        default="per_residue",
        choices=["per_residue", "mean_pooling", "max_pooling"],
        help="Pooling strategy for embeddings.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Max sequence length (ESMC limit: 2048).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="esmc_600m",
        choices=["esmc_300m", "esmc_600m"],
        help="ESMC model variant.",
    )
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    generator = EmbeddingsGenerator(
        batch_size=args.batch_size,
        pooling_type=args.pooling_type,
        max_length=args.max_length,
        model_name=args.model_name,
    )

    if args.input_path.endswith(".json"):
        generator.generate_from_json(args.input_path, args.output_path)
    else:
        generator.generate_from_file(args.input_path, args.output_path)


if __name__ == "__main__":
    main()
