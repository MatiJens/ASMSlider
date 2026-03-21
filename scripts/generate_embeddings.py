import argparse
import glob
import logging
import os
import sys
from pathlib import Path

import torch
from Bio import SeqIO
from esm.models.esmc import ESMC
from tqdm import tqdm


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


class EmbeddingsGenerator:
    def __init__(
        self,
        input_dir: list[str],
        output_path: str,
        batch_size: int = 2048,
        save_every: int = 1000,
        pooling_type: str = "per_residue",
    ):
        self.input_dir = input_dir
        self.output_path = output_path
        self.batch_size = batch_size
        self.save_every = save_every
        self.pooling_type = pooling_type

        self.embeddings_buffer: dict[str, torch.Tensor] = {}
        self.shard_counter: int = 0

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

    def _save_shard(self):
        shard_path = os.path.join(
            self.output_path,
            f"embedding_shard_{self.shard_counter}.pt",
        )
        torch.save(self.embeddings_buffer, shard_path)
        self.embeddings_buffer = {}
        self.shard_counter += 1
        torch.cuda.empty_cache()

    def _merge_shards(self, file_name):
        merged = {}
        embedding_path = os.path.join(
            self.output_path,
            file_name,
        )
        for i in range(self.shard_counter):
            current_shard_path = os.path.join(
                self.output_path,
                f"embedding_shard_{i}.pt",
            )
            current_shard = torch.load(
                current_shard_path,
                map_location="cpu",
                weights_only=False,
            )
            merged.update(current_shard)
            del current_shard
            os.remove(current_shard_path)
        torch.save(merged, embedding_path)
        logger.info(f"{file_name} saved under {embedding_path}")

    def generate(self):
        """Main method that load sequences, generates embeddings and save them as shard and finally merge all shards into one file."""
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
            self.shard_counter = 0
            self.embeddings_buffer = {}

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
                            case "mean_max_pooling":
                                mean_emb = seq_embedding.mean(dim=0)
                                max_emb = seq_embedding.max(dim=0).values
                                embedding = (
                                    torch.cat([mean_emb, max_emb], dim=0)
                                    .to(dtype=torch.float16)
                                    .cpu()
                                )
                            case _:
                                embedding = seq_embedding.to(dtype=torch.float16).cpu()
                        self.embeddings_buffer[seq_id] = embedding

                    del input_ids, output, batch_embeddings
                except Exception as e:
                    msg = f"Error while generating embedding: {e}.\nTry reducing batch_size."
                    logger.error(msg)
                    sys.exit(1)

                if len(self.embeddings_buffer) >= self.save_every:
                    self._save_shard()

            if self.embeddings_buffer:
                self._save_shard()
            self._merge_shards(f"{Path(directory).stem}.pt")

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
        "--save-every",
        type=int,
        default=1000,
        help="Number of embeddings generated per one shard. The more RAM you have the larger shards can be",
    )
    parser.add_argument(
        "--pooling-type",
        type=str,
        default="per_residue",
        choices=["per_residue", "mean_pooling", "max_pooling", "mean_max_pooling"],
        help="Type of pooling. Options: per_residue, mean_pooling, max_pooling, mean_max_pooling",
    )
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    generator = EmbeddingsGenerator(
        input_dir=args.input_dir,
        output_path=args.output_path,
        batch_size=args.batch_size,
        save_every=args.save_every,
        pooling_type=args.pooling_type,
    )
    generator.generate()


if __name__ == "__main__":
    main()
