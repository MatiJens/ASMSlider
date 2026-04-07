import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
from esm.models.esmc import ESMC
from tqdm import tqdm

from data_loader import SequenceLoader

logger = logging.getLogger(__name__)


class EmbeddingsGenerator:
    def __init__(
        self,
        batch_size: int = 2048,
        max_length: int = 2048,
        model_name: str = "esmc_600m",
    ):
        self.batch_size = batch_size
        if max_length > 2048:
            raise ValueError("max_length cannot be bigger than 2048.")
        self.max_length = max_length
        if model_name not in ("esmc_300m", "esmc_600m"):
            raise ValueError(
                f"{model_name} doesn't exist. Available: esmc_300m, esmc_600m."
            )
        self.model_name = model_name
        self._device = None
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model = ESMC.from_pretrained(self.model_name, device=self._device)
            self._model.eval()
            logger.info(f"Model {self.model_name} loaded on {self._device}")

    def _generate_emb(self, sequences):
        with torch.no_grad():
            input_ids = self._model._tokenize(sequences).to(self._device)
            output = self._model(input_ids)
            pad_idx = self._model.tokenizer.pad_token_id
            mask = input_ids != pad_idx
            del input_ids
            return [emb[m] for emb, m in zip(output.embeddings, mask)]

    def _process_batch(self, seq_list):
        self._ensure_model()
        all_keys = []
        all_embeddings = []

        for i in tqdm(range(0, len(seq_list), self.batch_size)):
            batch = seq_list[i : i + self.batch_size]
            seqs = [item["seq"][: self.max_length] for item in batch]
            ids = [item["id"] for item in batch]

            raw_embeddings = self._generate_emb(seqs)

            for seq_id, emb in zip(ids, raw_embeddings):
                if seq_id in all_keys:
                    logger.warning(f"Duplicate ID: {seq_id}, skipping.")
                    continue
                all_keys.append(seq_id)
                all_embeddings.append(
                    emb.mean(dim=0).to(dtype=torch.float16).cpu().numpy()
                )

            del raw_embeddings

        return all_keys, np.stack(all_embeddings, axis=0)

    def generate_from_file(self, input_file, output_path):
        """Load one FASTA file, generate embeddings and save as .npz."""
        seq_list = SequenceLoader.load_fasta(input_file)
        keys, embeddings = self._process_batch(seq_list)
        output_file = os.path.join(output_path, Path(input_file).stem + ".npz")
        SequenceLoader.save_embeddings(keys, embeddings, output_file)
        logger.info(f"Embeddings generated and saved under {output_path}")

    def generate_from_list(self, sequences):
        """Generate embeddings from a list of raw sequence strings. Returns np.ndarray float32."""
        self._ensure_model()
        all_embeddings = []

        for i in range(0, len(sequences), self.batch_size):
            batch_seqs = [
                s[: self.max_length] for s in sequences[i : i + self.batch_size]
            ]
            raw_embeddings = self._generate_emb(batch_seqs)

            for emb in raw_embeddings:
                all_embeddings.append(emb.mean(dim=0).to(dtype=torch.float16).cpu())

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
        help="Path to .fasta file.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        required=True,
        help="Output directory for .npz file path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Sequences per batch. Default 2048, reduce to 1024 for long sequences and to 128/256 for smaller GPUs.",
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = create_parser()
    args = parser.parse_args()

    generator = EmbeddingsGenerator(
        batch_size=args.batch_size,
        max_length=args.max_length,
        model_name=args.model_name,
    )

    generator.generate_from_file(args.input_path, args.output_path)


if __name__ == "__main__":
    main()
