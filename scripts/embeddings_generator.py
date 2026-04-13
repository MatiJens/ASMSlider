import argparse
import os
from pathlib import Path

import numpy as np
import torch

torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_cudnn_sdp(False)
from esm.models.esmc import ESMC

from sequence_loader import SequenceLoader


class EmbeddingsGenerator:
    _model = None
    _device = None

    @classmethod
    def _ensure_model(cls):
        if cls._model is None:
            cls._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cls._model = ESMC.from_pretrained("esmc_600m", device=cls._device).eval()

    @classmethod
    @torch.no_grad()
    def _process_batch(cls, seq_batch, pooling):
        input_ids = cls._model._tokenize(seq_batch).to(cls._device)
        with torch.autocast(device_type=cls._device.type, dtype=torch.bfloat16):
            output = cls._model(input_ids)
        pad_idx = cls._model.tokenizer.pad_token_id
        mask = input_ids != pad_idx
        mask[:, 0] = False
        del input_ids

        if pooling == "max":
            h = output.embeddings.float()
            h = h.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            return h.max(dim=1).values.cpu().numpy()
        else:
            h = output.embeddings.float()
            m = mask.float().unsqueeze(-1)
            summed = (h * m).sum(dim=1)
            counts = m.sum(dim=1).clamp(min=1)
            return (summed / counts).cpu().numpy()

    @classmethod
    def _generate_emb(cls, sequences, pooling, batch_size):
        all_embeddings = []

        for i in range(0, len(sequences), batch_size):
            seqs = [
                s[:2048] for s in sequences[i : i + batch_size]
            ]  # 2048 is maximum seq length for ESMC model
            all_embeddings.append(cls._process_batch(seqs, pooling))

        return np.concatenate(all_embeddings, axis=0)

    @classmethod
    def generate_from_file(cls, input_file, output_path, pooling, batch_size):
        """Load one FASTA file, generate embeddings and save as .npy."""
        cls._ensure_model()
        sequences = [item["seq"] for item in SequenceLoader.load_fasta(input_file)]
        embeddings = cls._generate_emb(sequences, pooling, batch_size)
        output_file = os.path.join(output_path, Path(input_file).stem)
        SequenceLoader.save_embeddings(embeddings, output_file)

    @classmethod
    def generate_from_list(cls, sequences, pooling, batch_size):
        """Generate embeddings from list."""
        cls._ensure_model()
        return cls._generate_emb(sequences, pooling, batch_size)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path", required=True, help="Path to input .fasta file."
    )
    parser.add_argument(
        "--output-path", required=True, help="Directory where results will be saved."
    )
    parser.add_argument("--pooling", default="mean", help="Type of pooling (mean/max).")
    parser.add_argument(
        "--batch-size", type=int, default=1024, help="Sequences per batch."
    )
    return parser


def main():
    args = create_parser().parse_args()
    EmbeddingsGenerator.generate_from_file(
        args.input_path, args.output_path, args.pooling, args.batch_size
    )


if __name__ == "__main__":
    main()
