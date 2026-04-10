import os
from pathlib import Path

import torch
from esm.models.esmc import ESMC

from scripts.sequence_loader import SequenceLoader


class EmbeddingsGenerator:
    @classmethod
    def _ensure_model(cls):
        cls._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls._model = ESMC.from_pretrained("esmc_600m", device=cls._device)
        cls._model.eval()

    @classmethod
    def _process_batch(cls, seq_batch, pooling):
        with torch.no_grad():
            input_ids = cls._model._tokenize(seq_batch).to(cls._device)
            output = cls._model(input_ids)
            pad_idx = cls._model.tokenizer.pad_token_id
            mask = input_ids != pad_idx
            del input_ids
            raw_emb = [emb[m] for emb, m in zip(output.embeddings, mask)]

            if pooling == "max":
                return (
                    torch.stack([emb.max(dim=0).values for emb in raw_emb])
                    .to(dtype=torch.float16)
                    .cpu()
                )
            return (
                torch.stack([emb.mean(dim=0) for emb in raw_emb])
                .to(dtype=torch.float16)
                .cpu()
            )

    @classmethod
    def _generate_emb(cls, sequences, pooling, batch_size):
        all_embeddings = []

        for i in range(0, len(sequences), batch_size):
            seqs = [
                s[:2048] for s in sequences[i : i + batch_size]
            ]  # 2048 is maximum seq length for ESMC model
            all_embeddings.append(cls._process_batch(seqs, pooling))

        return torch.cat(all_embeddings, dim=0).numpy()

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
