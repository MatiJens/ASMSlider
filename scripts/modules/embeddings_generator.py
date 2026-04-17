import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from esm.models.esmc import ESMC

from utils.sequence_loader import load_fasta

torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_cudnn_sdp(False)


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
        else:  # mean
            h = output.embeddings.float()
            m = mask.float().unsqueeze(-1)
            summed = (h * m).sum(dim=1)
            counts = m.sum(dim=1).clamp(min=1)
            return (summed / counts).cpu().numpy()

    @classmethod
    def generate_from_file(cls, input_file, pooling, batch_size):
        """Load one FASTA file and generate embeddings."""
        cls._ensure_model()
        sequences = [item["seq"] for item in load_fasta(input_file)]
        return cls._generate_emb(sequences, pooling, batch_size)

    @classmethod
    def generate_from_list(cls, sequences, pooling, batch_size):
        """Generate embeddings from list of sequences."""
        cls._ensure_model()
        return cls._generate_emb(sequences, pooling, batch_size)

    @classmethod
    def _generate_emb(cls, sequences, pooling, batch_size):
        all_embeddings = []

        for i in range(0, len(sequences), batch_size):
            seqs = [
                s[:2048] for s in sequences[i : i + batch_size]
            ]  # 2048 is maximum seq length for ESMC model
            all_embeddings.append(cls._process_batch(seqs, pooling))

        return np.concatenate(all_embeddings, axis=0)
