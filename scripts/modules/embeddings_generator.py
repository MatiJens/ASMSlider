import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from esm.models.esmc import ESMC

from utils.sequence_loader import load_fasta

torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_cudnn_sdp(False)

MAX_SEQ_LEN = 2048  # ESMC model limit


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
        mask = input_ids != cls._model.tokenizer.pad_token_id
        mask[:, 0] = False
        del input_ids

        h = output.embeddings.float()
        if pooling == "max":
            h = h.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            return h.max(dim=1).values.cpu().numpy()
        m = mask.float().unsqueeze(-1)
        return ((h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)).cpu().numpy()

    @classmethod
    def generate(cls, sequences, pooling, batch_size):
        """Embed a list of sequences."""
        cls._ensure_model()
        batches = [
            cls._process_batch([s[:MAX_SEQ_LEN] for s in sequences[i : i + batch_size]], pooling)
            for i in range(0, len(sequences), batch_size)
        ]
        return np.concatenate(batches, axis=0)

    @classmethod
    def generate_from_file(cls, input_file, pooling, batch_size):
        """Embed every sequence in a FASTA file."""
        sequences = [item["seq"] for item in load_fasta(input_file)]
        return cls.generate(sequences, pooling, batch_size)
