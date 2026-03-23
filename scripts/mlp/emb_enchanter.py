import argparse
import torch
import torch.nn as nn
import h5py
import json
import numpy as np
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


class EmbEnchanter:
    def __init__(self, input_path, output_path, encoder_config, encoder_weights):
        with open(encoder_config) as f:
            config = json.load(f)

        self.encoder = self._build_encoder(encoder_weights, **config)
        self.input_path = input_path
        self.output_path = output_path
        os.makedirs(Path(self.output_path).parent, exist_ok=True)

    def _build_encoder(self, encoder_weights, input_dim, hidden_dims, dropout):
        layers = []
        prev_dim = input_dim
        for i, dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout))
            prev_dim = dim
        encoder = nn.Sequential(*layers)
        encoder.load_state_dict(torch.load(encoder_weights, map_location="cpu"))
        encoder.eval()
        return encoder

    def _load_emb(self):
        with h5py.File(self.input_path, "r") as f:
            keys = list(f.keys())
            embeddings = np.stack([f[k][:] for k in keys], axis=0).astype(np.float32)
            return keys, embeddings

    @torch.no_grad()
    def _enchant_emb(self, embeddings):
        return self.encoder(torch.from_numpy(embeddings)).numpy()

    def _save_emb(self, keys, enchanted_embeddings):
        with h5py.File(self.output_path, "w") as f:
            for key, emb in zip(keys, enchanted_embeddings):
                f.create_dataset(key, data=emb)
            logger.info(f"Enchanted embedding saved under {self.output_path}")

    def enchant(self):
        """Load ESMC embeddings, enchant them and save."""
        keys, embeddings = self._load_emb()
        enchanted_embeddings = self._enchant_emb(embeddings)
        self._save_emb(keys, enchanted_embeddings)


def create_parser():
    parser = argparse.ArgumentParser(
        description="Script that create, teach and save MLP encoder and classifier."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="""Path to ESMC embeddings.""",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        required=True,
        help="Path where enchanted embeddings should be saved.",
    )
    parser.add_argument(
        "--encoder-config",
        type=str,
        required=True,
        help="Path to config file with encoder structure.",
    )
    parser.add_argument(
        "--encoder-weights",
        type=str,
        required=True,
        help="Path to file with model weights.",
    )
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    try:
        emb_enchanter = EmbEnchanter(
            input_path=args.input_path,
            output_path=args.output_path,
            encoder_config=args.encoder_config,
            encoder_weights=args.encoder_weights,
        )
        emb_enchanter.enchant()
    except Exception as e:
        logger.exception(f"Job failed: {e}")
        raise


if __name__ == "__main__":
    main()
