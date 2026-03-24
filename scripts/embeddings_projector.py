import argparse
import torch
import torch.nn as nn
import h5py
import json
import numpy as np
import logging
import os
from pathlib import Path
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


class EmbeddingsProjector:
    def __init__(self, encoder_config, encoder_weights):
        with open(encoder_config) as f:
            config = json.load(f)
        self.encoder = self._build_encoder(encoder_weights, **config)

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
        encoder.load_state_dict(
            torch.load(encoder_weights, map_location="cpu", weights_only=True)
        )
        encoder.eval()
        return encoder

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

    def _load_emb(self, filepath):
        with h5py.File(filepath, "r") as f:
            keys = list(f.keys())
            embeddings = np.stack([f[k][:] for k in keys], axis=0).astype(np.float32)
        return keys, embeddings

    @torch.no_grad()
    def _project_emb(self, embeddings):
        return self.encoder(torch.from_numpy(embeddings)).numpy()

    def _save_emb(self, keys, projected_embeddings, output_file):
        os.makedirs(Path(output_file).parent, exist_ok=True)
        with h5py.File(output_file, "w") as f:
            for key, emb in zip(keys, projected_embeddings):
                f.create_dataset(key, data=emb)
        logger.info(f"Saved to {output_file}")

    def project_from_json(self, input_path, output_path):
        """Load ESMC embeddings from JSON config, project them and save."""
        for split_name, class_name, filepath in self._iter_json_config(input_path):
            logger.info(f"[{split_name}/{class_name}] Processing {Path(filepath).stem}")

            keys, embeddings = self._load_emb(filepath)
            projected = self._project_emb(embeddings)

            output_file = os.path.join(
                output_path, class_name, Path(filepath).stem + ".h5"
            )
            self._save_emb(keys, projected, output_file)

        logger.info(f"All projected embeddings saved under {output_path}")

    def project_from_file(self, input_file, output_file):
        """Load one H5 file, project embeddings and save."""
        keys, embeddings = self._load_emb(input_file)
        projected = self._project_emb(embeddings)
        self._save_emb(keys, projected, output_file)


def create_parser():
    parser = argparse.ArgumentParser(
        description="Project embeddings through a trained MLP encoder."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to JSON config or single .h5 file with embeddings.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        required=True,
        help="Output directory (for JSON config) or output .h5 file path.",
    )
    parser.add_argument(
        "--encoder-config",
        type=str,
        required=True,
        help="Path to JSON config with encoder structure.",
    )
    parser.add_argument(
        "--encoder-weights",
        type=str,
        required=True,
        help="Path to .pt file with encoder weights.",
    )
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    projector = EmbeddingsProjector(
        encoder_config=args.encoder_config,
        encoder_weights=args.encoder_weights,
    )

    if args.input_path.endswith(".json"):
        projector.project_from_json(args.input_path, args.output_path)
    else:
        projector.project_from_file(args.input_path, args.output_path)


if __name__ == "__main__":
    main()
