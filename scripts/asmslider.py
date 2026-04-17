import argparse
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from Bio import SeqIO

from modules.embeddings_classifier import EmbeddingsClassifier
from modules.embeddings_encoder import EmbeddingsEncoder
from modules.embeddings_generator import EmbeddingsGenerator


class ASMSlider:
    _DEFAULT_WEIGHTS = Path(__file__).parent.parent / "models" / "best_model.pt"

    def __init__(self, batch_size=2048, checkpoint=None, pooling="mean",
                 encoder_checkpoint=None, latent_dim=128):
        self.batch_size = batch_size
        self.pooling = pooling
        checkpoint = checkpoint or str(self._DEFAULT_WEIGHTS)
        self.classifier = EmbeddingsClassifier(checkpoint_path=checkpoint)
        self.encoder = None
        if encoder_checkpoint:
            self.encoder = EmbeddingsEncoder(
                encoder_checkpoint, latent_dim=latent_dim
            )
        print("Models loaded.")

    def scan(
        self,
        input_fasta,
        output_dir,
        prefix="",
        window_size=30,
        stride=1,
        threshold=0.8,
        merge_distance=5,
    ):
        os.makedirs(output_dir, exist_ok=True)

        sequences = {r.id: str(r.seq) for r in SeqIO.parse(input_fasta, "fasta")}
        print(f"Loaded {len(sequences)} sequences from {input_fasta}.")

        all_results = {}
        for name, seq in sequences.items():
            print(f"Scanning {name} (len={len(seq)})...")
            hits = self._scan_sequence(
                seq, window_size, stride, threshold, merge_distance
            )
            if hits:
                all_results[name] = hits
            print(f"{name}: {len(hits)} hits found.")

        out_file = os.path.join(
            output_dir, f"{prefix}_{Path(input_fasta).stem}.json"
        )
        with open(out_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {out_file}")

    def _scan_sequence(self, sequence, window_size, stride, threshold, merge_distance):
        if window_size > len(sequence):
            return []

        starts = range(0, len(sequence) - window_size + 1, stride)
        fragments = [sequence[s : s + window_size] for s in starts]

        embeddings = EmbeddingsGenerator.generate_from_list(
            fragments, pooling=self.pooling, batch_size=self.batch_size
        )
        if self.encoder:
            embeddings = self.encoder.encode(embeddings)

        probas = self.classifier.predict(embeddings)
        print(
            f"Window {window_size}: min={probas.min():.4f}, max={probas.max():.4f}, mean={probas.mean():.4f}"
        )

        scores = np.zeros(len(sequence), dtype=np.float32)
        for s, prob in zip(starts, probas):
            np.maximum(
                scores[s : s + window_size], prob, out=scores[s : s + window_size]
            )

        return self._merge_hits(scores, threshold, merge_distance)

    @staticmethod
    def _merge_hits(scores, threshold, merge_distance):
        above = np.where(scores >= threshold)[0]
        if len(above) == 0:
            return []

        hits = []
        start = above[0]
        prev = above[0]

        for i in above[1:]:
            if i - prev > merge_distance + 1:
                hits.append({"start": int(start), "end": int(prev + 1)})
                start = i
            prev = i
        hits.append({"start": int(start), "end": int(prev + 1)})

        for hit in hits:
            region = scores[hit["start"] : hit["end"]]
            hit["max_probability"] = float(region.max())
            hit["mean_probability"] = float(region.mean())

        return hits


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-fasta", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--merge-distance", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to classifier checkpoint (.pt)."
    )
    parser.add_argument(
        "--pooling", type=str, default="mean",
        help="Embedding pooling: 'mean' or 'max' (default: mean)."
    )
    parser.add_argument(
        "--encoder-checkpoint", type=str, default=None,
        help="Path to trained autoencoder checkpoint. If provided, embeddings are reduced before classification.",
    )
    parser.add_argument(
        "--latent-dim", type=int, default=128,
        help="Encoder latent dimension (default: 128). Only used with --encoder-checkpoint.",
    )
    return parser


def main():
    args = create_parser().parse_args()
    slider = ASMSlider(
        batch_size=args.batch_size,
        checkpoint=args.checkpoint,
        pooling=args.pooling,
        encoder_checkpoint=args.encoder_checkpoint,
        latent_dim=args.latent_dim,
    )
    slider.scan(
        input_fasta=args.input_fasta,
        output_dir=args.output_dir,
        prefix=args.prefix,
        window_size=args.window_size,
        stride=args.stride,
        threshold=args.threshold,
        merge_distance=args.merge_distance,
    )


if __name__ == "__main__":
    main()
