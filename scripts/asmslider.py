import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from Bio import SeqIO

from modules.embeddings_classifier import EmbeddingsClassifier
from modules.embeddings_generator import EmbeddingsGenerator


class ASMSlider:
    def __init__(self, checkpoint, pooling="max", encoder_checkpoint=None,
                 latent_dim=128, batch_size=2048):
        self.batch_size = batch_size
        self.pooling = pooling
        self.classifier = EmbeddingsClassifier(
            checkpoint_path=checkpoint,
            encoder_path=encoder_checkpoint,
            latent_dim=latent_dim,
        )

    def scan(self, input_fasta, output_dir, prefix="", window_size=30,
             stride=1, threshold=0.8, merge_distance=5):
        os.makedirs(output_dir, exist_ok=True)
        sequences = {r.id: str(r.seq) for r in SeqIO.parse(input_fasta, "fasta")}
        print(f"Loaded {len(sequences)} sequences from {input_fasta}.")

        all_results = []
        for name, seq in sequences.items():
            print(f"Scanning {name} (len={len(seq)})...")
            hits = self._scan_sequence(seq, window_size, stride, threshold, merge_distance)
            for hit in hits:
                all_results.append({
                    "protein": name,
                    "location": f"{hit['start']}-{hit['end']}",
                    "sequence": seq[hit["start"]:hit["end"]],
                    "probability": hit["mean_probability"],
                })
            print(f"{name}: {len(hits)} hits found.")

        out_file = os.path.join(output_dir, f"{prefix}_{Path(input_fasta).stem}.json")
        with open(out_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {out_file}")

    def _scan_sequence(self, sequence, window_size, stride, threshold, merge_distance):
        if window_size > len(sequence):
            return []

        starts = range(0, len(sequence) - window_size + 1, stride)
        fragments = [sequence[s : s + window_size] for s in starts]
        embeddings = EmbeddingsGenerator.generate(fragments, self.pooling, self.batch_size)
        probas = self.classifier.predict(embeddings)

        scores = np.zeros(len(sequence), dtype=np.float32)
        for s, prob in zip(starts, probas):
            np.maximum(scores[s : s + window_size], prob, out=scores[s : s + window_size])

        return self._merge_hits(scores, threshold, merge_distance)

    @staticmethod
    def _merge_hits(scores, threshold, merge_distance):
        above = np.where(scores >= threshold)[0]
        if len(above) == 0:
            return []

        hits = []
        start = prev = above[0]
        for i in above[1:]:
            if i - prev > merge_distance + 1:
                hits.append({"start": int(start), "end": int(prev + 1)})
                start = i
            prev = i
        hits.append({"start": int(start), "end": int(prev + 1)})

        for hit in hits:
            region = scores[hit["start"] : hit["end"]]
            hit["mean_probability"] = float(region.mean())
        return hits


def create_parser():
    parser = argparse.ArgumentParser(description="Sliding-window ASM detection on FASTA sequences.")
    parser.add_argument("--input-fasta", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", required=True,
                        help="Classifier checkpoint (.pt file or directory with per-fold models).")
    parser.add_argument("--encoder-checkpoint", default=None,
                        help="Autoencoder checkpoint (.pt file or directory with per-fold AEs, "
                             "paired 1:1 with classifiers by sorted order).")
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--pooling", default="mean", choices=["mean", "max"])
    parser.add_argument("--prefix", default="")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--merge-distance", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2048)
    return parser


def main():
    args = create_parser().parse_args()
    slider = ASMSlider(
        checkpoint=args.checkpoint,
        encoder_checkpoint=args.encoder_checkpoint,
        latent_dim=args.latent_dim,
        pooling=args.pooling,
        batch_size=args.batch_size,
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
