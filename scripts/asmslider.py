import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from Bio import SeqIO

from models.ensemble import EnsembleModel
from modules.embeddings_generator import EmbeddingsGenerator


class ASMSlider:
    def __init__(self, checkpoint_dir, encoder_checkpoint_dir=None,
                 latent_dim=128, pooling="mean", batch_size=2048):
        self.batch_size = batch_size
        self.pooling = pooling
        self.model = EnsembleModel(
            mlp_dir=checkpoint_dir,
            encoder_dir=encoder_checkpoint_dir,
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
                    "variance": hit["mean_variance"],
                })
            print(f"{name}: {len(hits)} hits found.")

        stem = Path(input_fasta).stem
        filename = f"{prefix}_{stem}.json" if prefix else f"{stem}.json"
        out_file = os.path.join(output_dir, filename)
        with open(out_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {out_file}")

    def _scan_sequence(self, sequence, window_size, stride, threshold, merge_distance):
        if window_size > len(sequence):
            return []

        starts = range(0, len(sequence) - window_size + 1, stride)
        fragments = [sequence[s : s + window_size] for s in starts]
        embeddings = EmbeddingsGenerator.generate(fragments, self.pooling, self.batch_size)
        result = self.model.predict(embeddings)
        probas, variances = result[:, 0], result[:, 1]

        scores = np.zeros(len(sequence), dtype=np.float32)
        var_scores = np.zeros(len(sequence), dtype=np.float32)
        for s, prob, var in zip(starts, probas, variances):
            mask = prob > scores[s : s + window_size]
            region = slice(s, s + window_size)
            var_scores[region] = np.where(mask, var, var_scores[region])
            np.maximum(scores[region], prob, out=scores[region])

        return self._merge_hits(scores, var_scores, threshold, merge_distance)

    @staticmethod
    def _merge_hits(scores, var_scores, threshold, merge_distance):
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
            s, e = hit["start"], hit["end"]
            hit["mean_probability"] = float(scores[s:e].mean())
            hit["mean_variance"] = float(var_scores[s:e].mean())
        return hits


def create_parser():
    parser = argparse.ArgumentParser(description="Sliding-window ASM detection on FASTA sequences.")
    parser.add_argument("--input-fasta", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True,
                        help="Directory with per-fold MLP checkpoints "
                             "(fold_K/best_model.pt or fold_K.pt).")
    parser.add_argument("--encoder-checkpoint-dir", default=None,
                        help="Optional directory with per-fold AE checkpoints "
                             "(fold_K/best_autoencoder.pt).")
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
        checkpoint_dir=args.checkpoint_dir,
        encoder_checkpoint_dir=args.encoder_checkpoint_dir,
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
