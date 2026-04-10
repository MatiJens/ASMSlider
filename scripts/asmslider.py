import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
from Bio import SeqIO

from embeddings_classifier import EmbeddingsClassifier
from embeddings_generator import EmbeddingsGenerator

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


class ASMSlider:
    _MLP_WEIGHTS = Path(__file__).parent.parent / "model" / "best_model.pt"

    def __init__(self, batch_size=2048):
        self.generator = EmbeddingsGenerator(batch_size=batch_size)
        self.classifier = EmbeddingsClassifier(weights_path=str(self._MLP_WEIGHTS))
        logger.info("Models loaded.")

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
        logger.info(f"Loaded {len(sequences)} sequences from {input_fasta}.")

        all_results = {}
        for name, seq in sequences.items():
            logger.info(f"Scanning {name} (len={len(seq)})...")
            hits = self._scan_sequence(
                seq, window_size, stride, threshold, merge_distance
            )
            all_results[name] = hits
            logger.info(f"{name}: {len(hits)} hits found.")

        out_file = os.path.join(
            output_dir, f"{prefix}_mlp_{Path(input_fasta).stem}.json"
        )
        with open(out_file, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Results saved to {out_file}")

    def _scan_sequence(self, sequence, window_size, stride, threshold, merge_distance):
        if window_size > len(sequence):
            return []

        starts = range(0, len(sequence) - window_size + 1, stride)
        fragments = [sequence[s : s + window_size] for s in starts]

        probas = self.classifier.predict_batch(
            self.generator.generate_from_list(fragments)
        )
        logger.info(
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
    parser = argparse.ArgumentParser(description="Scan sequences for ASM regions.")
    parser.add_argument("--input-fasta", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--merge-distance", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2048)
    return parser


def main():
    args = create_parser().parse_args()
    slider = ASMSlider(batch_size=args.batch_size)
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
