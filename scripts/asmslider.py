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
    _WEIGHTS_DIR = Path(__file__).parent.parent / "weights"
    _MLP_WEIGHTS = _WEIGHTS_DIR / "mlp_classifier.pt"

    @classmethod
    def scan(
        cls,
        input_fasta,
        output_dir,
        prefix="",
        window_sizes=(15, 21, 30, 40),
        stride=1,
        threshold=0.8,
        merge_distance=5,
        batch_size=2048,
    ):
        os.makedirs(output_dir, exist_ok=True)

        generator = EmbeddingsGenerator(batch_size=batch_size)
        classifier = EmbeddingsClassifier(weights_path=str(cls._MLP_WEIGHTS))
        logger.info("Models loaded.")

        sequences = {}
        for record in SeqIO.parse(input_fasta, "fasta"):
            sequences[record.id] = str(record.seq)
        logger.info(f"Loaded {len(sequences)} sequences from {input_fasta}.")

        all_results = {}
        for seq_name, seq in sequences.items():
            logger.info(f"Scanning {seq_name} (len={len(seq)})...")
            results = cls._scan_sequence(
                seq,
                generator,
                classifier,
                window_sizes,
                stride,
                threshold,
                merge_distance,
            )
            all_results[seq_name] = results
            logger.info(f"{seq_name}: {len(results)} hits found.")

        results_file = os.path.join(
            output_dir, f"{prefix}_mlp_{Path(input_fasta).stem}.json"
        )
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Results saved to {results_file}")

    @classmethod
    def _scan_sequence(
        cls,
        sequence,
        generator,
        classifier,
        window_sizes,
        stride,
        threshold,
        merge_distance,
    ):
        seq_len = len(sequence)
        position_scores = np.zeros(seq_len, dtype=np.float32)

        for win_size in window_sizes:
            if win_size > seq_len:
                continue

            fragments = []
            positions = []
            for start in range(0, seq_len - win_size + 1, stride):
                fragments.append(sequence[start : start + win_size])
                positions.append((start, start + win_size))

            if not fragments:
                continue

            embeddings = generator.generate_from_list(fragments)
            probas = classifier.predict_batch(embeddings)

            logger.info(
                f"Window {win_size}: probas min={probas.min():.4f}, "
                f"max={probas.max():.4f}, mean={probas.mean():.4f}"
            )

            for (start, end), prob in zip(positions, probas):
                np.maximum(
                    position_scores[start:end], prob, out=position_scores[start:end]
                )

        return cls._find_hits(position_scores, threshold, merge_distance)

    @classmethod
    def _find_hits(cls, position_scores, threshold, merge_distance):
        in_hit = False
        hits = []

        for i, score in enumerate(position_scores):
            if score >= threshold and not in_hit:
                hit_start = i
                in_hit = True
            elif score < threshold and in_hit:
                hits.append({"start": hit_start, "end": i})
                in_hit = False

        if in_hit:
            hits.append({"start": hit_start, "end": len(position_scores)})

        merged = []
        for hit in hits:
            if merged and hit["start"] - merged[-1]["end"] <= merge_distance:
                merged[-1]["end"] = hit["end"]
            else:
                merged.append(hit.copy())

        for hit in merged:
            region = position_scores[hit["start"] : hit["end"]]
            hit["max_probability"] = float(region.max())
            hit["mean_probability"] = float(region.mean())

        return merged
