import os
import json
import logging
import warnings
from pathlib import Path
import numpy as np
from Bio import SeqIO

from scripts.embeddings.generator import Generator
from scripts.embeddings.projector import Projector
from scripts.embeddings.classifier import EmbeddingsClassifier

warnings.filterwarnings("ignore", message=".*older version of XGBoost.*")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


class ASMSlider:
    _WEIGHTS_DIR = Path(__file__).parent.parent / "weights"
    _ENCODER_CONFIG = _WEIGHTS_DIR / "mlp_config.json"
    _ENCODER_WEIGHTS = _WEIGHTS_DIR / "mlp_encoder.pt"
    _LR_MODEL = _WEIGHTS_DIR / "lr_model.pkl"
    _RF_MODEL = _WEIGHTS_DIR / "rf_model.pkl"
    _XGB_MODEL = _WEIGHTS_DIR / "xgb_model.pkl"

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

        generator = Generator(batch_size=batch_size, pooling_type="mean_pooling")
        projector = Projector(
            encoder_config=str(cls._ENCODER_CONFIG),
            encoder_weights=str(cls._ENCODER_WEIGHTS),
        )
        classifiers = {
            "xgb": EmbeddingsClassifier(model_path=str(cls._XGB_MODEL)),
        }
        logger.info("All models loaded.")

        sequences = {}
        for record in SeqIO.parse(input_fasta, "fasta"):
            sequences[record.id] = str(record.seq)
        logger.info(f"Loaded {len(sequences)} sequences from {input_fasta}.")

        for clf_name, classifier in classifiers.items():
            logger.info(f"Started scan with {clf_name}.")
            all_results = {}

            for seq_name, seq in sequences.items():
                logger.info(f"  [{clf_name}] Scanning {seq_name} (len={len(seq)})...")
                results = cls._scan_sequence(
                    seq,
                    generator,
                    projector,
                    classifier,
                    window_sizes,
                    stride,
                    threshold,
                    merge_distance,
                )
                all_results[seq_name] = results
                n_hits = len(results)
                logger.info(f"  [{clf_name}] {seq_name}: {n_hits} hits found.")

            results_file = os.path.join(
                output_dir, f"{prefix}_{clf_name}_{Path(input_fasta).stem}.json"
            )
            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=2)
            logger.info(f"Results saved to {results_file}")

    @classmethod
    def _scan_sequence(
        cls,
        sequence,
        generator,
        projector,
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
                end = start + win_size
                fragments.append(sequence[start:end])
                positions.append((start, end))

            if not fragments:
                continue

            embeddings = generator.generate_from_list(fragments)

            projected = projector.project(embeddings)

            probas = classifier.predict_batch(projected)

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
