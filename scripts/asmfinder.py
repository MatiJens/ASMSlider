import os
import tempfile
from pathlib import Path

from embeddings_generator import EmbeddingsGenerator
from embeddings_classifier import EmbeddingsClassifier


class ASMFinder:
    _WEIGHTS_DIR = Path(__file__).parent.parent / "weights"
    _MLP_WEIGHTS = _WEIGHTS_DIR / "mlp_classifier.pt"

    @classmethod
    def predict(cls, input_fasta, output_dir, prefix=""):
        os.makedirs(output_dir, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp:
            generator = EmbeddingsGenerator(batch_size=512)
            generator.generate_from_file(input_fasta, tmp)
            emb_file = os.path.join(tmp, Path(input_fasta).stem + ".npz")

            classifier = EmbeddingsClassifier(weights_path=str(cls._MLP_WEIGHTS))
            classifier.predict_from_file(
                emb_file, os.path.join(output_dir, f"{prefix}_mlp_results.json")
            )
