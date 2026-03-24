import os
import tempfile
import warnings
from pathlib import Path

from embeddings_generator import EmbeddingsGenerator
from embeddings_projector import EmbeddingsProjector
from embeddings_classifier import EmbeddingsClassifier

warnings.filterwarnings("ignore", message=".*older version of XGBoost.*")


class ASMFinder:
    _WEIGHTS_DIR = Path(__file__).parent.parent / "weights"
    _ENCODER_CONFIG = _WEIGHTS_DIR / "mlp_config.json"
    _ENCODER_WEIGHTS = _WEIGHTS_DIR / "mlp_encoder.pt"
    _LR_MODEL = _WEIGHTS_DIR / "lr_model.pkl"
    _RF_MODEL = _WEIGHTS_DIR / "rf_model.pkl"
    _XGB_MODEL = _WEIGHTS_DIR / "xgb_model.pkl"

    @classmethod
    def predict(cls, input_fasta, output_dir, prefix=""):
        os.makedirs(output_dir, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp:
            emb_file = os.path.join(tmp, "embeddings.h5")
            proj_file = os.path.join(tmp, "projected.h5")

            generator = EmbeddingsGenerator(batch_size=512, pooling_type="mean_pooling")
            generator.generate_from_file(input_fasta, emb_file)

            projector = EmbeddingsProjector(
                encoder_config=str(cls._ENCODER_CONFIG),
                encoder_weights=str(cls._ENCODER_WEIGHTS),
            )
            projector.project_from_file(emb_file, proj_file)

            lr = EmbeddingsClassifier(model_path=str(cls._LR_MODEL))
            lr.predict_from_file(
                proj_file, os.path.join(output_dir, f"{prefix}_lr_results.json")
            )

            rf = EmbeddingsClassifier(model_path=str(cls._RF_MODEL))
            rf.predict_from_file(
                proj_file, os.path.join(output_dir, f"{prefix}_rf_results.json")
            )

            xgb = EmbeddingsClassifier(model_path=str(cls._XGB_MODEL))
            xgb.predict_from_file(
                proj_file, os.path.join(output_dir, f"{prefix}_xgb_results.json")
            )
