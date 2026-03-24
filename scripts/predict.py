import logging
import json
import sys

from pathlib import Path

from asmfinder import ASMFinder

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)

input_json = "/mnt/magisterka/configs/predict_config.json"

try:
    with open(input_json, "r") as f:
        files_to_predict = json.load(f)
except FileNotFoundError:
    logger.error(f"Config file not found: {input_json}")
    sys.exit(1)
except json.JSONDecodeError as e:
    logger.error(f"Invalid JSON in {input_json}: {e}")
    sys.exit(1)

if not isinstance(files_to_predict, list):
    logger.error("Config must be a JSON list of [input, output, prefix] entries.")
    sys.exit(1)

for i, entry in enumerate(files_to_predict):
    if not isinstance(entry, list) or len(entry) != 3:
        logger.error(f"Row {i}: expected [input, output, prefix], got {entry}")
        sys.exit(1)
    if not Path(entry[0]).is_file():
        logger.error(f"Row {i}: file not found: {entry[0]}")
        sys.exit(1)

for input_file, output_dir, prefix in files_to_predict:
    logger.info(f"Predicting {Path(input_file).stem}")
    ASMFinder.predict(input_file, output_dir, prefix)
    logger.info(f"Predicted. Results saved under {output_dir}")
