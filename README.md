# ASMFinder

ASMFinder detects amyloid signaling motifs (ASMs) in fungal proteomes. It embeds
protein sequences with the [ESM-C](https://github.com/evolutionaryscale/esm) protein
language model, classifies windows with a trained ensemble, and scans whole proteomes
with a sliding window to locate candidate motifs. Predictions can then be evaluated and
visualised against curated ASM and PFAM domain references.

## Pipeline

```
FASTA -> ESM-C embeddings -> train / tune classifier -> sliding-window scan -> evaluate + plot
```

1. **Embeddings** -- turn sequences into ESM-C vectors.
2. **Training** -- train an MLP classifier (optionally with an autoencoder encoder) using
   cross-validation; tune hyperparameters with Optuna, tracked in MLflow.
3. **Scanning** -- slide a window over each protein and emit per-threshold predictions.
4. **Evaluation** -- compare predictions against references and render figures.

## Repository layout

```
scripts/
  generate_embeddings.py     ESM-C embedding generation from a FASTA file
  asmslider.py               sliding-window proteome scan
  models/                    MLP, autoencoder and ensemble model definitions
  training/                  classifier / autoencoder training and Optuna tuning
  evaluation/                model and slider benchmarking
  utils/                     shared dataset, loss and metric helpers
  tools/                     scan-result evaluation, plotting and data-prep utilities
Dockerfile                   CUDA + Python 3.12 runtime with ESM-C
```

## Setup

The project targets Python 3.12 with a CUDA-capable GPU. The provided `Dockerfile`
builds a reproducible environment (PyTorch, scikit-learn, BioPython, MLflow, Optuna and
ESM-C, with the ESM-C model weights baked in):

```
docker build -t asmfinder .
docker run --gpus all -it -v "$PWD":/mnt asmfinder
```

To run without Docker, install the same dependencies into a Python 3.12 virtual
environment (see the `pip install` steps in the `Dockerfile`).

## Usage

Generate embeddings:

```
python3 scripts/generate_embeddings.py \
    --input-path proteome.fasta --output-path embeddings/ --pooling mean
```

Train a classifier (cross-validated checkpoints land under the output directory):

```
python3 scripts/training/train_classifier.py ...
```

Scan a proteome with a sliding window:

```
python3 scripts/asmslider.py \
    --input-fasta proteome.fasta --output-dir scan_results/ \
    --checkpoint-dir models_weights/<run>/ \
    --window-size 30 --stride 1 --threshold 0.8 --merge-distance 5
```

Evaluate and plot the predictions: see [`scripts/tools/README.md`](scripts/tools/README.md)
for the per-threshold evaluation table and the prediction-vs-reference figures.

## References

- ESM-C protein language model: https://github.com/evolutionaryscale/esm
