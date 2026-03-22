#!/bin/bash
#SBATCH --job-name=visualize_embeddings
#SBATCH --partition=lem-gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:hopper:1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

LUSTRE_DIR="$HOME/pd-hpc-fi2/mjens"
SIF_IMAGE="$LUSTRE_DIR/asmfinder.sif"
SCRIPTS_DIR="$HOME/magisterka/scripts"
OUTPUT_DIR="$LUSTRE_DIR/embedding/mean_pooling/plots"

mkdir -p $OUTPUT_DIR

apptainer exec --nv \
    --bind "$LUSTRE_DIR":/mnt/lustre \
    --bind "$SCRIPTS_DIR":/mnt/scripts \
    "$SIF_IMAGE" \
    python /mnt/scripts/visualize_embeddings.py \
        --config /mnt/lustre/embedding/mean_pooling/config.json \
        -o /mnt/lustre/embedding/mean_pooling/plots \
        --use-gpu --title "Visualization of mean-pooling ESMC embeddings"

echo "Job finished at $(date)"
