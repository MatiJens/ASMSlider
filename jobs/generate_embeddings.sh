#!/bin/bash
#SBATCH --job-name=generate_embeddings
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
HOME_DIR="$HOME/magisterka"
OUTPUT_DIR="$LUSTRE_DIR/embedding/mean_pooling"

mkdir -p $OUTPUT_DIR

apptainer exec --nv \
    --bind "$LUSTRE_DIR":/mnt/lustre \
    --bind "$HOME_DIR":/mnt/magisterka \
    "$SIF_IMAGE" \
    python /mnt/magisterka/scripts/generate_embeddings.py \
        --input-path /mnt/magisterka/configs/esmc_config.json \
        -o /mnt/lustre/embedding \
        --pooling-type mean_pooling

echo "Job finished at $(date)"