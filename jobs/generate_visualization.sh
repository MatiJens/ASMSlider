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
HOME_DIR="$HOME/magisterka"
OUTPUT_DIR="$HOME_DIR/visualizations/pre_mlp"

mkdir -p $OUTPUT_DIR

apptainer exec --nv \
    --bind "$LUSTRE_DIR":/mnt/lustre \
    --bind "$HOME_DIR":/mnt/magisterka \
    "$SIF_IMAGE" \
    python /mnt/magisterka/scripts/visualize_embeddings.py \
        --config /mnt/magisterka/configs/pre_mlp_config.json \
        -o /mnt/magisterka/visualizations/pre_mlp \
        --use-gpu --title "Visualization of pre-MLP ESMC embeddings"

echo "Job finished at $(date)"
