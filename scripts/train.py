import argparse
import torch

import numpy as np

from focal_loss import FocalLoss
from mlp_model import MLPModel
from sequence_loader import SequenceLoader


# Path to data is hardcoded - it is easier to use than 6 args in parser.
POS_TRAIN = "/mnt/lustre/embeddings/mlp_train/positive/train"
NEG_TRAIN = "/mnt/lustre/embeddings/mlp_train/negative/train"

POS_VAL = "/mnt/lustre/embeddings/mlp_train/positive/val"
NEG_VAL = "/mnt/lustre/embeddings/mlp_train/negative/val"

POS_TEST = "/mnt/lustre/embeddings/mlp_train/positive/test"
NEG_TEST = "/mnt/lustre/embeddings/mlp_train/negative/test"

CHECKPOINT_DIR = "/mnt/magisterka/models"


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate of model. Higher lr mean faster learning, but can 'jump over' global minimum.",
    )
    parser.add_argument(
        "--weight-decay",
        default=1e-3,
        help="How much weight will be decreased. Higher wd mean smaller overfitting.",
    )
    parser.add_argument(
        "--alpha",
        default=0.8,
        help="Weight of positive class. Value range: (0, 1). Higher alpha increase importance of positive class.",
    )
    parser.add_argument(
        "--gamma",
        default=3.0,
        help="Control how much model ignore easy exaples.",
    )
    parser.add_argument(
        "--batch-size",
        default=256,
        help="Size of every batch. Smaller batch mean better generalization but slower learning and gradient instability.",
    )
    parser.add_argument(
        "--max-epochs",
        default=100,
        help="Max number of epochs.",
    )
    parser.add_argument(
        "--patience",
        default=15,
        help="After how many epochs without improvement model will stop learning.",
    )
    return parser


def create_loader(pos_path, neg_path):
    pos = SequenceLoader.load_embeddings(pos_path)
    neg = SequenceLoader.load_embeddings(neg_path)
    X = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])


def main():
    args = create_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPModel().to(device)

    criterion = FocalLoss(gamma=args.gamma, alpha=args.alpha)
    optimizer = torch.optim.AdamW(lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.max_epochs + 1):
        optimizer.zero_grad()
