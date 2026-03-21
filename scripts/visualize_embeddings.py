import gc
import numpy as np
import torch
import matplotlib
from umap import UMAP
import plotly.graph_objects as go
import argparse
import os
import json
import logging
import sys

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


class EmbeddingsVisualizer:
    def __init__(
        self,
        config_path: str,
        output_path: str,
        subsample_fit: int = 50000,
        batch_size: int = 50000,
        plot_max: int = 100000,
        title: str = "Visualization of embeddings",
    ):
        self.config = config_path
        self.output_path = output_path
        self.subsample_fit = subsample_fit
        self.batch_size = batch_size
        self.plot_max = plot_max
        self.title = title
        os.makedirs(self.output_path, exist_ok=True)

    def _load_files(self):
        req_keys = {"path", "color", "label"}
        all_embeddings = []
        all_labels = []
        all_colors = []
        self.labeled_embeddings = {}

        with open(self.config) as f:
            entries = json.load(f)

        for i, entry in enumerate(entries):
            missing = req_keys - entry.keys()
            if missing:
                msg = f"Missing {missing} in {i} row"
                logger.error(msg)
                sys.exit(1)
            if not entry["path"].endswith(".pt"):
                msg = f"Embedding are not *.pt file in {i} row"
                logger.error(msg)
                sys.exit(1)
            data = torch.load(entry["path"], map_location="cpu", weights_only=False)
            for _, emb in data.items():
                all_embeddings.append(emb.float().numpy())
                all_labels.append(entry["label"])
                all_colors.append(entry["color"])
            del data
            gc.collect()

        self.labeled_embeddings["label"] = np.array(all_labels)
        self.labeled_embeddings["color"] = np.array(all_colors)
        self.labeled_embeddings["embedding"] = np.vstack(all_embeddings).astype(
            np.float32
        )
        del all_embeddings
        gc.collect()

    def _umap_reduce(self):
        seed = 42
        rng = np.random.RandomState(seed)

        n = len(self.labeled_embeddings["embedding"])
        fit_n = min(self.subsample_fit, n)
        fit_idx = rng.choice(n, fit_n, replace=False)

        reducer_2d = UMAP(
            n_components=2,
            n_neighbors=15,
            metric="cosine",
            random_state=seed,
            low_memory=True,
        )
        reducer_2d.fit(self.labeled_embeddings["embedding"][fit_idx])

        self.coords_2d = np.empty((n, 2), dtype=np.float32)
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            self.coords_2d[start:end] = reducer_2d.transform(
                self.labeled_embeddings["embedding"][start:end]
            )

        reducer_3d = UMAP(
            n_components=3,
            n_neighbors=15,
            metric="cosine",
            random_state=seed,
            low_memory=True,
        )
        reducer_3d.fit(self.labeled_embeddings["embedding"][fit_idx])

        self.coords_3d = np.empty((n, 3), dtype=np.float32)
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            self.coords_3d[start:end] = reducer_3d.transform(
                self.labeled_embeddings["embedding"][start:end]
            )

    def _plot_2d(self):
        unique_labels = set(
            zip(self.labeled_embeddings["label"], self.labeled_embeddings["color"])
        )
        for label, color in unique_labels:
            mask = [l == label for l in self.labeled_embeddings["label"]]
            plt.scatter(
                self.coords_2d[mask, 0],
                self.coords_2d[mask, 1],
                c=color,
                label=label,
                s=2,
                alpha=0.3,
            )
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.title(f"{self.title} 2D")
        plt.legend()

        plot_path = os.path.join(self.output_path, "plot_2d.png")
        plt.savefig(plot_path)
        plt.close()

    def _plot_3d(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        unique_labels = set(
            zip(self.labeled_embeddings["label"], self.labeled_embeddings["color"])
        )
        for label, color in unique_labels:
            mask = [l == label for l in self.labeled_embeddings["label"]]
            ax.scatter(
                self.coords_3d[mask, 0],
                self.coords_3d[mask, 1],
                self.coords_3d[mask, 2],
                c=color,
                label=label,
                s=2,
                alpha=0.3,
            )
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_zlabel("UMAP 3")
        ax.set_title(f"{self.title} 3D")
        ax.legend()

        plot_path = os.path.join(self.output_path, "plot_3d.png")
        plt.savefig(plot_path)
        plt.close()

    def _interactive_plot(self):
        plotly_fig = go.Figure()

        unique_labels = set(
            zip(self.labeled_embeddings["label"], self.labeled_embeddings["color"])
        )
        for label, color in unique_labels:
            mask = [l == label for l in self.labeled_embeddings["label"]]
            plotly_fig.add_trace(
                go.Scatter3d(
                    x=self.coords_3d[mask, 0],
                    y=self.coords_3d[mask, 1],
                    z=self.coords_3d[mask, 2],
                    mode="markers",
                    marker=dict(size=2, color=color, opacity=0.3),
                    name=label,
                    hoverinfo="name",
                )
            )

        plotly_fig.update_layout(
            title=f"{self.title} - Interactive 3D",
            scene=dict(
                xaxis_title="UMAP 1",
                yaxis_title="UMAP 2",
                zaxis_title="UMAP 3",
            ),
            template="plotly_white",
        )
        plot_path = os.path.join(self.output_path, "plot_interactive.html")
        plotly_fig.write_html(plot_path)

    def visualize(self):
        """Main method that load data and create 2D, 3D and 3D-interactive plot."""
        self._load_files()
        self._umap_reduce()
        self._plot_2d()
        self._plot_3d()

        n = len(self.labeled_embeddings["label"])
        if n > self.plot_max:
            idx = np.random.RandomState(0).choice(n, self.plot_max, replace=False)
            self.coords_3d = self.coords_3d[idx]
            self.labeled_embeddings["label"] = [
                self.labeled_embeddings["label"][i] for i in idx
            ]
            self.labeled_embeddings["color"] = [
                self.labeled_embeddings["color"][i] for i in idx
            ]
            self.labeled_embeddings["embedding"] = [
                self.labeled_embeddings["embedding"][i] for i in idx
            ]

        self._interactive_plot()


def create_parser():
    parser = argparse.ArgumentParser(
        description="Script that generate embeddings by ESMC 600M model."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file that contains path to embeddings and assigned to them color and label.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        required=True,
        help="Path where embeddings should be saved.",
    )
    parser.add_argument(
        "--subsample-fit",
        type=int,
        default=50000,
        help="Max number of embeddings used to fit UMAP per batch.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50000,
        help="Max number of embeddings used to project points in space.",
    )
    parser.add_argument(
        "--plot-max",
        type=int,
        default=100000,
        help="Max number of points on interactive plot.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Visualization of embeddings",
        help="Custom title for plots.",
    )
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    visualizer = EmbeddingsVisualizer(
        config_path=args.config,
        output_path=args.output_path,
        subsample_fit=args.subsample_fit,
        batch_size=args.batch_size,
        plot_max=args.plot_max,
        title=args.title,
    )
    visualizer.visualize()


if __name__ == "__main__":
    main()
