import gc
import numpy as np
import h5py
import matplotlib
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
    format="%(asctime)s - %(module)s- %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


import gc
import json
import logging
import os
import sys

import h5py
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


class EmbeddingsVisualizer:
    def __init__(
        self,
        subsample_fit: int = 50000,
        batch_size: int = 50000,
        plot_max: int = 100000,
        title: str = "Visualization of embeddings",
        use_gpu: bool = False,
    ):
        self.subsample_fit = subsample_fit
        self.batch_size = batch_size
        self.plot_max = plot_max
        self.title = title
        self.use_gpu = use_gpu

    def _load_files(self, config_path):
        req_keys = {"path", "color", "label"}
        embeddings = []
        labels = []
        colors = []

        try:
            with open(config_path) as f:
                entries = json.load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_path}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {config_path}: {e}")
            sys.exit(1)

        if not isinstance(entries, list):
            logger.error("Config must be a JSON list of objects.")
            sys.exit(1)

        for i, entry in enumerate(entries):
            missing = req_keys - entry.keys()
            if missing:
                logger.error(f"Missing {missing} in row {i}")
                sys.exit(1)
            if not entry["path"].endswith(".h5"):
                logger.error(f"Not an .h5 file in row {i}: {entry['path']}")
                sys.exit(1)
            if not os.path.isfile(entry["path"]):
                logger.error(f"File not found in row {i}: {entry['path']}")
                sys.exit(1)

            with h5py.File(entry["path"], "r") as f:
                keys = list(f.keys())
                embs = np.stack([f[k][:] for k in keys], axis=0).astype(np.float32)
                embeddings.append(embs)
                labels.extend([entry["label"]] * len(keys))
                colors.extend([entry["color"]] * len(keys))
            gc.collect()

        data = {
            "embedding": np.vstack(embeddings).astype(np.float32),
            "label": np.array(labels),
            "color": np.array(colors),
        }
        del embeddings
        gc.collect()
        logger.info(f"Loaded {len(data['embedding'])} embeddings")
        return data

    def _umap_reduce(self, embeddings):
        seed = 67
        rng = np.random.RandomState(seed)
        n = len(embeddings)
        fit_idx = rng.choice(n, min(self.subsample_fit, n), replace=False)

        if self.use_gpu:
            return self._umap_gpu(embeddings, fit_idx, seed)
        return self._umap_cpu(embeddings, fit_idx, seed)

    def _umap_cpu(self, embeddings, fit_idx, seed):
        logger.info("Using UMAP CPU")
        from umap import UMAP

        umap_kwargs = dict(
            n_neighbors=15, metric="cosine", random_state=seed, low_memory=True
        )
        n = len(embeddings)
        results = {}

        for n_comp, key in [(2, "2d"), (3, "3d")]:
            reducer = UMAP(n_components=n_comp, **umap_kwargs)
            reducer.fit(embeddings[fit_idx])
            coords = np.empty((n, n_comp), dtype=np.float32)
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                coords[start:end] = reducer.transform(embeddings[start:end])
            results[key] = coords

        return results

    def _umap_gpu(self, embeddings, fit_idx, seed):
        logger.info("Using UMAP GPU")
        try:
            from cuml.manifold import UMAP
            import cupy as cp
        except ImportError:
            logger.error(
                "cuml/cupy not installed. Run without --use-gpu or install RAPIDS."
            )
            sys.exit(1)

        umap_kwargs = dict(n_neighbors=15, metric="cosine", random_state=seed)
        fit_data = cp.asarray(embeddings[fit_idx])
        all_data = cp.asarray(embeddings)
        results = {}

        for n_comp, key in [(2, "2d"), (3, "3d")]:
            reducer = UMAP(n_components=n_comp, **umap_kwargs)
            reducer.fit(fit_data)
            results[key] = reducer.transform(all_data).get()

        return results

    def _plot_2d(self, coords, data, output_path):
        unique_labels = set(zip(data["label"], data["color"]))
        for label, color in unique_labels:
            mask = data["label"] == label
            plt.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=color,
                label=label,
                s=2,
                alpha=0.3,
            )
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.title(f"{self.title} 2D")
        plt.legend(markerscale=3, bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        path = os.path.join(output_path, "plot_2d.png")
        plt.savefig(path)
        plt.close()
        logger.info(f"2D plot saved under {path}")

    def _plot_3d(self, coords, data, output_path):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        unique_labels = set(zip(data["label"], data["color"]))
        for label, color in unique_labels:
            mask = data["label"] == label
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                coords[mask, 2],
                c=color,
                label=label,
                s=2,
                alpha=0.3,
            )
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_zlabel("UMAP 3")
        ax.set_title(f"{self.title} 3D")
        ax.legend(markerscale=3, bbox_to_anchor=(1.15, 1), loc="upper left")
        plt.tight_layout()

        path = os.path.join(output_path, "plot_3d.png")
        plt.savefig(path)
        plt.close()
        logger.info(f"3D plot saved under {path}")

    def _plot_interactive(self, coords, data, output_path):
        fig = go.Figure()
        unique_labels = set(zip(data["label"], data["color"]))
        for label, color in unique_labels:
            mask = data["label"] == label
            fig.add_trace(
                go.Scatter3d(
                    x=coords[mask, 0],
                    y=coords[mask, 1],
                    z=coords[mask, 2],
                    mode="markers",
                    marker=dict(size=2, color=color, opacity=0.3),
                    name=label,
                    hoverinfo="name",
                )
            )
        fig.update_layout(
            title=f"{self.title} - Interactive 3D",
            scene=dict(
                xaxis_title="UMAP 1", yaxis_title="UMAP 2", zaxis_title="UMAP 3"
            ),
            template="plotly_white",
        )

        path = os.path.join(output_path, "plot_interactive.html")
        fig.write_html(path)
        logger.info(f"Interactive plot saved under {path}")

    def _subsample(self, coords, data):
        n = len(data["label"])
        if n <= self.plot_max:
            return coords, data
        idx = np.random.RandomState(0).choice(n, self.plot_max, replace=False)
        return coords[idx], {k: v[idx] for k, v in data.items()}

    def visualize(self, config_path, output_path):
        """Load data and create 2D, 3D and interactive 3D plots."""
        os.makedirs(output_path, exist_ok=True)

        data = self._load_files(config_path)
        coords = self._umap_reduce(data["embedding"])

        self._plot_2d(coords["2d"], data, output_path)
        self._plot_3d(coords["3d"], data, output_path)

        coords_sub, data_sub = self._subsample(coords["3d"], data)
        self._plot_interactive(coords_sub, data_sub, output_path)


def create_parser():
    parser = argparse.ArgumentParser(
        description="Visualize pooled embeddings using UMAP."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON config with paths, colors and labels for embeddings.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        required=True,
        help="Directory where plots should be saved.",
    )
    parser.add_argument(
        "--subsample-fit",
        type=int,
        default=50000,
        help="Max number of embeddings used to fit UMAP.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50000,
        help="Batch size for UMAP transform.",
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
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU-accelerated UMAP (requires cuml and cupy).",
    )
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    visualizer = EmbeddingsVisualizer(
        subsample_fit=args.subsample_fit,
        batch_size=args.batch_size,
        plot_max=args.plot_max,
        title=args.title,
        use_gpu=args.use_gpu,
    )
    visualizer.visualize(
        config_path=args.config,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
