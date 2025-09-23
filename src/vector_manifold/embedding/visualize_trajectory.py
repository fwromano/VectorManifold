"""Visualize embedding trajectories using PCA (and optional UMAP).

Usage example:
    python scripts/embedding_trajectory_visualizer.py \
        --embeddings artifacts/runs/car_rotation3_fastvit/embeddings.npy \
        --metadata artifacts/runs/car_rotation3_fastvit/metadata.csv \
        --out-dir artifacts/runs/car_rotation3_fastvit
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

try:
    import umap  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    umap = None


@dataclass
class SampleInfo:
    uid: str
    source: str
    angle_deg: Optional[float]


def parse_metadata(path: Path) -> List[SampleInfo]:
    records: List[SampleInfo] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            angle_str = row.get("angle_deg") or ""
            angle = float(angle_str) if angle_str else None
            records.append(SampleInfo(uid=row["uid"], source=row["source"], angle_deg=angle))
    return records


def compute_pca_traj(embeddings: np.ndarray, n_components: int = 3) -> np.ndarray:
    if embeddings.ndim != 2:
        raise ValueError("Embeddings array must be 2D")
    if embeddings.shape[0] < n_components:
        raise ValueError("Need at least as many samples as PCA components")
    mean = embeddings.mean(axis=0, keepdims=True)
    centered = embeddings - mean
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components]
    projection = centered @ components.T
    return projection


def compute_umap_traj(embeddings: np.ndarray, seed: int = 1234) -> Optional[np.ndarray]:
    if umap is None:
        return None
    reducer = umap.UMAP(random_state=seed, n_neighbors=15, min_dist=0.1)
    return reducer.fit_transform(embeddings)


def plot_trajectory(points: np.ndarray, out_path: Path, title: str) -> None:
    if points.shape[1] < 2:
        raise ValueError("Need at least 2D points to plot trajectory")
    plt.figure(figsize=(7, 5))
    x, y = points[:, 0], points[:, 1]
    plt.plot(x, y, color="#999999", linewidth=1.0, alpha=0.7)
    scatter = plt.scatter(x, y, c=np.arange(len(points)), cmap="viridis", s=25)
    plt.colorbar(scatter, label="Frame index")
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_pc_timeseries(points: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(points[:, 0], label="PC1")
    if points.shape[1] > 1:
        plt.plot(points[:, 1], label="PC2")
    if points.shape[1] > 2:
        plt.plot(points[:, 2], label="PC3")
    plt.xlabel("Frame index")
    plt.ylabel("Component value")
    plt.title("Principal components over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize embedding manifolds over time.")
    parser.add_argument("--embeddings", type=Path, required=True, help="Path to embeddings.npy")
    parser.add_argument("--metadata", type=Path, required=True, help="Path to metadata.csv")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory to write plots")
    parser.add_argument("--skip-umap", action="store_true", help="Disable UMAP computation")
    args = parser.parse_args()

    embeddings = np.load(args.embeddings)
    samples = parse_metadata(args.metadata)
    if embeddings.shape[0] != len(samples):
        raise ValueError("Number of embeddings does not match metadata entries")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    pca_points = compute_pca_traj(embeddings, n_components=3)
    plot_trajectory(pca_points[:, :2], args.out_dir / "trajectory_pca.png", "PCA trajectory")
    plot_pc_timeseries(pca_points, args.out_dir / "pca_components_over_time.png")
    np.save(args.out_dir / "pca_projection.npy", pca_points)

    if not args.skip_umap:
        umap_points = compute_umap_traj(embeddings)
        if umap_points is not None:
            plot_trajectory(umap_points, args.out_dir / "trajectory_umap.png", "UMAP trajectory")
            np.save(args.out_dir / "umap_projection.npy", umap_points)


if __name__ == "__main__":
    main()
