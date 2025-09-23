"""Visualize canonical rotation vectors and their relationship to embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

Array = np.ndarray


def load_canonical(run_dir: Path, num_angles: int) -> dict:
    analysis_dir = run_dir / "rotation_analysis"
    export_dir = analysis_dir / "canonical_angle_vectors"
    export_dir.mkdir(parents=True, exist_ok=True)
    npz_path = export_dir / "canonical_angle_vectors.npz"
    if not npz_path.exists():
        from .canonical_vectors import canonical_vectors, determine_angles

        angles = determine_angles(run_dir, explicit=None, num=num_angles)
        payload = canonical_vectors(run_dir, angles)
        np.savez(npz_path, **payload)
        meta = {"angles_deg": payload["angles_deg"].tolist(), "generated": True}
        (export_dir / "canonical_angle_vectors.json").write_text(json.dumps(meta, indent=2))
    data = np.load(npz_path)
    return {key: data[key] for key in data}


def ensure_unit(vectors: Array) -> Array:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return vectors / norms


def angular_difference(pred: Array, true: Array) -> Array:
    diff = (pred - true + 180.0) % 360.0 - 180.0
    return diff


def plot_pose_circle(out_path: Path, pose_codes: Array, angles_deg: Array, actual_coords: Optional[Array]) -> None:
    plt.figure(figsize=(5, 5))
    plt.plot(pose_codes[:, 0], pose_codes[:, 1], color="lightgray", linewidth=1.2, label="Canonical circle")
    plt.scatter(pose_codes[::4, 0], pose_codes[::4, 1], c=angles_deg[::4], cmap="hsv", s=20)
    if actual_coords is not None:
        plt.scatter(actual_coords[:, 0], actual_coords[:, 1], c=np.linspace(0, 1, actual_coords.shape[0]), cmap="Greens", s=12, alpha=0.6, label="Projected embeddings")
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.gca().set_aspect("equal", "box")
    plt.xlabel("Axis 1")
    plt.ylabel("Axis 2")
    plt.title("Canonical rotation circle")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_similarity_heatmap(out_path: Path, vectors: Array, angles_deg: Array) -> None:
    vectors_unit = ensure_unit(vectors)
    sim = vectors_unit @ vectors_unit.T
    plt.figure(figsize=(6, 5))
    extent = [angles_deg[0], angles_deg[-1], angles_deg[0], angles_deg[-1]]
    plt.imshow(sim, origin="lower", cmap="coolwarm", vmin=-1.0, vmax=1.0, extent=extent, aspect="auto")
    plt.colorbar(label="Cosine similarity")
    plt.xlabel("Angle (deg)")
    plt.ylabel("Angle (deg)")
    plt.title("Canonical vector similarity matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_basis_components(out_path: Path, aligned_basis: Array, top_k: int = 20) -> None:
    col0 = aligned_basis[:, 0]
    col1 = aligned_basis[:, 1]
    idx0 = np.argsort(np.abs(col0))[::-1][:top_k]
    idx1 = np.argsort(np.abs(col1))[::-1][:top_k]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.bar(np.arange(top_k), col0[idx0])
    plt.xticks(np.arange(top_k), idx0, rotation=90)
    plt.title("Top components of u1")
    plt.ylabel("Weight")

    plt.subplot(1, 2, 2)
    plt.bar(np.arange(top_k), col1[idx1], color="orange")
    plt.xticks(np.arange(top_k), idx1, rotation=90)
    plt.title("Top components of u2")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_alignment_heatmap(out_path: Path, vectors: Array, z_rot: Array, angles_deg: Array, frame_angles: Array) -> Array:
    vectors_unit = ensure_unit(vectors)
    z_rot_unit = ensure_unit(z_rot)
    sim = vectors_unit @ z_rot_unit.T  # (A, T)
    plt.figure(figsize=(7, 5))
    extent = [0, frame_angles.size - 1, angles_deg[0], angles_deg[-1]]
    plt.imshow(sim, origin="lower", cmap="viridis", vmin=-1.0, vmax=1.0, aspect="auto", extent=extent)
    plt.colorbar(label="Cosine similarity")
    plt.xlabel("Frame index")
    plt.ylabel("Canonical angle (deg)")
    plt.title("Alignment of canonical vectors with rotation subspace")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

    best_idx = np.argmax(sim, axis=0)
    best_angles = angles_deg[best_idx]
    return best_angles


def plot_true_vs_best(out_path: Path, best_angles: Array, frame_angles: Array) -> None:
    diff = angular_difference(best_angles, frame_angles)
    plt.figure(figsize=(6, 4))
    plt.scatter(frame_angles, best_angles, s=18, alpha=0.7)
    plt.plot([0, 360], [0, 360], linestyle="--", color="black", linewidth=1)
    plt.xlabel("True angle (deg)")
    plt.ylabel("Best canonical angle (deg)")
    plt.title("True vs. nearest canonical angle")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

    plt.figure(figsize=(5, 3.5))
    plt.hist(diff, bins=30, color="steelblue", alpha=0.85)
    plt.xlabel("Angle error (deg)")
    plt.ylabel("Count")
    plt.title("Canonical angle error distribution")
    plt.tight_layout()
    plt.savefig(out_path.with_name("canonical_angle_error_hist.png"), dpi=180)
    plt.close()


def visualize(run_dir: Path, num_angles: int) -> None:
    analysis_dir = run_dir / "rotation_analysis"
    viz_dir = analysis_dir / "rotation_vector_viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    data = load_canonical(run_dir, num_angles)
    angles_deg = data["angles_deg"]
    pose_codes = data["pose_codes"]
    aligned_basis = data["aligned_basis"]
    vectors = data["vectors"]

    pose_code_aligned_path = analysis_dir / "pose_code_aligned.npy"
    actual_coords = np.load(pose_code_aligned_path) if pose_code_aligned_path.exists() else None

    z_rot_path = analysis_dir / "Z_rot.npy"
    if not z_rot_path.exists():
        raise FileNotFoundError(f"Missing rotation-only embeddings at {z_rot_path}")
    z_rot = np.load(z_rot_path)

    angles_path = run_dir / "angles.npy"
    if not angles_path.exists():
        raise FileNotFoundError(f"Missing frame angles at {angles_path}")
    frame_angles = np.mod(np.load(angles_path), 360.0)

    plot_pose_circle(viz_dir / "canonical_circle_overlay.png", pose_codes, angles_deg, actual_coords)
    plot_similarity_heatmap(viz_dir / "canonical_similarity_heatmap.png", vectors, angles_deg)
    plot_basis_components(viz_dir / "rotation_basis_top_components.png", aligned_basis)
    best_angles = plot_alignment_heatmap(viz_dir / "canonical_to_embedding_heatmap.png", vectors, z_rot, angles_deg, frame_angles)
    plot_true_vs_best(viz_dir / "canonical_vs_true_angle.png", best_angles, frame_angles)

    summary = {
        "angles_sampled": angles_deg.tolist(),
        "mean_alignment_error_deg": float(np.abs(angular_difference(best_angles, frame_angles)).mean()),
        "max_alignment_error_deg": float(np.abs(angular_difference(best_angles, frame_angles)).max()),
        "viz_files": [
            "canonical_circle_overlay.png",
            "canonical_similarity_heatmap.png",
            "rotation_basis_top_components.png",
            "canonical_to_embedding_heatmap.png",
            "canonical_vs_true_angle.png",
            "canonical_angle_error_hist.png",
        ],
    }
    (viz_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved rotation vector visualizations to {viz_dir}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize canonical rotation vectors")
    parser.add_argument("run", type=Path, help="Run directory containing rotation_analysis/")
    parser.add_argument("--num-angles", type=int, default=360, help="Number of canonical angles to visualise")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    visualize(args.run, args.num_angles)


if __name__ == "__main__":
    main()
