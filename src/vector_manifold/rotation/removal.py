"""Remove rotation components from embeddings and compare cosine similarity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

Array = np.ndarray


def load_run(run_dir: Path) -> tuple[Array, Array, float]:
    embeddings = np.load(run_dir / "embeddings.npy").astype(np.float64)
    angles = np.mod(np.load(run_dir / "angles.npy").astype(np.float64), 360.0)
    metrics_path = run_dir / "rotation_analysis" / "metrics.json"
    ridge_lambda = 0.1
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        ridge_lambda = float(metrics.get("ridge_lambda", ridge_lambda))
    return embeddings, angles, ridge_lambda


def construct_targets(angles_deg: Array, harmonics: int) -> Array:
    theta = np.deg2rad(angles_deg)
    cols = []
    for k in range(1, harmonics + 1):
        cols.append(np.cos(k * theta))
        cols.append(np.sin(k * theta))
    return np.stack(cols, axis=1)


def fit_basis(Z: Array, targets: Array, ridge_lambda: float) -> Array:
    n, d = Z.shape
    if d <= n:
        gram = Z.T @ Z + ridge_lambda * np.eye(d)
        weights = np.linalg.solve(gram, Z.T @ targets)
    else:
        gram = Z @ Z.T + ridge_lambda * np.eye(n)
        alpha = np.linalg.solve(gram, targets)
        weights = Z.T @ alpha
    q, _ = np.linalg.qr(weights)
    return q[:, : targets.shape[1]]


def project(Z: Array, basis: Array) -> tuple[Array, Array]:
    projector = basis @ basis.T
    Z_rot = Z @ projector
    return Z_rot, Z - Z_rot


def cosine_matrix(Z: Array) -> Array:
    Z_norm = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    return Z_norm @ Z_norm.T


def upper_triangular_values(M: Array) -> Array:
    idx = np.triu_indices(M.shape[0], k=1)
    return M[idx]


def angle_mae(Z: Array, angles_deg: Array, ridge_lambda: float) -> float:
    theta = np.deg2rad(angles_deg)
    targets = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    n, d = Z.shape
    if d <= n:
        weights = np.linalg.solve(Z.T @ Z + ridge_lambda * np.eye(d), Z.T @ targets)
    else:
        alpha = np.linalg.solve(Z @ Z.T + ridge_lambda * np.eye(n), targets)
        weights = Z.T @ alpha
    preds = Z @ weights
    pred_angles = np.degrees(np.arctan2(preds[:, 1], preds[:, 0])) % 360.0
    err = np.abs(((pred_angles - angles_deg + 180.0) % 360.0) - 180.0)
    return float(err.mean())


def plot_heatmap(matrix: Array, path: Path, vmin: float, vmax: float, title: str) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="coolwarm", vmin=vmin, vmax=vmax)
    plt.colorbar(label="Cosine similarity")
    plt.xlabel("Frame index")
    plt.ylabel("Frame index")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def analyse(run_dir: Path, harmonics: int) -> dict:
    Z, angles_deg, ridge_lambda = load_run(run_dir)
    targets = construct_targets(angles_deg, harmonics)
    basis = fit_basis(Z, targets, ridge_lambda)
    Z_rot, Z_resid = project(Z, basis)

    cos_full = cosine_matrix(Z)
    cos_resid = cosine_matrix(Z_resid)

    vmin = float(np.min(cos_full))
    vmax = float(np.max(cos_full))

    analysis_dir = run_dir / "rotation_analysis"
    out_dir = analysis_dir / f"rotation_removed_h{harmonics}"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "basis.npy", basis)
    np.save(out_dir / "Z_rot.npy", Z_rot)
    np.save(out_dir / "Z_residual.npy", Z_resid)
    np.save(out_dir / "cosine_full.npy", cos_full)
    np.save(out_dir / "cosine_residual.npy", cos_resid)

    plot_heatmap(cos_full, out_dir / "cosine_full_heatmap.png", vmin, vmax, "Cosine similarity (full)")
    plot_heatmap(cos_resid, out_dir / "cosine_residual_heatmap.png", vmin, vmax, "Cosine similarity (rotation removed)")

    idx = np.triu_indices(Z.shape[0], k=1)
    cos_full_vec = cos_full[idx]
    cos_resid_vec = cos_resid[idx]

    diff_vec = cos_resid_vec - cos_full_vec

    mae_full = angle_mae(Z, angles_deg, ridge_lambda)
    mae_resid = angle_mae(Z_resid, angles_deg, ridge_lambda)

    summary = {
        "harmonics": harmonics,
        "dim_removed": basis.shape[1],
        "ridge_lambda": ridge_lambda,
        "rot_energy_fraction": float(np.linalg.norm(Z_rot) ** 2 / np.linalg.norm(Z) ** 2),
        "mean_cos_full": float(cos_full_vec.mean()),
        "mean_cos_residual": float(cos_resid_vec.mean()),
        "std_cos_full": float(cos_full_vec.std()),
        "std_cos_residual": float(cos_resid_vec.std()),
        "mean_improvement": float(diff_vec.mean()),
        "max_improvement": float(diff_vec.max()),
        "min_improvement": float(diff_vec.min()),
        "angle_mae_full_deg": mae_full,
        "angle_mae_residual_deg": mae_resid,
        "vmin": vmin,
        "vmax": vmax,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved outputs to {out_dir}")
    print(json.dumps(summary, indent=2))
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove rotation components and analyse cosine similarity")
    parser.add_argument("run", type=Path, help="Embedding run directory")
    parser.add_argument("--harmonics", type=int, default=1, help="Number of harmonics to include (1 -> cos/sin only)")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    analyse(args.run, max(1, args.harmonics))


if __name__ == "__main__":
    main()
