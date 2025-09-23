"""Utilities for projecting embeddings onto the orientation-neutral subspace."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import json
import numpy as np

Array = np.ndarray


@dataclass
class OrientationFreeResult:
    residual: Array
    rotational: Array
    basis: Array
    projector: Array
    rot_energy_fraction: float
    mean_cos_full: float
    mean_cos_residual: float
    mean_diff_norm_full: float
    mean_diff_norm_residual: float


def _pairwise_cosine(embeddings: Array) -> Array:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    normalized = embeddings / norms
    return normalized @ normalized.T


def _pairwise_euclidean(embeddings: Array) -> Array:
    sq_norm = np.sum(embeddings ** 2, axis=1, keepdims=True)
    distances_sq = sq_norm + sq_norm.T - 2.0 * (embeddings @ embeddings.T)
    np.maximum(distances_sq, 0.0, out=distances_sq)
    return np.sqrt(distances_sq, out=distances_sq)


def _mean_off_diagonal(matrix: Array) -> float:
    idx = np.triu_indices(matrix.shape[0], k=1)
    return float(matrix[idx].mean()) if idx[0].size else float("nan")


def _diff_norms(embeddings: Array) -> Array:
    if embeddings.shape[0] < 2:
        return np.empty((0,), dtype=embeddings.dtype)
    diffs = np.diff(embeddings, axis=0)
    return np.linalg.norm(diffs, axis=1)


def compute_orientation_free(
    run_dir: Path,
    *,
    basis_path: Optional[Path] = None,
    normalize: bool = True,
) -> OrientationFreeResult:
    embeddings = np.load(run_dir / "embeddings.npy").astype(np.float64)
    basis_file = basis_path if basis_path is not None else run_dir / "rotation_analysis" / "basis_U.npy"
    basis = np.load(basis_file)
    projector = basis @ basis.T
    rotational = embeddings @ projector
    residual = embeddings - rotational
    if normalize:
        norms = np.linalg.norm(residual, axis=1, keepdims=True)
        residual = residual / np.clip(norms, 1e-12, None)
    cosine_full = _pairwise_cosine(embeddings)
    cosine_residual = _pairwise_cosine(residual)
    diff_full = _diff_norms(embeddings)
    diff_resid = _diff_norms(residual)

    rot_energy_fraction = float(np.linalg.norm(rotational) ** 2 / np.linalg.norm(embeddings) ** 2)

    return OrientationFreeResult(
        residual=residual,
        rotational=rotational,
        basis=basis,
        projector=projector,
        rot_energy_fraction=rot_energy_fraction,
        mean_cos_full=_mean_off_diagonal(cosine_full),
        mean_cos_residual=_mean_off_diagonal(cosine_residual),
        mean_diff_norm_full=float(diff_full.mean()) if diff_full.size else float("nan"),
        mean_diff_norm_residual=float(diff_resid.mean()) if diff_resid.size else float("nan"),
    )


def save_orientation_free_outputs(run_dir: Path, result: OrientationFreeResult) -> None:
    out_dir = run_dir / "orientation_free"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", result.residual.astype(np.float32))
    np.save(out_dir / "rotational_component.npy", result.rotational.astype(np.float32))
    np.save(out_dir / "projector.npy", result.projector.astype(np.float32))
    cosine_resid = _pairwise_cosine(result.residual)
    np.save(out_dir / "cosine_similarity.npy", cosine_resid.astype(np.float32))
    euclid_resid = _pairwise_euclidean(result.residual)
    np.save(out_dir / "euclidean_distance.npy", euclid_resid.astype(np.float32))
    diff_norms = _diff_norms(result.residual)
    np.save(out_dir / "diff_norms.npy", diff_norms.astype(np.float32))
    summary = {
        "rot_energy_fraction": result.rot_energy_fraction,
        "mean_cos_full": result.mean_cos_full,
        "mean_cos_residual": result.mean_cos_residual,
        "mean_diff_norm_full": result.mean_diff_norm_full,
        "mean_diff_norm_residual": result.mean_diff_norm_residual,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
