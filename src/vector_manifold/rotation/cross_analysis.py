"""Cross-run analysis for orientation and colour subspaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

from vector_manifold.rotation.subspace_analysis import compute_targets, solve_ridge

Array = np.ndarray


def _pairwise_cosine(embeddings: Array) -> Array:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    normalized = embeddings / norms
    return normalized @ normalized.T


def _diff_norms(embeddings: Array) -> Array:
    if embeddings.shape[0] < 2:
        return np.empty((0,), dtype=embeddings.dtype)
    diffs = np.diff(embeddings, axis=0)
    return np.linalg.norm(diffs, axis=1)


def _load_run(run_dir: Path) -> tuple[Array, Array]:
    embeddings = np.load(run_dir / "embeddings.npy").astype(np.float64)
    angles = np.load(run_dir / "angles.npy").astype(np.float64)
    return embeddings, angles


@dataclass
class OrientationSummary:
    run: str
    energy_fraction: float
    mean_cos_full: float
    mean_cos_residual: float
    diff_norm_full: float
    diff_norm_residual: float


@dataclass
class OrientationResult:
    basis: Array
    projector: Array
    summaries: List[OrientationSummary]


@dataclass
class ColourResult:
    colour_means: Array
    colour_basis: Array
    explained_variance: Array
    labels: List[str]


def compute_cross_orientation_basis(
    run_dirs: Sequence[Path],
    *,
    ridge_lambda: float = 0.1,
) -> OrientationResult:
    embeddings_list: List[Array] = []
    angles_list: List[Array] = []
    names: List[str] = []
    for run_dir in run_dirs:
        embeddings, angles = _load_run(run_dir)
        embeddings_list.append(embeddings)
        angles_list.append(angles)
        label = run_dir.name.replace("_clip", "")
        names.append(label)

    stacked_embeddings = np.vstack(embeddings_list)
    stacked_angles = np.concatenate(angles_list)
    targets = compute_targets(np.deg2rad(stacked_angles))
    weights_t = solve_ridge(stacked_embeddings, targets, ridge_lambda)
    q, _ = np.linalg.qr(weights_t)
    basis = q[:, : targets.shape[1]]
    projector = basis @ basis.T

    summaries: List[OrientationSummary] = []
    offset = 0
    for name, embeddings in zip(names, embeddings_list):
        n = embeddings.shape[0]
        subset = stacked_embeddings[offset : offset + n]
        offset += n
        rotational = subset @ projector
        residual = subset - rotational
        energy_fraction = float(np.linalg.norm(rotational) ** 2 / np.linalg.norm(subset) ** 2)
        cos_full = _pairwise_cosine(subset)
        cos_res = _pairwise_cosine(residual)
        idx = np.triu_indices(subset.shape[0], k=1)
        mean_cos_full = float(cos_full[idx].mean()) if idx[0].size else float("nan")
        mean_cos_res = float(cos_res[idx].mean()) if idx[0].size else float("nan")
        diff_full = _diff_norms(subset)
        diff_res = _diff_norms(residual)
        summaries.append(
            OrientationSummary(
                run=name,
                energy_fraction=energy_fraction,
                mean_cos_full=mean_cos_full,
                mean_cos_residual=mean_cos_res,
                diff_norm_full=float(diff_full.mean()) if diff_full.size else float("nan"),
                diff_norm_residual=float(diff_res.mean()) if diff_res.size else float("nan"),
            )
        )

    return OrientationResult(basis=basis, projector=projector, summaries=summaries)


def compute_colour_basis(
    run_dirs: Sequence[Path],
    *,
    projector: Array,
    normalise_means: bool = True,
) -> ColourResult:
    colour_means: List[Array] = []
    labels: List[str] = []
    for run_dir in run_dirs:
        embeddings, _ = _load_run(run_dir)
        residual = embeddings - embeddings @ projector
        norms = np.linalg.norm(residual, axis=1, keepdims=True)
        residual = residual / np.clip(norms, 1e-12, None)
        mean_vec = residual.mean(axis=0)
        if normalise_means:
            mean_vec = mean_vec / np.clip(np.linalg.norm(mean_vec), 1e-12, None)
        colour_means.append(mean_vec)
        labels.append(run_dir.name)

    colour_matrix = np.vstack(colour_means)
    centred = colour_matrix - colour_matrix.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(centred, full_matrices=False)
    colour_basis = vt[: min(centred.shape[0], 3)]
    total = (s ** 2).sum()
    explained = (s ** 2) / total if total > 0 else np.zeros_like(s)
    return ColourResult(
        colour_means=colour_matrix,
        colour_basis=colour_basis,
        explained_variance=explained,
        labels=labels,
    )
