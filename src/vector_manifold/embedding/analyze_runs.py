"""Produce diagnostic plots and summaries for one or more embedding runs.

Example:
    python scripts/analyze_embedding_runs.py \
        --run artifacts/runs/car_rotation3_fastvit \
        --run artifacts/runs/car_rotation3_fastvit_cropped
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


@dataclass
class RunArtifacts:
    name: str
    root: Path
    embeddings: np.ndarray
    cosine: np.ndarray
    euclidean: np.ndarray
    angles: Optional[np.ndarray]
    angles_unwrapped: Optional[np.ndarray]


@dataclass
class RunSummary:
    name: str
    num_samples: int
    embedding_dim: int
    cosine_mean: float
    cosine_std: float
    cosine_min: float
    cosine_max: float
    euclid_mean: float
    euclid_std: float
    norm_mean: float
    norm_std: float
    diff_norm_mean: float
    diff_norm_std: float
    diff_norm_min: float
    diff_norm_max: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze embedding similarity structure.")
    parser.add_argument(
        "--run",
        type=Path,
        action="append",
        required=True,
        help="Path to a directory containing embeddings.npy and cosine_similarity.npy",
    )
    parser.add_argument(
        "--max-frame-delta",
        type=int,
        default=None,
        help="If set, limit frame-difference aggregation to this delta",
    )
    return parser.parse_args()


def load_run(path: Path) -> RunArtifacts:
    embeddings = np.load(path / "embeddings.npy")
    cosine = np.load(path / "cosine_similarity.npy")
    euclidean = np.load(path / "euclidean_distance.npy")
    angles: Optional[np.ndarray] = None
    angles_unwrapped: Optional[np.ndarray] = None
    metadata_path = path / "metadata.csv"
    if metadata_path.exists():
        with metadata_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            angle_values: List[float] = []
            have_angles = True
            for row in reader:
                angle_str = (row.get("angle_deg") or "").strip()
                if angle_str:
                    angle_values.append(float(angle_str))
                else:
                    have_angles = False
                    break
            if have_angles and len(angle_values) == embeddings.shape[0]:
                angles = np.asarray(angle_values, dtype=float)
                unwrapped_path = path / "angles_unwrapped.npy"
                if unwrapped_path.exists():
                    angles_unwrapped = np.load(unwrapped_path)
                else:
                    angles_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(angles)))
    return RunArtifacts(
        name=path.name,
        root=path,
        embeddings=embeddings,
        cosine=cosine,
        euclidean=euclidean,
        angles=angles,
        angles_unwrapped=angles_unwrapped,
    )


def summarize_run(run: RunArtifacts) -> RunSummary:
    cosine = run.cosine
    euclid = run.euclidean
    triu = np.triu_indices_from(cosine, k=1)
    cosine_vals = cosine[triu]
    euclid_vals = euclid[triu]
    norms = np.linalg.norm(run.embeddings, axis=1)
    if run.embeddings.shape[0] > 1:
        diffs = np.diff(run.embeddings, axis=0)
        diff_norms = np.linalg.norm(diffs, axis=1)
        diff_norm_mean = float(diff_norms.mean())
        diff_norm_std = float(diff_norms.std())
        diff_norm_min = float(diff_norms.min())
        diff_norm_max = float(diff_norms.max())
    else:
        diff_norm_mean = diff_norm_std = diff_norm_min = diff_norm_max = float("nan")
    return RunSummary(
        name=run.name,
        num_samples=run.embeddings.shape[0],
        embedding_dim=run.embeddings.shape[1],
        cosine_mean=float(cosine_vals.mean()),
        cosine_std=float(cosine_vals.std()),
        cosine_min=float(cosine_vals.min()),
        cosine_max=float(cosine_vals.max()),
        euclid_mean=float(euclid_vals.mean()),
        euclid_std=float(euclid_vals.std()),
        norm_mean=float(norms.mean()),
        norm_std=float(norms.std()),
        diff_norm_mean=diff_norm_mean,
        diff_norm_std=diff_norm_std,
        diff_norm_min=diff_norm_min,
        diff_norm_max=diff_norm_max,
    )


def plot_heatmap(matrix: np.ndarray, out_path: Path, title: str, vmin: float | None = None, vmax: float | None = None) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="magma", interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.colorbar(label=title)
    plt.title(title)
    plt.xlabel("Frame index")
    plt.ylabel("Frame index")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _subset_single_revolution(angles_unwrapped: np.ndarray) -> np.ndarray:
    valid_idx = np.where(~np.isnan(angles_unwrapped))[0]
    if valid_idx.size == 0:
        return np.ones_like(angles_unwrapped, dtype=bool)
    nonneg = valid_idx[angles_unwrapped[valid_idx] >= 0.0]
    if nonneg.size:
        start_idx = nonneg[0]
    else:
        start_idx = valid_idx[0]
    start = float(angles_unwrapped[start_idx])
    end = start + 360.0
    mask = (angles_unwrapped >= start) & (angles_unwrapped <= end + 1e-6)
    if mask.sum() < 2:
        return np.ones_like(angles_unwrapped, dtype=bool)
    return mask


def plot_heatmap_by_angle(
    matrix: np.ndarray,
    angles_wrapped: np.ndarray,
    out_path: Path,
    vmin: float,
    vmax: float,
    angles_unwrapped: Optional[np.ndarray] = None,
) -> None:
    if angles_unwrapped is not None:
        mask = _subset_single_revolution(angles_unwrapped)
        matrix = matrix[np.ix_(mask, mask)]
        angles_wrapped = angles_wrapped[mask]
        angles_unwrapped = angles_unwrapped[mask]

    sort_key = angles_unwrapped if angles_unwrapped is not None else angles_wrapped
    order = np.argsort(sort_key)
    sorted_matrix = matrix[order][:, order]
    sorted_angles = angles_wrapped[order]
    plt.figure(figsize=(6, 5))
    plt.imshow(sorted_matrix, cmap="magma", interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.colorbar(label="Cosine similarity")
    plt.title("Cosine similarity (sorted by angle)")
    tick_positions = np.linspace(0, len(order) - 1, num=6, dtype=int)
    tick_angles = sorted_angles[tick_positions]
    plt.xticks(tick_positions, [f"{a:.0f}°" for a in tick_angles])
    plt.yticks(tick_positions, [f"{a:.0f}°" for a in tick_angles])
    plt.xlabel("Angle-sorted index")
    plt.ylabel("Angle-sorted index")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _find_front_indices(angles_unwrapped: np.ndarray) -> tuple[int, int, int]:
    valid = np.where(~np.isnan(angles_unwrapped))[0]
    if valid.size == 0:
        n = len(angles_unwrapped)
        return 0, n - 1, n - 1

    start_idx = valid[0]
    for idx in valid[1:]:
        if angles_unwrapped[idx - 1] < 0.0 <= angles_unwrapped[idx]:
            start_idx = idx
            break

    first_end = start_idx
    max_idx = int(np.nanargmax(angles_unwrapped))
    if max_idx >= start_idx:
        first_end = max_idx

    return_idx = first_end
    base_angle = angles_unwrapped[start_idx]
    candidates = [idx for idx in valid if idx > first_end and abs(angles_unwrapped[idx] - base_angle) < 1e-3]
    if candidates:
        return_idx = candidates[-1]

    return start_idx, first_end, return_idx


def plot_heatmap_angle_segment(
    matrix: np.ndarray,
    angles: np.ndarray,
    start: int,
    end: int,
    out_path: Path,
    vmin: float,
    vmax: float,
) -> None:
    sub = matrix[start : end + 1, start : end + 1]
    sub_angles = angles[start : end + 1]
    plt.figure(figsize=(6, 5))
    plt.imshow(sub, cmap="magma", interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.colorbar(label="Cosine similarity")
    plt.title("Cosine similarity (angle segment)")
    ticks = min(6, sub.shape[0])
    tick_positions = np.linspace(0, sub.shape[0] - 1, num=ticks, dtype=int)
    tick_angles = sub_angles[tick_positions]
    plt.xticks(tick_positions, [f"{a:.0f}°" for a in tick_angles])
    plt.yticks(tick_positions, [f"{a:.0f}°" for a in tick_angles])
    plt.xlabel("Frame index within segment")
    plt.ylabel("Frame index within segment")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def aggregate_by_frame_delta(matrix: np.ndarray, max_delta: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    n = matrix.shape[0]
    deltas = np.arange(1, n)
    if max_delta is not None:
        deltas = deltas[deltas <= max_delta]
    means = []
    for delta in deltas:
        values = matrix.diagonal(offset=delta)
        if values.size:
            means.append(values.mean())
        else:
            means.append(np.nan)
    return deltas, np.asarray(means)


def plot_delta_curve(deltas: np.ndarray, values: np.ndarray, out_path: Path, ylabel: str, title: str) -> None:
    plt.figure(figsize=(6, 3.5))
    plt.plot(deltas, values, marker="o")
    plt.xlabel("Frame difference (Δ)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def dump_summary_json(summary: RunSummary, out_path: Path, extras: Optional[Dict[str, float]] = None) -> None:
    data = summary.__dict__.copy()
    if extras:
        data.update(extras)
    with out_path.open("w") as f:
        json.dump(data, f, indent=2)


def write_global_summary(summaries: List[RunSummary], out_path: Path) -> None:
    fieldnames = list(summaries[0].__dict__.keys())
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary.__dict__)


def write_delta_metrics(out_path: Path, deltas: np.ndarray, cosine: np.ndarray, euclid: np.ndarray) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_delta", "cosine_mean", "euclidean_mean"])
        for d, c_val, e_val in zip(deltas, cosine, euclid):
            writer.writerow([int(d), float(c_val), float(e_val)])


def plot_norm_timeseries(embeddings: np.ndarray, out_path: Path) -> None:
    norms = np.linalg.norm(embeddings, axis=1)
    plt.figure(figsize=(8, 3.5))
    plt.plot(norms)
    plt.xlabel("Frame index")
    plt.ylabel("Embedding norm")
    plt.title("Embedding norm over time")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return norms


def plot_stepwise_cosine(embeddings: np.ndarray, out_path: Path) -> None:
    a = embeddings[:-1]
    b = embeddings[1:]
    dot = np.sum(a * b, axis=1)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    cosine = dot / (norm_a * norm_b + 1e-8)
    plt.figure(figsize=(8, 3.5))
    plt.plot(cosine)
    plt.xlabel("Frame index")
    plt.ylabel("Cosine(frame_t, frame_{t+1})")
    plt.title("Successive-frame cosine similarity")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return cosine


def plot_cosine_hist(cosine_matrix: np.ndarray, out_path: Path) -> None:
    triu = np.triu_indices_from(cosine_matrix, k=1)
    values = cosine_matrix[triu]
    plt.figure(figsize=(6, 3.5))
    plt.hist(values, bins=40, color="#4c72b0", alpha=0.85)
    plt.xlabel("Cosine similarity")
    plt.ylabel("Count")
    plt.title("Pairwise cosine distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_pca_3d(embeddings: np.ndarray, out_path: Path) -> None:
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ vt[:3].T
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2], c=np.arange(len(proj)), cmap="viridis", s=12)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("PCA trajectory (3D)")
    fig.colorbar(scatter, ax=ax, shrink=0.6, label="Frame index")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def compute_diff_vectors(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.shape[0] < 2:
        return np.empty((0, embeddings.shape[1]))
    return np.diff(embeddings, axis=0)


def plot_diff_norm_timeseries(diff_norms: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(8, 3.5))
    plt.plot(diff_norms)
    plt.xlabel("Frame index (t → t+1)")
    plt.ylabel("‖Δembedding‖")
    plt.title("Embedding difference norm over time")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_diff_norm_hist(diff_norms: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(6, 3.5))
    plt.hist(diff_norms, bins=40, color="#c44e52", alpha=0.85)
    plt.xlabel("‖Δembedding‖")
    plt.ylabel("Count")
    plt.title("Distribution of embedding difference norms")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_diff_pca(diffs: np.ndarray, out_path: Path) -> None:
    if diffs.shape[0] == 0:
        return None
    centered = diffs - diffs.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ vt[:2].T
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(proj[:, 0], proj[:, 1], c=np.arange(len(proj)), cmap="plasma", s=18)
    plt.colorbar(scatter, label="Frame index")
    plt.xlabel("ΔPC1")
    plt.ylabel("ΔPC2")
    plt.title("Principal directions of embedding differences")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return proj


def plot_diff_norm_vs_angle(diff_norms: np.ndarray, angle_steps: np.ndarray, out_path: Path) -> None:
    if diff_norms.size == 0 or angle_steps.size != diff_norms.size:
        return
    plt.figure(figsize=(6, 3.5))
    plt.scatter(angle_steps, diff_norms, alpha=0.35, s=14, color="#8172b2")
    plt.xlabel("Angle step (deg)")
    plt.ylabel("‖Δembedding‖")
    plt.title("Embedding difference norm vs. angle change")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_metric_vs_angle(
    angles_unwrapped: np.ndarray,
    metric: np.ndarray,
    start: int,
    end: int,
    out_path: Path,
    ylabel: str,
    title: str,
) -> None:
    if start >= end:
        return
    x = angles_unwrapped[start : end + 1]
    y = metric[start : end + 1]
    plt.figure(figsize=(7, 3.5))
    plt.plot(x - x[0], y)
    plt.xlabel("Angle offset (deg)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_metric_vs_angle_midpoints(
    angles_unwrapped: np.ndarray,
    metric: np.ndarray,
    start: int,
    end: int,
    out_path: Path,
    ylabel: str,
    title: str,
) -> None:
    if start >= end:
        return
    idx = slice(start, end)
    mids = 0.5 * (angles_unwrapped[start:end] + angles_unwrapped[start + 1 : end + 1])
    vals = metric[idx]
    plt.figure(figsize=(7, 3.5))
    plt.plot(mids - mids[0], vals)
    plt.xlabel("Angle offset (deg)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_angle_over_time(angles: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(8, 3.5))
    plt.plot(angles)
    plt.xlabel("Frame index")
    plt.ylabel("Angle (deg)")
    plt.title("Estimated car orientation")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_angle_hist(angles: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(6, 3.5))
    plt.hist(angles % 360, bins=36, color="#55a868", alpha=0.85)
    plt.xlabel("Angle (deg)")
    plt.ylabel("Count")
    plt.title("Angle coverage")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def minimal_angle_differences(angles: np.ndarray) -> np.ndarray:
    diff = np.abs(angles[:, None] - angles[None, :])
    diff = np.mod(diff, 360.0)
    return np.minimum(diff, 360.0 - diff)


def plot_cosine_vs_angle_delta(cosine: np.ndarray, angles: np.ndarray, out_path: Path) -> None:
    angle_delta = minimal_angle_differences(angles)
    triu = np.triu_indices_from(cosine, k=1)
    plt.figure(figsize=(6.5, 4))
    plt.scatter(angle_delta[triu], cosine[triu], alpha=0.25, s=10)
    plt.xlabel("Angle difference (deg)")
    plt.ylabel("Cosine similarity")
    plt.title("Cosine vs. angle difference")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def compute_angle_metrics(angles: np.ndarray) -> Dict[str, float]:
    wrapped = angles % 360
    diff = np.diff(angles)
    wrapped_diff = ((diff + 180) % 360) - 180
    return {
        "angle_min": float(wrapped.min()),
        "angle_max": float(wrapped.max()),
        "angle_span": float((wrapped.max() - wrapped.min()) % 360),
        "angle_step_mean": float(np.mean(np.abs(wrapped_diff))),
        "angle_step_std": float(np.std(wrapped_diff)),
    }


def cosine_angle_correlation(cosine: np.ndarray, angles: np.ndarray) -> float:
    angle_delta = minimal_angle_differences(angles)
    triu = np.triu_indices_from(cosine, k=1)
    vals = cosine[triu]
    deltas = angle_delta[triu]
    if vals.size == 0:
        return float("nan")
    cov = np.cov(deltas, vals)
    denom = np.sqrt(cov[0, 0] * cov[1, 1])
    if denom == 0:
        return float("nan")
    return float(cov[0, 1] / denom)


def main() -> None:
    args = parse_args()
    runs = [load_run(path) for path in args.run]

    summaries: List[RunSummary] = []
    for run in runs:
        summary = summarize_run(run)
        summaries.append(summary)

        cosine_path = run.root / "cosine_heatmap.png"
        plot_heatmap(
            run.cosine,
            cosine_path,
            "Cosine similarity",
            vmin=float(run.cosine.min()),
            vmax=float(run.cosine.max()),
        )

        euclid_path = run.root / "euclidean_heatmap.png"
        plot_heatmap(run.euclidean, euclid_path, "Euclidean distance")

        deltas, cosine_means = aggregate_by_frame_delta(run.cosine, args.max_frame_delta)
        plot_delta_curve(deltas, cosine_means, run.root / "cosine_vs_frame_delta.png", "Cosine", "Cosine vs. frame delta")

        euclid_deltas, euclid_means = aggregate_by_frame_delta(run.euclidean, args.max_frame_delta)
        plot_delta_curve(
            euclid_deltas,
            euclid_means,
            run.root / "euclidean_vs_frame_delta.png",
            "Euclidean distance",
            "Euclidean vs. frame delta",
        )

        write_delta_metrics(run.root / "frame_delta_metrics.csv", deltas, cosine_means, euclid_means)

        norms = plot_norm_timeseries(run.embeddings, run.root / "norm_over_time.png")
        successive_cos = plot_stepwise_cosine(run.embeddings, run.root / "successive_cosine.png")
        plot_cosine_hist(run.cosine, run.root / "cosine_histogram.png")
        plot_pca_3d(run.embeddings, run.root / "trajectory_pca_3d.png")

        diffs = compute_diff_vectors(run.embeddings)
        diff_norms = np.linalg.norm(diffs, axis=1)
        if diff_norms.size:
            plot_diff_norm_timeseries(diff_norms, run.root / "diff_norm_over_time.png")
            plot_diff_norm_hist(diff_norms, run.root / "diff_norm_hist.png")
            plot_diff_pca(diffs, run.root / "diff_pca.png")

        extras = None
        if run.angles is not None:
            plot_angle_over_time(run.angles, run.root / "angle_over_time.png")
            plot_angle_hist(run.angles, run.root / "angle_histogram.png")
            plot_cosine_vs_angle_delta(run.cosine, run.angles, run.root / "cosine_vs_angle_delta.png")
            extras = compute_angle_metrics(run.angles)
            extras["cosine_angle_corr"] = cosine_angle_correlation(run.cosine, run.angles)
            plot_heatmap_by_angle(
                run.cosine,
                run.angles,
                run.root / "cosine_heatmap_by_angle.png",
                float(run.cosine.min()),
                float(run.cosine.max()),
                run.angles_unwrapped,
            )
            if run.angles_unwrapped is not None:
                start_idx, first_end, return_idx = _find_front_indices(run.angles_unwrapped)
                first_end = min(first_end, run.embeddings.shape[0] - 1)
                return_idx = min(return_idx, run.embeddings.shape[0] - 1)
                plot_metric_vs_angle(
                    run.angles_unwrapped,
                    norms,
                    start_idx,
                    first_end,
                    run.root / "norm_over_angle_one_rev.png",
                    "Embedding norm",
                    "Embedding norm vs. angle (one rev)",
                )
                plot_metric_vs_angle(
                    run.angles_unwrapped,
                    norms,
                    start_idx,
                    return_idx,
                    run.root / "norm_over_angle_out_back.png",
                    "Embedding norm",
                    "Embedding norm vs. angle (out & back)",
                )
                plot_metric_vs_angle_midpoints(
                    run.angles_unwrapped,
                    successive_cos,
                    start_idx,
                    first_end,
                    run.root / "successive_cosine_over_angle_one_rev.png",
                    "Cosine(frame_t, frame_{t+1})",
                    "Successive cosine vs. angle (one rev)",
                )
                plot_metric_vs_angle_midpoints(
                    run.angles_unwrapped,
                    successive_cos,
                    start_idx,
                    return_idx,
                    run.root / "successive_cosine_over_angle_out_back.png",
                    "Cosine(frame_t, frame_{t+1})",
                    "Successive cosine vs. angle (out & back)",
                )
                plot_heatmap_angle_segment(
                    run.cosine,
                    run.angles,
                    start_idx,
                    first_end,
                    run.root / "cosine_heatmap_angle_one_rev.png",
                    float(run.cosine.min()),
                    float(run.cosine.max()),
                )
                if return_idx > first_end:
                    plot_heatmap_angle_segment(
                        run.cosine,
                        run.angles,
                        start_idx,
                        return_idx,
                        run.root / "cosine_heatmap_angle_out_back.png",
                        float(run.cosine.min()),
                        float(run.cosine.max()),
                    )
                if diff_norms.size:
                    angle_steps = np.mod(np.abs(run.angles[1:] - run.angles[:-1]), 360.0)
                    angle_steps = np.minimum(angle_steps, 360.0 - angle_steps)
                    plot_diff_norm_vs_angle(
                        diff_norms,
                        angle_steps,
                        run.root / "diff_norm_vs_angle_step.png",
                    )
                    start_diff = start_idx
                    first_diff_end = max(first_end - 1, start_diff)
                    return_diff_end = max(return_idx - 1, start_diff)
                    plot_metric_vs_angle(
                        run.angles_unwrapped[:-1],
                        diff_norms,
                        start_diff,
                        first_diff_end,
                        run.root / "diff_norm_over_angle_one_rev.png",
                        "‖Δembedding‖",
                        "Embedding difference vs. angle (one rev)",
                    )
                    plot_metric_vs_angle(
                        run.angles_unwrapped[:-1],
                        diff_norms,
                        start_diff,
                        return_diff_end,
                        run.root / "diff_norm_over_angle_out_back.png",
                        "‖Δembedding‖",
                        "Embedding difference vs. angle (out & back)",
                    )

        dump_summary_json(summary, run.root / "analysis_summary.json", extras)

    summary_dir = Path("artifacts") / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    write_global_summary(summaries, summary_dir / "embedding_run_summary.csv")


if __name__ == "__main__":
    main()
