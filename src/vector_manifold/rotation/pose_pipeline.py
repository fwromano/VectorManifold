"""Run pose-subspace experiments (linear probe, retrieval, harmonics, spectra)."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from pose_subspace import (
    Fold,
    decompose,
    default_folds,
    eval_pose,
    eval_retrieval,
    fit_probe,
    harmonics_probe,
    predict_angles,
    projector,
    residual_spectrum,
)

Array = np.ndarray


@dataclass
class RunData:
    name: str
    path: Path
    embeddings: Array
    angles_deg: Array
    angles_unwrapped: Array
    obj_id: Array


def load_run(path: Path) -> RunData:
    embeddings = np.load(path / "embeddings.npy")
    angles_deg = np.load(path / "angles.npy")
    unwrapped_path = path / "angles_unwrapped.npy"
    if unwrapped_path.exists():
        angles_unwrapped = np.load(unwrapped_path)
    else:
        angles_unwrapped = np.unwrap(np.deg2rad(angles_deg))
        angles_unwrapped = np.rad2deg(angles_unwrapped)

    metadata_path = path / "metadata.csv"
    if metadata_path.exists():
        with metadata_path.open() as f:
            next(f)
            obj_ids = [1 for _ in f]
    else:
        obj_ids = [1] * len(angles_deg)
    obj_id = np.array(obj_ids)
    if obj_id.size != len(angles_deg):
        obj_id = np.ones(len(angles_deg), dtype=int)
    return RunData(path.name, path, embeddings, angles_deg, angles_unwrapped, obj_id)


def find_segments(angles_unwrapped: Array) -> Dict[str, int]:
    valid = np.where(~np.isnan(angles_unwrapped))[0]
    if valid.size == 0:
        start = 0
    else:
        start_candidates = valid[angles_unwrapped[valid] >= 0.0]
        start = start_candidates[0] if start_candidates.size else valid[0]
    base_angle = angles_unwrapped[start]
    one_rev = np.where(angles_unwrapped >= base_angle + 359.0)[0]
    one_rev_end = one_rev[0] if one_rev.size else len(angles_unwrapped) - 1
    return_candidates = np.where(np.abs(angles_unwrapped - base_angle) < 1.0)[0]
    return_candidates = return_candidates[return_candidates > one_rev_end]
    return_end = return_candidates[-1] if return_candidates.size else len(angles_unwrapped) - 1
    return {"start": start, "one_rev_end": one_rev_end, "return_end": return_end}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def angle_error(true_deg: Array, pred_deg: Array) -> Array:
    return np.abs(((pred_deg - true_deg + 180.0) % 360.0) - 180.0)


def plot_series_with_angle(
    angle_axis: Array,
    values: Array,
    path: Path,
    title: str,
    ylabel: str,
) -> None:
    plt.figure(figsize=(7, 3.5))
    plt.plot(angle_axis - angle_axis[0], values)
    plt.xlabel("Angle offset (deg)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def diff_norms(Z: Array) -> Array:
    if Z.shape[0] < 2:
        return np.empty((0,), dtype=float)
    diff = np.diff(Z, axis=0)
    return np.linalg.norm(diff, axis=1)


def summarise(values: Array) -> Dict[str, float]:
    if values.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "p95": float("nan")}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p95": float(np.percentile(values, 95)),
    }


def save_json(path: Path, data: Dict) -> None:
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def run_linear_probe(run: RunData, out_dir: Path) -> Dict:
    ensure_dir(out_dir)
    theta_rad = np.deg2rad(run.angles_deg)
    W = fit_probe(run.embeddings, theta_rad)
    Z_rot, Z_perp = decompose(run.embeddings, W)
    preds = predict_angles(W, run.embeddings)

    np.save(out_dir / "W.npy", W)
    np.save(out_dir / "Z_rot.npy", Z_rot)
    np.save(out_dir / "Z_perp.npy", Z_perp)
    np.save(out_dir / "pred_angles.npy", preds)

    circle = run.embeddings @ W.T
    plt.figure(figsize=(5, 5))
    sc = plt.scatter(circle[:, 0], circle[:, 1], c=run.angles_deg, cmap="hsv", s=12)
    plt.colorbar(sc, label="Angle (deg)")
    plt.xlabel("Axis 1")
    plt.ylabel("Axis 2")
    plt.title("Latent circle (linear probe)")
    plt.gca().set_aspect("equal", "box")
    plt.tight_layout()
    plt.savefig(out_dir / "latent_circle.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 3.5))
    plt.scatter(run.angles_deg, preds, s=10, alpha=0.6)
    plt.plot([0, 360], [0, 360], linestyle="--", color="black", linewidth=1)
    plt.xlabel("True angle (deg)")
    plt.ylabel("Predicted angle (deg)")
    plt.title("Angle prediction (linear probe)")
    plt.tight_layout()
    plt.savefig(out_dir / "true_vs_pred.png", dpi=160)
    plt.close()

    err = angle_error(run.angles_deg, preds)
    save_json(out_dir / "pose_metrics.json", {
        "mae_deg": float(np.mean(err)),
        "median_deg": float(np.median(err)),
        "max_deg": float(np.max(err)),
    })

    return {"W": W, "Z_rot": Z_rot, "Z_perp": Z_perp, "pred": preds, "errors": err}


def evaluate_splits(run: RunData, out_dir: Path) -> Dict:
    ensure_dir(out_dir)
    splits = {"kfold": default_folds(len(run.embeddings), k=5, shuffle=False)}
    sparse = build_even_odd_folds(len(run.embeddings))
    if sparse:
        splits["even_odd"] = sparse
    cross_obj = build_leave_object_out(run.obj_id)
    if cross_obj:
        splits["leave_object"] = cross_obj

    results = {}
    for name, folds in splits.items():
        res = eval_pose(run.embeddings, run.angles_deg, folds)
        results[name] = res
        save_json(out_dir / f"{name}_pose.json", res)
    return results


def build_even_odd_folds(n: int) -> List[Fold]:
    even = np.arange(0, n, 2)
    odd = np.arange(1, n, 2)
    folds: List[Fold] = []
    if even.size > 0 and odd.size > 0:
        folds.append(Fold(name="train-even", train_idx=even, test_idx=odd))
        folds.append(Fold(name="train-odd", train_idx=odd, test_idx=even))
    return folds


def build_leave_object_out(obj_id: Array) -> List[Fold]:
    unique = np.unique(obj_id)
    folds = []
    if unique.size > 1:
        for uid in unique:
            train_idx = np.where(obj_id != uid)[0]
            test_idx = np.where(obj_id == uid)[0]
            if test_idx.size > 0 and train_idx.size > 0:
                folds.append(Fold(name=f"leave-{uid}", train_idx=train_idx, test_idx=test_idx))
    return folds


def plot_diff_metrics(
    run: RunData,
    Z_rot: Array,
    Z_perp: Array,
    base_dir: Path,
    segments: Dict[str, int],
) -> Dict:
    ensure_dir(base_dir)
    diff_full = diff_norms(run.embeddings)
    diff_rot = diff_norms(Z_rot)
    diff_perp = diff_norms(Z_perp)

    stats = {
        "full": summarise(diff_full),
        "rot": summarise(diff_rot),
        "perp": summarise(diff_perp),
    }
    save_json(base_dir / "diff_stats.json", stats)

    plt.figure(figsize=(7, 3.5))
    plt.plot(diff_full, label="full")
    if diff_perp.size:
        plt.plot(diff_perp, label="perp")
    plt.xlabel("Frame index (t → t+1)")
    plt.ylabel("‖Δembedding‖")
    plt.title("Embedding difference norms")
    plt.legend()
    plt.tight_layout()
    plt.savefig(base_dir / "diff_norm_over_time.png", dpi=160)
    plt.close()

    angle_steps = np.abs(np.diff(run.angles_deg))
    if angle_steps.size:
        plt.figure(figsize=(6, 3.5))
        plt.scatter(angle_steps, diff_full, s=12, alpha=0.4, label="full")
        if diff_perp.size:
            plt.scatter(angle_steps, diff_perp, s=12, alpha=0.4, label="perp")
        plt.xlabel("Angle step (deg)")
        plt.ylabel("‖Δembedding‖")
        plt.title("Embedding difference vs. angle step")
        plt.legend()
        plt.grid(True, alpha=0.3, linestyle="--")
        plt.tight_layout()
        plt.savefig(base_dir / "diff_norm_vs_angle_step.png", dpi=160)
        plt.close()

    start = segments["start"]
    one_rev_end = segments["one_rev_end"]
    return_end = segments["return_end"]
    if diff_full.size:
        plot_series_with_angle(
            run.angles_unwrapped[start : one_rev_end + 1],
            run.embeddings[start : one_rev_end + 1].dot(run.embeddings[start : one_rev_end + 1].T).diagonal(),
            base_dir / "norm_over_angle_one_rev.png",
            "Embedding norm vs. angle (one rev)",
            "⟨z,z⟩",
        )
        plot_series_with_angle(
            run.angles_unwrapped[start : return_end + 1],
            run.embeddings[start : return_end + 1].dot(run.embeddings[start : return_end + 1].T).diagonal(),
            base_dir / "norm_over_angle_out_back.png",
            "Embedding norm vs. angle (out & back)",
            "⟨z,z⟩",
        )
    if diff_full.size:
        plot_series_with_angle(
            run.angles_unwrapped[start : one_rev_end],
            diff_full[start : one_rev_end],
            base_dir / "diff_norm_over_angle_one_rev.png",
            "Embedding diff vs. angle (one rev)",
            "‖Δembedding‖",
        )
        plot_series_with_angle(
            run.angles_unwrapped[start : return_end],
            diff_full[start : return_end],
            base_dir / "diff_norm_over_angle_out_back.png",
            "Embedding diff vs. angle (out & back)",
            "‖Δembedding‖",
        )

    return {
        "diff_full": diff_full,
        "diff_perp": diff_perp,
    }


def run_experiments_for_run(run_path: Path, report_dir: Path) -> Dict:
    run = load_run(run_path)
    run_dir = report_dir / run.name
    ensure_dir(run_dir)

    segments = find_segments(run.angles_unwrapped)

    probe = run_linear_probe(run, run_dir / "linear_probe")

    pose_eval_results = evaluate_splits(run, run_dir / "pose_eval")

    retrieval = eval_retrieval(run.embeddings, probe["Z_perp"], run.obj_id, run.angles_deg)
    save_json(run_dir / "retrieval.json", retrieval)

    harmonics = harmonics_probe(run.embeddings, np.deg2rad(run.angles_deg))
    save_json(run_dir / "harmonics.json", harmonics)

    residual = {
        "full": residual_spectrum(run.embeddings, run.angles_deg),
        "perp": residual_spectrum(probe["Z_perp"], run.angles_deg),
    }
    save_json(run_dir / "residual_spectrum.json", residual)

    diff_outputs = plot_diff_metrics(run, probe["Z_rot"], probe["Z_perp"], run_dir / "diff_metrics", segments)

    outliers = identify_outliers(run, probe["pred"], threshold=10.0)
    save_json(run_dir / "outliers.json", outliers)

    summary = {
        "pose_eval": pose_eval_results,
        "retrieval": retrieval,
        "harmonics": harmonics,
        "residual": residual,
        "diff_stats": {
            "full": summarise(diff_outputs["diff_full"]),
            "perp": summarise(diff_outputs["diff_perp"]),
        },
        "outliers": outliers,
        "segments": segments,
    }
    save_json(run_dir / "summary.json", summary)
    return summary


def identify_outliers(run: RunData, pred_angles: Array, threshold: float) -> Dict:
    errors = angle_error(run.angles_deg, pred_angles)
    idx = np.where(errors > threshold)[0]
    return {
        "threshold": threshold,
        "count": int(idx.size),
        "indices": idx.tolist(),
        "angles": run.angles_deg[idx].tolist(),
        "errors": errors[idx].tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pose-subspace experiments")
    parser.add_argument("--run", action="append", type=Path, required=True, help="Run directory")
    parser.add_argument("--report-dir", type=Path, default=Path("docs/reports/pose_analysis"))
    args = parser.parse_args()

    report_dir = args.report_dir
    ensure_dir(report_dir)

    aggregate = {}
    for run_path in args.run:
        summary = run_experiments_for_run(run_path, report_dir)
        aggregate[run_path.name] = summary

    save_json(report_dir / "aggregate_summary.json", aggregate)


if __name__ == "__main__":
    main()
