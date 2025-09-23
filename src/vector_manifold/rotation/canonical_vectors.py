"""Export canonical rotation-plane vectors for selected angles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

Array = np.ndarray


def load_metrics(run_dir: Path) -> dict:
    metrics_path = run_dir / "rotation_analysis" / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found at {metrics_path}")
    return json.loads(metrics_path.read_text())


def parse_angle_list(values: Optional[Sequence[str]]) -> Optional[np.ndarray]:
    if values is None:
        return None
    parsed: List[float] = []
    for item in values:
        if "," in item:
            parts = item.split(",")
        else:
            parts = [item]
        for part in parts:
            stripped = part.strip()
            if not stripped:
                continue
            parsed.append(float(stripped))
    if not parsed:
        return None
    return np.unique(np.mod(parsed, 360.0))


def angle_grid(num: int) -> np.ndarray:
    if num <= 0:
        raise ValueError("Number of angles must be positive")
    return np.linspace(0.0, 360.0, num, endpoint=False)


def canonical_vectors(run_dir: Path, angles_deg: np.ndarray) -> dict:
    analysis_dir = run_dir / "rotation_analysis"
    basis = np.load(analysis_dir / "basis_U.npy")  # d x 2
    metrics = load_metrics(run_dir)
    R = np.asarray(metrics["procrustes"]["R"], dtype=np.float64)
    aligned_basis = basis @ R  # d x 2
    # Re-orthonormalise to avoid drift
    q, _ = np.linalg.qr(aligned_basis)
    aligned_basis = q[:, :2]

    theta = np.deg2rad(angles_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    vectors = cos_t[:, None] * aligned_basis[:, 0][None, :] + sin_t[:, None] * aligned_basis[:, 1][None, :]
    pose_codes = np.stack([cos_t, sin_t], axis=1)

    return {
        "angles_deg": angles_deg,
        "aligned_basis": aligned_basis,
        "vectors": vectors,
        "pose_codes": pose_codes,
    }


def save_outputs(payload: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "canonical_angle_vectors.npz", **payload)
    index = {"angles_deg": payload["angles_deg"].tolist()}
    (out_dir / "canonical_angle_vectors.json").write_text(json.dumps(index, indent=2))


def determine_angles(run_dir: Path, explicit: Optional[np.ndarray], num: Optional[int]) -> np.ndarray:
    if explicit is not None:
        return explicit
    if num is not None:
        return angle_grid(num)
    angles_path = run_dir / "angles.npy"
    if angles_path.exists():
        angles = np.load(angles_path)
        return np.unique(np.mod(angles, 360.0))
    raise FileNotFoundError(f"Could not infer angles; provide --angles or --num-angles (checked {angles_path})")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export canonical rotation vectors for given angles")
    parser.add_argument("run", type=Path, help="Run directory containing rotation_analysis/")
    parser.add_argument(
        "--angles",
        nargs="*",
        help="Specific angles in degrees (space- or comma-separated). Defaults to unique angles in angles.npy.",
    )
    parser.add_argument("--num-angles", type=int, help="Generate a uniform grid of this many angles")
    parser.add_argument(
        "--out-name",
        type=str,
        default="canonical_angle_vectors",
        help="Subdirectory name under rotation_analysis for the export",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run_dir = args.run
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    angles = determine_angles(run_dir, parse_angle_list(args.angles), args.num_angles)
    payload = canonical_vectors(run_dir, angles)
    out_dir = run_dir / "rotation_analysis" / args.out_name
    save_outputs(payload, out_dir)
    print(f"Saved canonical vectors for {len(angles)} angles to {out_dir}")


if __name__ == "__main__":
    main()
