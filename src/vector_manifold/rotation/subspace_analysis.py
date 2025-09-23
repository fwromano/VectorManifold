"""Rotation subspace extraction following the linear ridge recipe."""

from __future__ import annotations

import argparse
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from numpy.fft import rfft

ArrayF = NDArray[np.float64]


def orthogonal_procrustes(A: ArrayF, B: ArrayF) -> Tuple[ArrayF, float]:
    M = A.T @ B
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    scale = float(s.sum())
    return R, scale


def subspace_angles(A: ArrayF, B: ArrayF) -> ArrayF:
    M = A.T @ B
    _, s, _ = np.linalg.svd(M, full_matrices=False)
    s = np.clip(s, -1.0, 1.0)
    return np.arccos(s)


@dataclass
class DecodeResult:
    mae_deg: float
    pred_deg: ArrayF
    weights: ArrayF


@dataclass
class ProbeResult:
    weights: ArrayF  # shape (2, d)
    weights_t: ArrayF  # shape (d, 2)
    basis: ArrayF  # shape (d, 2)
    projector: ArrayF  # shape (d, d)
    pose_code: ArrayF  # shape (T, 2)
    pose_code_aligned: ArrayF  # shape (T, 2)
    pred_deg: ArrayF  # shape (T,)
    ridge_lambda: float
    procrustes_R: ArrayF  # shape (2, 2)
    procrustes_scale: float


@dataclass
class FourierStats:
    k_indices: Tuple[int, int]
    power_rot: float
    power_perp: float
    total_rot: float
    total_perp: float

    @property
    def fraction_rot(self) -> float:
        return self.power_rot / self.total_rot if self.total_rot > 0 else 0.0

    @property
    def fraction_perp(self) -> float:
        return self.power_perp / self.total_perp if self.total_perp > 0 else 0.0

    @property
    def ratio_rot_perp(self) -> float:
        frac_perp = self.fraction_perp
        if frac_perp <= 0:
            return float("inf")
        return self.fraction_rot / frac_perp


@dataclass
class BootstrapStats:
    angles_deg: ArrayF  # shape (B, 2)

    @property
    def mean_deg(self) -> float:
        return float(np.mean(self.angles_deg[:, 0]))

    @property
    def p95_deg(self) -> float:
        return float(np.percentile(self.angles_deg[:, 0], 95))


@dataclass
class UnsupervisedBases:
    ssa: ArrayF
    dft: ArrayF
    dft_dims: List[int]
    ssa_angles_deg: ArrayF
    dft_angles_deg: ArrayF


@dataclass
class CovarianceStats:
    eigenvalues: ArrayF

    @property
    def eigenvalue_gap(self) -> float:
        return float(self.eigenvalues[1] - self.eigenvalues[0]) if self.eigenvalues.size >= 2 else float("nan")


@dataclass
class CrossObjectResult:
    per_object: List[Dict[str, float]]
    mean_mae_deg: float


# ---------------------------------------------------------------------------
# Core math helpers
# ---------------------------------------------------------------------------

def load_run(run_dir: Path) -> Tuple[ArrayF, ArrayF, Optional[ArrayF]]:
    embeddings = np.load(run_dir / "embeddings.npy").astype(np.float64)
    angles = np.load(run_dir / "angles.npy").astype(np.float64)
    metadata_path = run_dir / "metadata.csv"
    object_ids = load_object_ids(metadata_path, expected=len(angles))
    if embeddings.shape[0] != angles.shape[0]:
        raise ValueError("Embeddings and angles length mismatch")
    return embeddings, angles, object_ids


def load_object_ids(path: Path, expected: int) -> Optional[ArrayF]:
    if not path.exists():
        return None
    with path.open() as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "object_id" not in reader.fieldnames:
            return None
        ids: List[str] = []
        for row in reader:
            ids.append(row["object_id"])
    if not ids:
        return None
    if len(ids) != expected:
        return None
    uniq = {val: idx for idx, val in enumerate(sorted(set(ids)))}
    mapped = np.array([uniq[val] for val in ids], dtype=np.int32)
    return mapped.astype(np.float64)


def compute_targets(theta_rad: ArrayF) -> ArrayF:
    return np.stack([np.cos(theta_rad), np.sin(theta_rad)], axis=1)


def angular_error_deg(pred_deg: ArrayF, true_deg: ArrayF) -> ArrayF:
    diff = (pred_deg - true_deg + 180.0) % 360.0 - 180.0
    return np.abs(diff)


def coords_to_angles(coords: ArrayF) -> ArrayF:
    return np.degrees(np.arctan2(coords[:, 1], coords[:, 0])) % 360.0


def solve_ridge(Z: ArrayF, Y: ArrayF, ridge_lambda: float) -> ArrayF:
    n, d = Z.shape
    if d <= n:
        gram = Z.T @ Z
        gram += ridge_lambda * np.eye(d, dtype=np.float64)
        XY = Z.T @ Y
        weights_t = np.linalg.solve(gram, XY)
    else:
        gram = Z @ Z.T
        gram += ridge_lambda * np.eye(n, dtype=np.float64)
        alpha = np.linalg.solve(gram, Y)
        weights_t = Z.T @ alpha
    return weights_t  # shape (d, 2)


def predict_from_weights(Z: ArrayF, weights_t: ArrayF) -> ArrayF:
    return Z @ weights_t


def decode_with_ridge(Z: ArrayF, angles_deg: ArrayF, ridge_lambda: float) -> DecodeResult:
    theta = np.deg2rad(angles_deg)
    Y = compute_targets(theta)
    weights_t = solve_ridge(Z, Y, ridge_lambda)
    coords = predict_from_weights(Z, weights_t)
    pred_deg = coords_to_angles(coords)
    mae = float(angular_error_deg(pred_deg, angles_deg).mean())
    return DecodeResult(mae_deg=mae, pred_deg=pred_deg, weights=weights_t.T)


def fit_ridge_probe(Z: ArrayF, angles_deg: ArrayF, ridge_lambda: float) -> ProbeResult:
    theta = np.deg2rad(angles_deg)
    targets = compute_targets(theta)
    weights_t = solve_ridge(Z, targets, ridge_lambda)
    weights = weights_t.T
    basis, _ = np.linalg.qr(weights_t)
    projector = basis @ basis.T
    pose_code = Z @ basis
    coords = predict_from_weights(Z, weights_t)
    pred_deg = coords_to_angles(coords)
    R, scale = orthogonal_procrustes(pose_code, targets)
    pose_code_aligned = pose_code @ R
    return ProbeResult(
        weights=weights,
        weights_t=weights_t,
        basis=basis,
        projector=projector,
        pose_code=pose_code,
        pose_code_aligned=pose_code_aligned,
        pred_deg=pred_deg,
        ridge_lambda=ridge_lambda,
        procrustes_R=R,
        procrustes_scale=float(scale),
    )


def project_subspaces(Z: ArrayF, projector: ArrayF) -> Tuple[ArrayF, ArrayF]:
    Z_rot = Z @ projector
    Z_perp = Z - Z_rot
    return Z_rot, Z_perp


def compute_fourier_stats(pose_code: ArrayF, Z_perp: ArrayF, basis: ArrayF) -> FourierStats:
    rot_centered = pose_code - pose_code.mean(axis=0, keepdims=True)
    fft_rot = rfft(rot_centered, axis=0)
    perp_code = Z_perp @ basis
    perp_centered = perp_code - perp_code.mean(axis=0, keepdims=True)
    fft_perp = rfft(perp_centered, axis=0)
    k1 = 1 if fft_rot.shape[0] > 1 else 0
    k2 = 2 if fft_rot.shape[0] > 2 else k1
    power_rot = float(np.sum(np.abs(fft_rot[k1]) ** 2) + np.sum(np.abs(fft_rot[k2]) ** 2))
    power_perp = float(np.sum(np.abs(fft_perp[k1]) ** 2) + np.sum(np.abs(fft_perp[k2]) ** 2))
    total_rot = float(np.sum(np.abs(fft_rot) ** 2))
    total_perp = float(np.sum(np.abs(fft_perp) ** 2))
    return FourierStats(
        k_indices=(k1, k2),
        power_rot=power_rot,
        power_perp=power_perp,
        total_rot=total_rot,
        total_perp=total_perp,
    )


def isolate_harmonics(Z: ArrayF, harmonics: Sequence[int] = (1, 2)) -> ArrayF:
    n = Z.shape[0]
    spectrum = rfft(Z - Z.mean(axis=0, keepdims=True), axis=0)
    filtered = np.zeros_like(spectrum)
    for h in harmonics:
        if 0 <= h < spectrum.shape[0]:
            filtered[h] = spectrum[h]
    return np.fft.irfft(filtered, n=n, axis=0)


def bootstrap_stability(
    Z: ArrayF,
    angles_deg: ArrayF,
    ridge_lambda: float,
    reference_basis: ArrayF,
    *,
    n_bootstrap: int = 128,
    sample_ratio: float = 0.7,
    seed: int = 0,
) -> BootstrapStats:
    rng = np.random.default_rng(seed)
    n = Z.shape[0]
    sample_size = max(2, int(sample_ratio * n))
    collected = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=sample_size, replace=True)
        probe = fit_ridge_probe(Z[idx], angles_deg[idx], ridge_lambda)
        angles = np.rad2deg(subspace_angles(reference_basis, probe.basis))
        collected.append(angles)
    angles_stack = np.asarray(collected, dtype=np.float64)
    return BootstrapStats(angles_deg=angles_stack)


def lagged_covariance_basis(Z: ArrayF, max_lag: Optional[int] = None) -> ArrayF:
    n, d = Z.shape
    if n < 3:
        return np.eye(d, 2)
    Zc = Z - Z.mean(axis=0, keepdims=True)
    if max_lag is None:
        max_lag = n - 1
    max_lag = min(max_lag, n - 1)
    if max_lag <= 0:
        cov = (Zc.T @ Zc) / n
        eigvals, eigvecs = np.linalg.eigh(cov)
        return eigvecs[:, np.argsort(eigvals)[::-1][:2]]
    cov0 = (Zc.T @ Zc) / n
    accum = cov0.copy()
    for delta in range(1, max_lag + 1):
        weight = np.cos(2.0 * np.pi * delta / n)
        limit = n - delta
        if limit <= 0:
            break
        left = Zc[:limit]
        right = Zc[delta : delta + limit]
        C_delta = (left.T @ right) / limit
        accum += weight * (C_delta + C_delta.T)
    eigvals, eigvecs = np.linalg.eigh(accum)
    order = np.argsort(eigvals)[::-1]
    basis = eigvecs[:, order[:2]]
    basis, _ = np.linalg.qr(basis)
    return basis


def fourier_screened_basis(Z: ArrayF, top_k: int = 16) -> Tuple[ArrayF, List[int]]:
    n, d = Z.shape
    Zc = Z - Z.mean(axis=0, keepdims=True)
    fft = rfft(Zc, axis=0)
    k1 = 1 if fft.shape[0] > 1 else 0
    k2 = 2 if fft.shape[0] > 2 else k1
    power = np.abs(fft[k1]) ** 2 + np.abs(fft[k2]) ** 2
    order = np.argsort(power)[::-1]
    k = min(top_k, d)
    dims = order[:k].tolist()
    if not dims:
        return np.eye(d, 2), []
    selected = Zc[:, dims]
    cov = np.cov(selected, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    principal = eigvecs[:, idx[:2]]
    basis = np.zeros((d, 2), dtype=np.float64)
    basis[dims, :] = principal
    for i in range(2):
        norm = np.linalg.norm(basis[:, i])
        if norm > 0:
            basis[:, i] /= norm
    return basis, dims


def compute_unsupervised_bases(Z: ArrayF, reference_basis: ArrayF) -> UnsupervisedBases:
    dft_basis, dft_dims = fourier_screened_basis(Z)
    if dft_dims:
        subset = isolate_harmonics(Z[:, dft_dims])
        ssa_local = lagged_covariance_basis(subset)
        ssa_basis = np.zeros((Z.shape[1], 2), dtype=np.float64)
        ssa_basis[dft_dims, :] = ssa_local
        for i in range(2):
            norm = np.linalg.norm(ssa_basis[:, i])
            if norm > 0:
                ssa_basis[:, i] /= norm
    else:
        ssa_basis = lagged_covariance_basis(isolate_harmonics(Z))
    ssa_angles = np.rad2deg(subspace_angles(reference_basis, ssa_basis))
    dft_angles = np.rad2deg(subspace_angles(reference_basis, dft_basis))
    return UnsupervisedBases(
        ssa=ssa_basis,
        dft=dft_basis,
        dft_dims=dft_dims,
        ssa_angles_deg=ssa_angles,
        dft_angles_deg=dft_angles,
    )


def covariance_stats(Z: ArrayF, basis: ArrayF) -> CovarianceStats:
    Zc = Z - Z.mean(axis=0, keepdims=True)
    cov = np.cov(Zc, rowvar=False)
    cov_proj = basis.T @ cov @ basis
    eigvals, _ = np.linalg.eigh(cov_proj)
    order = np.argsort(eigvals)[::-1]
    return CovarianceStats(eigenvalues=eigvals[order])


def cross_object_generalisation(
    Z: ArrayF,
    angles_deg: ArrayF,
    object_ids: ArrayF,
    ridge_lambda: float,
) -> Optional[CrossObjectResult]:
    if object_ids is None:
        return None
    ids = object_ids.astype(int)
    unique = np.unique(ids)
    if unique.size < 2:
        return None
    results: List[Dict[str, float]] = []
    for obj_id in unique:
        train_idx = ids != obj_id
        test_idx = ids == obj_id
        if np.sum(train_idx) < 5 or np.sum(test_idx) < 2:
            continue
        weights_t = solve_ridge(Z[train_idx], compute_targets(np.deg2rad(angles_deg[train_idx])), ridge_lambda)
        preds = coords_to_angles(predict_from_weights(Z[test_idx], weights_t))
        mae = float(angular_error_deg(preds, angles_deg[test_idx]).mean())
        results.append({
            "object": int(obj_id),
            "mae_deg": mae,
            "n_test": int(np.sum(test_idx)),
            "n_train": int(np.sum(train_idx)),
        })
    if not results:
        return None
    mean_mae = float(np.mean([r["mae_deg"] for r in results]))
    return CrossObjectResult(per_object=results, mean_mae_deg=mean_mae)


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

def plot_latent_circle(path: Path, coords: ArrayF, angles_deg: ArrayF) -> None:
    plt.figure(figsize=(5, 5))
    sc = plt.scatter(coords[:, 0], coords[:, 1], c=angles_deg, cmap="hsv", s=14)
    plt.colorbar(sc, label="Angle (deg)")
    plt.xlabel("Axis 1")
    plt.ylabel("Axis 2")
    plt.title("Pose code (aligned)")
    plt.gca().set_aspect("equal", "box")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_true_vs_pred(path: Path, true_deg: ArrayF, pred_deg: ArrayF) -> None:
    plt.figure(figsize=(6, 3.5))
    plt.scatter(true_deg, pred_deg, s=12, alpha=0.6)
    plt.plot([0, 360], [0, 360], linestyle="--", color="black", linewidth=1)
    plt.xlabel("True angle (deg)")
    plt.ylabel("Predicted angle (deg)")
    plt.title("Angle prediction (ridge probe)")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Dict) -> None:
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# Main analysis routine
# ---------------------------------------------------------------------------

def analyse_run(run_dir: Path, ridge_lambda: float, n_bootstrap: int, sample_ratio: float, seed: int) -> Dict:
    embeddings, angles_deg, object_ids = load_run(run_dir)
    out_dir = run_dir / "rotation_analysis"
    ensure_dir(out_dir)

    probe = fit_ridge_probe(embeddings, angles_deg, ridge_lambda)
    Z_rot, Z_perp = project_subspaces(embeddings, probe.projector)

    decode_full = DecodeResult(mae_deg=float(angular_error_deg(probe.pred_deg, angles_deg).mean()), pred_deg=probe.pred_deg, weights=probe.weights)
    decode_rot = decode_with_ridge(Z_rot, angles_deg, ridge_lambda)
    decode_perp = decode_with_ridge(Z_perp, angles_deg, ridge_lambda)

    fourier = compute_fourier_stats(probe.pose_code_aligned, Z_perp, probe.basis)
    bootstrap = bootstrap_stability(
        embeddings,
        angles_deg,
        ridge_lambda,
        probe.basis,
        n_bootstrap=n_bootstrap,
        sample_ratio=sample_ratio,
        seed=seed,
    )
    unsupervised = compute_unsupervised_bases(embeddings, probe.basis)
    cov_stats = covariance_stats(embeddings, probe.basis)
    cross_object = cross_object_generalisation(embeddings, angles_deg, object_ids, ridge_lambda)

    # Persist arrays for downstream inspection
    np.save(out_dir / "weights.npy", probe.weights)
    np.save(out_dir / "basis_U.npy", probe.basis)
    np.save(out_dir / "projector.npy", probe.projector)
    np.save(out_dir / "pose_code.npy", probe.pose_code)
    np.save(out_dir / "pose_code_aligned.npy", probe.pose_code_aligned)
    np.save(out_dir / "pred_angles.npy", probe.pred_deg)
    np.save(out_dir / "Z_rot.npy", Z_rot)
    np.save(out_dir / "Z_perp.npy", Z_perp)
    np.save(out_dir / "bootstrap_principal_angles_deg.npy", bootstrap.angles_deg)
    np.save(out_dir / "ssa_basis.npy", unsupervised.ssa)
    np.save(out_dir / "dft_basis.npy", unsupervised.dft)

    plot_latent_circle(out_dir / "latent_circle.png", probe.pose_code_aligned, angles_deg)
    plot_true_vs_pred(out_dir / "true_vs_pred.png", angles_deg, probe.pred_deg)

    metrics = {
        "ridge_lambda": ridge_lambda,
        "decode": {
            "mae_full_deg": decode_full.mae_deg,
            "mae_rot_deg": decode_rot.mae_deg,
            "mae_perp_deg": decode_perp.mae_deg,
            "mae_rot_to_perp_ratio": decode_perp.mae_deg / decode_rot.mae_deg if decode_rot.mae_deg > 0 else float("inf"),
        },
        "fourier": {
            "k_indices": fourier.k_indices,
            "power_rot": fourier.power_rot,
            "power_perp": fourier.power_perp,
            "fraction_rot": fourier.fraction_rot,
            "fraction_perp": fourier.fraction_perp,
            "power_ratio_rot_perp": fourier.ratio_rot_perp,
        },
        "bootstrap": {
            "n": int(bootstrap.angles_deg.shape[0]),
            "sample_ratio": sample_ratio,
            "mean_principal_angle_deg": bootstrap.mean_deg,
            "p95_principal_angle_deg": bootstrap.p95_deg,
        },
        "unsupervised": {
            "ssa_principal_angles_deg": unsupervised.ssa_angles_deg.tolist(),
            "dft_principal_angles_deg": unsupervised.dft_angles_deg.tolist(),
            "dft_top_dims": unsupervised.dft_dims,
        },
        "covariance": {
            "eigenvalues": cov_stats.eigenvalues.tolist(),
        },
        "procrustes": {
            "scale": probe.procrustes_scale,
            "R": probe.procrustes_R.tolist(),
        },
        "cross_object": None,
    }

    if cross_object is not None:
        metrics["cross_object"] = {
            "mean_mae_deg": cross_object.mean_mae_deg,
            "per_object": cross_object.per_object,
        }
    else:
        metrics["cross_object"] = None

    save_json(out_dir / "metrics.json", metrics)
    print(f"Run {run_dir.name}: MAE full={decode_full.mae_deg:.3f}°, rot={decode_rot.mae_deg:.3f}°, perp={decode_perp.mae_deg:.3f}°")
    print(f"  Fourier ratio rot/perp={fourier.ratio_rot_perp:.2f}, bootstrap mean angle={bootstrap.mean_deg:.2f}°")
    print(f"  SSA angles={unsupervised.ssa_angles_deg.round(2)}, DFT angles={unsupervised.dft_angles_deg.round(2)}")
    return metrics


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract the rotation subspace from embedding runs")
    parser.add_argument("--run", action="append", type=Path, required=True, help="Run directory containing embeddings.npy and angles.npy")
    parser.add_argument("--ridge-lambda", type=float, default=1e-1, help="Ridge regularisation strength")
    parser.add_argument("--bootstrap", type=int, default=128, help="Number of bootstrap resamples")
    parser.add_argument("--bootstrap-ratio", type=float, default=0.7, help="Fraction of frames per bootstrap sample")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for bootstrap sampling")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    aggregate: Dict[str, Dict] = {}
    for run_dir in args.run:
        metrics = analyse_run(run_dir, args.ridge_lambda, args.bootstrap, args.bootstrap_ratio, args.seed)
        aggregate[run_dir.name] = metrics
    docs_dir = Path("docs/reports/rotation_subspace")
    ensure_dir(docs_dir)
    save_json(docs_dir / "aggregate_metrics.json", aggregate)


if __name__ == "__main__":
    main()
