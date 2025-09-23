#!/usr/bin/env python3
"""Cross-run orientation and colour subspace analysis."""

from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from vector_manifold.rotation.cross_analysis import (
    compute_colour_basis,
    compute_cross_orientation_basis,
)


def main() -> None:  # pragma: no cover
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Cross-run rotation and colour analysis")
    parser.add_argument("runs", nargs="+", type=Path, help="Embedding run directories")
    parser.add_argument("--ridge", type=float, default=0.1, help="Ridge coefficient for orientation fitting")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/runs/cross_analysis"))
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    orientation = compute_cross_orientation_basis(args.runs, ridge_lambda=args.ridge)
    np.save(out_dir / "orientation_basis.npy", orientation.basis.astype(np.float32))
    np.save(out_dir / "orientation_projector.npy", orientation.projector.astype(np.float32))
    summary_payload = [
        {
            "run": item.run,
            "energy_fraction": item.energy_fraction,
            "mean_cos_full": item.mean_cos_full,
            "mean_cos_residual": item.mean_cos_residual,
            "diff_norm_full": item.diff_norm_full,
            "diff_norm_residual": item.diff_norm_residual,
        }
        for item in orientation.summaries
    ]
    (out_dir / "orientation_summary.json").write_text(json.dumps(summary_payload, indent=2))

    colour = compute_colour_basis(args.runs, projector=orientation.projector)
    np.save(out_dir / "colour_means.npy", colour.colour_means.astype(np.float32))
    np.save(out_dir / "colour_basis.npy", colour.colour_basis.astype(np.float32))
    np.save(out_dir / "colour_explained_variance.npy", colour.explained_variance.astype(np.float32))
    colour_payload = {
        "labels": colour.labels,
        "explained_variance": colour.explained_variance.tolist(),
    }
    (out_dir / "colour_summary.json").write_text(json.dumps(colour_payload, indent=2))
    print(f"Saved cross-run analysis to {out_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
