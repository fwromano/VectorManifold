#!/usr/bin/env python3
"""Project embeddings onto the orientation-neutral subspace."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from vector_manifold.embedding.orientation_free import compute_orientation_free, save_orientation_free_outputs


def main() -> None:  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Derive orientation-neutral embeddings for a run")
    parser.add_argument("run", type=Path, help="Embedding run directory")
    parser.add_argument("--basis", type=Path, help="Optional explicit rotation basis path")
    parser.add_argument("--no-normalize", action="store_true", help="Skip per-vector normalization")
    args = parser.parse_args()

    result = compute_orientation_free(
        args.run,
        basis_path=args.basis,
        normalize=not args.no_normalize,
    )
    save_orientation_free_outputs(args.run, result)
    print(f"Saved orientation-neutral embeddings to {args.run / 'orientation_free'}")
    print(
        f"Mean cosine full={result.mean_cos_full:.4f} → residual={result.mean_cos_residual:.4f}; "
        f"rotation energy fraction={result.rot_energy_fraction:.4f}"
    )


if __name__ == "__main__":
    main()
