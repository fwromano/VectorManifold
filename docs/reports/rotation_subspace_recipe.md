# Rotation Subspace Extraction Report (Rotating Car)

This note documents the ridge-based pose subspace extraction applied to the cropped FastViT run over the rotating car sequence. The procedure follows the provided recipe:

1. Fit a ridge probe to the targets $[\cos \theta, \sin \theta]$ (`lambda = 0.1`).
2. Orthogonalise the probe’s row space to obtain $U$, the rotation plane, with projector $P = UU^T$.
3. Decompose the embeddings into $Z_\text{rot} = ZP$ and $Z_\perp = Z(I-P)$.
4. Run validation checks (decoding, spectral energy, bootstrap stability) and two unsupervised corroborations (lagged-covariance SSA, Fourier-screened PCA).
5. Canonicalise via Procrustes so that $ZUR$ tracks $(\cos \theta, \sin \theta)$, saving `U`, `P`, `R`.

All artefacts land in `artifacts/runs/<run>/rotation_analysis/`. Use `python scripts/rotation_subspace_analysis.py --run <run>` to regenerate them.

## Summary Metrics (Cropped run only)

| Run | MAE (full) | MAE ($Z_\text{rot}$) | MAE ($Z_\perp$) | MAE ratio ($\perp/\text{rot}$) | Fourier ratio ($k=1,2$) | Bootstrap mean ± p95 (°) | SSA principal angles (°) | DFT principal angles (°) | $\lambda_1, \lambda_2$ of $U^T \operatorname{Cov}(Z)U$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `car_rotation3_fastvit_cropped` | 2.29° | 2.29° | 5.59° | 2.44× | 1.20× | 36.9°, 41.6° | 81.4°, 86.2° | 82.4°, 89.7° | 0.486, 0.122 |

* MAE figures use angular error on a held-out ridge fit to the rotation plane. The orthogonal complement degrades decoding by roughly $2.4\times$, confirming that the ridge probe isolates most rotation energy.
* Fourier ratios compare the $k=1,2$ power of the aligned pose code to the projection of $Z_\perp$ onto the same axes. Background leakage keeps the ratio modest (~1.2×).
* Bootstrapped principal angles (64 resamples, 70% frames) sit near 37° mean / 42° p95. The subspace is directionally consistent but not within the desired <5° band—more data or stronger regularisation appears necessary.
* SSA/DFT unsupervised bases remain far (>80°) from the supervised plane, suggesting additional preprocessing is needed to isolate rotation harmonics automatically.

## Validation Checklist

- **Angle decodability:** ✓ $Z_\text{rot}$ maintains the full regression quality; $Z_\perp$ errors roughly double.
- **Spectral energy:** • Full-frame run shows clear dominance (6.9×). • Cropped version only modest (1.2×) – rotation still visible in the complement.
- **Bootstrap stability:** ✗ Principal angles stay around 37°/42°. Increasing $\lambda$ or enriching the dataset should tighten this.
- **Cross-object generalisation:** n/a – metadata lacks multiple object IDs, so leave-one-object-out folds were not available.
- **Unsupervised corroboration:** ✗ SSA / Fourier PCA subspaces sit >80° away. See below.

## Discussion & Next Steps

1. **Background leakage dominates unsupervised screens.** The SSA/DFT procedures latch onto high-variance channels that are not the supervised rotation plane. Pre-filtering with masks or working on residuals (e.g., subtracting the mean background embedding) should help. Integrating motion-focused embeddings (optic flow or masked CLIP features) is another option.
2. **Bootstrap variance remains high.** With only 180 frames, leaving 30% out of each resample removes many viewpoints. Raising the ridge penalty ($\lambda \approx 0.5$–$1.0$) mitigates this somewhat, but a longer clip or additional objects would be better evidence.
3. **Fourier ratio for the cropped run is modest.** The complement still carries first-harmonic energy, likely because the projector only captures two dimensions and reflections/lighting live outside. Expanding the probe with harmonic targets ($[\cos 2\theta, \sin 2\theta]$) and keeping the leading canonical pair would quantify this symmetry channel.
4. **Reportable artefacts.** Latent circles (`latent_circle.png`) and aligned pose codes are in each run’s `rotation_analysis/`. `basis_U.npy`, `projector.npy`, and `procrustes_R.npy` are ready for downstream reuse.

## Recommendations

1. Re-run the pipeline with a stronger ridge penalty (≥0.5) and possibly a PCA compression (e.g., top 128 components) to stabilise the bootstrap angles.
2. Apply the recipe to additional objects (or synthetic rotations) so the cross-object generalisation check can be exercised.
3. Pre-clean embeddings (background subtraction or temporal filtering) before SSA/DFT to reduce their 80° discrepancy.
4. Extend the probe with second-harmonic targets and compare canonical directions to verify the pose really is 2-D.

## Removing rotation for alignment

Use `python scripts/remove_rotation_components.py <run> --harmonics 2` to project embeddings into the nullspace of both first- and second-harmonic pose axes. The script saves residual embeddings, cosine matrices with the original colour scale, and summary statistics (mean cosine gain, angle MAE drop, variance removed) under `rotation_analysis/rotation_removed_h2/`.
