# Rotating Car Embedding Report

## Data Overview
| Run folder | Frames | Approx. duration | Native aspect | Notes |
| --- | --- | --- | --- | --- |
| `artifacts/runs/car_rotation3_fastvit_cropped` | 180 | ~18.0 s (@ 10 FPS) | 173×123 crop | Bounding-box crop around YOLO mask (+5 px) with raw pixels preserved (preview: `.../preview.mp4`) |

All frames were sampled at 10 FPS with the longer side scaled to ~512 px before computing embeddings with `fastvit_sa12` on CPU. The cropped run uses YOLOv8 segmentation to compute a per-frame mask, then crops the tight bounding rectangle plus a 5 px safety margin—keeping the raw pixels and avoiding any black borders. Per-frame angles were interpolated from the provided key frames (front/side/back landmarks) and written to `angle_deg` / `angles.npy`, enabling angle-aware diagnostics.

## Embedding Statistics
| Run | Mean cosine (±SD) | Cosine min / max | Mean Euclidean (±SD) | Cosine @ Δ10 | Euclidean @ Δ10 | Corr(cosine, Δangle) |
| --- | --- | --- | --- | --- | --- | --- |
| `car_rotation3_fastvit_cropped` | 0.766 ± 0.098 | 0.516 / 1.000 | 7.35 ± 1.90 | 0.780 | 7.07 | −0.38 |

*Δ10 means the mean similarity/distance between frames separated by 10 timesteps (≈1 s at 10 FPS).* Raw numbers originate from `artifacts/summaries/embedding_run_summary.csv` and each run’s `frame_delta_metrics.csv`.

## Visual Diagnostics Produced
For every run, the analysis suite writes:
- `trajectory_pca.png` & `pca_components_over_time.png`: 2D latent trajectory projections and temporal traces.
- `cosine_heatmap.png` / `euclidean_heatmap.png`: pairwise similarity matrices revealing periodic structure.
- `cosine_vs_frame_delta.png` / `euclidean_vs_frame_delta.png`: average similarity decay as temporal separation grows.
- `norm_over_time.png` / `successive_cosine.png`: stability of embedding magnitudes and consecutive-frame cosine scores.
- `cosine_histogram.png`: distribution of all pairwise cosine similarities.
- `trajectory_pca_3d.png`: PCA trajectory in 3D, coloured by frame index.
- `angle_over_time.png` / `angle_histogram.png`: estimated viewing angle as a function of frame and coverage.
- `cosine_vs_angle_delta.png`: cosine similarity plotted against absolute angle difference.
- `cosine_heatmap_by_angle.png`: cosine matrix restricted to the first full rotation and ordered by angle.
- `cosine_heatmap_angle_one_rev.png`: chronological cosine matrix for the initial front→front sweep.
- `cosine_heatmap_angle_out_back.png`: chronological cosine matrix spanning the outward revolution plus the return path.
- `diff_norm_over_time.png` / `diff_norm_hist.png`: magnitude of successive embedding differences, visualised over time and as a distribution.
- `diff_pca.png`: principal directions of the difference vectors in the top-2 components.
- `diff_norm_vs_angle_step.png`: relationship between orientation change and embedding delta.
- `norm_over_angle_one_rev.png` / `norm_over_angle_out_back.png`: embedding norm plotted against angle for forward-only and out/back passes.
- `successive_cosine_over_angle_one_rev.png` / `successive_cosine_over_angle_out_back.png`: consecutive-frame cosine similarity versus angle.
- `diff_norm_over_angle_one_rev.png` / `diff_norm_over_angle_out_back.png`: embedding change magnitude versus angle for the same segments.
- `analysis_summary.json` & `frame_delta_metrics.csv`: machine-readable stats for notebooks or further reporting.

Cross-run aggregates live in `artifacts/summaries/embedding_run_summary.csv` for quick sorting.

## Observations
- **YOLO-cropped fly-around** (`car_rotation3_fastvit_cropped`): The mask-driven crop removes most background while preserving reflections and tyres. Cosine similarity remains high (Δ10 ≈0.78) yet Euclidean distance grows faster than in the full-frame baseline, signalling that latent movement is now dominated by body rotation. `cosine_vs_angle_delta.png` shows a clear decay with larger orientation differences (Pearson ≈ −0.38), while the angle-specific heatmaps (`cosine_heatmap_by_angle.png`, `cosine_heatmap_angle_one_rev.png`, `cosine_heatmap_angle_out_back.png`), `diff_norm_over_time.png`, `diff_norm_vs_angle_step.png`, and the PCA plots highlight the smoother latent loop and how vector differences track the car’s rotation.

## Rotation subspace experiments
| Experiment | `car_rotation3_fastvit_cropped` |
| --- | --- |
| Linear probe MAE (full / rot-only / perp-only) | 0.12° / 0.12° / **0.71°** |
| Fourier-selected MAE (top 10 dims) | 26.7° |
| Diff-norm mean (‖z<sub>t+1</sub>−z<sub>t</sub>‖) | 2.86 |

- The **linear probe** finds an almost-perfect 2D rotation plane. Projecting onto its orthogonal complement raises angle MAE to ≥0.5°, confirming rotation energy is concentrated in that subspace. `rotation_analysis/linear_probe_latent_circle.png` shows embeddings forming an almost exact circle.
- The **Fourier screen** identifies repeated high-energy dimensions with strong $k=1$ and $k=2$ harmonics, but a naïve reconstruction (top-10 dims) still leaves ≈26° MAE—useful for ranking rotation-aligned channels yet insufficient on its own.
- Difference-norm plots against angle (`diff_norm_over_angle_*.png`) reveal that large embedding jumps coincide with quarter-turn transitions; the cropped run exhibits larger jumps, matching its higher rotation sensitivity.

## Recommendations & Next Steps
1. **Angle refinement**: swap in higher-precision angle estimates (e.g., optical flow, ArUco markers) and re-run `scripts/assign_car_angles.py` to tighten the cosine-vs-angle relationship.
2. **Temporal smoothing**: apply a moving average to embeddings and compare decay curves to quantify noise reduction, especially for the cropped run.
3. **Model comparison**: rerun the pipeline with an alternative backbone (e.g., `eva02_large_patch14`) to test whether tighter crops remain discriminative.
4. **Add more modalities**: capture exterior night shots or interior cabin spins to broaden the manifold coverage.

Report generated on-demand via the outputs dated in `artifacts/runs/car_rotation*_fastvit/`.
