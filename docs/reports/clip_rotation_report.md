# CLIP Embedding Analysis of Jeep/Polaris Rotations

## Overview
We extracted 2 FPS, native-resolution crops for eight rotating vehicle videos (Jeep colorways + Polaris Red), yielding 287–360 frames per sequence after YOLO segmentation. Each frame was embedded with `open_clip` ViT-L/14 (OpenAI weights, L2-normalised). We computed pairwise cosine/Euclidean matrices, per-frame norms, successive step deltas, and angle-aligned diagnostics before generating the figure suite below.

## Aggregate Metrics
- **Frames per sequence:** 287 (Jeep_Green) → 360 (Jeep_Yellow); see [frame_counts_bar.png](../../artifacts/reports/clip_summary/frame_counts_bar.png).
- **Mean pairwise cosine:** 0.931–0.943 across runs; the highest cohesion is Jeep_Violet (0.943). Plot: [cosine_mean_bar.png](../../artifacts/reports/clip_summary/cosine_mean_bar.png).
- **Mean successive ‖Δembedding‖:** ranges 0.14–0.18; Jeep_Green shows the largest frame-to-frame motion (0.180) due to intermittent YOLO drops. Plot: [diff_norm_mean_bar.png](../../artifacts/reports/clip_summary/diff_norm_mean_bar.png).
- **Cosine decay vs. frame separation:** all curves decrease smoothly with Δ; see overlay [cosine_vs_delta_overlay.png](../../artifacts/reports/clip_summary/cosine_vs_delta_overlay.png).

| Run | Frames | Mean cosine | Mean ‖Δ‖ |
| --- | ---: | ---: | ---: |
| Jeep_Blue | 346 | 0.9315 | 0.1689 |
| Jeep_Green | 287 | 0.9348 | 0.1802 |
| Jeep_Indigo | 353 | 0.9401 | 0.1433 |
| Jeep_Orange | 356 | 0.9380 | 0.1480 |
| Jeep_Red | 353 | 0.9344 | 0.1679 |
| Jeep_Violet | 357 | 0.9434 | 0.1491 |
| Jeep_Yellow | 360 | 0.9400 | 0.1398 |
| Polaris_Red | 294 | 0.9366 | 0.1486 |

## Per-Run Highlights
- **Heatmaps:** `cosine_heatmap.png` and `cosine_heatmap_by_angle.png` reveal strong block structure with angle-aligned diagonals for each run.
- **Angle-aligned norms:** `norm_over_angle_one_rev.png` shows mild norm modulation (~±0.02) as the vehicles rotate.
- **Step deltas:** `diff_norm_over_angle_out_back.png` surfaces the sections with the fastest latent motion (Jeep_Green spikes near frames where YOLO loses track).
- **Trajectory PCA:** `trajectory_pca_3d.png` visualises a smooth loop in CLIP space; Jeep_Violet and Jeep_Yellow exhibit especially consistent loops.

All figures live under `artifacts/runs/<run>_clip/`. Key examples:
- Jeep_Blue: [Cosine heatmap](../../artifacts/runs/Jeep_Blue_clip/cosine_heatmap.png), [Angle-sorted heatmap](../../artifacts/runs/Jeep_Blue_clip/cosine_heatmap_by_angle.png), [PCA loop](../../artifacts/runs/Jeep_Blue_clip/trajectory_pca_3d.png).
- Jeep_Green: [Diff norm over time](../../artifacts/runs/Jeep_Green_clip/diff_norm_over_time.png), [Cosine vs angle](../../artifacts/runs/Jeep_Green_clip/cosine_vs_angle_delta.png).
- Polaris_Red: [Successive cosine](../../artifacts/runs/Polaris_Red_clip/successive_cosine.png), [Angle histogram](../../artifacts/runs/Polaris_Red_clip/angle_histogram.png).

## Interpretation
1. **CLIP remains rotation-aware.** Pairwise cosines stay >0.93, yet drift monotonically with angle separation (overlay plot). Latent loops are circular with minimal hysteresis, echoing prior FastViT findings.
2. **Color variations shift dynamics subtly.** Jeep_Violet and Jeep_Yellow maintain the tightest trajectories (lowest ‖Δ‖), while Jeep_Green’s masks induce sharper jumps (diff_norm_max ≈0.394).
3. **Polaris vs. Jeep.** Polaris_Red retains high cosine despite a distinct shape; its cosine decay mirrors Jeep curves, confirming CLIP’s shape invariance under color/style changes.
4. **Mask quality affects smoothness.** Runs with occasional segmentation loss (Jeep_Green, Jeep_Red) show higher variance in successive cosine metrics—worth revisiting YOLO thresholds if smoother embeddings are desired.

## Semantic Attribute Likelihoods
- **Color inference:** The CLIP text prompts cleanly recover the expected paint scheme for each rotation; heatmap: [color_likelihood_heatmap.png](../../artifacts/reports/clip_summary/color_likelihood_heatmap.png).
- **Vehicle class:** All sequences favour the SUV prompt (>0.39 probability), per stacked bars in [vehicle_likelihood_stacked.png](../../artifacts/reports/clip_summary/vehicle_likelihood_stacked.png).
- **View distribution:** Averaged likelihoods across the loop remain balanced, but left/right dominate in most runs (see [view_likelihood_heatmap.png](../../artifacts/reports/clip_summary/view_likelihood_heatmap.png)), mirroring the camera arc.
- **Aggregated JSON:** Detailed per-run means sit in `artifacts/runs/<run>_clip/semantic_summary.json`; the combined digest is [orientation_semantic_overview.json](../../artifacts/reports/clip_summary/orientation_semantic_overview.json).

## Orientation-Neutral Embeddings
- **Energy budget:** Only ~0.7–1.1% of CLIP embedding energy lies in the rotation subspace (bar plot: [rotation_energy_fraction.png](../../artifacts/reports/clip_summary/rotation_energy_fraction.png)).
- **Cosine boost:** Projecting out the learned rotation basis lifts mean pairwise cosine by 0.006–0.009 across runs—see [cosine_comparison.png](../../artifacts/reports/clip_summary/cosine_comparison.png) and CSV [orientation_summary_table.csv](../../artifacts/reports/clip_summary/orientation_summary_table.csv).
- **Outputs:** Orientation-free embeddings live under `artifacts/runs/<run>_clip/orientation_free/`, ready for downstream similarity comparisons that ignore viewpoint.

## Cross-Color Consistency
- **Color-to-color cosine:** Aligning frames across chromatic variants shows CLIP remains highly consistent; mean cosines stay ≥0.95 for every pair (matrix: [cosine_color_matrix.png](../../artifacts/reports/clip_summary/cosine_color_matrix.png)). Selected distributions (e.g. Blue↔Red, Blue↔Green) are plotted in [cosine_color_hist_selected.png](../../artifacts/reports/clip_summary/cosine_color_hist_selected.png).
- **Jeep subset:** The Jeep-only mean cosine heatmap ([cosine_color_heatmap_jeep.png](../../artifacts/reports/clip_summary/cosine_color_heatmap_jeep.png)) shows the colorways stay tightly aligned (≥0.95) when frames are synchronized across runs.
- **Method:** We truncated each run to the shortest sequence (287 frames), L2-normalised embeddings, and computed per-frame dot products with matching indices across colors. Summary statistics live in [color_pair_cosine_stats.csv](../../artifacts/reports/clip_summary/color_pair_cosine_stats.csv).
- **Distribution check:** [cosine_color_distribution_boxplot.png](../../artifacts/reports/clip_summary/cosine_color_distribution_boxplot.png) compares cross-color (same angle) and same-color (all angles) cosines. Cross-color median ≈0.915 (99th percentile ≈0.975) remains below same-color values (median ≈0.94, 99th percentile ≈0.988), confirming cross-color pairs never fall outside the high-similarity band.

![Cross vs. same color cosine distributions](../../artifacts/reports/clip_summary/cosine_color_distribution_boxplot.png)

## Cross-Run Subspace Analysis
- **Shared orientation basis:** Fitting a single ridge probe across all eight runs yields a global basis capturing ~0.12–0.18 % of energy per sequence (bar: [cross_orientation_energy.png](../../artifacts/reports/clip_summary/cross_orientation_energy.png), JSON: `artifacts/runs/cross_analysis/orientation_summary.json`). Residual cosines climb ~0.0015–0.0020 beyond the run-specific figures ([cross_orientation_cosines.png](../../artifacts/reports/clip_summary/cross_orientation_cosines.png)).
- **Color subspace:** Averaging the orientation-free embeddings by color and running SVD highlights two dominant color axes explaining ≈60 % of variance (`color_summary.json`). The orientation-neutral mean cosine map is in [color_mean_cosine_heatmap.png](../../artifacts/reports/clip_summary/color_mean_cosine_heatmap.png), confirming color clustering once rotation is removed.
- **Outputs:** Global artefacts reside in `artifacts/runs/cross_analysis/` (`orientation_basis.npy`, `color_basis.npy`, etc.), ready for reuse when normalising new runs.

## Likelihood Mapping Pipeline
1. **Optional rotation removal.** We project each embedding onto the residual of a learned rotation subspace (run-specific or pooled) when we want viewpoint invariance; the “raw” variant skips this step for comparison.
2. **Normalisation.** Vectors are L2-normalised so CLIP dot products become cosine similarities against the cached text dictionary (ViT-L/14 prompts covering colors, vehicle classes, views, etc.).
3. **Temperature softmax.** For each prompt group we treat the cosine similarities as logits: take the `cos(frame, prompt)` values, divide by τ = 0.1 to control sharpness, subtract the largest logit for numerical stability, and exponentiate. The resulting vector is then normalised so the probabilities sum to 1. Formally,
   
   `p_i = exp((cos_i/τ) - max_j (cos_j/τ)) / Σ_k exp((cos_k/τ) - max_j (cos_j/τ))`.
   
   This produces a full distribution over prompts (color, vehicle type, etc.) for each frame. We repeat the procedure with and without rotation removal to expose viewpoint bias, then average the per-frame probabilities to obtain the run-level summaries and comparison heatmaps.
4. **Automation.** `semantic_describer.py describe` wraps the workflow and emits both frame-wise details (`semantic_details*.json`) and run-level means (`semantic_summary*.json`).
5. **Temperature sweeps.** We also run the describer at τ ∈ {0.01, 0.03, 0.05, 0.1, 0.2} (without rotation removal) to gauge sensitivity. Lower τ sharpens the distribution (higher dominant-color confidence), while higher τ spreads mass across prompts. See [temperature_dominant_color.png](../../artifacts/reports/clip_summary/temperature_dominant_color.png), [temperature_suv_probability.png](../../artifacts/reports/clip_summary/temperature_suv_probability.png), and [semantic_temperature_effects.csv](../../artifacts/reports/clip_summary/semantic_temperature_effects.csv) for the trend lines and table.
5. **Temperature sweeps.** We also run the describer at τ ∈ {0.05, 0.1, 0.2} (without rotation removal) to gauge sensitivity. Lower τ sharpens the distribution, raising the dominant-color confidence; higher τ smooths it. See [temperature_dominant_color.png](../../artifacts/reports/clip_summary/temperature_dominant_color.png), [temperature_suv_probability.png](../../artifacts/reports/clip_summary/temperature_suv_probability.png), and [semantic_temperature_effects.csv](../../artifacts/reports/clip_summary/semantic_temperature_effects.csv) for details.

## Semantic Distributions
- **Color likelihoods:** The aggregated probabilities (temperature 0.1) zero in on the expected paint scheme for each rotation; see [color_likelihood_heatmap.png](../../artifacts/reports/clip_summary/color_likelihood_heatmap.png). A Jeep-only view is provided in [color_likelihood_heatmap_jeep.png](../../artifacts/reports/clip_summary/color_likelihood_heatmap_jeep.png).
- **Vehicle type bias:** CLIP prefers the “SUV” prompt for all sequences (stacked bars: [vehicle_likelihood_stacked.png](../../artifacts/reports/clip_summary/vehicle_likelihood_stacked.png)); the Jeep subset is in [vehicle_likelihood_stacked_jeep.png](../../artifacts/reports/clip_summary/vehicle_likelihood_stacked_jeep.png).
- **Tables:** Full distributions sit in [semantic_distributions.csv](../../artifacts/reports/clip_summary/semantic_distributions.csv); Jeep-only extracts plus top picks live in [semantic_distributions_jeep.csv](../../artifacts/reports/clip_summary/semantic_distributions_jeep.csv) and [semantic_top_jeep.csv](../../artifacts/reports/clip_summary/semantic_top_jeep.csv).
- **Rotation vs. raw comparison:** Removing the rotation component slightly sharpens the distributions—see the color deltas in [color_likelihood_delta_heatmap_jeep.png](../../artifacts/reports/clip_summary/color_likelihood_delta_heatmap_jeep.png) and the SUV likelihood comparison in [vehicle_suv_comparison_jeep.png](../../artifacts/reports/clip_summary/vehicle_suv_comparison_jeep.png). Raw vs. compensated values are tabulated in [semantic_distribution_comparison_jeep.csv](../../artifacts/reports/clip_summary/semantic_distribution_comparison_jeep.csv).

## Next Steps
1. Compare CLIP vs. FastViT directly (e.g., overlay cosine-vs-angle curves across models).
2. Feed these embeddings to the semantic describer / evaluator to test text alignment along the rotation.
3. Adjust YOLO confidence for Green/Red to reduce spikes, then regenerate embeddings for cleaner curves.

All assets are ready under `artifacts/reports/clip_summary/` and `artifacts/runs/*_clip/` for further exploration.
