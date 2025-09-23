# CLIP Analysis of Cropped Jeep Rotations

## 1. Data Preparation Pipeline
1. **Frame extraction.** Each Jeep colorway video (Blue, Green, Indigo, Orange, Red, Violet, Yellow) is re-sampled at 2 FPS from the native-resolution source. We downsample the entire clip to 361 frames (≈180 seconds).
2. **YOLO segmentation & cropping.** For every frame we run YOLOv8 segmentation with car + truck classes, take the dominant mask, and crop to the tight bounding box (margin = 5 pixels). This removes background clutter while preserving the vehicle.
3. **Deduplication.** A normalized mean-absolute-difference (MAD) filter removes near-duplicate frames. After deduping we retain ~287–357 frames per colorway (see Table 1).
4. **Outputs.** Cropped frames live in `data/processed/frames/Jeep_<Color>_cropped/`. These images feed every downstream experiment in this report.

**Table 1. Frame counts per colorway (post-dedup, native resolution)**

| Colorway | Frames |
| --- | ---: |
| Blue | 346 |
| Green | 287 |
| Indigo | 353 |
| Orange | 356 |
| Red | 353 |
| Violet | 357 |
| Yellow | 360 |

## 2. CLIP Embedding Overview
- **Model.** `open_clip` ViT-L/14 (OpenAI weights), embeddings are 768-D and L2-normalized.
- **Why CLIP?** The multimodal training encourages semantic alignment: visually similar frames should land near text prompts like “a photo of a green vehicle”.
- **Artifacts.** Per-run embeddings and metadata: `artifacts/runs/Jeep_<Color>_clip/embeddings.npy` etc.

## 3. Cross-Color Cosine Similarity
- **Aligned frames.** We truncate each run to the minimum length (287 frames) and compute cosine similarities between same-index frames across colorways.
- **Heatmap.** High values (≥0.95) show CLIP’s robustness to paint changes.

![Jeep cross-color cosine heatmap](../../artifacts/reports/clip_summary/cosine_color_heatmap_jeep.png)

- **Distribution check.** [cosine_color_distribution_boxplot.png](../../artifacts/reports/clip_summary/cosine_color_distribution_boxplot.png) compares cross-color (same angle) and same-color (all angles) cosines. Cross-color similarities remain within the high-similarity band seen intra-color (median ≈0.915 vs. ≈0.94; 99th percentile ≈0.975 vs. ≈0.988).

![Cross vs. same color cosine distributions](../../artifacts/reports/clip_summary/cosine_color_distribution_boxplot.png)

- **Angle structure.** The distributions hide where the outliers live, so we average each same-color angle×angle cosine matrix and plot the per-angle cross-color cosines alongside it. The top panel highlights the bright diagonal ridge (matched viewpoints) and how similarity decays at quarter/half turns, while the lower panel shows which color pairs dip for specific headings.

![Angular cosine structure](../../artifacts/reports/clip_summary/cosine_angle_structure.png)

## 4. Cross-Run Orientation Subspace (summary)
- **Global basis.** Fitting a pooled orientation ridge probe yields a tiny rotation subspace (<0.18% energy). Removing it boosts cosine similarity slightly across colors.
- **Color means.** Orientation-free color means cluster tightly (heatmap in main report).

## 5. Semantic Likelihood Mapping
### 5.1 Process
1. **Optional rotation removal.** For viewpoint-invariant scores we subtract the rotation component (using either per-run or pooled basis). Raw variant leaves embeddings untouched.
2. **L2 normalisation.** Ensures cosine = dot product.
3. **Cosine vs. prompt.** Compare embeddings against a cached text dictionary (color & vehicle prompts).
4. **Temperature softmax.** Convert cosines to probabilities with `p_i = exp((cos_i/τ) - max_j)/Σ exp((cos_k/τ) - max_j)`. Smaller τ sharpens distributions.
5. **Aggregation.** Average per-frame probabilities across the sequence.

**Prompt inventory used for likelihood mapping**

| Group | Prompts |
| --- | --- |
| Colors | white, black, gray, red, blue, green, orange, yellow, purple vehicles |
| Vehicle type | car, truck, SUV |
| Views | front, right side, rear, left side, driver side, passenger side, front three-quarter, rear three-quarter, top |

### 5.2 Color & Vehicle Distributions (τ = 0.1, rotation removed)

**Figure 1. Color likelihoods (Jeep only)**

![Color heatmap](../../artifacts/reports/clip_summary/color_likelihood_heatmap_jeep.png)

**Figure 2. Vehicle-type likelihoods (Jeep only)**

![Vehicle stacked bar](../../artifacts/reports/clip_summary/vehicle_likelihood_stacked_jeep.png)

**Table 2. Top color & vehicle per run (τ = 0.1, rotation removed)**

| Run | Top color (prob) | Top vehicle (prob) |
| --- | --- | ---: |
| Blue | blue (0.186) | SUV (0.422) |
| Green | green (0.218) | SUV (0.398) |
| Indigo | purple (0.169) | SUV (0.417) |
| Orange | yellow (0.174) | SUV (0.414) |
| Red | red (0.182) | SUV (0.420) |
| Violet | purple (0.227) | SUV (0.405) |
| Yellow | yellow (0.180) | SUV (0.408) |

Raw data: `artifacts/reports/clip_summary/semantic_distributions_jeep.csv`

### 5.3 Temperature Sweep (raw embeddings)
- Lower τ (0.01–0.03) yields very sharp distributions (dominant color probability → 1.0).
- Higher τ (0.1–0.2) spreads probability mass.

**Figure 3. Dominant color probability vs. τ**

![Temperature color](../../artifacts/reports/clip_summary/temperature_dominant_color.png)

**Figure 4. SUV probability vs. τ**

![Temperature SUV](../../artifacts/reports/clip_summary/temperature_suv_probability.png)

**Table 3. Sample color distributions (τ = 0.05, raw)**

| Run | Top color (prob) | SUV prob |
| --- | --- | ---: |
| Blue | blue (0.291) | 0.513 |
| Green | green (0.385) | 0.472 |
| Indigo | purple (0.240) | 0.503 |
| Orange | yellow (0.253) | 0.497 |
| Red | red (0.281) | 0.509 |
| Violet | purple (0.403) | 0.479 |
| Yellow | yellow (0.272) | 0.484 |

**Table 4. Sample color distributions (τ = 0.01, raw)**

| Run | Top color (prob) | SUV prob |
| --- | --- | ---: |
| Blue | blue (0.966) | 0.939 |
| Green | green (0.996) | 0.853 |
| Indigo | purple (0.582) | 0.923 |
| Orange | yellow (0.727) | 0.902 |
| Red | red (0.949) | 0.933 |
| Violet | purple (0.997) | 0.878 |
| Yellow | yellow (0.841) | 0.873 |

Full table: `artifacts/reports/clip_summary/semantic_temperature_effects.csv`

## 6. Rotation Removal vs. Raw
- Removing the rotation component concentrates color predictions and slightly boosts SUV confidence.
- The learned rotation subspace accounts for only ~0.12–0.18% of embedding energy across runs, so these deltas are on the order of 10⁻³ but directionally consistent.
- The learned rotation subspace accounts for only ~0.12–0.18% of embedding energy across runs, leaving cosine structure essentially unchanged (see cross-run orientation summary).

**Figure 5. Color probability delta (rot removed − raw, values ×10⁻³)**

![Color delta](../../artifacts/reports/clip_summary/color_likelihood_delta_heatmap_jeep.png)

**Figure 6. SUV probability delta (rotation removed − raw, ×10⁻³)**

![SUV delta bar](../../artifacts/reports/clip_summary/vehicle_suv_delta_bar.png)

**Table 5. Delta summary (τ = 0.1)**

| Run | Δ color peak | Δ SUV |
| --- | ---: | ---: |
| Blue | +0.0005 | +0.0000 |
| Green | −0.0010 | −0.0038 |
| Indigo | +0.0002 | +0.0003 |
| Orange | +0.0004 | +0.0004 |
| Red | +0.0003 | +0.0003 |
| Violet | +0.0005 | +0.0003 |
| Yellow | +0.0004 | +0.0004 |

Detailed CSV: `semantic_distribution_comparison_jeep.csv`

## 7. Key Takeaways
- CLIP embeddings stay highly similar across Jeep colorways (cos ≥0.95 when frames align).
- The pooled rotation basis removes viewpoint bias with negligible energy loss (<0.2%).
- Temperature controls how decisive the semantic likelihoods become; τ = 0.1 balances certainty and robustness.
- Rotation removal slightly sharpens color/vehicle predictions, but even raw embeddings strongly identify color and vehicle type.

All plots/tables referenced here live under `artifacts/reports/clip_summary/` or `artifacts/runs/Jeep_<Color>_clip/`.

## 8. Potential Extensions
- Benchmark the same workflow with rotation-compensated CLIP embeddings to confirm negligible changes across larger prompt libraries (e.g., body style, accessories).
- Add angular alignment when comparing colors (e.g., front vs. front only) to quantify any remaining viewpoint sensitivity.
- Train lightweight probes on the pooled orientation-free embeddings to separate color vs. texture subspaces explicitly.
- Incorporate confidence intervals by bootstrapping frame subsets, providing uncertainty bands on the likelihood plots.
