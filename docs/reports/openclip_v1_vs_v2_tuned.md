# CLIP v1 vs. OpenCLIP v2 (Tuned) – Comparison

## Dataset & Metrics
- Clips: seven Jeep colorways (`Jeep_*`), plus `Polaris_Red` and `car_rotation3` for the tuned v2 pass.
- Features: CLIP v1 used rotation-removed OpenCLIP embeddings from the earlier semantic study, while the tuned v2 run reuses the same image tower with projector removal, negative-aware pooling, and τ=0.07 decoding.
- Statistics: mean top probability and mean top-vs-runner margin per prompt group (`artifacts/reports/v2_tuned/aggregate_group_metrics.csv:1-5`). CLIP v1 baselines pulled from `artifacts/reports/clip_summary/semantic_distributions_jeep.csv`.

## Group-Level Summary
| Group | v1 Top p | v1 Margin | v2 Top p | v2 Margin |
| --- | --- | --- | --- | --- |
| color | 0.191 | 0.060 | 0.234 | 0.108 |
| type | 0.412 | 0.095 | 0.433 | 0.217 |
| occupancy | 0.501 | 0.001 | 0.610 | 0.219 |
| armament | 0.503 | 0.006 | 0.636 | 0.272 |
| side | 0.251 | 0.001 | 0.290 | 0.037 |

- v2 margins dwarf the original translator across every group thanks to the negative-subtraction and projector stages.
- Occupancy/armament moved from coin flips to decisive splits (>0.63 vs. >0.36), unlocking downstream calibration.
- Side-view logits remain the softest, but the tuned pass still quadruples the average margin relative to v1.

## Color Accuracy
- CLIP v1 guessed four of seven Jeep colors exactly; indigo was mapped to “purple” and orange to “yellow”.
- The tuned pipeline hits six of nine runs (Jeep blue/green/red/yellow/violet plus Polaris) and raises the average top probability by +4.3 points. Remaining misses mirror v1’s trouble spots (orange vs. yellow, indigo vs. blue) and the showroom `car_rotation3` clip where white ties with green; see `artifacts/reports/v2_tuned/color_run_summary.csv`.

## Implementation Highlights
- Prompt bank revamp with body-paint phrasing and cross-color negatives (`docs/prompts/multi_attribute.json:3-139`).
- Orientation removal and negative-aware pooling before decoding (`scripts_v2/score.py:16-60`, `src/vector_manifold/v2/scoring.py:23-90`).
- Tunable temperature/negative subtraction at decode time (`scripts_v2/decode.py:74-123`, `src/vector_manifold/v2/decode.py:18-71`).

## Remaining Gaps
- Orange vs. yellow still needs either labelled calibration frames or stronger illumination modelling.
- White showroom clip is indistinguishable from pale green under current prompts; we recommend synthesising targeted negatives (“a vehicle reflecting green foliage”) or adding polarization-aware augmentations.
- Collecting a small dev split would enable proper vector scaling (rather than the uniform τ) so we can quantify confidence calibration head-to-head with the v1 translator.
