# OpenCLIP Prompt Pipeline v2 – Tuned Iteration

## Overview
- **Objective.** Address the weak logits reported in the first v2 sweep by enriching the prompt bank, removing the rotation subspace before scoring, and turning on temperature-scaled decoding so that the OpenCLIP workflow surpasses the legacy CLIP v1 pipeline rather than tying it.
- **Runs covered.** `Jeep_*`, `Polaris_Red`, and `car_rotation3` sequences under `artifacts/runs/*_v2_tuned/`, reusing the original image towers but rewriting the scoring/decoding stage.
- **Key artifacts.** Updated prompt bank `artifacts/text_dictionary_v2.npz`, pooled scores `artifacts/runs/*_v2_tuned/scores_pooled.npz`, decoded probabilities `artifacts/runs/*_v2_tuned/decoded/*.npy`, and aggregate metrics `artifacts/reports/v2_tuned/*`.

## What Changed
- Added explicit negative prompt tracking (`negative_indices`) so subtraction is possible during pooling; see `src/vector_manifold/v2/promptbank.py:39-146`.
- Extended pooling and decode logic to subtract negative prompts and apply a temperature hyper-parameter (`src/vector_manifold/v2/scoring.py:23-90`, `src/vector_manifold/v2/decode.py:18-71`).
- Updated the CLI wrappers to expose projector removal, negative subtraction, and temperature flags (`scripts_v2/score.py:16-60`, `scripts_v2/decode.py:74-123`).
- Rebuilt `docs/prompts/multi_attribute.json:3-172` with richer color synonyms (body-paint phrasing, side-by-side language) and color-aware negatives to punish confounders (yellow vs. orange, red tail lights, etc.).
- Applied the learned rotation projector from the original CLIP analysis before scoring each run.
- Decoded logits with `τ = 0.07` to sharpen distributions, saved calibrated scores alongside raw logits for downstream inspection.

## Color Results Per Run
| Run | Expected | Predicted | Top p | Margin | Note |
| --- | --- | --- | --- | --- | --- |
| Jeep_Blue | blue | blue | 0.275 | 0.152 | ✅ |
| Jeep_Green | green | green | 0.308 | 0.200 | ✅ |
| Jeep_Indigo | indigo | blue | 0.238 | 0.078 | ⚠️ |
| Jeep_Orange | orange | yellow | 0.225 | 0.094 | ⚠️ |
| Jeep_Red | red | red | 0.231 | 0.116 | ✅ |
| Jeep_Violet | violet | violet | 0.241 | 0.137 | ✅ |
| Jeep_Yellow | yellow | yellow | 0.278 | 0.157 | ✅ |
| Polaris_Red | red | red | 0.174 | 0.040 | ✅ |
| car_rotation3 | white | green | 0.132 | 0.001 | ⚠️ |

- Mean top probability rose to 0.234 (was 0.205) and the mean top-vs-runner margin jumped from 0.003 to 0.108, indicating far more decisive logits (`artifacts/reports/v2_tuned/aggregate_group_metrics.csv:1`).
- Remaining misses mirror the original CLIP behaviour: the orange Jeep leans yellow and the indigo clip often reads as “blue”. The white showcase still ties with green; margins are effectively zero, so further calibration will need labelled negatives or photometric augmentation.

## Other Prompt Groups
- Type, occupancy, armament, and side classes all gained sharpness: mean top probabilities now sit between 0.29 and 0.64 with 0.22–0.32 margins (`artifacts/reports/v2_tuned/aggregate_group_metrics.csv:1-5`). Baseline margins were <0.01.
- The side-by-side Polaris clip now cleanly prefers the `side_by_side` bucket (0.324 top p) while SUVs average 0.44; see per-run distributions in `artifacts/runs/*_v2_tuned/decoded/type_probs.npy`.
- Occupancy and armament distributions finally separate the trivial coin flips (0.61/0.64 vs. 0.50), which is critical for downstream calibration and abstention logic.

## Notes & Next Steps
- Orange vs. yellow remains the hardest split; adding illumination-aware negatives or per-run white-balance normalisation should be explored next. Capturing additional orange exemplars would also help.
- The white showroom clip scores nearly uniform across several colors; a targeted prompt such as “a glossy white SUV with chrome trim” already exists, but additional negatives emphasising green reflections may be required.
- With the new subtraction hooks in place, collecting labelled dev frames will allow us to fit proper vector scaling (`scripts_v2/calibrate.py`) instead of the heuristic τ. That should unlock a fair comparison against the v1 calibrated translator.
