# FastViT Semantic Readout Evaluation

## Overview
- Backbone: `fastvit_sa12` embeddings from `artifacts/runs/car_rotation3_fastvit_cropped`
- Semantic paths: (A) logistic probes on FastViT features, (B) FastViT→CLIP linear translator with cached text dictionary; translator cosine scores are converted to softmax likelihoods per prompt group (temperature = 0.1).
- Rotation handling: 2D pose plane learned via ridge regression on `(cos θ, sin θ)` and projected out before evaluation
- Resources: evaluation bundle `artifacts/eval/car_rotation_semantic.npz`, dictionary `artifacts/text_dictionary.npz`
- Labels: `side`, `type`, `occupancy`, `armament` (heuristics derived from viewing angle; replace with real annotations for production)

-## Metrics
- Pose plane: dev $R^2$ drops $0.64 \to -0.86$ after projecting the 2D rotation basis; same-object cosine rises slightly (`0.767 → 0.772`).
- Translator alignment stays high (`0.998 → 0.996` train, `0.980 → 0.974` dev), confirming the FastViT→CLIP map remains consistent post-derotation.

### Probe accuracies (dev/test)
| Label | Raw (dev / test) | Derotated (dev / test) |
| --- | --- | --- |
| side | 0.861 / 0.639 | **0.889** / 0.639 |
| type | **0.806** / **0.917** | 0.722 / 0.861 |
| occupancy | **0.944** / **0.972** | 0.917 / 0.972 |
| armament | 0.556 / **1.000** | 0.556 / **1.000** |

### Translator accuracies (dev/test)
| Label | Raw | Derotated |
| --- | --- | --- |
| side | 0.611 / 0.222 | 0.444 / 0.222 |
| type | 0.444 / 0.000 | 0.444 / 0.000 |
| occupancy | 0.306 / 0.167 | 0.306 / 0.167 |
| armament | **0.694 / 1.000** | **0.694 / 0.972** |

- Accuracy scores use the heuristic labels; real annotations should replace them for meaningful evaluation.

## Visual Diagnostics
Saved under `artifacts/runs/car_rotation3_fastvit_cropped/semantic_eval/`:
- `pose_r2.png`: bar chart of pose predictability pre/post.
- `same_object_cosine.png`: mean cosine between paired cross-views.
- `probe_accuracy_dev.png`, `probe_accuracy_test.png`: probe comparison charts.
- `translator_alignment.png`: CLIP alignment bars.
- `translator_accuracy_dev.png`, `translator_accuracy_test.png`: translator accuracy bars.

## Example Frame Descriptions
Translator predictions (after rotation removal) expressed as softmax likelihoods over each prompt group. Percentages denote the probability mass assigned to the top options; remaining prompts in the group share the leftover probability.

| Frame | Angle (deg) | Type likelihood | Occupancy likelihood | Armament likelihood | Side likelihood | Vehicle likelihood | Color likelihood |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 275.3 | car 46.0%, truck 31.6%, person 22.3% | empty 53.2%, driver 46.8% | unarmed 51.7%, armed 48.3% | right 27.8%, front 26.9%, left 25.4% | car 37.5%, suv 36.8%, truck 25.7% | white 25.2%, gray 23.4%, black 19.2% |
| 30 | 74.1 | car 44.8%, truck 31.5%, person 23.8% | driver 52.1%, empty 47.9% | unarmed 52.1%, armed 47.9% | right 25.8%, left 25.4%, front 24.9% | suv 39.3%, car 35.6%, truck 25.1% | white 25.1%, gray 24.8%, black 18.8% |
| 60 | 255.0 | car 45.7%, truck 29.9%, person 24.4% | driver 51.6%, empty 48.4% | unarmed 51.5%, armed 48.5% | right 25.8%, left 25.4%, front 24.9% | suv 38.7%, car 37.0%, truck 24.3% | white 25.7%, gray 23.8%, black 19.0% |
| 90 | 313.2 | car 45.9%, truck 33.0%, person 21.2% | empty 50.9%, driver 49.1% | unarmed 50.3%, armed 49.7% | right 28.9%, front 27.6%, left 24.7% | car 37.5%, suv 35.6%, truck 26.9% | gray 26.3%, white 22.6%, black 19.4% |
| 120 | 140.6 | car 46.5%, truck 29.7%, person 23.7% | empty 51.2%, driver 48.8% | unarmed 51.0%, armed 49.0% | right 26.8%, left 26.4%, front 23.4% | car 38.2%, suv 37.4%, truck 24.4% | gray 24.0%, white 23.9%, black 19.4% |
| 150 | 27.0 | car 46.6%, truck 31.2%, person 22.1% | empty 52.5%, driver 47.5% | unarmed 51.1%, armed 48.9% | right 26.7%, front 26.0%, left 25.6% | suv 38.5%, car 36.8%, truck 24.7% | gray 24.2%, white 23.8%, black 18.8% |
| 170 | 327.0 | car 45.7%, truck 31.8%, person 22.5% | empty 52.4%, driver 47.6% | unarmed 51.0%, armed 49.0% | front 27.4%, right 27.1%, left 25.5% | suv 37.6%, car 36.8%, truck 25.6% | white 24.1%, gray 23.3%, black 19.7% |

Likelihoods use a temperature of 0.1 for sharper decoding. Dataset-level averages appear in `translator_distribution_{type,occupancy,armament,side}_{raw,derotated}.png`. Residual ambiguity reflects the heuristic labels—curated annotations or additional prompts would further tighten the distributions.

## Notes
- Replace the heuristic CSVs with annotated data for credible metrics, especially for `type`, `occupancy`, and `armament`.
- Update `docs/prompts/multi_attribute.json` to match your label inventory; rebuild the dictionary afterwards (`semantic_describer.py build-dict`).
- Translator accuracy drops highlight the need for balanced training pairs—collect more CLIP-image samples that cover each class to stabilise the linear map.
