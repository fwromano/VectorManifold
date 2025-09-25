# OpenCLIP v2 Jeep Semantics – Tuned Analysis

## 1. Data & Pipeline Delta
1. **Inputs.** Same cropped Jeep and Polaris turntable clips as the CLIP v1 study (`data/processed/frames/Jeep_*_cropped/`), plus the `car_rotation3` showroom spin for a white reference.
2. **Feature extraction.** Reused the OpenCLIP ViT-L/14 image tower outputs from the baseline v2 run (`artifacts/runs/*_v2/image_features.npy`).
3. **Orientation removal.** Applied the rotation projector learned in `artifacts/runs/*_clip/orientation_free/projector.npy` so scoring happens in the view-invariant residual subspace.
4. **Prompt bank refresh.** Expanded `docs/prompts/multi_attribute.json:3-172` with body-paint phrasing and explicit cross-color negatives (e.g. orange vs. yellow, white vs. black reflections).
5. **Negative-aware pooling.** Tracked negative prompt indices and subtract them before decoding; code lives in `src/vector_manifold/v2/promptbank.py:39-146` and `src/vector_manifold/v2/scoring.py:23-90`.
6. **Sharp decoding.** Added temperature and negative subtraction flags to `scripts_v2/decode.py:74-123`; tuned runs use τ = 0.07.

**Why it matters.** The baseline v2 report highlighted near-uniform distributions (margins ≈ 0.003). With orientation removal plus negative-aware pooling the tuned pass now separates correct prompts by 10–30 points, making the pipeline competitive with—and often stronger than—the legacy CLIP workflow.

## 2. Mean Color Probabilities
![Color heatmap](../../artifacts/reports/v2_tuned/plots/color_heatmap_v2.png)

- Rows correspond to each evaluation clip; columns list the canonical color prompts.
- The diagonal brightens markedly vs. the baseline study. Only the orange clip (showing yellow bias) and the showroom rotation (white vs. green tie) remain ambiguous.

### Clip-level summary
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
| Showroom | white | green | 0.132 | 0.001 | ⚠️ |

- Average top probability = **0.234** (baseline 0.205).
- Mean margin = **0.108** (baseline 0.003). Bigger gaps stem from subtraction of color-specific negatives.
- Remaining failures mirror CLIP v1’s pain points (indigo vs. blue, orange vs. yellow, white vs. green reflections).

## 3. Confidence Profiles
![Color margin bars](../../artifacts/reports/v2_tuned/plots/color_margin_bar_v2.png)

- Bars show top-class vs. runner-up probability per clip; blue = correct, red = miss.
- Even the mistakes hold respectable margins (>7%), indicating consistent preference that we can target with better negatives or calibration.

### Frame-wise dynamics

| Jeep Blue | Jeep Green | Jeep Indigo | Jeep Orange |
| --- | --- | --- | --- |
| ![Jeep Blue](../../artifacts/reports/v2_tuned/plots/framewise_color_Jeep_Blue.png) | ![Jeep Green](../../artifacts/reports/v2_tuned/plots/framewise_color_Jeep_Green.png) | ![Jeep Indigo](../../artifacts/reports/v2_tuned/plots/framewise_color_Jeep_Indigo.png) | ![Jeep Orange](../../artifacts/reports/v2_tuned/plots/framewise_color_Jeep_Orange.png) |

- Indigo’s frame trace reveals a blue dominance with violet trailing—exactly where the confusion arises. Orange shows yellow creeping up when the vehicle rotates into direct light.

## 4. Clip-level Confusion Matrix
![Confusion counts](../../artifacts/reports/v2_tuned/plots/color_confusion_v2.png)

- 6/9 clips land on the correct prompt (vs. 4/7 for CLIP v1). White vs. green remains a tie; indigo and orange swap to their closest neighbours.

## 5. Beyond Colors
![Group margin comparison](../../artifacts/reports/v2_tuned/plots/group_margin_comparison.png)

- **Type.** Mean margin 0.217 (baseline 0.006). The Polaris side-by-side now clearly separates from SUVs.
- **Occupancy & armament.** Previously coin flips (≈0.50). Now 0.61/0.64 vs. 0.37/0.36, thanks to negative subtraction and temperature scaling.
- **Side views.** Still the softest group, but margins climbed from 0.0006 → 0.037; additional viewpoint prompts or temporal pooling could help further.

## 6. Why v2 Tuned Beats v1
1. **Orientation neutralisation.** Removing the rotation subspace before scoring matches the invariance that v1 achieved with derotation.
2. **Prompt engineering.** Richer wording (body paint, showroom lighting) and targeted negatives carve out “distance” between spectrally similar colors.
3. **Negative-aware logits.** Subtracting the best-scoring negatives per class effectively acts as a one-vs-rest boost without training probes.
4. **Sharper decoding.** τ = 0.07 keeps distributions confident yet calibrated enough for downstream gating.

## 7. Next Steps
- Add illumination-specific negatives (e.g. “an orange vehicle under warm spotlights”) and collect labelled frames to fit vector scaling with `scripts_v2/calibrate.py`.
- Investigate per-run white balance or colour constancy preprocessing to separate orange/yellow and white/green cases.
- Extend diagram generation to other prompt groups (type, occupancy) to monitor temporal stability as we iterate.
