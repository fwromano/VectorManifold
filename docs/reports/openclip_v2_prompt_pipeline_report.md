# OpenCLIP Prompt Pipeline v2 – Experiment Report

## 1. Goal, Data, and Tooling
- **Objective.** Evaluate the v2 dual-encoder (OpenCLIP) workflow for translating rotation videos of vehicles into prompt-level probabilities, replacing the legacy FastViT→translator stack.
- **Datasets.** Nine cropped frame sequences (`car_rotation3` plus eight Jeep/Polaris runs), 180–360 frames each under `data/processed/frames/*_cropped`.
- **Models & prompts.** OpenCLIP ViT-L/14 (OpenAI weights) for both image and text towers; prompt bank built from `docs/prompts/multi_attribute.json` spanning color, type, occupancy, armament, and side-view groups.
- **Outputs.** For every run we generated `image_features.npy`, `scores_raw.npy`, synonym-pooled `scores_pooled.npz`, per-frame probability JSON + `{group}_probs.npy` via the new `scripts_v2/decode.py`, and aggregate summaries under `artifacts/reports/v2/`.

## 2. Workflow Summary
1. **Feature extraction.** `scripts_v2/extract_img_feats.py --model ViT-L-14 --pretrained openai --device cpu --no-fp16` across all `_cropped` frame folders, writing to `artifacts/runs/<run>_v2/`.
2. **Prompt bank.** Added list-handling to `vector_manifold.v2.promptbank.merge_prompt_configs` and built `artifacts/text_dictionary_v2.npz` from `multi_attribute.json`.
3. **Scoring.** `scripts_v2/score.py --text-dict artifacts/text_dictionary_v2.npz` to compute raw and synonym-pooled cosine scores per run.
4. **Quality metrics.** `scripts_v2/score_quality.py` logged Laplacian focus, bounding-box span, and mask occupancy (all masks span the frame post-cropping).
5. **Decoding.** New CLI `scripts_v2/decode.py` wraps `vector_manifold.v2.decode.decode_probabilities`, emitting per-frame JSON and group-wise probability tensors; calibration hooks are in place but not activated (no labelled dev split available yet).
6. **Summaries.** Aggregated mean probabilities, top-class margins, and quality stats into CSVs under `artifacts/reports/v2/` for reporting.

## 3. Experimental Findings
### 3.1 Color Recognition Is Weak Without Additional Cues
Using only positive color prompts yields nearly uniform distributions (mean top prob ≈0.205, mean margin 0.0031). The model occasionally lands on the correct hue (Blue/Red cases) but confuses green/orange/violet/yellow runs for “gray/black” vehicles.

| Run | Expected | Predicted | Top prob | Δ vs. uniform |
| --- | --- | --- | ---: | ---: |
| Jeep_Blue | blue | blue | 0.209 | 0.009 |
| Jeep_Green | green | black | 0.202 | 0.002 |
| Jeep_Indigo | indigo | blue | 0.208 | 0.008 |
| Jeep_Orange | orange | gray | 0.203 | 0.003 |
| Jeep_Red | red | red | 0.207 | 0.007 |
| Jeep_Violet | violet | gray | 0.203 | 0.003 |
| Jeep_Yellow | yellow | gray | 0.203 | 0.003 |
| Polaris_Red | red | red | 0.204 | 0.004 |
| car_rotation3 | white | white | 0.205 | 0.005 |

**Takeaway.** Without negatives, compositional prompts, or calibration, OpenCLIP treats color classes as near-equivalent; probability gains over the uniform baseline (<1%) are too small for reliable inference.

### 3.2 Vehicle Type Predictions Are Slightly More Decisive
Type prompts (person / car / truck) show marginal separation (mean margin ≈0.006). All Jeep runs favour “car”, the Polaris side-by-side tips toward “truck” but with only a 0.003 margin.

| Run | Expected | Predicted | Top prob | Margin |
| --- | --- | --- | ---: | ---: |
| Jeep_Blue | car | car | 0.342 | 0.005 |
| Jeep_Green | car | car | 0.343 | 0.006 |
| Jeep_Indigo | car | car | 0.342 | 0.005 |
| Jeep_Orange | car | car | 0.342 | 0.006 |
| Jeep_Red | car | car | 0.342 | 0.005 |
| Jeep_Violet | car | car | 0.343 | 0.005 |
| Jeep_Yellow | car | car | 0.342 | 0.006 |
| Polaris_Red | truck | truck | 0.339 | 0.003 |
| car_rotation3 | car | car | 0.345 | 0.013 |

**Takeaway.** Signals align with expectations but remain soft; a 0.34 probability ceiling indicates the need for sharper prompts or calibration scaling.

### 3.3 Occupancy, Armament, and Side Groups Are Near Uniform
- **Occupancy / armament.** Top probabilities hover at 0.501 ± 0.001 with margins below 0.003, essentially coin flips.
- **Side views.** Distributions cluster around 0.25 per class (mean margin 0.0006). Without rotation compensation or angle labels, the model lacks viewpoint certainty.

### 3.4 Image Quality Metrics Confirm Clean Inputs
`artifacts/reports/v2/quality_summary.csv` shows mask coverage at 1.0 (cropped frames fill the bounding box) and well-focused imagery (Laplacian variance 470–1850 depending on clip). The polished `car_rotation3` clip is the sharpest (1853 variance), while Jeep Orange is softest (~467).

## 4. Limitations & Gaps
- **Calibration skipped.** The `v2` calibration stack (global τ, τ(x), energy gate) requires labelled dev splits; none were available for these runs, so decoded probabilities are raw softmax scores.
- **Prompt coverage.** The base prompt file lacks negatives, multi-word descriptors, or templates tying color to context (e.g., “a blue SUV” vs. “a gray SUV”), leading to near-uniform cosines.
- **Viewpoint invariance.** No temporal pooling or unsupervised derotation was enabled; side-group predictions stay ambiguous.
- **Occupancy/armament supervision.** Without targeted prompts or annotations, these groups provide no actionable signal.

## 5. Recommended Next Steps
1. **Augment prompt bank.** Add color-specific negatives (“not a gray vehicle”), multi-attribute prompts (“a bright yellow Jeep SUV”), and class aliases; rebuild `artifacts/text_dictionary_v2.npz` and rescore.
2. **Collect labelled dev data.** Derive color/type/side labels per frame (or coarse clip-level annotations) to unlock the calibration stages and quantify reliability.
3. **Enable temporal smoothing.** Implement `scripts_v2/pool_video.py` (EMA) or derotation to stabilise side predictions before decoding.
4. **Evaluate OOD gating.** Once calibration assets exist, run `scripts_v2/calibrate.py` with energy thresholds and negatives to benchmark abstention behaviour.

## 6. Artifact Index
- Per-run outputs: `artifacts/runs/<run>_v2/{image_features.npy,scores_raw.npy,scores_pooled.npz,quality_v2.csv,decoded/}`.
- Aggregate summaries: `artifacts/reports/v2/{prompt_summary.csv,quality_summary.csv,<group>_distributions.csv}`.
- New CLI decoder: `scripts_v2/decode.py`.

The pipeline is operational end-to-end for v2 (feature extraction → scoring → decoding). Accuracy now hinges on richer prompts and calibration data; the current configuration serves as a baseline for future iterations.
