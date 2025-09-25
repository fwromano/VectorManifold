# Video-to-Prompt Probability Flow v2 (Drop-In Replacement)

## 1. Guiding Principles
- **Interface parity:** Inputs remain raw video or frame folders (optionally with masks). Outputs keep the `P(prompt | frame)` JSON schema and artifact layout used by v1.
- **Translator-free:** Image embeddings now come directly from a CLIP-family dual encoder; prompt vectors live in the same space, eliminating the FastViT→CLIP translator and manual alignment metrics.
- **Label-free pose handling:** Viewpoint compensation relies on unsupervised feature-space analysis or temporal pooling—no angle annotations required.
- **Principled calibration:** Temperatures and confidence thresholds are learned from quality features and OOD splits, replacing fixed softmax scaling.
- **Temporal coherence:** Optional smoothing treats frame distributions as a Dirichlet belief updated over time, reducing label flicker.

## 2. Old → New Component Map
| Legacy component | v2 default | Benefit |
| --- | --- | --- |
| FastViT embeddings + linear translator | OpenCLIP (e.g., ViT-L/14) image features used directly against matching text embeddings | Higher zero-shot accuracy, simpler cosine geometry |
| Hand-labelled angles + ridge rotation plane | Unsupervised rotation removal (PCA/SSA) or temporal pooling | No angle labels; robust to pose drift |
| YOLOv8 masks (optional) | SAM-distill or YOLO-Seg masks with detector fallback | Cleaner foreground, less background bias |
| Global temperature | Learned calibration stack: global τ₀ → τ(x) from quality features → vector scaling → energy/OOD gate → optional conformal sets | Reliable probabilities with abstention capability |
| Single-frame decoding | Temporal fusion (EMA / Dirichlet filter) | Smoother semantics over time |

## 3. Data & Preprocessing (additions highlighted)
1. Extract frames via `ffmpeg` (10 FPS, ≤512 px long side) into `data/processed/frames/<run>/`.
2. Generate masks (`scripts_v2/segment.py`) prioritising SAM-distill → YOLO-Seg → detector box, storing binaries beside frames.
3. Score per-frame quality (`scripts_v2/score_quality.py`): `mask_area_norm`, `bbox_h`, `laplacian_var`, `detector_conf`, optional `occlusion_ratio`, `track_stability`.

## 4. Dual-Encoder Feature Extraction
- **Image encoder:** OpenCLIP ViT-L/14 (default) or EVA-CLIP g/14 for HQ preset. Produces L2-normalised `image_features.npy` under `artifacts/runs/<run>/`.
- **Text encoder:** Same model; `scripts_v2/build_promptbank.py` outputs `artifacts/text_dictionary_v2.npz` holding prompts, group indices, and text features.
- **Promptbank enhancements:** Support synonyms, optional negatives ("not a bus"), and compositional templates per prompt group.

## 5. Pose Handling Without Labels
- **Temporal pooling (default):** `scripts_v2/pool_video.py --agg ema` saves smoothed prototypes while preserving frame-wise features.
- **Unsupervised derotation (optional):** `scripts_v2/derotate.py` isolates cyclic principal components via PCA/SSA and writes `image_features_derot.npy`.

## 6. Prompt Scoring
- Cosine scores computed directly: `scores = Z_img @ Z_text^T` (L2 normalised, translator removed).
- Synonym handling: configurable max/mean pooling per class across synonym prompts.
- Outputs stored as `scores_raw.npy` (frames × prompts) plus per-group breakdowns.

## 7. Calibration & OOD Defences
1. **Global temperature τ₀** via `scripts_v2/calibrate.py --method temp` (dev split).
2. **Feature-conditioned τ(x):** softplus regression on quality features (`tau_model.joblib`).
3. **Vector scaling:** optional class-wise affine correction for large prompt sets.
4. **Energy gate:** threshold on negative log-sum-exp scores with OOD negatives (`energy.json`).
5. **Conformal prediction:** optional set-valued guarantees at target coverage (`conformal.json`).
6. **Decode:** `scripts_v2/decode.py` applies priors, calibration, gates, and emits calibrated probabilities (`decoded_probs.npy`) and JSON distributions matching v1 schema; conformal mode can emit abstentions.

## 8. Temporal Fusion
- `scripts_v2/fuse_tracks.py` maintains Dirichlet posteriors per track/clip, yielding smoothed probabilities and entropy diagnostics (`track_semantics.json`).
- For clip-level only, an EMA (decay 0.9) suffices.

## 9. Reporting & Artifacts
- Heatmaps, PCA trajectories, rotation summaries (if derotation used), calibration curves, and OOD ROC/PR remain under `artifacts/runs/<run>/` with `_v2` suffixes.
- Additional artifacts: `image_features.npy`, optional `image_features_derot.npy`, `scores_raw.npy`, `calib/` directory (temperatures, tau model, vector scaling, energy threshold, conformal stats), `decoded_probs.npy`, `decoded_sets.json` (when conformal active).

## 10. Minimal CLI Example
```bash
# 1) Frames + masks
python scripts_v2/extract_frames.py --video data/raw/video/<run>.mp4 --out data/processed/frames/<run> --fps 10 --max_side 512
python scripts_v2/segment.py --frames data/processed/frames/<run> --sam  # fallback to --yoloseg or detector

# 2) Image & text features
python scripts_v2/extract_img_feats.py --model openclip_vit_l14 --frames data/processed/frames/<run> --masks --out artifacts/runs/<run>/
python scripts_v2/build_promptbank.py --prompts docs/prompts/*.json --model openclip_vit_l14 --out artifacts/text_dictionary_v2.npz

# 3) Optional pose handling
python scripts_v2/pool_video.py --run <run> --agg ema
# python scripts_v2/derotate.py --run <run>

# 4) Raw scores
python scripts_v2/score.py --run <run> --text_dict artifacts/text_dictionary_v2.npz --out artifacts/runs/<run>/scores_raw.npy

# 5) Calibration
python scripts_v2/score_quality.py --run <run>
python scripts_v2/calibrate.py --run <run> --split dev --method temp
python scripts_v2/calibrate.py --run <run> --split dev --method tau_model --features mask_area_norm,bbox_h,laplacian_var,detector_conf
python scripts_v2/calibrate.py --run <run> --split dev --method energy --dev_ood artifacts/dev/negatives.json

# 6) Decode & export JSON
python scripts_v2/decode.py --run <run> --temperature_map calib/temperature.json \
  --tau_model calib/tau_model.joblib --energy calib/energy.json --priors calib/priors.json --out_json
```

## 11. Repository Changes Required
- Remove translator-specific utilities (`fit_translator`, `mean_cos_aligned`, translator artifacts) and manual angle labelling scripts.
- Add new `scripts_v2/` modules: feature extraction, promptbank builder, quality scoring, calibration suite, decoding, temporal pooling/derotation.
- Update `semantic_describer.py` to prefer v2 artifacts (dual-encoder features) while retaining backward compatibility with v1 FastViT workflows.

## 12. Defaults That “Just Work”
- Model: OpenCLIP ViT-L/14, FP16, cosine similarity.
- Masks: apply SAM-distill if available; otherwise YOLO-Seg → detector box.
- Calibration: τ₀ from dev split; τ(x) using `mask_area_norm`, `bbox_h`, `laplacian_var`.
- OOD: energy gate tuned for 5% FPR on curated negatives.
- Decoding: softmax with priors, top-k=5; EMA temporal fusion with decay 0.9.

## 13. Failure Modes & Mitigations
- **Small/blurred masks:** τ(x) down-weights confidence; temporal pooling smooths predictions.
- **Duplicate prompts:** prune or merge prompts with cosine >0.95 in promptbank.
- **Open-set surprises:** rely on energy gate + "unknown" bucket + conformal sets.
- **Domain shift (e.g. night footage):** re-run calibration on in-domain dev clips; no retraining needed.

## 14. HQ Preset (Optional)
- Swap to EVA-CLIP g/14, high-quality SAM masks, enable vector scaling + conformal sets, add test-time augmentation (horizontal flip), optionally integrate attribute-specific pixel heads for color prompts.

## 15. Implementation Checklist
- [ ] Implement dual-encoder feature extraction.
- [ ] Build promptbank loader with synonym/negative support.
- [ ] Generate quality feature script.
- [ ] Add calibration routines: τ₀, τ(x), vector scaling, energy, conformal.
- [ ] Provide decode wrapper maintaining v1 JSON schema.
- [ ] (Optional) Add EMA/Dirichlet temporal fusion utilities.
