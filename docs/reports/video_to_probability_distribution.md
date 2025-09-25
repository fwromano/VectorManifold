# Video-to-Prompt Probability Pipeline

## Abstract
We transform raw rotating-vehicle video into calibrated probability distributions over a curated prompt dictionary. The workflow combines classical preprocessing (frame extraction, optional segmentation), FastViT embeddings, rotation-plane modelling, and CLIP-aligned semantic decoding. This note documents each stage so new collaborators can reproduce the pipeline and interpret the resulting distributions $P(\text{prompt}\mid\text{frame})$.

## 1. Data Ingestion and Preprocessing
- **Source acquisition.** Short 360° dealership clips are downloaded with `yt-dlp` (README §1). The raw assets land under `data/raw/video/<run>/`.
- **Frame extraction.** `ffmpeg` produces uniformly sampled PNGs (default 10 FPS, ≤512 px long side) in `data/processed/frames/<run>/`. A lightweight `preview.mp4` is stored beside each embedding run for quick inspection.
- **Optional segmentation.** `python scripts/crop_frames.py` applies YOLOv8 masks to keep only vehicle pixels, reducing background leakage prior to embedding.
- **Angle annotation.** `python scripts/assign_angles.py` interpolates user-labelled key frames to fill `angles.npy/angles_unwrapped.npy`, providing a continuous pose reference for later rotation modelling.

## 2. FastViT Embedding Generation
- **Backbone.** Frames are embedded with `python scripts/extract_embeddings.py` using `fastvit_sa12` (a FastViT-HD analogue). Outputs live in `artifacts/runs/<run>/` and include `embeddings.npy`, cosine/euclidean matrices, metadata, and run configuration.
- **Normalisation.** Embeddings are L2-normalised (see `vector_manifold/semantic/describer.ensure_l2`) so cosine similarity equals dot product, enabling subsequent ridge fits and CLIP alignment.

## 3. Rotation Subspace Modelling
- **Ridge probe.** `python scripts/rotation_subspace_analysis.py --run <run>` learns a ridge regression from embeddings to $(\cos\theta, \sin\theta)$ using annotated angles.
- **Plane extraction.** The probe’s row space is orthogonalised to obtain a 2-D rotation basis $U$ and projector $P = UU^T`. Artefacts (`basis_U.npy`, `projector.npy`, diagnostic plots) reside under `rotation_analysis/`.
- **Derotation.** Subsequent stages optionally remove viewpoint drift by projecting embeddings into the nullspace of $P$ (`semantic_describer.derotate`), improving attribute discrimination.

## 4. Semantic Dictionary and Translator
- **Text dictionary.** `python scripts/semantic_describer.py build-dict` uses OpenCLIP (`ViT-L-14@openai`) to embed curated prompt groups (`docs/prompts/*.json`). The saved archive `artifacts/text_dictionary.npz` stores prompts, embeddings, and group indices.
- **Evaluation dataset.** `python scripts/create_semantic_eval_npz.py` assembles aligned FastViT embeddings, CLIP image features, angle labels, and optional heuristic labels into `artifacts/eval/<name>.npz` with train/dev/test splits.
- **Translator training.** `evaluate_readouts.fit_translator` fits a ridge map $A$ such that $Z_{\text{FastViT}} A \approx Z_{\text{CLIP}}$. Alignment quality is scored via mean cosine (`mean_cos_aligned`).

## 5. Probability Distribution Construction
- **Cosine scores.** For each frame, `compute_group_distributions` (in `semantic_describer.py`) multiplies FastViT embeddings (raw or derotated) against the text dictionary to produce cosine similarities per prompt group.
- **Softmax decoding.** Group scores are temperature-scaled and normalised into probabilities: $P_i = \operatorname{softmax}(\text{cosine}_i / T)$. Temperatures can be globally set (`--temperature`) or calibrated per group on the dev split (`evaluate_readouts.calibrate_temperature_map`).
- **Outputs.** `semantic_describer.py describe` stores per-frame JSON records with ranked distributions and optional linear-probe scores, plus aggregate summaries (`summary_out`) averaging $P(\text{prompt} \mid \text{frame})$ across the dataset. `evaluate_semantic_readouts.py` additionally emits distribution plots (`translator_distribution_{group}_{raw,derotated}.png`).

## 6. Evaluation and Reporting
- **Probe baselines.** Logistic probes (`train-probe`) provide supervised baselines on raw and derotated embeddings, enabling accuracy comparison against translator-driven probabilities.
- **Translator diagnostics.** `evaluate_semantic_readouts.py` reports alignment cosines, classification accuracy per label, and probability distributions for each semantic group.
- **Visual artefacts.** Heatmaps, PCA trajectories, and rotation summaries accompany every run under `artifacts/runs/<run>/`, contextualising the probability outputs.

## 7. Limitations and Future Work
- Duplicate or weakly distinguished prompts (see `docs/reports/prompt_cosine_similarity.md`) compress probability mass, necessitating prompt rephrasing before deployment.
- Rotation removal depends on accurate angle annotations; noisy keys degrade the learned subspace.
- Translator performance hinges on CLIP feature quality—additional augmentation or multi-run training may be required for robust generalisation.

## Reproduction Checklist
1. Download video → extract frames (`ffmpeg`).
2. (Optional) Crop with YOLOv8 and annotate angles.
3. Embed frames via FastViT (`extract_embeddings.py`).
4. Run rotation subspace analysis (`rotation_subspace_analysis.py`).
5. Build text dictionary (`semantic_describer.py build-dict`).
6. Assemble evaluation bundle with CLIP image features (`create_semantic_eval_npz.py`).
7. Fit translator & evaluate (`evaluate_semantic_readouts.py`).
8. Generate per-frame prompt distributions (`semantic_describer.py describe`).

Following these steps yields probability distributions $P(\text{prompt}\mid\text{frame})$ ready for analysis or downstream semantic labelling.
