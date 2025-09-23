# FastViT Embedding Explorer

This project scripts a workflow for studying how viewpoint changes of an object map onto a FastViT embedding manifold. You can ingest pre-rendered images, or build a dataset from a rotating-car video, then analyse the latent trajectory with PCA/UMAP diagnostics. The maintained pipeline now defaults to the cropped-car variant (`car_rotation3_fastvit_cropped`) so every analysis step assumes cropped inputs.

## Features
## Repository layout
- `data/`: raw videos under `data/raw/` and processed frame sequences under `data/processed/`.
- `assets/models/`: cached third-party weights (e.g., YOLOv8).
- `src/`: Python modules (`vector_manifold`) for data prep, embedding extraction, rotation logic, and semantic readout.
- `scripts/`: Small CLI wrappers that forward to the package modules (so every tool is available via `python scripts/<name>.py`).
- `artifacts/runs/`: generated embeddings, diagnostics, and analysis outputs.
- `artifacts/summaries/`: cross-run tables produced by the analysis suite.
- `docs/reports/`: written summaries and figures.

- Pull one or more rotating-car clips (e.g. dealership turntables) and extract frame sequences.
- Save lightweight `preview.mp4` renders so you can quickly inspect each source video from the CLI.
- Batch frames through a pretrained FastViT backbone (`fastvit_sa12`, a public analogue to "FastViT-HD").
- Persist embeddings plus cosine/Euclidean similarity matrices for downstream work.
- Optional YOLOv8 segmentation stage isolates just the car pixels before embedding.
- Run an analysis suite that produces PCA trajectories, similarity heatmaps, and per-frame-delta charts for clearer interpretation.

## Prerequisites
- Python 3.10+
- Packages: `torch`, `timm`, `numpy`, `matplotlib`, `Pillow`; optional `umap-learn` for UMAP plots. For semantic readout, install `open_clip_torch`.
- For data acquisition: `ffmpeg` (preinstalled on Ubuntu) and `yt-dlp` (`pip install yt-dlp`).
- Internet access the first time you download FastViT weights from the Hugging Face Hub.

## 1. Acquire rotating car imagery
```bash
# 1) Download a rotating-car showcase clip (first match from a YouTube search)
yt-dlp -f "bestvideo[height<=720]+bestaudio/best[height<=720]" \
  -o data/raw/video/car_rotation3/car_rotation.%(ext)s "ytsearch1:car rotating display 360"

# 2) (Optional) repeat with alternate search queries to gather more clips

# 3) Extract frames at 10 FPS, scaling the longer side to ~512 px
ffmpeg -y -i data/raw/video/car_rotation3/car_rotation.mkv \
  -vf "fps=10,scale=512:-2" data/processed/frames/car_rotation3/frame_%05d.png

# 4) Create a downscaled preview so you can replay the clip locally (optional quick look)
ffmpeg -y -i data/raw/video/car_rotation3/car_rotation.mkv \
  -vf "scale=640:-2" -an artifacts/runs/car_rotation3_fastvit_cropped/preview.mp4
```

Feel free to substitute your own curated footage. Higher FPS yields finer angular resolution but also more frames to embed. The generated `preview.mp4` clips live alongside each embedding run under `artifacts/runs/<run>/`, so you can replay them with `ffplay artifacts/runs/<run>/preview.mp4`.

### Optional: isolate the vehicle with YOLOv8 segmentation
```bash
pip install ultralytics  # one-time
python scripts/crop_frames.py \
  --input-dir data/processed/frames/car_rotation3 \
  --output-dir data/processed/frames/car_rotation3_cropped \
  --confidence 0.35 --margin 5
```

The script crops the tight YOLOv8 mask-aligned bounding box (plus a small safety margin) while preserving raw pixels—no black borders or aspect squeezing. Adjust `--margin` to keep more or less context.

### Optional: annotate viewing angles
```bash
python scripts/assign_angles.py
```

This interpolates the user-provided key frames (front/side/back) across the entire sequence, writing per-frame `angle_deg` values (and storing continuous `angles.npy` + `angles_unwrapped.npy`).

## 2. Generate FastViT embeddings
```bash
python scripts/extract_embeddings.py \
  --image-dir data/processed/frames/car_rotation3_cropped \
  --output-dir artifacts/runs/car_rotation3_fastvit_cropped \
  --model fastvit_sa12 \
  --device cpu \
  --batch-size 32
```

Key outputs under `artifacts/runs/car_rotation3_fastvit_cropped/`:
- `embeddings.npy`: array of shape `(num_frames, embedding_dim)`.
- `cosine_similarity.npy` / `euclidean_distance.npy`: pairwise metrics.
- `metadata.csv`: frame IDs and source files.
- `run_info.json`: model configuration snapshot.
- `preview.mp4`: lightweight version of the source video for convenience.

If you also generate synthetic rotations, provide `--base-image/--angles` instead of `--image-dir`.

## 3. Visualise the embedding trajectory
```bash
python scripts/visualize_trajectory.py \
  --embeddings artifacts/runs/car_rotation3_fastvit_cropped/embeddings.npy \
  --metadata artifacts/runs/car_rotation3_fastvit_cropped/metadata.csv \
  --out-dir artifacts/runs/car_rotation3_fastvit_cropped
```

This creates:
- `trajectory_pca.png`: PCA-reduced path coloured by frame index.
- `pca_components_over_time.png`: Principal component curves highlighting periodicity.
- `pca_projection.npy`: Saved PCA coordinates (optionally feed into other tools).
- If `umap-learn` is installed, an additional `trajectory_umap.png` is generated.

## 4. Run the analysis suite (recommended)
```bash
python scripts/analyze_runs.py \
  --run artifacts/runs/car_rotation3_fastvit_cropped

# Optional: rotation-subspace experiments (linear probe & Fourier screening)
python scripts/rotation_subspace_analysis.py \
  --run artifacts/runs/car_rotation3_fastvit_cropped

# Export canonical rotation vectors (saved under rotation_analysis/canonical_angle_vectors)
python scripts/export_canonical_vectors.py \
  artifacts/runs/car_rotation3_fastvit_cropped --num-angles 360

# Visualise the canonical vectors vs. embeddings
python scripts/visualize_rotation_vectors.py \
  artifacts/runs/car_rotation3_fastvit_cropped --num-angles 360

# Remove rotation components (harmonics=2 removes cos/sin and cos2/sin2 subspaces)
python scripts/remove_rotation_components.py \
  artifacts/runs/car_rotation3_fastvit_cropped --harmonics 2

# Build cached CLIP text dictionary (once)
python scripts/semantic_describer.py build-dict \
  --out artifacts/text_dictionary.npz

# Produce semantic summaries (rotation plane removed by default)
python scripts/semantic_describer.py describe \
  artifacts/runs/car_rotation3_fastvit_cropped \
  --dict artifacts/text_dictionary.npz \
  --topk 2 \
  --out artifacts/runs/car_rotation3_fastvit_cropped/semantic_descriptions.json

# (Optional) Fit a ridge probe (CSV needs columns index,label)
python scripts/semantic_describer.py train-probe \
  artifacts/runs/car_rotation3_fastvit_cropped \
  --labels data/labels/color_red.csv \
  --out artifacts/probes/color_red.npz
```

## Compare probe vs. translator readouts
Bundle your FastViT embeddings, angle labels, optional CLIP image features, and supervised splits into an `.npz`, then run:

```bash
python scripts/evaluate_semantic_readouts.py \
  artifacts/eval/car_rotation_semantic.npz \
  --dictionary artifacts/text_dictionary.npz \
  --label-prompts docs/prompts/default_label_map.json \
  --labels side \
  --out artifacts/runs/car_rotation3_fastvit_cropped/semantic_eval
```

The script reports pose $R^2$, cross-view cosine gains, and accuracy tables before/after derotation, plus bar charts (`probe_*`, `translator_*`) so you can visually compare the probes-only baseline with the FastViT→CLIP translator. Adjust the JSON lists so their order matches the numeric class IDs in your evaluation `.npz`.

- Use `--temperature <val>` to control the translator softmax temperature (defaults to 1.0). Smaller values (e.g. `0.1`) produce sharper likelihoods for the distribution plots.
- Add `--calibrate-temperature` to fit per-group temperatures on the dev split using cross-entropy.
- Additional diagrams (`translator_distribution_<group>_{raw,derotated}.png`) capture the dataset-average likelihoods for each prompt group.

### Adding type / occupancy / armament labels
Supply extra label CSVs via `scripts/create_semantic_eval_npz.py` and swap in the prompt list at `docs/prompts/multi_attribute.json`:

```bash
python scripts/create_semantic_eval_npz.py \
  artifacts/runs/car_rotation3_fastvit_cropped \
  data/processed/frames/car_rotation3_cropped \
  --label type=data/labels/type_heuristic.csv \
  --label occupancy=data/labels/occupancy_heuristic.csv \
  --label armament=data/labels/armament_heuristic.csv \
  --out artifacts/eval/car_rotation_semantic.npz

python scripts/semantic_describer.py build-dict \
  --out artifacts/text_dictionary.npz

python scripts/evaluate_semantic_readouts.py \
  artifacts/eval/car_rotation_semantic.npz \
  --dictionary artifacts/text_dictionary.npz \
  --label-prompts docs/prompts/multi_attribute.json \
  --labels side type occupancy armament \
  --temperature 0.1 \
  --out artifacts/runs/car_rotation3_fastvit_cropped/semantic_eval
```

Replace the sample `_heuristic` CSVs with real annotations—each file must cover every frame index. Update the JSON prompt lists if your class order or phrasing differs.

Each run folder will gain:
- `cosine_heatmap.png` / `euclidean_heatmap.png`: similarity structure at a glance.
- `cosine_vs_frame_delta.png` / `euclidean_vs_frame_delta.png`: average similarity as frame spacing grows.
- `analysis_summary.json`: scalar stats per sequence.
- `norm_over_time.png` / `successive_cosine.png`: how embedding magnitude and frame-to-frame similarity evolve.
- `cosine_histogram.png`: distribution of pairwise cosine scores.
- `trajectory_pca.png` / `trajectory_pca_3d.png`: 2D and 3D projections of the latent loop.
- `angle_over_time.png` / `angle_histogram.png` / `cosine_vs_angle_delta.png`: available once `angle_deg` is populated.
- `cosine_heatmap_by_angle.png`: cosine matrix restricted to the first full revolution, sorted by viewing angle.
- `cosine_heatmap_angle_one_rev.png`: chronological cosine matrix for the initial front→front sweep (~frames 17–78).
- `cosine_heatmap_angle_out_back.png`: chronological cosine matrix covering the outward revolution plus return pass (~frames 17–159).
- `diff_norm_over_time.png` / `diff_norm_hist.png`: magnitude of successive embedding differences, over time and as a distribution.
- `diff_pca.png`: principal directions of the difference vectors in the top-2 components.
- `diff_norm_vs_angle_step.png`: scatter of difference norms against per-frame angle change.
- `norm_over_angle_one_rev.png` / `norm_over_angle_out_back.png`: embedding norm plotted versus angle for the forward-only and out/back passes.
- `successive_cosine_over_angle_one_rev.png` / `successive_cosine_over_angle_out_back.png`: consecutive-frame cosine similarity against angle.
- `diff_norm_over_angle_one_rev.png` / `diff_norm_over_angle_out_back.png`: embedding difference magnitudes against angle for the same segments.

Cross-run summaries aggregate into `artifacts/summaries/embedding_run_summary.csv` for quick comparison (e.g., which clip is most stable under rotation).

## Notes on "FastViT-HD"
The original Apple paper introduces FastViT variants trained on higher-resolution data, commonly referenced as "FastViT-HD". Within `timm`, the `fastvit_sa12` checkpoint is the closest public release. If you have alternative checkpoints, place the weights in your Hugging Face cache and pass the corresponding model name with `--model`.

## Extending
- Estimate per-frame rotation angles (e.g., assuming a full revolution) and append to `metadata.csv` for cosine vs. angle plots.
- Swap PCA for spectral methods or compute geodesic distances along the manifold.
- Feed `embeddings.npy` into notebooks for clustering, recurrence plots, or auto-correlation analysis of the latent trajectory.

Happy embedding exploring!
