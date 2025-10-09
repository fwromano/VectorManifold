#!/usr/bin/env python3
"""Calibration utilities for the v2 flow."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from vector_manifold.v2.calibration import CalibrationBundle, EnergyGate, TemperatureScaler, TauModel, load_calibration_bundle
from vector_manifold.v2.promptbank import PromptBank


@dataclass
class DevData:
    logits: Dict[str, np.ndarray]
    labels: Dict[str, np.ndarray]
    features: np.ndarray


def load_dev_data(run_dir: Path, promptbank_path: Path, eval_npz: Path, quality_csv: Path) -> DevData:
    scores_path = run_dir / "scores_pooled.npz"
    if not scores_path.exists():
        raise FileNotFoundError(scores_path)
    scores_npz = np.load(scores_path)

    promptbank = PromptBank.from_npz(promptbank_path)

    data = np.load(eval_npz)
    quality = pd.read_csv(quality_csv)

    first_key = next(iter(scores_npz.files))
    total_frames = scores_npz[first_key].shape[0]
    train_end = int(0.6 * total_frames)
    dev_end = int(0.8 * total_frames)
    dev_slice = slice(train_end, dev_end)

    logits = {}
    labels = {}
    for group_name in promptbank.config.groups.keys():
        if group_name not in scores_npz:
            continue
        logits[group_name] = scores_npz[group_name][dev_slice]
        label_key = f"{group_name}_dev"
        if label_key in data:
            labels[group_name] = data[label_key]
        else:
            # Allow missing labels (e.g., prompts without annotations)
            labels[group_name] = np.full(logits[group_name].shape[0], -1, dtype=np.int64)

    features = quality.iloc[dev_slice, :]
    feature_cols = [
        col
        for col in [
            "mask_area_norm",
            "bbox_span_norm",
            "laplacian_var",
            "detector_conf",
            "occlusion_ratio",
            "track_stability",
        ]
        if col in features.columns
    ]
    feature_matrix = features[feature_cols].fillna(features[feature_cols].mean()).to_numpy(dtype=np.float32)
    if feature_matrix.shape[1] == 0:
        raise ValueError('quality features missing required columns')

    return DevData(logits=logits, labels=labels, features=feature_matrix)


def nll_with_temperature(logits: np.ndarray, labels: np.ndarray, tau: float) -> float:
    valid = labels >= 0
    if not np.any(valid):
        return 0.0
    scaled = logits[valid] / tau
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    probs = exp / exp.sum(axis=1, keepdims=True)
    nll = -np.log(np.clip(probs[np.arange(probs.shape[0]), labels[valid]], 1e-8, 1.0))
    return float(np.mean(nll))


def fit_temperature(dev: DevData) -> float:
    candidates = np.linspace(0.05, 2.5, 60)
    best_tau = 1.0
    best_loss = float("inf")
    for tau in candidates:
        losses = []
        for group, logits in dev.logits.items():
            losses.append(nll_with_temperature(logits, dev.labels[group], tau))
        loss = np.mean(losses)
        if loss < best_loss:
            best_loss = loss
            best_tau = float(tau)
    return best_tau


def fit_tau_model(dev: DevData) -> TauModel:
    features = torch.from_numpy(dev.features).float()
    weights = torch.zeros(features.shape[1], dtype=torch.float32, requires_grad=True)
    bias = torch.zeros(1, dtype=torch.float32, requires_grad=True)
    optimiser = optim.Adam([weights, bias], lr=1e-2)

    def _loss_fn() -> torch.Tensor:
        loss = 0.0
        total = 0
        tau = torch.log1p(torch.exp(features @ weights + bias)) + 1e-6
        for group, logits in dev.logits.items():
            labels = dev.labels[group]
            valid = labels >= 0
            if not np.any(valid):
                continue
            logit_tensor = torch.from_numpy(logits[valid]).float()
            label_tensor = torch.from_numpy(labels[valid]).long()
            scaled = logit_tensor / (tau[valid].unsqueeze(1))
            loss += torch.nn.functional.cross_entropy(scaled, label_tensor)
            total += 1
        if total == 0:
            return torch.tensor(0.0, requires_grad=True)
        return loss / total

    for _ in range(400):
        optimiser.zero_grad()
        loss = _loss_fn()
        loss.backward()
        optimiser.step()

    with torch.no_grad():
        w = weights.detach().numpy()
        b = bias.detach().item()
    return TauModel(weights=w, bias=float(b))


def fit_energy_gate(dev: DevData) -> EnergyGate:
    stacked = np.concatenate(list(dev.logits.values()), axis=1)
    energy = -np.log(np.sum(np.exp(stacked), axis=1))
    threshold = float(np.percentile(energy, 95))
    return EnergyGate(threshold=threshold)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate v2 prompt scores")
    parser.add_argument("--run", type=Path, required=True, help="Run directory")
    parser.add_argument("--promptbank", type=Path, required=True, help="Prompt bank npz")
    parser.add_argument("--eval", type=Path, required=True, help="Evaluation dataset (npz) with labels")
    parser.add_argument("--quality", type=Path, required=True, help="Quality CSV")
    parser.add_argument(
        "--method",
        choices=["temp", "tau_model", "energy"],
        required=True,
        help="Calibration method to apply",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calib_dir = args.run / "calib"
    if calib_dir.exists():
        try:
            bundle = load_calibration_bundle(calib_dir)
        except FileNotFoundError:
            bundle = CalibrationBundle()
    else:
        bundle = CalibrationBundle()

    dev_data = load_dev_data(args.run, args.promptbank, args.eval, args.quality)

    if args.method == "temp":
        tau = fit_temperature(dev_data)
        bundle.temperature = TemperatureScaler(temperature=tau)
        print(f"Fitted global temperature: {tau:.3f}")
    elif args.method == "tau_model":
        tau_model = fit_tau_model(dev_data)
        bundle.tau_model = tau_model
        print("Fitted tau(x) model")
    elif args.method == "energy":
        gate = fit_energy_gate(dev_data)
        bundle.energy_gate = gate
        print(f"Set energy threshold to {gate.threshold:.3f}")

    bundle.to_disk(calib_dir)
    print(f"Updated calibration bundle at {calib_dir}")



if __name__ == "__main__":
    main()
