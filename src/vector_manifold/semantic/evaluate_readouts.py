"""Evaluate FastViT semantic readout paths with rotation subtraction diagnostics."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .describer import TextDictionary

try:  # imported lazily so unit usage can stub dependencies
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import accuracy_score, r2_score
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover - toolchain variant
    raise ImportError(
        "evaluate_readouts requires scikit-learn; install 'scikit-learn' before running"
    ) from exc


Array = np.ndarray


@dataclass
class LabelBundle:
    name: str
    train: Array
    dev: Array
    test: Array

    @property
    def num_classes(self) -> int:
        return int(np.max(self.train)) + 1 if self.train.size else 0

    def has_multiple_classes(self) -> bool:
        return np.unique(self.train).size > 1


@dataclass
class EvaluationDataset:
    Z_train: Array
    Z_dev: Array
    Z_test: Array
    theta_train: Array
    theta_dev: Array
    labels: Dict[str, LabelBundle]
    Z_clip_train: Optional[Array] = None
    Z_clip_dev: Optional[Array] = None
    same_pairs_dev: Optional[Array] = None

    @staticmethod
    def from_npz(path: Path, label_names: Sequence[str]) -> "EvaluationDataset":
        data = np.load(path, allow_pickle=True)

        def fetch(key: str) -> Array:
            if key not in data:
                raise KeyError(f"dataset missing key '{key}'")
            arr = data[key]
            return np.asarray(arr, dtype=np.float32)

        Z_train = ensure_l2(fetch("Z_train"))
        Z_dev = ensure_l2(fetch("Z_dev"))
        Z_test = ensure_l2(fetch("Z_test"))
        theta_train = np.asarray(data["theta_train"], dtype=np.float32)
        theta_dev = np.asarray(data["theta_dev"], dtype=np.float32)

        labels: Dict[str, LabelBundle] = {}
        for name in label_names:
            train_key = f"{name}_train"
            dev_key = f"{name}_dev"
            test_key = f"{name}_test"
            if train_key not in data:
                raise KeyError(f"dataset missing key '{train_key}' for label '{name}'")
            labels[name] = LabelBundle(
                name,
                np.asarray(data[train_key], dtype=np.int64),
                np.asarray(data[dev_key], dtype=np.int64),
                np.asarray(data[test_key], dtype=np.int64),
            )

        Z_clip_train = ensure_optional_l2(data, "Z_clip_train")
        Z_clip_dev = ensure_optional_l2(data, "Z_clip_dev")

        same_pairs = None
        if "pairs_dev" in data:
            same_pairs = np.asarray(data["pairs_dev"], dtype=np.int64)
            if same_pairs.ndim != 2 or same_pairs.shape[1] != 2:
                raise ValueError("pairs_dev must have shape (N, 2)")

        return EvaluationDataset(
            Z_train=Z_train,
            Z_dev=Z_dev,
            Z_test=Z_test,
            theta_train=theta_train,
            theta_dev=theta_dev,
            labels=labels,
            Z_clip_train=Z_clip_train,
            Z_clip_dev=Z_clip_dev,
            same_pairs_dev=same_pairs,
        )


@dataclass
class RotationModel:
    basis: Array
    ridge_weights: Array
    r2_pre: float
    r2_post: float
    same_cos_pre: Optional[float]
    same_cos_post: Optional[float]


@dataclass
class ProbeMetrics:
    accuracy_dev: Dict[str, float]
    accuracy_test: Dict[str, float]


@dataclass
class TranslatorMetrics:
    aligned_cos_dev: float
    aligned_cos_train: float
    accuracy_dev: Optional[Dict[str, float]]
    accuracy_test: Optional[Dict[str, float]]
    temperature_map: Optional[Dict[str, float]] = None


@dataclass
class EvaluationSummary:
    rotation: RotationModel
    probe_raw: ProbeMetrics
    probe_derotated: ProbeMetrics
    translator_raw: Optional[TranslatorMetrics]
    translator_derotated: Optional[TranslatorMetrics]
    translator_probs_raw: Optional[Dict[str, Dict[str, float]]]
    translator_probs_derotated: Optional[Dict[str, Dict[str, float]]]


def ensure_l2(Z: Array, eps: float = 1e-8) -> Array:
    Z = np.asarray(Z, dtype=np.float32)
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    norms = np.where(norms > eps, norms, 1.0)
    return Z / norms


def ensure_optional_l2(data: Mapping[str, Array], key: str) -> Optional[Array]:
    if key not in data:
        return None
    return ensure_l2(data[key])


def softmax_temperature(logits: Array, temperature: float) -> Array:
    temp = max(float(temperature), 1e-4)
    shifted = logits / temp
    shifted = shifted - shifted.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    denom = exp.sum(axis=1, keepdims=True)
    return exp / np.where(denom == 0.0, 1.0, denom)


def qr_ortho(W: Array) -> Array:
    q, _ = np.linalg.qr(W)
    return q[:, : W.shape[1]]


def project_out(Z: Array, U: Array) -> Array:
    proj = U @ U.T
    residual = Z - Z @ proj
    return ensure_l2(residual)


def learn_rotation_plane(
    dataset: EvaluationDataset,
    alpha: float,
) -> RotationModel:
    def build_targets(theta: Array) -> Array:
        return np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)

    ridge = Ridge(alpha=alpha, fit_intercept=False)
    Y_train = build_targets(dataset.theta_train)
    ridge.fit(dataset.Z_train, Y_train)
    Wpose = ridge.coef_.T.astype(np.float32)
    basis = qr_ortho(Wpose)

    r2_pre = pose_r2(dataset.Z_dev, build_targets(dataset.theta_dev), Wpose)
    Z_dev_star = project_out(dataset.Z_dev, basis)
    r2_post = pose_r2(Z_dev_star, build_targets(dataset.theta_dev), Wpose)

    same_cos_pre = None
    same_cos_post = None
    if dataset.same_pairs_dev is not None and dataset.same_pairs_dev.size:
        same_cos_pre = mean_cos(dataset.Z_dev, dataset.same_pairs_dev)
        same_cos_post = mean_cos(Z_dev_star, dataset.same_pairs_dev)

    return RotationModel(
        basis=basis,
        ridge_weights=Wpose,
        r2_pre=float(r2_pre),
        r2_post=float(r2_post),
        same_cos_pre=same_cos_pre,
        same_cos_post=same_cos_post,
    )


def pose_r2(Z: Array, Y: Array, W: Array) -> float:
    preds = Z @ W
    return float(r2_score(Y, preds))


def mean_cos(Z: Array, pairs: Array) -> float:
    sims = []
    for i, j in pairs:
        sims.append(float(Z[int(i)] @ Z[int(j)]))
    return float(np.mean(sims)) if sims else float("nan")


def fit_probe(Z_tr: Array, y_tr: Array, C: float) -> tuple[LogisticRegression, StandardScaler]:
    scaler = StandardScaler(with_mean=False)
    Z_tr_scaled = scaler.fit_transform(Z_tr)
    clf = LogisticRegression(
        C=C,
        max_iter=300,
        n_jobs=8,
        penalty="l2",
        fit_intercept=False,
        multi_class="auto",
    )
    clf.fit(Z_tr_scaled, y_tr)
    return clf, scaler


def evaluate_probes(
    Z_tr: Array,
    Z_dev: Array,
    Z_te: Array,
    labels: Dict[str, LabelBundle],
    C: float,
) -> ProbeMetrics:
    acc_dev: Dict[str, float] = {}
    acc_test: Dict[str, float] = {}
    for name, bundle in labels.items():
        if not bundle.has_multiple_classes():
            acc_dev[name] = float("nan")
            acc_test[name] = float("nan")
            continue
        clf, scaler = fit_probe(Z_tr, bundle.train, C)
        dev_preds = clf.predict(scaler.transform(Z_dev))
        test_preds = clf.predict(scaler.transform(Z_te))
        acc_dev[name] = float(accuracy_score(bundle.dev, dev_preds))
        acc_test[name] = float(accuracy_score(bundle.test, test_preds))
    return ProbeMetrics(acc_dev, acc_test)


def fit_translator(Z_f: Array, Z_c: Array, alpha: float) -> Array:
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(Z_f, Z_c)
    return ridge.coef_.T.astype(np.float32)


def mean_cos_aligned(Z_f: Array, Z_c: Array, A: Array) -> float:
    mapped = ensure_l2(Z_f @ A)
    return float(np.mean(np.sum(mapped * Z_c, axis=1)))


def translator_classification(
    Z_f: Array,
    A: Array,
    dictionary: TextDictionary,
    label_prompts: Mapping[str, Sequence[str]],
) -> Dict[str, Array]:
    Z_clip_like = ensure_l2(Z_f @ A)
    sims = Z_clip_like @ dictionary.embeddings.T

    results: Dict[str, Array] = {}
    for group, prompts in label_prompts.items():
        group_rows = prompt_indices(dictionary, group, prompts)
        group_scores = sims[:, group_rows]
        results[group] = group_scores
    return results


def prompt_indices(dictionary: TextDictionary, group: str, prompts: Sequence[str]) -> np.ndarray:
    prompt_list = dictionary.prompts.get(group)
    group_indices = dictionary.group_indices.get(group)
    if prompt_list is None or group_indices is None:
        raise KeyError(f"dictionary missing group '{group}'")
    positions: List[int] = []
    for prompt in prompts:
        if prompt not in prompt_list:
            raise KeyError(f"prompt '{prompt}' not found in dictionary group '{group}'")
        positions.append(int(group_indices[prompt_list.index(prompt)]))
    return np.asarray(positions, dtype=int)


def accuracy_from_scores(scores: Array, labels: Array) -> float:
    preds = np.argmax(scores, axis=1)
    return float(accuracy_score(labels, preds))


def calibrate_temperature_map(
    dataset: EvaluationDataset,
    dictionary: TextDictionary,
    label_prompts: Mapping[str, Sequence[str]],
    A: Array,
    search: Sequence[float],
) -> Dict[str, float]:
    Z_f = dataset.Z_dev
    Z_clip_like = ensure_l2(Z_f @ A)
    sims = Z_clip_like @ dictionary.embeddings.T
    temps: Dict[str, float] = {}
    for group, prompts in label_prompts.items():
        if group not in dataset.labels:
            continue
        bundle = dataset.labels[group]
        if not bundle.has_multiple_classes():
            continue
        rows = prompt_indices(dictionary, group, prompts)
        logits = sims[:, rows]
        y = bundle.dev
        best_temp = 1.0
        best_ll = float("-inf")
        for temp in search:
            probs = softmax_temperature(logits, temp)
            ll = float(np.mean(np.log(np.clip(probs[np.arange(probs.shape[0]), y], 1e-8, 1.0))))
            if ll > best_ll:
                best_ll = ll
                best_temp = float(temp)
        temps[group] = best_temp
    return temps


def evaluate_translator(
    dataset: EvaluationDataset,
    A: Array,
    dictionary: TextDictionary,
    label_prompts: Mapping[str, Sequence[str]],
    split: str,
) -> Optional[Dict[str, float]]:
    if not label_prompts:
        return None
    if split == "dev":
        Z_f = dataset.Z_dev
        labels = {name: bundle.dev for name, bundle in dataset.labels.items()}
    elif split == "train":
        Z_f = dataset.Z_train
        labels = {name: bundle.train for name, bundle in dataset.labels.items()}
    elif split == "test":
        Z_f = dataset.Z_test
        labels = {name: bundle.test for name, bundle in dataset.labels.items()}
    else:
        raise ValueError(f"unknown split '{split}'")

    scores = translator_classification(Z_f, A, dictionary, label_prompts)
    return {
        name: accuracy_from_scores(scores[name], labels[name])
        for name in scores.keys()
    }


def collate_translator_metrics(
    dataset: EvaluationDataset,
    dictionary: TextDictionary,
    label_prompts: Mapping[str, Sequence[str]],
    A: Array,
    temperature_map: Optional[Mapping[str, float]] = None,
) -> TranslatorMetrics:
    aligned_train = mean_cos_aligned(dataset.Z_train, dataset.Z_clip_train, A)
    aligned_dev = mean_cos_aligned(dataset.Z_dev, dataset.Z_clip_dev, A)
    acc_dev = evaluate_translator(dataset, A, dictionary, label_prompts, "dev")
    acc_test = evaluate_translator(dataset, A, dictionary, label_prompts, "test")
    return TranslatorMetrics(
        aligned_cos_dev=aligned_dev,
        aligned_cos_train=aligned_train,
        accuracy_dev=acc_dev,
        accuracy_test=acc_test,
        temperature_map=dict(temperature_map) if temperature_map else None,
    )


def compute_mean_probabilities(
    Z_f: Array,
    A: Array,
    dictionary: TextDictionary,
    label_prompts: Mapping[str, Sequence[str]],
    temperature_map: Optional[Mapping[str, float]] = None,
) -> Dict[str, Dict[str, float]]:
    Z_clip_like = ensure_l2(Z_f @ A)
    sims = Z_clip_like @ dictionary.embeddings.T
    result: Dict[str, Dict[str, float]] = {}
    for group, prompts in label_prompts.items():
        rows = prompt_indices(dictionary, group, prompts)
        logits = sims[:, rows]
        temp = temperature_map[group] if temperature_map and group in temperature_map else 1.0
        probs = softmax_temperature(logits, temp)
        mean_probs = probs.mean(axis=0)
        result[group] = {prompt: float(prob) for prompt, prob in zip(prompts, mean_probs)}
    return result


def run_evaluation(
    dataset: EvaluationDataset,
    dictionary: TextDictionary,
    label_prompts: Mapping[str, Sequence[str]],
    *,
    out_dir: Path,
    pose_alpha: float,
    logreg_C: float,
    translator_alpha: float,
    skip_probes: bool,
    skip_translator: bool,
    temperature_default: float,
    calibrate_temperature: bool,
) -> EvaluationSummary:
    out_dir.mkdir(parents=True, exist_ok=True)

    rotation = learn_rotation_plane(dataset, pose_alpha)

    probe_raw = ProbeMetrics({}, {})
    probe_star = ProbeMetrics({}, {})
    if not skip_probes:
        probe_raw = evaluate_probes(
            dataset.Z_train,
            dataset.Z_dev,
            dataset.Z_test,
            dataset.labels,
            logreg_C,
        )
        Z_train_star = project_out(dataset.Z_train, rotation.basis)
        Z_dev_star = project_out(dataset.Z_dev, rotation.basis)
        Z_test_star = project_out(dataset.Z_test, rotation.basis)
        probe_star = evaluate_probes(
            Z_train_star,
            Z_dev_star,
            Z_test_star,
            dataset.labels,
            logreg_C,
        )

    translator_raw: Optional[TranslatorMetrics] = None
    translator_star: Optional[TranslatorMetrics] = None
    translator_probs_raw: Optional[Dict[str, Dict[str, float]]] = None
    translator_probs_star: Optional[Dict[str, Dict[str, float]]] = None
    temperature_map: Dict[str, float] = {}
    if not skip_translator:
        if dataset.Z_clip_train is None or dataset.Z_clip_dev is None:
            raise ValueError("translator evaluation requested but CLIP features missing in dataset")
        A_raw = fit_translator(dataset.Z_train, dataset.Z_clip_train, translator_alpha)
        if label_prompts:
            temperature_map = {group: float(temperature_default) for group in label_prompts.keys()}
            if calibrate_temperature:
                search_grid = np.linspace(0.05, 1.5, num=30)
                temperature_map = calibrate_temperature_map(dataset, dictionary, label_prompts, A_raw, search_grid)
        translator_raw = collate_translator_metrics(dataset, dictionary, label_prompts, A_raw, temperature_map)
        if label_prompts:
            translator_probs_raw = compute_mean_probabilities(
                dataset.Z_dev,
                A_raw,
                dictionary,
                label_prompts,
                temperature_map,
            )

        Z_train_star = project_out(dataset.Z_train, rotation.basis)
        Z_dev_star = project_out(dataset.Z_dev, rotation.basis)
        if dataset.Z_clip_train.shape[1] != A_raw.shape[1]:
            raise ValueError("CLIP embedding dimensionality mismatch")
        A_star = fit_translator(Z_train_star, dataset.Z_clip_train, translator_alpha)
        dataset_star = EvaluationDataset(
            Z_train=Z_train_star,
            Z_dev=Z_dev_star,
            Z_test=project_out(dataset.Z_test, rotation.basis),
            theta_train=dataset.theta_train,
            theta_dev=dataset.theta_dev,
            labels=dataset.labels,
            Z_clip_train=dataset.Z_clip_train,
            Z_clip_dev=dataset.Z_clip_dev,
            same_pairs_dev=dataset.same_pairs_dev,
        )
        translator_star = collate_translator_metrics(dataset_star, dictionary, label_prompts, A_star, temperature_map)
        if label_prompts:
            translator_probs_star = compute_mean_probabilities(
                dataset_star.Z_dev,
                A_star,
                dictionary,
                label_prompts,
                temperature_map,
            )

    summary = EvaluationSummary(
        rotation=rotation,
        probe_raw=probe_raw,
        probe_derotated=probe_star,
        translator_raw=translator_raw,
        translator_derotated=translator_star,
        translator_probs_raw=translator_probs_raw,
        translator_probs_derotated=translator_probs_star,
    )

    write_summary(summary, out_dir)
    plot_diagnostics(summary, out_dir)
    return summary


def write_summary(summary: EvaluationSummary, out_dir: Path) -> None:
    data = {
        "rotation": {
            "r2_pre": summary.rotation.r2_pre,
            "r2_post": summary.rotation.r2_post,
            "same_cos_pre": summary.rotation.same_cos_pre,
            "same_cos_post": summary.rotation.same_cos_post,
            "basis_dim": int(summary.rotation.basis.shape[1]),
        },
        "probes": {
            "raw_dev": summary.probe_raw.accuracy_dev,
            "raw_test": summary.probe_raw.accuracy_test,
            "derotated_dev": summary.probe_derotated.accuracy_dev,
            "derotated_test": summary.probe_derotated.accuracy_test,
        },
    }
    if summary.translator_raw:
        data["translator_raw"] = {
            "aligned_cos_train": summary.translator_raw.aligned_cos_train,
            "aligned_cos_dev": summary.translator_raw.aligned_cos_dev,
            "accuracy_dev": summary.translator_raw.accuracy_dev,
            "accuracy_test": summary.translator_raw.accuracy_test,
        }
        if summary.translator_raw.temperature_map:
            data["translator_raw"]["temperature"] = summary.translator_raw.temperature_map
    if summary.translator_derotated:
        data["translator_derotated"] = {
            "aligned_cos_train": summary.translator_derotated.aligned_cos_train,
            "aligned_cos_dev": summary.translator_derotated.aligned_cos_dev,
            "accuracy_dev": summary.translator_derotated.accuracy_dev,
            "accuracy_test": summary.translator_derotated.accuracy_test,
        }
        if summary.translator_derotated.temperature_map:
            data["translator_derotated"]["temperature"] = summary.translator_derotated.temperature_map
    if summary.translator_probs_raw:
        data["translator_probabilities_raw"] = summary.translator_probs_raw
    if summary.translator_probs_derotated:
        data["translator_probabilities_derotated"] = summary.translator_probs_derotated
    (out_dir / "summary.json").write_text(json.dumps(data, indent=2))


def plot_diagnostics(summary: EvaluationSummary, out_dir: Path) -> None:
    plt.figure(figsize=(4.5, 3.5))
    plt.bar(["pre", "post"], [summary.rotation.r2_pre, summary.rotation.r2_post], color=["#d55e00", "#0072b2"])
    plt.ylabel("Pose R$^2$")
    plt.title("Pose predictability")
    plt.tight_layout()
    plt.savefig(out_dir / "pose_r2.png", dpi=160)
    plt.close()

    if summary.rotation.same_cos_pre is not None and summary.rotation.same_cos_post is not None:
        plt.figure(figsize=(4.5, 3.5))
        plt.bar(["pre", "post"], [summary.rotation.same_cos_pre, summary.rotation.same_cos_post], color=["#cc79a7", "#009e73"])
        plt.ylabel("Mean cosine")
        plt.title("Same-object cross-view consistency")
        plt.tight_layout()
        plt.savefig(out_dir / "same_object_cosine.png", dpi=160)
        plt.close()

    if summary.probe_raw.accuracy_dev:
        plot_probe_metrics(summary, out_dir)
    if summary.translator_raw:
        plot_translator_metrics(summary, out_dir)
    if summary.translator_probs_raw:
        plot_probability_distributions(summary.translator_probs_raw, out_dir, suffix="raw")
    if summary.translator_probs_derotated:
        plot_probability_distributions(summary.translator_probs_derotated, out_dir, suffix="derotated")


def plot_probe_metrics(summary: EvaluationSummary, out_dir: Path) -> None:
    labels = list(summary.probe_raw.accuracy_dev.keys())
    x = np.arange(len(labels))
    width = 0.35

    dev_raw = [summary.probe_raw.accuracy_dev[label] for label in labels]
    dev_star = [summary.probe_derotated.accuracy_dev.get(label, np.nan) for label in labels]

    plt.figure(figsize=(6, 3.5))
    plt.bar(x - width / 2, dev_raw, width, label="raw")
    plt.bar(x + width / 2, dev_star, width, label="derotated")
    plt.xticks(x, labels)
    plt.ylim(0, 1.0)
    plt.ylabel("Dev accuracy")
    plt.title("Probe accuracies")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "probe_accuracy_dev.png", dpi=160)
    plt.close()

    test_raw = [summary.probe_raw.accuracy_test[label] for label in labels]
    test_star = [summary.probe_derotated.accuracy_test.get(label, np.nan) for label in labels]

    plt.figure(figsize=(6, 3.5))
    plt.bar(x - width / 2, test_raw, width, label="raw")
    plt.bar(x + width / 2, test_star, width, label="derotated")
    plt.xticks(x, labels)
    plt.ylim(0, 1.0)
    plt.ylabel("Test accuracy")
    plt.title("Probe accuracies")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "probe_accuracy_test.png", dpi=160)
    plt.close()


def plot_translator_metrics(summary: EvaluationSummary, out_dir: Path) -> None:
    raw = summary.translator_raw
    star = summary.translator_derotated
    if raw is None:
        return

    plt.figure(figsize=(4.5, 3.5))
    values = [raw.aligned_cos_train, raw.aligned_cos_dev]
    labels = ["train", "dev"]
    if star is not None:
        values += [star.aligned_cos_train, star.aligned_cos_dev]
        labels += ["train*", "dev*"]
    plt.bar(labels, values, color=["#56b4e9", "#009e73", "#8073ac", "#b2abd2"][: len(labels)])
    plt.ylabel("Mean cosine")
    plt.title("Translator alignment")
    plt.tight_layout()
    plt.savefig(out_dir / "translator_alignment.png", dpi=160)
    plt.close()

    if raw.accuracy_dev:
        groups = list(raw.accuracy_dev.keys())
        x = np.arange(len(groups))
        width = 0.35
        plt.figure(figsize=(6, 3.5))
        raw_dev = [raw.accuracy_dev[g] for g in groups]
        bars = [raw_dev]
        labels_list = ["raw"]
        if star and star.accuracy_dev:
            bars.append([star.accuracy_dev.get(g, np.nan) for g in groups])
            labels_list.append("derotated")
        for idx, label in enumerate(labels_list):
            offset = (idx - (len(labels_list) - 1) / 2) * width
            plt.bar(x + offset, bars[idx], width, label=label)
        plt.xticks(x, groups)
        plt.ylim(0, 1.0)
        plt.ylabel("Dev accuracy")
        plt.title("Translator group accuracies")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "translator_accuracy_dev.png", dpi=160)
        plt.close()

        if raw.accuracy_test:
            plt.figure(figsize=(6, 3.5))
            raw_test = [raw.accuracy_test[g] for g in groups]
            bars = [raw_test]
            labels_list = ["raw"]
            if star and star.accuracy_test:
                bars.append([star.accuracy_test.get(g, np.nan) for g in groups])
                labels_list.append("derotated")
            for idx, label in enumerate(labels_list):
                offset = (idx - (len(labels_list) - 1) / 2) * width
                plt.bar(x + offset, bars[idx], width, label=label)
            plt.xticks(x, groups)
            plt.ylim(0, 1.0)
            plt.ylabel("Test accuracy")
            plt.title("Translator group accuracies")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "translator_accuracy_test.png", dpi=160)
            plt.close()


def plot_probability_distributions(
    probabilities: Dict[str, Dict[str, float]],
    out_dir: Path,
    *,
    suffix: str,
) -> None:
    for group, probs in probabilities.items():
        labels = list(probs.keys())
        values = [probs[label] for label in labels]
        plt.figure(figsize=(6, 3.5))
        plt.bar(np.arange(len(labels)), values, color="#56b4e9")
        plt.xticks(np.arange(len(labels)), labels, rotation=30, ha="right")
        plt.ylim(0, 1.0)
        plt.ylabel("Mean likelihood")
        plt.title(f"{group} likelihood ({suffix})")
        plt.tight_layout()
        path = out_dir / f"translator_distribution_{group}_{suffix}.png"
        plt.savefig(path, dpi=160)
        plt.close()


def parse_label_prompts(path: Optional[Path]) -> Dict[str, List[str]]:
    if not path:
        return {}
    spec = json.loads(path.read_text())
    return {k: list(v) for k, v in spec.items()}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate probe vs translator semantic readouts")
    parser.add_argument("dataset", type=Path, help="NPZ file containing FastViT/CLIP features and labels")
    parser.add_argument("--dictionary", type=Path, required=True, help="Cached CLIP text dictionary npz")
    parser.add_argument("--label-prompts", type=Path, help="JSON mapping label name -> ordered list of prompt strings")
    parser.add_argument("--labels", nargs="+", default=["side"], help="Label field prefixes present in dataset")
    parser.add_argument("--out", type=Path, default=Path("artifacts/semantic_eval"))
    parser.add_argument("--pose-alpha", type=float, default=0.1, help="Ridge penalty for pose plane")
    parser.add_argument("--logreg-C", type=float, default=4.0, help="Inverse regularisation strength for probes")
    parser.add_argument("--translator-alpha", type=float, default=1e-2, help="Ridge penalty for translator fit")
    parser.add_argument("--skip-probes", action="store_true", help="Skip probe evaluation")
    parser.add_argument("--skip-translator", action="store_true", help="Skip translator evaluation")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature for translator likelihoods")
    parser.add_argument(
        "--calibrate-temperature",
        action="store_true",
        help="Estimate per-group temperature on the dev split",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    dataset = EvaluationDataset.from_npz(args.dataset, args.labels)
    dictionary = TextDictionary.load(args.dictionary)
    label_prompts = parse_label_prompts(args.label_prompts)
    if label_prompts:
        label_prompts = {k: v for k, v in label_prompts.items() if k in dataset.labels}

    if not args.skip_translator and label_prompts and not set(label_prompts.keys()).issubset(dataset.labels.keys()):
        missing = set(label_prompts.keys()) - set(dataset.labels.keys())
        raise ValueError(f"label prompts provided for unknown labels: {sorted(missing)}")

    run_evaluation(
        dataset,
        dictionary,
        label_prompts,
        out_dir=args.out,
        pose_alpha=args.pose_alpha,
        logreg_C=args.logreg_C,
        translator_alpha=args.translator_alpha,
        skip_probes=args.skip_probes,
        skip_translator=args.skip_translator,
        temperature_default=args.temperature,
        calibrate_temperature=args.calibrate_temperature,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
