from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .calibration import CalibrationBundle
from .promptbank import PromptBank
from .scoring import batch_synonym_pool


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def decode_probabilities(
    scores: np.ndarray,
    promptbank: PromptBank,
    *,
    calibration: Optional[CalibrationBundle] = None,
    quality_features: Optional[np.ndarray] = None,
    topk: int = 5,
    pooling: str = "max",
    temperature: float = 1.0,
    subtract_negatives: bool = False,
) -> List[Dict]:
    logits = scores.copy()
    if calibration is not None:
        logits = calibration.apply(logits, features=quality_features)

    pooled = batch_synonym_pool(logits, promptbank, mode=pooling, subtract_negatives=subtract_negatives)
    records: List[Dict] = []

    # optional energy gate relies on pooled logits stacked by group
    energy_mask = None
    if calibration is not None and calibration.energy_gate is not None:
        stacked = np.concatenate(list(pooled.values()), axis=1)
        energy_mask = calibration.energy_gate.is_ood(stacked)

    tau = max(temperature, 1e-6)

    for idx in range(scores.shape[0]):
        entry: Dict = {"index": int(idx)}
        if energy_mask is not None:
            entry["ood"] = bool(energy_mask[idx])
        for group_name, group in promptbank.config.groups.items():
            group_logits = pooled[group_name][idx : idx + 1]
            scaled_logits = group_logits / tau
            probs = softmax(scaled_logits)
            distributions: List[Dict[str, float]] = []
            class_names = list(group.classes.keys())
            for cls_idx, class_name in enumerate(class_names):
                synonyms = group.classes[class_name]
                canonical = synonyms.prompts[0]
                distributions.append(
                    {
                        "prompt": canonical,
                        "score": float(group_logits[0, cls_idx]),
                        "calibrated_score": float(scaled_logits[0, cls_idx]),
                        "prob": float(probs[0, cls_idx]),
                    }
                )
            distributions.sort(key=lambda item: item["prob"], reverse=True)
            entry[group_name] = {
                "topk": distributions[: min(topk, len(distributions))],
                "distribution": distributions,
            }
        records.append(entry)
    return records


__all__ = ["decode_probabilities"]
