from __future__ import annotations

from typing import Dict, Literal, Mapping

import numpy as np

from .promptbank import PromptBank

PoolMode = Literal["max", "mean"]


def cosine_scores(image_features: np.ndarray, promptbank: PromptBank) -> np.ndarray:
    if image_features.ndim != 2:
        raise ValueError("image_features must be 2-D")
    text = promptbank.text_embeddings
    if text.ndim != 2:
        raise ValueError("prompt embeddings must be 2-D")
    if text.shape[1] != image_features.shape[1]:
        raise ValueError("feature and prompt dimensions mismatch")
    return image_features @ text.T


def synonym_pool(
    scores: np.ndarray,
    promptbank: PromptBank,
    *,
    mode: PoolMode = "max",
    subtract_negatives: bool = False,
) -> Dict[str, Dict[str, float]]:
    pooled: Dict[str, Dict[str, float]] = {}
    for group_name, group in promptbank.config.groups.items():
        group_scores: Dict[str, float] = {}
        for class_name in group.classes:
            idx = promptbank.class_prompt_indices(group_name, class_name)
            class_scores = scores[idx]
            if class_scores.size == 0:
                value = float("nan")
            else:
                if mode == "max":
                    value = float(np.nanmax(class_scores))
                elif mode == "mean":
                    value = float(np.nanmean(class_scores))
                else:
                    raise ValueError(f"unknown pooling mode '{mode}'")
                if subtract_negatives:
                    neg_idx = promptbank.class_negative_indices(group_name, class_name)
                    if neg_idx.size:
                        neg_scores = scores[neg_idx]
                        if mode == "max":
                            value -= float(np.nanmax(neg_scores))
                        else:
                            value -= float(np.nanmean(neg_scores))
            group_scores[class_name] = value
        pooled[group_name] = group_scores
    return pooled


def batch_synonym_pool(
    scores: np.ndarray,
    promptbank: PromptBank,
    *,
    mode: PoolMode = "max",
    subtract_negatives: bool = False,
) -> Dict[str, np.ndarray]:
    if scores.ndim != 2:
        raise ValueError("scores must be 2-D (frames x prompts)")
    pooled: Dict[str, np.ndarray] = {}
    for group_name, group in promptbank.config.groups.items():
        group_matrix = np.zeros((scores.shape[0], len(group.classes)), dtype=np.float32)
        for idx_col, class_name in enumerate(group.classes):
            idx = promptbank.class_prompt_indices(group_name, class_name)
            class_scores = scores[:, idx]
            if class_scores.shape[1] == 0:
                group_matrix[:, idx_col] = np.nan
                continue
            if mode == "max":
                group_matrix[:, idx_col] = np.max(class_scores, axis=1)
            elif mode == "mean":
                group_matrix[:, idx_col] = np.mean(class_scores, axis=1)
            else:
                raise ValueError(f"unknown pooling mode '{mode}'")
            if subtract_negatives:
                neg_idx = promptbank.class_negative_indices(group_name, class_name)
                if neg_idx.size:
                    neg_scores = scores[:, neg_idx]
                    if mode == "max":
                        group_matrix[:, idx_col] -= np.max(neg_scores, axis=1)
                    else:
                        group_matrix[:, idx_col] -= np.mean(neg_scores, axis=1)
        pooled[group_name] = group_matrix
    return pooled


__all__ = ["cosine_scores", "synonym_pool", "batch_synonym_pool"]
