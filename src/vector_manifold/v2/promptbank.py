from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import json
import numpy as np

from .config import PromptBankConfig, PromptGroup, PromptSynonyms


def _config_to_json(config: PromptBankConfig) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
    payload: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
    for group_name, group in config.groups.items():
        payload[group_name] = {}
        for class_name, synonyms in group.classes.items():
            payload[group_name][class_name] = {
                "synonyms": list(synonyms.prompts),
                "negatives": list(synonyms.negatives),
            }
    return payload


def _config_from_json(payload: Dict[str, Dict[str, Dict[str, List[str]]]]) -> PromptBankConfig:
    groups: Dict[str, PromptGroup] = {}
    for group_name, classes in payload.items():
        cfg_classes = {}
        for class_name, entry in classes.items():
            cfg_classes[class_name] = PromptSynonyms(
                name=class_name,
                prompts=entry.get("synonyms", []),
                negatives=entry.get("negatives", []),
            )
        groups[group_name] = PromptGroup(name=group_name, classes=cfg_classes)
    return PromptBankConfig(groups=groups)


@dataclass
class PromptBank:
    config: PromptBankConfig
    prompt_list: List[str]
    group_indices: Dict[str, np.ndarray]
    class_indices: Dict[Tuple[str, str], np.ndarray]
    negative_indices: Dict[Tuple[str, str], np.ndarray]
    text_embeddings: np.ndarray

    def to_npz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "prompt_list": np.array(self.prompt_list, dtype=object),
            "group_indices": np.array(
                [json.dumps({k: v.tolist() for k, v in self.group_indices.items()})],
                dtype=object,
            ),
            "class_indices": np.array(
                [json.dumps({f"{g}::{c}": v.tolist() for (g, c), v in self.class_indices.items()})],
                dtype=object,
            ),
            "class_negatives": np.array(
                [
                    json.dumps({f"{g}::{c}": v.tolist() for (g, c), v in self.negative_indices.items()})
                ],
                dtype=object,
            ),
            "text_embeddings": self.text_embeddings.astype(np.float32),
            "config": np.array([json.dumps(_config_to_json(self.config))], dtype=object),
        }
        np.savez_compressed(path, **payload)

    @classmethod
    def from_npz(cls, path: Path, config: Optional[PromptBankConfig] = None) -> "PromptBank":
        data = np.load(path, allow_pickle=True)
        prompt_list = data["prompt_list"].tolist()
        group_indices_json = json.loads(data["group_indices"][0])
        class_indices_json = json.loads(data["class_indices"][0])
        neg_indices_json = json.loads(data["class_negatives"][0]) if "class_negatives" in data else {}
        group_indices = {k: np.asarray(v, dtype=int) for k, v in group_indices_json.items()}
        class_indices = {}
        negative_indices = {}
        for key, values in class_indices_json.items():
            group, cls_name = key.split("::", 1)
            class_indices[(group, cls_name)] = np.asarray(values, dtype=int)
        for key, values in neg_indices_json.items():
            group, cls_name = key.split("::", 1)
            negative_indices[(group, cls_name)] = np.asarray(values, dtype=int)
        embeddings = np.asarray(data["text_embeddings"], dtype=np.float32)
        if config is None:
            if "config" not in data:
                raise ValueError("prompt bank missing embedded config; provide config manually")
            config_payload = json.loads(data["config"][0])
            config = _config_from_json(config_payload)
        return cls(
            config=config,
            prompt_list=prompt_list,
            group_indices=group_indices,
            class_indices=class_indices,
            negative_indices=negative_indices,
            text_embeddings=embeddings,
        )

    @staticmethod
    def build(config: PromptBankConfig, text_encoder) -> "PromptBank":
        prompts: List[str] = []
        group_indices: Dict[str, List[int]] = {}
        class_indices: Dict[Tuple[str, str], List[int]] = {}
        negative_indices: Dict[Tuple[str, str], List[int]] = {}

        for group_name, group in config.groups.items():
            group_positions: List[int] = []
            for class_name, synonyms in group.classes.items():
                class_positions: List[int] = []
                neg_positions: List[int] = []
                for prompt in synonyms.prompts:
                    class_positions.append(len(prompts))
                    group_positions.append(len(prompts))
                    prompts.append(prompt)
                for neg in synonyms.negatives:
                    neg_positions.append(len(prompts))
                    group_positions.append(len(prompts))
                    prompts.append(neg)
                class_indices[(group_name, class_name)] = class_positions
                negative_indices[(group_name, class_name)] = neg_positions
            group_indices[group_name] = group_positions

        embeddings = text_encoder(prompts)
        if embeddings.shape[0] != len(prompts):
            raise ValueError("text encoder returned mismatched embedding count")
        return PromptBank(
            config=config,
            prompt_list=prompts,
            group_indices={k: np.asarray(v, dtype=int) for k, v in group_indices.items()},
            class_indices={k: np.asarray(v, dtype=int) for k, v in class_indices.items()},
            negative_indices={k: np.asarray(v, dtype=int) for k, v in negative_indices.items()},
            text_embeddings=np.asarray(embeddings, dtype=np.float32),
        )

    def class_prompt_indices(self, group: str, class_name: str) -> np.ndarray:
        key = (group, class_name)
        if key not in self.class_indices:
            raise KeyError(f"class '{class_name}' missing for group '{group}'")
        return self.class_indices[key]

    def class_negative_indices(self, group: str, class_name: str) -> np.ndarray:
        key = (group, class_name)
        return self.negative_indices.get(key, np.asarray([], dtype=int))

    def group_prompt_indices(self, group: str) -> np.ndarray:
        if group not in self.group_indices:
            raise KeyError(f"group '{group}' missing")
        return self.group_indices[group]


def merge_prompt_configs(paths: Iterable[Path]) -> PromptBankConfig:
    merged: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
    for path in paths:
        with path.open() as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Prompt config {path} must contain a JSON object")
        for group, classes in data.items():
            merged_group = merged.setdefault(group, {})
            if isinstance(classes, list):
                class_items = {prompt: {"synonyms": [prompt], "negatives": []} for prompt in classes}
            else:
                class_items = classes
            for class_name, entry in class_items.items():
                if isinstance(entry, dict):
                    synonyms = entry.get("synonyms", [])
                    negatives = entry.get("negatives", [])
                elif isinstance(entry, (list, tuple, set)):
                    synonyms = list(entry)
                    negatives = []
                else:
                    synonyms = [entry]
                    negatives = []
                merged_class = merged_group.setdefault(class_name, {"synonyms": [], "negatives": []})
                for synonym in synonyms:
                    if synonym not in merged_class["synonyms"]:
                        merged_class["synonyms"].append(synonym)
                for neg in negatives:
                    if neg not in merged_class["negatives"]:
                        merged_class["negatives"].append(neg)
    return _config_from_json(merged)


__all__ = ["PromptBank", "merge_prompt_configs"]
