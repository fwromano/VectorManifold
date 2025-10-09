from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import json


@dataclass
class PromptSynonyms:
    name: str
    prompts: Sequence[str]
    negatives: Sequence[str] = ()


@dataclass
class PromptGroup:
    name: str
    classes: Dict[str, PromptSynonyms]


@dataclass
class PromptBankConfig:
    groups: Dict[str, PromptGroup] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: Path) -> "PromptBankConfig":
        with path.open() as f:
            payload = json.load(f)
        groups: Dict[str, PromptGroup] = {}
        for group_name, entries in payload.items():
            classes: Dict[str, PromptSynonyms] = {}
            for class_name, prompts in entries.items():
                if isinstance(prompts, Mapping):
                    synonyms = prompts.get("synonyms", [])
                    negatives = prompts.get("negatives", [])
                else:
                    synonyms = list(prompts)
                    negatives = []
                if not synonyms:
                    raise ValueError(f"group '{group_name}' class '{class_name}' missing prompts")
                classes[class_name] = PromptSynonyms(name=class_name, prompts=synonyms, negatives=negatives)
            groups[group_name] = PromptGroup(name=group_name, classes=classes)
        return cls(groups=groups)

    def to_json(self, path: Path) -> None:
        payload: Dict[str, Dict[str, Dict[str, Sequence[str]]]] = {}
        for group_name, group in self.groups.items():
            payload[group_name] = {}
            for class_name, synonyms in group.classes.items():
                payload[group_name][class_name] = {
                    "synonyms": list(synonyms.prompts),
                    "negatives": list(synonyms.negatives),
                }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(payload, f, indent=2)

    def class_names(self, group: str) -> List[str]:
        if group not in self.groups:
            raise KeyError(f"group '{group}' not defined")
        return list(self.groups[group].classes.keys())

    def synonym_prompts(self, group: str, class_name: str) -> PromptSynonyms:
        if group not in self.groups:
            raise KeyError(f"group '{group}' not defined")
        classes = self.groups[group].classes
        if class_name not in classes:
            raise KeyError(f"class '{class_name}' not found in group '{group}'")
        return classes[class_name]

    def total_prompt_count(self) -> int:
        total = 0
        for group in self.groups.values():
            for synonyms in group.classes.values():
                total += len(synonyms.prompts)
                total += len(synonyms.negatives)
        return total


__all__ = ["PromptBankConfig", "PromptGroup", "PromptSynonyms"]
