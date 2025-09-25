from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import json
import numpy as np


@dataclass
class TemperatureScaler:
    temperature: float

    def apply(self, logits: np.ndarray) -> np.ndarray:
        tau = max(self.temperature, 1e-6)
        return logits / tau

    def to_json(self) -> Dict[str, float]:
        return {"temperature": float(self.temperature)}

    @classmethod
    def from_json(cls, data: Dict[str, float]) -> "TemperatureScaler":
        return cls(temperature=float(data["temperature"]))


@dataclass
class TauModel:
    weights: np.ndarray
    bias: float

    def tau(self, features: np.ndarray) -> np.ndarray:
        logits = features @ self.weights + self.bias
        return np.log1p(np.exp(logits)) + 1e-6

    def apply(self, logits: np.ndarray, features: np.ndarray) -> np.ndarray:
        scale = self.tau(features)[:, None]
        return logits / scale

    def to_json(self) -> Dict[str, list]:
        return {"weights": self.weights.tolist(), "bias": float(self.bias)}

    @classmethod
    def from_json(cls, data: Dict[str, list]) -> "TauModel":
        return cls(weights=np.asarray(data["weights"], dtype=np.float32), bias=float(data["bias"]))


@dataclass
class VectorScaling:
    weights: np.ndarray  # shape (num_classes,)
    bias: np.ndarray

    def apply(self, logits: np.ndarray) -> np.ndarray:
        return logits * self.weights + self.bias

    def to_json(self) -> Dict[str, list]:
        return {"weights": self.weights.tolist(), "bias": self.bias.tolist()}

    @classmethod
    def from_json(cls, data: Dict[str, list]) -> "VectorScaling":
        weights = np.asarray(data["weights"], dtype=np.float32)
        bias = np.asarray(data["bias"], dtype=np.float32)
        return cls(weights=weights, bias=bias)


@dataclass
class EnergyGate:
    threshold: float

    def is_ood(self, scores: np.ndarray) -> np.ndarray:
        energy = -np.log(np.sum(np.exp(scores), axis=1))
        return energy > self.threshold

    def to_json(self) -> Dict[str, float]:
        return {"threshold": float(self.threshold)}

    @classmethod
    def from_json(cls, data: Dict[str, float]) -> "EnergyGate":
        return cls(threshold=float(data["threshold"]))


@dataclass
class CalibrationBundle:
    temperature: Optional[TemperatureScaler] = None
    tau_model: Optional[TauModel] = None
    vector_scaling: Optional[VectorScaling] = None
    energy_gate: Optional[EnergyGate] = None
    priors: Optional[Dict[str, np.ndarray]] = None

    def apply(self, logits: np.ndarray, *, features: Optional[np.ndarray] = None) -> np.ndarray:
        adjusted = logits
        if self.temperature is not None:
            adjusted = self.temperature.apply(adjusted)
        if self.tau_model is not None:
            if features is None:
                raise ValueError("features required for tau_model")
            adjusted = self.tau_model.apply(adjusted, features)
        if self.vector_scaling is not None:
            adjusted = self.vector_scaling.apply(adjusted)
        return adjusted

    def to_disk(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Dict] = {}
        if self.temperature:
            payload["temperature"] = self.temperature.to_json()
        if self.tau_model:
            payload["tau_model"] = self.tau_model.to_json()
        if self.vector_scaling:
            payload["vector_scaling"] = self.vector_scaling.to_json()
        if self.energy_gate:
            payload["energy"] = self.energy_gate.to_json()
        with (directory / "calibration_bundle.json").open("w") as f:
            json.dump(payload, f, indent=2)


def load_calibration_bundle(directory: Path) -> CalibrationBundle:
    path = directory / "calibration_bundle.json"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open() as f:
        data = json.load(f)
    bundle = CalibrationBundle()
    if "temperature" in data:
        bundle.temperature = TemperatureScaler.from_json(data["temperature"])
    if "tau_model" in data:
        bundle.tau_model = TauModel.from_json(data["tau_model"])
    if "vector_scaling" in data:
        bundle.vector_scaling = VectorScaling.from_json(data["vector_scaling"])
    if "energy" in data:
        bundle.energy_gate = EnergyGate.from_json(data["energy"])
    return bundle


__all__ = [
    "TemperatureScaler",
    "TauModel",
    "VectorScaling",
    "EnergyGate",
    "CalibrationBundle",
    "load_calibration_bundle",
]
