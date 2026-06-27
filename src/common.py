"""Shared types and helpers for multimodal emotion recognition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional


EMOTION_CLASSES = ["happy", "sad", "angry", "neutral", "fear", "surprise", "disgust"]

LABEL_ALIASES = {
    "anger": "angry",
    "angry": "angry",
    "annoyance": "angry",
    "joy": "happy",
    "happiness": "happy",
    "happy": "happy",
    "sadness": "sad",
    "sad": "sad",
    "neutral": "neutral",
    "calm": "neutral",
    "fear": "fear",
    "fearful": "fear",
    "surprise": "surprise",
    "surprised": "surprise",
    "disgust": "disgust",
    "disgusted": "disgust",
}


def normalize_label(label: str) -> str:
    """Map model-specific labels into the seven project emotion classes."""

    cleaned = str(label).strip().lower().replace("_", " ").replace("-", " ")
    if cleaned.startswith("label_"):
        cleaned = cleaned.replace("label_", "")
    return LABEL_ALIASES.get(cleaned, cleaned)


def empty_distribution() -> Dict[str, float]:
    return {emotion: 0.0 for emotion in EMOTION_CLASSES}


def uniform_distribution() -> Dict[str, float]:
    value = 1.0 / len(EMOTION_CLASSES)
    return {emotion: value for emotion in EMOTION_CLASSES}


def normalize_probabilities(scores: Mapping[str, float]) -> Dict[str, float]:
    """Aggregate aliases, clamp negatives, and normalize over project classes."""

    probabilities = empty_distribution()
    for label, score in scores.items():
        normalized = normalize_label(label)
        if normalized in probabilities:
            probabilities[normalized] += max(0.0, float(score))

    total = sum(probabilities.values())
    if total <= 0:
        return uniform_distribution()
    return {label: value / total for label, value in probabilities.items()}


def top_label(probabilities: Mapping[str, float]) -> str:
    return max(EMOTION_CLASSES, key=lambda label: probabilities.get(label, 0.0))


@dataclass
class ModalityPrediction:
    """Prediction returned by a single modality recognizer."""

    label: str
    probabilities: Dict[str, float]
    confidence: float
    modality: str
    is_available: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @classmethod
    def from_scores(
        cls,
        scores: Mapping[str, float],
        modality: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ModalityPrediction":
        probabilities = normalize_probabilities(scores)
        label = top_label(probabilities)
        return cls(
            label=label,
            probabilities=probabilities,
            confidence=float(probabilities[label]),
            modality=modality,
            metadata=metadata or {},
        )

    @classmethod
    def unavailable(cls, modality: str, reason: str) -> "ModalityPrediction":
        return cls(
            label="unavailable",
            probabilities=empty_distribution(),
            confidence=0.0,
            modality=modality,
            is_available=False,
            metadata={},
            error=reason,
        )


@dataclass
class FusedPrediction:
    """Final reliability-aware prediction over all available modalities."""

    label: str
    probabilities: Dict[str, float]
    confidence: float
    weights: Dict[str, float]
    modality_predictions: Dict[str, ModalityPrediction]
    reliability_scores: Dict[str, float]
