"""Detect emotion disagreement across available modalities."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping

from .common import ModalityPrediction


@dataclass
class ConflictResult:
    conflict_detected: bool
    message: str
    agreeing_modalities: Dict[str, List[str]]
    dominant_emotion: str | None = None


def detect_emotion_conflict(predictions: Mapping[str, ModalityPrediction]) -> ConflictResult:
    """Detect whether available modalities predict different emotions.

    Missing or unavailable modalities are ignored. A conflict is reported when
    two or more available modalities produce different top emotions.
    """

    available = {
        modality: prediction.label
        for modality, prediction in predictions.items()
        if prediction.is_available and prediction.label not in {"", "unavailable", "missing"}
    }

    if len(available) <= 1:
        return ConflictResult(
            conflict_detected=False,
            message="No cross-modal conflict can be assessed with fewer than two available modalities.",
            agreeing_modalities=_group_by_emotion(available),
            dominant_emotion=next(iter(available.values()), None),
        )

    grouped = _group_by_emotion(available)
    if len(grouped) == 1:
        emotion = next(iter(grouped))
        return ConflictResult(
            conflict_detected=False,
            message=f"No emotion conflict detected. Available modalities agree on {emotion}.",
            agreeing_modalities=grouped,
            dominant_emotion=emotion,
        )

    counts = Counter(available.values())
    dominant_emotion, dominant_count = counts.most_common(1)[0]
    minority = {
        modality: emotion
        for modality, emotion in available.items()
        if emotion != dominant_emotion
    }
    minority_text = ", ".join(f"{modality}={emotion}" for modality, emotion in minority.items())
    dominant_modalities = ", ".join(grouped[dominant_emotion])

    if dominant_count > 1:
        message = (
            f"Emotion conflict detected. {dominant_modalities} suggest {dominant_emotion}, "
            f"while {minority_text}. This may indicate emotional masking or modality disagreement."
        )
    else:
        message = (
            "Emotion conflict detected. Each available modality points to a different emotion. "
            "This may indicate ambiguity, noise, emotional masking, or modality disagreement."
        )

    return ConflictResult(
        conflict_detected=True,
        message=message,
        agreeing_modalities=grouped,
        dominant_emotion=dominant_emotion,
    )


def _group_by_emotion(modality_to_emotion: Mapping[str, str]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for modality, emotion in modality_to_emotion.items():
        grouped.setdefault(emotion, []).append(modality)
    return grouped
