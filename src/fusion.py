"""Reliability-aware fusion for multimodal emotion probabilities."""

from __future__ import annotations

from typing import Dict, Mapping

from .common import EMOTION_CLASSES, FusedPrediction, ModalityPrediction, normalize_probabilities, top_label, uniform_distribution
from .reliability import ReliabilityResult


class ReliabilityAwareFusion:
    """Fuse modality predictions by reliability-normalized probability averaging."""

    def __init__(self, min_reliability: float = 0.05) -> None:
        self.min_reliability = min_reliability

    def fuse(
        self,
        predictions: Mapping[str, ModalityPrediction],
        reliabilities: Mapping[str, float | ReliabilityResult],
    ) -> FusedPrediction:
        available = {
            name: prediction
            for name, prediction in predictions.items()
            if prediction.is_available and sum(prediction.probabilities.values()) > 0
        }

        if not available:
            probabilities = uniform_distribution()
            label = top_label(probabilities)
            return FusedPrediction(
                label=label,
                probabilities=probabilities,
                confidence=probabilities[label],
                weights={name: 0.0 for name in predictions},
                modality_predictions=dict(predictions),
                reliability_scores={name: 0.0 for name in predictions},
            )

        reliability_scores = {
            name: self._as_score(reliabilities.get(name, 0.0))
            for name in predictions
        }
        active_scores = {
            name: score
            for name, score in reliability_scores.items()
            if name in available and score >= self.min_reliability
        }

        if not active_scores:
            active_scores = {name: 1.0 for name in available}

        total_reliability = sum(active_scores.values())
        weights = {
            name: (active_scores[name] / total_reliability if name in active_scores else 0.0)
            for name in predictions
        }

        fused_scores = {emotion: 0.0 for emotion in EMOTION_CLASSES}
        for name, prediction in available.items():
            weight = weights.get(name, 0.0)
            for emotion in EMOTION_CLASSES:
                fused_scores[emotion] += weight * prediction.probabilities.get(emotion, 0.0)

        probabilities = normalize_probabilities(fused_scores)
        label = top_label(probabilities)
        return FusedPrediction(
            label=label,
            probabilities=probabilities,
            confidence=probabilities[label],
            weights=weights,
            modality_predictions=dict(predictions),
            reliability_scores=reliability_scores,
        )

    @staticmethod
    def _as_score(value: float | ReliabilityResult) -> float:
        if isinstance(value, ReliabilityResult):
            return float(value.score)
        return float(value or 0.0)
