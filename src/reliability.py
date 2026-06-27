"""Reliability scoring for text, audio, and facial predictions."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Mapping

from .common import EMOTION_CLASSES, ModalityPrediction


@dataclass
class ReliabilityResult:
    modality: str
    score: float
    components: Dict[str, float] = field(default_factory=dict)


class ReliabilityScorer:
    """Estimate how trustworthy each modality prediction is."""

    def score(self, prediction: ModalityPrediction, raw_input: object | None = None) -> ReliabilityResult:
        if prediction.modality == "text":
            return self.score_text(prediction, str(raw_input or ""))
        if prediction.modality == "audio":
            return self.score_audio(prediction)
        if prediction.modality in {"face", "video"}:
            return self.score_face(prediction)
        return ReliabilityResult(prediction.modality, 0.0, {"available": 0.0})

    def score_text(self, prediction: ModalityPrediction, text: str = "") -> ReliabilityResult:
        base = _probability_reliability(prediction)
        text = (text or "").strip()
        word_count = len(text.split()) or int(prediction.metadata.get("word_count", 0) or 0)
        length_score = _clip(word_count / 8.0) if word_count < 8 else 1.0
        if word_count > 80:
            length_score = _clip(1.0 - ((word_count - 80) / 160.0))

        score = _weighted_mean({"confidence": base, "length": length_score}, {"confidence": 0.75, "length": 0.25})
        return ReliabilityResult("text", score, {"confidence": base, "length": length_score})

    def score_audio(self, prediction: ModalityPrediction) -> ReliabilityResult:
        base = _probability_reliability(prediction)
        metadata = prediction.metadata
        duration = float(metadata.get("duration_seconds", 0.0) or 0.0)
        snr_db = float(metadata.get("snr_db", 0.0) or 0.0)
        clipping_ratio = float(metadata.get("clipping_ratio", 1.0) or 0.0)
        rms = float(metadata.get("rms", 0.0) or 0.0)

        duration_score = _triangular(duration, low=0.5, ideal_low=1.5, ideal_high=8.0, high=15.0)
        snr_score = _clip((snr_db - 5.0) / 25.0)
        clipping_score = _clip(1.0 - (clipping_ratio / 0.08))
        energy_score = _triangular(rms, low=0.005, ideal_low=0.03, ideal_high=0.35, high=0.8)

        components = {
            "confidence": base,
            "duration": duration_score,
            "snr": snr_score,
            "clipping": clipping_score,
            "energy": energy_score,
        }
        weights = {"confidence": 0.45, "duration": 0.15, "snr": 0.2, "clipping": 0.1, "energy": 0.1}
        return ReliabilityResult("audio", _weighted_mean(components, weights), components)

    def score_face(self, prediction: ModalityPrediction) -> ReliabilityResult:
        base = _probability_reliability(prediction)
        metadata = prediction.metadata
        face_detected = float(metadata.get("face_detected", 1.0 if prediction.is_available else 0.0))
        blur_variance = float(metadata.get("blur_variance", 0.0) or 0.0)
        brightness = float(metadata.get("brightness", 0.0) or 0.0)
        face_area_ratio = float(metadata.get("face_area_ratio", 0.0) or 0.0)

        blur_score = _clip((blur_variance - 30.0) / 170.0)
        brightness_score = _triangular(brightness, low=25.0, ideal_low=70.0, ideal_high=190.0, high=240.0)
        face_size_score = _triangular(face_area_ratio, low=0.01, ideal_low=0.06, ideal_high=0.45, high=0.85)

        components = {
            "confidence": base,
            "face_detected": _clip(face_detected),
            "blur": blur_score,
            "brightness": brightness_score,
            "face_size": face_size_score,
        }
        weights = {
            "confidence": 0.45,
            "face_detected": 0.2,
            "blur": 0.15,
            "brightness": 0.1,
            "face_size": 0.1,
        }
        return ReliabilityResult(prediction.modality, _weighted_mean(components, weights), components)


def _probability_reliability(prediction: ModalityPrediction) -> float:
    if not prediction.is_available:
        return 0.0

    n_classes = len(EMOTION_CLASSES)
    confidence_floor = 1.0 / n_classes
    confidence_score = _clip((prediction.confidence - confidence_floor) / (1.0 - confidence_floor))
    entropy_score = 1.0 - _normalized_entropy(prediction.probabilities)
    return _clip(0.65 * confidence_score + 0.35 * entropy_score)


def _normalized_entropy(probabilities: Mapping[str, float]) -> float:
    values = [max(float(probabilities.get(label, 0.0)), 1e-12) for label in EMOTION_CLASSES]
    entropy = -sum(value * math.log(value) for value in values)
    return _clip(entropy / math.log(len(EMOTION_CLASSES)))


def _weighted_mean(values: Mapping[str, float], weights: Mapping[str, float]) -> float:
    numerator = sum(_clip(values[key]) * weights[key] for key in values)
    denominator = sum(weights[key] for key in values)
    return _clip(numerator / denominator) if denominator else 0.0


def _triangular(value: float, low: float, ideal_low: float, ideal_high: float, high: float) -> float:
    if value <= low or value >= high:
        return 0.0
    if ideal_low <= value <= ideal_high:
        return 1.0
    if value < ideal_low:
        return _clip((value - low) / (ideal_low - low))
    return _clip((high - value) / (high - ideal_high))


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))
