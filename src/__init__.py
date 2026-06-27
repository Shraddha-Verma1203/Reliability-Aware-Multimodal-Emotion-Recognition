"""Reliability-aware multimodal emotion recognition package."""

from .common import EMOTION_CLASSES, FusedPrediction, ModalityPrediction, normalize_label
from .conflict_detector import ConflictResult, detect_emotion_conflict
from .fusion import ReliabilityAwareFusion
from .reliability import ReliabilityScorer

__all__ = [
    "EMOTION_CLASSES",
    "ConflictResult",
    "FusedPrediction",
    "ModalityPrediction",
    "ReliabilityAwareFusion",
    "ReliabilityScorer",
    "detect_emotion_conflict",
    "normalize_label",
]
