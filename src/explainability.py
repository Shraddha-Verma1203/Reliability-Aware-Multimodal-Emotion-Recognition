"""Generate human-readable explanations for reliability-aware MER outputs."""

from __future__ import annotations

from typing import Dict, List, Mapping

from .common import FusedPrediction, ModalityPrediction
from .conflict_detector import ConflictResult
from .reliability import ReliabilityResult


def modality_reason(
    modality: str,
    prediction: ModalityPrediction | None,
    reliability: ReliabilityResult | None,
    fusion_weight: float,
) -> str:
    """Explain one modality's prediction, reliability, and contribution."""

    if prediction is None:
        return f"{modality.title()} not provided. Re-weighting available modalities."
    if not prediction.is_available:
        return f"{modality.title()} unavailable: {prediction.error}. Its fusion weight is 0."

    reliability_score = reliability.score if reliability else 0.0
    engine = prediction.metadata.get("engine", "model")
    return (
        f"{modality.title()} predicted {prediction.label} with confidence {prediction.confidence:.2f}. "
        f"Reliability is {reliability_score:.2f}, so its final contribution weight is {fusion_weight:.2f}. "
        f"Backend: {engine}."
    )


def fusion_reason(
    fused: FusedPrediction,
    conflict: ConflictResult,
) -> str:
    """Explain why the fused output was selected."""

    active = [
        (modality, weight, fused.modality_predictions[modality])
        for modality, weight in fused.weights.items()
        if weight > 0 and modality in fused.modality_predictions
    ]
    active.sort(key=lambda item: item[1], reverse=True)

    if not active:
        return "No reliable modality was available, so the system returned a uniform fallback distribution."

    top_modality, top_weight, top_prediction = active[0]
    support = [
        f"{modality}={prediction.label} ({weight:.2f})"
        for modality, weight, prediction in active
    ]

    if conflict.conflict_detected:
        return (
            f"Final prediction leaned toward {fused.label} after resolving modality disagreement. "
            f"The largest contribution came from {top_modality}, which predicted {top_prediction.label} "
            f"with weight {top_weight:.2f}. Contributions: {', '.join(support)}."
        )

    return (
        f"Final prediction is {fused.label} because the most reliable available modalities supported this class. "
        f"Top contribution: {top_modality} with weight {top_weight:.2f}. "
        f"Contributions: {', '.join(support)}."
    )


def build_explainability_panel(
    predictions: Mapping[str, ModalityPrediction],
    reliabilities: Mapping[str, ReliabilityResult],
    fused: FusedPrediction,
    conflict: ConflictResult,
) -> Dict[str, str | List[str]]:
    """Return structured explanation text for the UI."""

    modality_reasons = [
        modality_reason(
            modality,
            predictions.get(modality),
            reliabilities.get(modality),
            fused.weights.get(modality, 0.0),
        )
        for modality in ("text", "audio", "video")
    ]
    return {
        "modality_reasons": modality_reasons,
        "fusion_reason": fusion_reason(fused, conflict),
        "conflict_reason": conflict.message,
    }
