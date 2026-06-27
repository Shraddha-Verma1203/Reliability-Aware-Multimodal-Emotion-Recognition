"""Text emotion recognition using a pretrained Hugging Face Transformer."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional

from .common import EMOTION_CLASSES, ModalityPrediction, normalize_probabilities


class TextEmotionRecognizer:
    """Lazy Hugging Face text emotion recognizer.

    The model is loaded only on first prediction so importing this module stays
    fast and does not require network access.
    """

    def __init__(
        self,
        model_name: str = "bhadresh-savani/bert-base-uncased-emotion",
        device: int = -1,
        use_transformer: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.use_transformer = use_transformer
        self._pipeline: Optional[Any] = None

    @property
    def pipeline(self) -> Any:
        if self._pipeline is None:
            from transformers import pipeline

            try:
                self._pipeline = pipeline(
                    "text-classification",
                    model=self.model_name,
                    tokenizer=self.model_name,
                    top_k=None,
                    device=self.device,
                )
            except TypeError:
                self._pipeline = pipeline(
                    "text-classification",
                    model=self.model_name,
                    tokenizer=self.model_name,
                    return_all_scores=True,
                    device=self.device,
                )
        return self._pipeline

    def predict(self, text: str) -> ModalityPrediction:
        text = (text or "").strip()
        if not text:
            return ModalityPrediction.unavailable("text", "No text was provided.")

        metadata = {
            "model_name": self.model_name,
            "algorithm": "BERT text branch",
            "feature_source": "token embeddings / text semantics",
            "text_length": len(text),
            "word_count": len(text.split()),
        }
        if not self.use_transformer:
            metadata["engine"] = "lexical_fallback"
            metadata["fallback_reason"] = "Fast demo mode is using the local lexical predictor."
            return lexical_emotion_prediction(text, metadata)

        try:
            raw_output = self.pipeline(text)
            scores = self._scores_from_pipeline_output(raw_output)
            metadata["engine"] = "transformer"
            metadata["backend_status"] = "real model"
            return ModalityPrediction.from_scores(scores, modality="text", metadata=metadata)
        except Exception as exc:
            return ModalityPrediction(
                label="unavailable",
                probabilities={emotion: 0.0 for emotion in EMOTION_CLASSES},
                confidence=0.0,
                modality="text",
                is_available=False,
                metadata={
                    **metadata,
                    "engine": "model_error",
                    "backend_status": "real model failed",
                },
                error=f"Text Transformer model failed: {exc}",
            )

    @staticmethod
    def _scores_from_pipeline_output(raw_output: Any) -> Dict[str, float]:
        if isinstance(raw_output, list) and raw_output and isinstance(raw_output[0], list):
            rows: List[Dict[str, Any]] = raw_output[0]
        elif isinstance(raw_output, list):
            rows = raw_output
        else:
            rows = [raw_output]

        scores: Dict[str, float] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            label = str(row.get("label", ""))
            score = float(row.get("score", 0.0))
            scores[label] = score
        return scores


LEXICON = {
    "happy": {
        "happy", "joy", "joyful", "excited", "great", "good", "excellent", "amazing",
        "love", "glad", "proud", "delighted", "smile", "success", "won", "awesome",
    },
    "sad": {
        "sad", "upset", "down", "depressed", "cry", "crying", "hurt", "lonely",
        "lost", "tired", "hopeless", "bad", "unhappy", "miss", "failed",
        "depression", "miserable", "empty", "broken", "worthless", "grief",
        "devastated", "disappointed", "low",
    },
    "angry": {
        "angry", "mad", "furious", "annoyed", "irritated", "hate", "unfair",
        "rage", "frustrated", "terrible", "stupid", "blame", "wrong",
        "resentful", "outraged", "stressed", "stress",
    },
    "neutral": {
        "okay", "fine", "normal", "regular", "average", "noted", "information",
        "meeting", "schedule", "today", "report", "update",
    },
    "fear": {
        "afraid", "scared", "fear", "worried", "anxious", "panic", "danger",
        "nervous", "unsafe", "threat", "risk", "terrified", "anxiety",
        "stressed", "stress", "overwhelmed",
    },
    "surprise": {
        "surprised", "surprise", "wow", "unexpected", "suddenly", "shocked",
        "unbelievable", "amazed", "astonished", "can't believe", "did not expect",
    },
    "disgust": {
        "disgust", "gross", "nasty", "awful", "sick", "dirty", "horrible",
        "repulsive", "revolting", "yuck", "pathetic",
    },
}

NEGATIONS = {"not", "never", "no", "hardly", "barely", "without"}
INTENSIFIERS = {"very", "really", "extremely", "so", "too", "highly", "deeply", "absolutely"}
CONTRAST_WORDS = {"but", "however", "still", "though", "although", "yet", "nevertheless"}
POSITIVE_EMOTIONS = {"happy", "surprise"}
NEGATIVE_EMOTIONS = {"sad", "angry", "fear", "disgust"}


def lexical_emotion_prediction(text: str, metadata: Optional[Dict[str, Any]] = None) -> ModalityPrediction:
    """Deterministic text fallback used when a Transformer is unavailable.

    This is not meant to replace a trained model. It keeps the demo usable and
    makes reliability/fusion behavior visible without internet/model downloads.
    """

    lowered = text.lower()
    tokens = re.findall(r"[a-z']+", lowered)
    scores = {emotion: 0.05 for emotion in EMOTION_CLASSES}
    metadata = dict(metadata or {"engine": "lexical_fallback"})
    contrast_index = _first_contrast_index(tokens)
    cue_counts = {emotion: 0 for emotion in EMOTION_CLASSES}

    for emotion, keywords in LEXICON.items():
        for keyword in keywords:
            if " " in keyword:
                for match in re.finditer(re.escape(keyword), lowered):
                    count_weight = 1.4 * _contrast_weight(tokens, contrast_index, _token_index_at_char(tokens, lowered, match.start()))
                    scores[emotion] += count_weight
                    cue_counts[emotion] += 1
            else:
                for idx, token in enumerate(tokens):
                    if token != keyword:
                        continue
                    weight = _contrast_weight(tokens, contrast_index, idx)
                    window = tokens[max(0, idx - 3):idx]
                    if any(word in INTENSIFIERS for word in window):
                        weight += 0.4
                    if any(word in NEGATIONS for word in window):
                        weight *= 0.35
                    scores[emotion] += weight
                    cue_counts[emotion] += 1

    punctuation_boost = min(text.count("!") * 0.08, 0.24)
    if punctuation_boost:
        scores["surprise"] += punctuation_boost
        scores["angry"] += punctuation_boost / 2

    if "?" in text and max(scores.values()) < 0.6:
        scores["neutral"] += 0.25

    if max(scores.values()) <= 0.1:
        scores["neutral"] += 0.8

    positive_strength = sum(scores[emotion] for emotion in POSITIVE_EMOTIONS)
    negative_strength = sum(scores[emotion] for emotion in NEGATIVE_EMOTIONS)
    positive_hits = sum(cue_counts[emotion] for emotion in POSITIVE_EMOTIONS)
    negative_hits = sum(cue_counts[emotion] for emotion in NEGATIVE_EMOTIONS)
    mixed_cues = positive_hits > 0 and negative_hits > 0
    contrast_detected = contrast_index is not None
    if mixed_cues:
        scores = _flatten_scores(scores, factor=0.38)

    probabilities = normalize_probabilities(_softmax(scores))
    label = max(probabilities, key=probabilities.get)
    if mixed_cues:
        probabilities = _cap_distribution(probabilities, cap=0.55)
        label = max(probabilities, key=probabilities.get)

    metadata.update(
        {
            "contrast_detected": contrast_detected,
            "contrast_word": tokens[contrast_index] if contrast_index is not None else "",
            "mixed_emotional_cues": mixed_cues,
            "positive_cue_strength": round(positive_strength, 3),
            "negative_cue_strength": round(negative_strength, 3),
            "positive_cue_count": positive_hits,
            "negative_cue_count": negative_hits,
            "cue_counts": cue_counts,
        }
    )
    if mixed_cues:
        metadata["ambiguity_note"] = (
            "Mixed emotional cues detected; confidence was reduced because text alone may be ambiguous."
        )

    return ModalityPrediction(
        label=label,
        probabilities=probabilities,
        confidence=probabilities[label],
        modality="text",
        metadata=metadata,
    )


def _softmax(scores: Dict[str, float]) -> Dict[str, float]:
    peak = max(scores.values())
    exp_scores = {label: math.exp(value - peak) for label, value in scores.items()}
    total = sum(exp_scores.values())
    return {label: value / total for label, value in exp_scores.items()}


def _first_contrast_index(tokens: List[str]) -> Optional[int]:
    for idx, token in enumerate(tokens):
        if token in CONTRAST_WORDS:
            return idx
    return None


def _contrast_weight(tokens: List[str], contrast_index: Optional[int], token_index: int) -> float:
    if contrast_index is None:
        return 1.0
    if token_index > contrast_index:
        return 2.0
    if token_index < contrast_index:
        return 0.55
    return 0.0


def _token_index_at_char(tokens: List[str], lowered_text: str, char_index: int) -> int:
    cursor = 0
    for idx, token in enumerate(tokens):
        found = lowered_text.find(token, cursor)
        if found < 0:
            continue
        if found <= char_index < found + len(token):
            return idx
        cursor = found + len(token)
    return len(tokens) - 1


def _flatten_scores(scores: Dict[str, float], factor: float) -> Dict[str, float]:
    mean_score = sum(scores.values()) / len(scores)
    return {
        label: mean_score + (value - mean_score) * factor
        for label, value in scores.items()
    }


def _cap_distribution(probabilities: Dict[str, float], cap: float) -> Dict[str, float]:
    top = max(probabilities, key=probabilities.get)
    if probabilities[top] <= cap:
        return probabilities

    excess = probabilities[top] - cap
    adjusted = dict(probabilities)
    adjusted[top] = cap
    others = [label for label in adjusted if label != top]
    other_total = sum(adjusted[label] for label in others)
    if other_total <= 0:
        share = excess / len(others)
        for label in others:
            adjusted[label] += share
    else:
        for label in others:
            adjusted[label] += excess * (adjusted[label] / other_total)
    return normalize_probabilities(adjusted)
