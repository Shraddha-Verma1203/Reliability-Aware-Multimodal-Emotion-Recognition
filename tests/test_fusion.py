from src.common import ModalityPrediction
from src.fusion import ReliabilityAwareFusion
from src.reliability import ReliabilityScorer


def prediction(modality, label, confidence):
    scores = {label: confidence, "neutral": 1.0 - confidence}
    return ModalityPrediction.from_scores(scores, modality=modality)


def test_reliability_weighted_fusion_prefers_reliable_modality():
    text = prediction("text", "happy", 0.9)
    audio = prediction("audio", "sad", 0.9)

    fused = ReliabilityAwareFusion().fuse(
        {"text": text, "audio": audio},
        {"text": 0.9, "audio": 0.1},
    )

    assert fused.label == "happy"
    assert fused.weights["text"] > fused.weights["audio"]


def test_missing_modalities_do_not_break_fusion():
    text = prediction("text", "angry", 0.8)
    face = ModalityPrediction.unavailable("face", "No face detected.")

    fused = ReliabilityAwareFusion().fuse(
        {"text": text, "face": face},
        {"text": 0.7, "face": 0.0},
    )

    assert fused.label == "angry"
    assert fused.weights["face"] == 0.0


def test_reliability_penalizes_short_text():
    scorer = ReliabilityScorer()
    strong = prediction("text", "happy", 0.95)

    short_score = scorer.score_text(strong, "ok").score
    useful_score = scorer.score_text(strong, "I am very happy about the project results today").score

    assert useful_score > short_score


def test_audio_reliability_penalizes_clipping():
    scorer = ReliabilityScorer()
    clean = prediction("audio", "happy", 0.9)
    clean.metadata = {"duration_seconds": 3.0, "snr_db": 25.0, "clipping_ratio": 0.0, "rms": 0.08}
    clipped = prediction("audio", "happy", 0.9)
    clipped.metadata = {"duration_seconds": 3.0, "snr_db": 25.0, "clipping_ratio": 0.2, "rms": 0.08}

    assert scorer.score_audio(clean).score > scorer.score_audio(clipped).score


def test_fusion_excludes_low_reliability_noisy_modality():
    text = prediction("text", "happy", 0.85)
    audio = prediction("audio", "angry", 0.9)

    fused = ReliabilityAwareFusion(min_reliability=0.2).fuse(
        {"text": text, "audio": audio},
        {"text": 0.75, "audio": 0.03},
    )

    assert fused.label == "happy"
    assert fused.weights["audio"] == 0.0
