from src.common import ModalityPrediction
from src.conflict_detector import detect_emotion_conflict
from src.evaluation import demo_metrics
from src.explainability import build_explainability_panel
from src.reliability import ReliabilityResult


def pred(modality, label, confidence=0.8):
    return ModalityPrediction.from_scores({label: confidence, "neutral": 1 - confidence}, modality)


def test_conflict_detector_flags_disagreement():
    result = detect_emotion_conflict(
        {
            "text": pred("text", "happy"),
            "audio": pred("audio", "sad"),
            "video": pred("video", "sad"),
        }
    )

    assert result.conflict_detected is True
    assert result.dominant_emotion == "sad"


def test_demo_metrics_are_available():
    metrics = demo_metrics()

    assert metrics["label"] == "demo/example metrics"
    assert "confusion_matrix" in metrics
    assert 0 <= metrics["accuracy"] <= 1


def test_explainability_panel_includes_missing_modality_reason():
    predictions = {"text": pred("text", "happy")}
    reliabilities = {"text": ReliabilityResult("text", 0.7)}
    from src.fusion import ReliabilityAwareFusion

    fused = ReliabilityAwareFusion().fuse(predictions, reliabilities)
    conflict = detect_emotion_conflict(predictions)
    panel = build_explainability_panel(predictions, reliabilities, fused, conflict)

    assert any("Audio not provided" in item for item in panel["modality_reasons"])


def test_video_files_are_not_loaded_as_images():
    from src.face_model import FacialEmotionRecognizer
    from src.media_utils import is_video_file

    recognizer = FacialEmotionRecognizer(use_pretrained=False)

    assert is_video_file("sample.mp4") is True
    assert recognizer.load_image("sample.mp4") is None


def test_audio_recognizer_limits_audio_to_first_ten_seconds():
    import numpy as np

    from src.audio_model import AudioEmotionRecognizer

    recognizer = AudioEmotionRecognizer(use_pretrained=False, target_sample_rate=16000)
    signal = np.ones(16000 * 15, dtype="float32")
    prediction = recognizer.predict(audio_signal=signal, sample_rate=16000)

    assert prediction.metadata["duration_seconds"] == 10.0
    assert prediction.metadata["processed_duration_cap_seconds"] == 10.0
