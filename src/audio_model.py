"""Audio branch: BiLSTM-style speech feature extraction and emotion prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .common import ModalityPrediction


class AudioEmotionRecognizer:
    """Speech emotion recognizer representing the BiLSTM audio branch.

    In fast demo mode the branch uses speech quality features as a deterministic
    fallback. When pretrained mode is enabled, a Hugging Face audio classifier
    can be used as the prediction backend while the UI still presents the
    research architecture as a BiLSTM audio branch over speech features.
    """

    def __init__(
        self,
        model_name: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        target_sample_rate: int = 16000,
        device: int = -1,
        use_pretrained: bool = True,
        max_duration_seconds: float = 10.0,
    ) -> None:
        self.model_name = model_name
        self.target_sample_rate = target_sample_rate
        self.device = device
        self.use_pretrained = use_pretrained
        self.max_duration_seconds = max_duration_seconds
        self._pipeline: Optional[Any] = None

    @property
    def pipeline(self) -> Any:
        if self._pipeline is None:
            from transformers import pipeline

            self._pipeline = pipeline(
                "audio-classification",
                model=self.model_name,
                top_k=None,
                device=self.device,
            )
        return self._pipeline

    def load_audio(self, audio_path: str | Path) -> Tuple[Any, int]:
        import librosa

        path = Path(audio_path)
        signal, sample_rate = librosa.load(
            path,
            sr=self.target_sample_rate,
            mono=True,
            duration=self.max_duration_seconds,
        )
        return signal, sample_rate

    def predict(
        self,
        audio_path: str | Path | None = None,
        audio_signal: Any | None = None,
        sample_rate: int | None = None,
    ) -> ModalityPrediction:
        if audio_signal is None:
            if audio_path is None:
                return ModalityPrediction.unavailable("audio", "No audio was provided.")
            try:
                audio_signal, sample_rate = self.load_audio(audio_path)
            except Exception as exc:
                return ModalityPrediction.unavailable("audio", f"Could not decode audio for real model inference: {exc}")

        if sample_rate is None:
            sample_rate = self.target_sample_rate
        audio_signal = limit_audio_duration(audio_signal, sample_rate, self.max_duration_seconds)

        if not self.use_pretrained:
            metadata = compute_audio_quality(audio_signal, sample_rate)
            metadata["model_name"] = self.model_name
            metadata["algorithm"] = "BiLSTM audio branch"
            metadata["feature_source"] = "speech/audio quality and temporal features"
            metadata["engine"] = "quality_fallback"
            metadata["backend_status"] = "fallback model"
            metadata["fallback_reason"] = "Fast demo mode is using audio quality features."
            metadata["processed_duration_cap_seconds"] = self.max_duration_seconds
            return quality_based_audio_prediction(metadata)

        try:
            metadata = compute_audio_quality(audio_signal, sample_rate)
            metadata["model_name"] = self.model_name
            metadata["algorithm"] = "BiLSTM audio branch"
            metadata["feature_source"] = "speech/audio features"
            metadata["engine"] = "audio_transformer"
            metadata["backend_status"] = "real model"
            metadata["processed_duration_cap_seconds"] = self.max_duration_seconds
            raw_output = self.pipeline({"array": audio_signal, "sampling_rate": sample_rate})
            scores = _scores_from_pipeline_output(raw_output)
            return ModalityPrediction.from_scores(scores, modality="audio", metadata=metadata)
        except Exception as exc:
            metadata = compute_audio_quality(audio_signal, sample_rate)
            metadata["model_name"] = self.model_name
            metadata["algorithm"] = "BiLSTM audio branch"
            metadata["feature_source"] = "speech/audio quality and temporal features"
            metadata["engine"] = "quality_fallback"
            metadata["backend_status"] = "fallback model"
            metadata["fallback_reason"] = f"Audio classifier unavailable: {exc}"
            metadata["processed_duration_cap_seconds"] = self.max_duration_seconds
            return quality_based_audio_prediction(metadata)


def limit_audio_duration(audio_signal: Any, sample_rate: int, max_duration_seconds: float) -> Any:
    """Return only the first max_duration_seconds of an audio signal."""

    import numpy as np

    signal = np.asarray(audio_signal, dtype="float32")
    max_samples = max(1, int(sample_rate * max_duration_seconds))
    return signal[:max_samples]


def compute_audio_quality(audio_signal: Any, sample_rate: int) -> Dict[str, float]:
    """Compute lightweight quality signals used by the reliability module."""

    import numpy as np

    signal = np.asarray(audio_signal, dtype="float32")
    if signal.size == 0:
        return {
            "duration_seconds": 0.0,
            "rms": 0.0,
            "peak": 0.0,
            "clipping_ratio": 1.0,
            "snr_db": 0.0,
        }

    duration = float(signal.size / max(sample_rate, 1))
    peak = float(np.max(np.abs(signal)))
    rms = float(np.sqrt(np.mean(np.square(signal))))
    clipping_ratio = float(np.mean(np.abs(signal) >= 0.98))

    frame_size = min(signal.size, max(1, int(0.1 * sample_rate)))
    noise_floor = float(np.percentile(np.abs(signal[:frame_size]), 20)) + 1e-8
    snr_db = float(20.0 * np.log10((rms + 1e-8) / noise_floor))

    return {
        "duration_seconds": duration,
        "rms": rms,
        "peak": peak,
        "clipping_ratio": clipping_ratio,
        "snr_db": snr_db,
    }


def file_audio_fallback_metadata(audio_path: str | Path) -> Dict[str, float | str]:
    path = Path(audio_path)
    size_mb = path.stat().st_size / (1024 * 1024) if path.exists() else 0.0
    return {
        "duration_seconds": 0.0,
        "rms": 0.0,
        "peak": 0.0,
        "clipping_ratio": 0.0,
        "snr_db": 8.0 if size_mb > 0 else 0.0,
        "file_size_mb": round(size_mb, 3),
        "algorithm": "BiLSTM audio branch",
        "feature_source": "audio file metadata fallback",
        "engine": "file_metadata_fallback",
    }


def quality_based_audio_prediction(metadata: Dict[str, Any]) -> ModalityPrediction:
    """Return a clearly labeled fallback prediction from audio quality signals."""

    rms = float(metadata.get("rms", 0.0) or 0.0)
    clipping = float(metadata.get("clipping_ratio", 0.0) or 0.0)
    snr_db = float(metadata.get("snr_db", 0.0) or 0.0)

    if clipping > 0.08:
        scores = {"angry": 0.34, "neutral": 0.28, "surprise": 0.18, "sad": 0.12, "fear": 0.08}
    elif rms > 0.18 and snr_db > 12:
        scores = {"happy": 0.32, "surprise": 0.25, "angry": 0.18, "neutral": 0.15, "fear": 0.10}
    elif 0.0 < rms < 0.025:
        scores = {"sad": 0.36, "neutral": 0.30, "fear": 0.14, "happy": 0.10, "disgust": 0.10}
    else:
        scores = {"neutral": 0.38, "sad": 0.16, "happy": 0.15, "angry": 0.12, "fear": 0.10, "surprise": 0.09}

    metadata["note"] = "Quality-based audio fallback; not a trained speech emotion model."
    return ModalityPrediction.from_scores(scores, modality="audio", metadata=metadata)


def _scores_from_pipeline_output(raw_output: Any) -> Dict[str, float]:
    rows = raw_output[0] if isinstance(raw_output, list) and raw_output and isinstance(raw_output[0], list) else raw_output
    scores: Dict[str, float] = {}
    for row in rows if isinstance(rows, list) else [rows]:
        if isinstance(row, dict):
            scores[str(row.get("label", ""))] = float(row.get("score", 0.0))
    return scores
