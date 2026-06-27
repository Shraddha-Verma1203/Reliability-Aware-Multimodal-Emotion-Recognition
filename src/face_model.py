"""Video branch: CNN-style facial/visual emotion recognition."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .common import EMOTION_CLASSES, ModalityPrediction, normalize_probabilities, top_label
from .media_utils import is_video_file, sample_video_frames


class FacialEmotionRecognizer:
    """Detect a face/frame, estimate visual quality, and classify emotion.

    The project architecture calls this the video branch. For a practical
    Streamlit prototype, uploaded images are treated as representative video
    frames and processed with CNN-style visual/facial features.
    """

    def __init__(
        self,
        model_name: str = "dima806/facial_emotions_image_detection",
        device: int = -1,
        use_pretrained: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.use_pretrained = use_pretrained
        self._pipeline: Optional[Any] = None
        self._face_cascade: Optional[Any] = None

    @property
    def pipeline(self) -> Any:
        if self._pipeline is None:
            from transformers import pipeline

            self._pipeline = pipeline(
                "image-classification",
                model=self.model_name,
                top_k=None,
                device=self.device,
            )
        return self._pipeline

    @property
    def face_cascade(self) -> Any:
        if self._face_cascade is None:
            import cv2

            cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            self._face_cascade = cv2.CascadeClassifier(str(cascade_path))
        return self._face_cascade

    def predict(self, image_path: str | Path | None = None, image: Any | None = None) -> ModalityPrediction:
        try:
            if image_path is not None and is_video_file(image_path):
                return self.predict_video(image_path)

            cv_image = image if image is not None else self.load_image(image_path)
            if cv_image is None:
                if image_path is not None:
                    return ModalityPrediction.unavailable("video", "OpenCV could not decode the image/frame.")
                return ModalityPrediction.unavailable("video", "No video frame/image was provided.")

            face_crop, face_box, quality = self.extract_primary_face(cv_image)
            if face_crop is None:
                quality["model_name"] = self.model_name
                quality["algorithm"] = "CNN video branch"
                quality["feature_source"] = "facial/visual frame quality"
                quality["engine"] = "no_face_detected"
                quality["backend_status"] = "no face detected"
                return ModalityPrediction.unavailable("video", "No face was detected in the visual input.")

            pil_image = self._cv_bgr_to_pil(face_crop)
            quality["model_name"] = self.model_name
            quality["algorithm"] = "CNN video branch"
            quality["feature_source"] = "facial/visual frame features"
            quality["face_box"] = face_box
            if not self.use_pretrained:
                quality["engine"] = "quality_fallback"
                quality["backend_status"] = "fallback model"
                quality["fallback_reason"] = "Fast demo mode is using face/image quality features."
                return quality_based_face_prediction(quality, face_detected=True)
            try:
                raw_output = self.pipeline(pil_image)
                scores = _scores_from_pipeline_output(raw_output)
                quality["engine"] = "image_transformer"
                quality["backend_status"] = "real model"
            except Exception as exc:
                quality["engine"] = "quality_fallback"
                quality["backend_status"] = "fallback model"
                quality["fallback_reason"] = f"Face classifier unavailable: {exc}"
                return quality_based_face_prediction(quality, face_detected=True)
            return ModalityPrediction.from_scores(scores, modality="video", metadata=quality)
        except Exception as exc:
            if image_path is not None:
                if is_video_file(image_path):
                    return ModalityPrediction.unavailable("video", f"Video frame emotion model failed: {exc}")
                return pil_face_fallback_prediction(image_path, str(exc))
            return ModalityPrediction.unavailable("video", f"Video model failed: {exc}")

    @staticmethod
    def load_image(image_path: str | Path | None) -> Any:
        if image_path is None:
            return None
        import cv2

        path = Path(image_path)
        if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
            return None
        return cv2.imread(str(path))

    def predict_video(self, video_path: str | Path, max_frames: int = 5) -> ModalityPrediction:
        """Run facial emotion inference on sampled video frames and aggregate scores."""

        frames = sample_video_frames(video_path, max_frames=max_frames)
        if not frames:
            return ModalityPrediction.unavailable("video", "Could not extract frames from video.")

        aggregated_scores = {emotion: 0.0 for emotion in EMOTION_CLASSES}
        frame_predictions: List[Dict[str, Any]] = []
        detected_faces = 0
        sampled_frames = len(frames)

        for frame_index, frame in enumerate(frames):
            face_crop, face_box, quality = self.extract_primary_face(frame)
            if face_crop is None:
                frame_predictions.append(
                    {"frame_index": frame_index, "face_detected": False, "label": "no_face", "confidence": 0.0}
                )
                continue

            detected_faces += 1
            if not self.use_pretrained:
                frame_prediction = quality_based_face_prediction(
                    {
                        **quality,
                        "engine": "quality_fallback",
                        "backend_status": "fallback model",
                        "fallback_reason": "Fast demo mode is using frame quality features.",
                    },
                    face_detected=True,
                )
            else:
                try:
                    pil_image = self._cv_bgr_to_pil(face_crop)
                    raw_output = self.pipeline(pil_image)
                    scores = _scores_from_pipeline_output(raw_output)
                    frame_prediction = ModalityPrediction.from_scores(
                        scores,
                        modality="video",
                        metadata={"engine": "image_transformer", "backend_status": "real model"},
                    )
                except Exception as exc:
                    frame_prediction = quality_based_face_prediction(
                        {
                            **quality,
                            "engine": "quality_fallback",
                            "backend_status": "fallback model",
                            "fallback_reason": f"Frame classifier unavailable: {exc}",
                        },
                        face_detected=True,
                    )

            for emotion, probability in frame_prediction.probabilities.items():
                aggregated_scores[emotion] += probability
            frame_predictions.append(
                {
                    "frame_index": frame_index,
                    "face_detected": True,
                    "face_box": face_box,
                    "label": frame_prediction.label,
                    "confidence": frame_prediction.confidence,
                    "engine": frame_prediction.metadata.get("engine", "model"),
                }
            )

        if detected_faces == 0:
            return ModalityPrediction.unavailable("video", "No faces were detected in sampled video frames.")

        probabilities = normalize_probabilities(
            {emotion: score / detected_faces for emotion, score in aggregated_scores.items()}
        )
        label = top_label(probabilities)
        engines = {item.get("engine", "") for item in frame_predictions if item.get("face_detected")}
        fallback_used = any("fallback" in engine for engine in engines)
        metadata = {
            "model_name": self.model_name,
            "algorithm": "CNN video branch",
            "feature_source": "sampled video frames",
            "engine": "video_frame_aggregation",
            "backend_status": "fallback model" if fallback_used else "real model",
            "sampled_frames": sampled_frames,
            "detected_faces": detected_faces,
            "frame_predictions": frame_predictions,
            "fallback_used": fallback_used,
        }
        return ModalityPrediction(
            label=label,
            probabilities=probabilities,
            confidence=probabilities[label],
            modality="video",
            metadata=metadata,
        )

    def extract_primary_face(self, image: Any) -> Tuple[Any | None, Tuple[int, int, int, int] | None, Dict[str, float]]:
        import cv2

        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())

        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        quality: Dict[str, float] = {
            "image_width": float(width),
            "image_height": float(height),
            "blur_variance": blur_variance,
            "brightness": brightness,
            "face_detected": 0.0,
            "face_area_ratio": 0.0,
        }

        if len(faces) == 0:
            return None, None, quality

        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        padding = int(0.15 * max(w, h))
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(width, x + w + padding)
        y1 = min(height, y + h + padding)
        crop = image[y0:y1, x0:x1]

        quality["face_detected"] = 1.0
        quality["face_area_ratio"] = float((w * h) / max(width * height, 1))
        return crop, (int(x), int(y), int(w), int(h)), quality

    @staticmethod
    def _cv_bgr_to_pil(image: Any) -> Any:
        import cv2
        from PIL import Image

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)


def _scores_from_pipeline_output(raw_output: Any) -> Dict[str, float]:
    rows = raw_output[0] if isinstance(raw_output, list) and raw_output and isinstance(raw_output[0], list) else raw_output
    scores: Dict[str, float] = {}
    for row in rows if isinstance(rows, list) else [rows]:
        if isinstance(row, dict):
            scores[str(row.get("label", ""))] = float(row.get("score", 0.0))
    return scores


def pil_face_fallback_prediction(image_path: str | Path, reason: str) -> ModalityPrediction:
    """Fallback that extracts basic image quality using Pillow only."""

    from PIL import Image, ImageStat

    try:
        image = Image.open(image_path).convert("L")
        stat = ImageStat.Stat(image)
        brightness = float(stat.mean[0])
        extrema = image.getextrema()
        contrast = float(extrema[1] - extrema[0])
        quality = {
            "image_width": float(image.width),
            "image_height": float(image.height),
            "blur_variance": contrast,
            "brightness": brightness,
            "face_detected": 0.25,
            "face_area_ratio": 0.08,
            "algorithm": "CNN video branch",
            "feature_source": "facial/visual frame quality",
            "engine": "pil_quality_fallback",
            "fallback_reason": reason,
        }
        return quality_based_face_prediction(quality, face_detected=False)
    except Exception as exc:
        return ModalityPrediction.unavailable("video", f"Image fallback failed: {exc}")


def quality_based_face_prediction(metadata: Dict[str, Any], face_detected: bool) -> ModalityPrediction:
    """Return a clearly labeled fallback prediction from face/image quality."""

    brightness = float(metadata.get("brightness", 128.0) or 128.0)
    blur = float(metadata.get("blur_variance", 0.0) or 0.0)

    if not face_detected:
        scores = {"neutral": 0.42, "sad": 0.16, "happy": 0.14, "fear": 0.10, "angry": 0.09, "surprise": 0.09}
    elif blur < 45:
        scores = {"neutral": 0.36, "sad": 0.18, "happy": 0.16, "fear": 0.12, "angry": 0.10, "surprise": 0.08}
    elif brightness < 55:
        scores = {"sad": 0.30, "neutral": 0.27, "fear": 0.17, "angry": 0.10, "happy": 0.09, "surprise": 0.07}
    elif brightness > 200:
        scores = {"surprise": 0.28, "neutral": 0.24, "happy": 0.20, "fear": 0.12, "angry": 0.09, "sad": 0.07}
    else:
        scores = {"neutral": 0.30, "happy": 0.24, "sad": 0.14, "surprise": 0.12, "angry": 0.09, "fear": 0.07, "disgust": 0.04}

    metadata["note"] = "Quality-based face fallback; not a trained facial expression model."
    metadata["algorithm"] = "CNN video branch"
    metadata["feature_source"] = metadata.get("feature_source", "facial/visual frame quality")
    return ModalityPrediction.from_scores(scores, modality="video", metadata=metadata)
