"""CLI evaluation pipeline for reliability-aware MER."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from .audio_model import AudioEmotionRecognizer
from .common import EMOTION_CLASSES, FusedPrediction, ModalityPrediction, normalize_label
from .face_model import FacialEmotionRecognizer
from .fusion import ReliabilityAwareFusion
from .reliability import ReliabilityScorer
from .text_model import TextEmotionRecognizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate reliability-aware multimodal emotion recognition.")
    parser.add_argument("--manifest", required=True, help="CSV with label and optional text/audio_path/video_path columns.")
    parser.add_argument("--output", default="results/evaluation.json", help="JSON metrics output path.")
    parser.add_argument("--predictions-output", default="results/predictions.csv", help="Per-sample CSV output path.")
    parser.add_argument("--text-model", default="bhadresh-savani/bert-base-uncased-emotion")
    parser.add_argument("--audio-model", default="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
    parser.add_argument("--video-model", default="dima806/facial_emotions_image_detection")
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> Dict[str, object]:
    import pandas as pd
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    manifest_path = Path(args.manifest)
    manifest = pd.read_csv(manifest_path)
    if "label" not in manifest.columns:
        raise ValueError("Manifest must contain a 'label' column.")

    text_model = TextEmotionRecognizer(args.text_model)
    audio_model = AudioEmotionRecognizer(args.audio_model)
    video_model = FacialEmotionRecognizer(args.video_model)
    scorer = ReliabilityScorer()
    fusion = ReliabilityAwareFusion()

    rows: List[Dict[str, object]] = []
    y_true: List[str] = []
    y_pred: List[str] = []

    for index, row in manifest.iterrows():
        true_label = normalize_label(row["label"])
        predictions: Dict[str, ModalityPrediction] = {}
        reliabilities = {}

        text = _optional_string(row, "text")
        if text:
            predictions["text"] = text_model.predict(text)
            reliabilities["text"] = scorer.score_text(predictions["text"], text)

        audio_path = _resolve_path(manifest_path.parent, _optional_string(row, "audio_path"))
        if audio_path:
            predictions["audio"] = audio_model.predict(audio_path=audio_path)
            reliabilities["audio"] = scorer.score_audio(predictions["audio"])

        video_path_value = _optional_string(row, "video_path") or _optional_string(row, "image_path")
        video_path = _resolve_path(manifest_path.parent, video_path_value)
        if video_path:
            predictions["video"] = video_model.predict(image_path=video_path)
            reliabilities["video"] = scorer.score(predictions["video"])

        fused = fusion.fuse(predictions, reliabilities)
        y_true.append(true_label)
        y_pred.append(fused.label)
        rows.append(_prediction_row(index, true_label, fused))

    output = {
        "accuracy": accuracy_score(y_true, y_pred) if y_true else 0.0,
        "macro_f1": f1_score(y_true, y_pred, labels=EMOTION_CLASSES, average="macro", zero_division=0) if y_true else 0.0,
        "labels": EMOTION_CLASSES,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=EMOTION_CLASSES).tolist() if y_true else [],
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=EMOTION_CLASSES,
            output_dict=True,
            zero_division=0,
        )
        if y_true
        else {},
        "num_samples": len(y_true),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    predictions_path = Path(args.predictions_output)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(predictions_path, index=False)
    return output


def _optional_string(row: object, key: str) -> str:
    value = row.get(key, "") if hasattr(row, "get") else ""
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none"} else text


def _resolve_path(base_dir: Path, value: str) -> str:
    if not value:
        return ""
    path = Path(value)
    return str(path if path.is_absolute() else base_dir / path)


def _prediction_row(index: int, true_label: str, fused: FusedPrediction) -> Dict[str, object]:
    row: Dict[str, object] = {
        "index": index,
        "true_label": true_label,
        "predicted_label": fused.label,
        "confidence": fused.confidence,
    }
    for emotion, probability in fused.probabilities.items():
        row[f"prob_{emotion}"] = probability
    for modality in ("text", "audio", "video"):
        row[f"weight_{modality}"] = fused.weights.get(modality, 0.0)
        row[f"reliability_{modality}"] = fused.reliability_scores.get(modality, 0.0)
        prediction = fused.modality_predictions.get(modality)
        row[f"{modality}_label"] = prediction.label if prediction else ""
    return row


def main() -> None:
    args = parse_args()
    metrics = evaluate(args)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
