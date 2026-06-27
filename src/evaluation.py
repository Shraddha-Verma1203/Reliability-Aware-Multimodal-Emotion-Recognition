"""Demo evaluation metrics for the Streamlit research prototype."""

from __future__ import annotations

from typing import Dict, List

from .common import EMOTION_CLASSES


DEMO_TRUE = ["happy", "sad", "angry", "neutral", "fear", "surprise", "disgust", "sad"]
DEMO_PRED = ["happy", "sad", "angry", "neutral", "fear", "happy", "disgust", "neutral"]


def demo_metrics() -> Dict[str, object]:
    """Return clearly labeled demo/example metrics.

    These are not research results. They exist so the UI has a clean metrics
    section until a MELD or custom evaluation manifest is added.
    """

    return compute_classification_metrics(DEMO_TRUE, DEMO_PRED)


def compute_classification_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, object]:
    labels = EMOTION_CLASSES
    matrix = [[0 for _ in labels] for _ in labels]
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    for truth, pred in zip(y_true, y_pred):
        if truth in label_to_idx and pred in label_to_idx:
            matrix[label_to_idx[truth]][label_to_idx[pred]] += 1

    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[idx][idx] for idx in range(len(labels)))
    accuracy = correct / total if total else 0.0

    precisions = []
    recalls = []
    f1s = []
    per_class = {}
    for idx, label in enumerate(labels):
        tp = matrix[idx][idx]
        fp = sum(matrix[row][idx] for row in range(len(labels)) if row != idx)
        fn = sum(matrix[idx][col] for col in range(len(labels)) if col != idx)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        per_class[label] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }

    return {
        "label": "demo/example metrics",
        "accuracy": round(accuracy, 3),
        "precision": round(sum(precisions) / len(labels), 3),
        "recall": round(sum(recalls) / len(labels), 3),
        "f1_score": round(sum(f1s) / len(labels), 3),
        "labels": labels,
        "confusion_matrix": matrix,
        "per_class": per_class,
    }
