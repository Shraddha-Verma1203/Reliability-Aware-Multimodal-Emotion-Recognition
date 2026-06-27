# Reliability-Aware Multimodal Emotion Recognition

## Abstract

Emotion recognition is commonly studied using text, speech, or facial
expressions. Real-world communication is multimodal, but the quality of each
modality can vary substantially. Audio may be noisy, a face may be blurred or
partially missing, and text may be too short to be emotionally informative.
This project proposes a reliability-aware multimodal emotion recognition system
that dynamically estimates the trustworthiness of each modality before fusion.

## Problem Statement

Many multimodal emotion recognition systems combine text, audio, and facial
features using fixed or learned fusion strategies. A limitation of equal or
static fusion is that it can over-trust weak inputs. For example, a noisy audio
clip should contribute less than a clear text statement, and a blurred face
should not dominate the final decision.

The objective of this project is to build a complete MER pipeline for seven
emotion classes:

`happy`, `sad`, `angry`, `neutral`, `fear`, `surprise`, `disgust`.

## Proposed Contribution

The core contribution is a reliability scoring module. Instead of treating all
modalities equally, the system estimates a reliability score for each available
modality and uses those scores to compute fusion weights.

Reliability is based on two categories of evidence:

1. Prediction certainty, measured using confidence and probability entropy.
2. Input quality, measured with modality-specific signals.

For the live prototype, the application includes a fast demo mode. In this mode,
text is handled by a deterministic lexical emotion recognizer, while optional
audio and face uploads can use transparent quality-based fallback predictions.
This keeps the system runnable during mentor review while preserving the same
reliability-aware fusion mechanism. Pretrained models can be enabled when the
full ML environment and model weights are available.

## Methodology

### Text Emotion Recognition

The text module uses a pretrained Transformer text classifier. Model outputs are
mapped into the seven project emotion classes. Text reliability combines model
confidence with simple text quality signals such as word count.

### Audio Emotion Recognition

The audio module loads speech using Librosa and performs inference with a
pretrained speech emotion recognition model. Audio reliability uses confidence,
duration, estimated SNR, clipping ratio, and signal energy.

### Facial Emotion Recognition

The face module uses OpenCV Haar cascade detection to locate the primary face in
an image. A pretrained image emotion classifier predicts the facial emotion.
Face reliability uses confidence, whether a face was detected, blur variance,
brightness, and face size.

### Reliability-Aware Fusion

Each modality produces a probability distribution over the seven emotion
classes. The reliability module assigns a score in the range `[0, 1]`. The
fusion module normalizes these scores into modality weights and computes a
weighted average of emotion probabilities.

This allows the system to reduce the influence of weak signals:

- Noisy audio receives a lower fusion weight.
- Blurry or face-missing images receive a lower fusion weight.
- High-confidence text receives a higher fusion weight.

## Evaluation Plan

The evaluation pipeline reads a CSV manifest containing labels and optional
text, audio, and image paths. It reports:

- Accuracy
- Macro-F1 score
- Confusion matrix
- Per-sample predictions
- Per-modality reliability scores
- Final fusion weights

Macro-F1 is important because emotion datasets are often imbalanced.

## Practical Datasets

The project is designed for a 6-week undergraduate internship, so it uses
pretrained models and practical public datasets:

- Text: GoEmotions or a curated seven-class utterance set.
- Audio: RAVDESS, CREMA-D, TESS, or SAVEE.
- Face: FER2013, RAF-DB, AffectNet subset, or curated image samples.

The repository documents dataset format instead of committing dataset files,
because these datasets are large and may have redistribution restrictions.

## Expected Outcomes

The expected outcome is a runnable MER system that demonstrates how reliability
can improve robustness when one or more modalities are degraded. The Streamlit
demo helps visualize predictions, reliability scores, and fusion weights for
mentor review.

## Limitations

- Pretrained models may use label sets that do not perfectly match the seven
  target emotions.
- Reliability scoring is heuristic in this version and should be validated with
  controlled noise, blur, and missing-modality experiments.
- Audio and facial emotion recognition can be sensitive to dataset domain shift.

## Future Work

- Train a small learned reliability network using degradation-augmented data.
- Add video support with temporal face aggregation.
- Calibrate modality probabilities using validation data.
- Compare equal fusion, confidence-only fusion, and reliability-aware fusion.
- Add ablation studies for each reliability component.
