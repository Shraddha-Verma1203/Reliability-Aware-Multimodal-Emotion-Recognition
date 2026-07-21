# 🧠 Reliability-Aware Multimodal Emotion Recognition

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-red?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)


> **Research prototype developed during my Summer Research Internship at NIT Goa, implementing Reliability-Aware Multimodal Emotion Recognition using text, audio and facial expression fusion.**

[![Live Demo](https://img.shields.io/badge/🚀-Live%20Demo-success?style=for-the-badge)](https://reliability-aware-mer.streamlit.app)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github)](https://github.com/Shraddha-Verma1203/Reliability-Aware-Multimodal-Emotion-Recognition)

---

## 🌐 Live Demo

Click the **Live Demo** button above to explore the application.

---

Mentor-ready research prototype for NIT Goa.

## Project Goal

This project upgrades a basic emotion prediction demo into a
**Reliability-Aware Multimodal Emotion Recognition (MER)** system. It accepts
text, audio, and video/frame input, predicts emotion for each available
modality, estimates how reliable each modality is, detects cross-modal conflict,
and produces an explainable final prediction through adaptive weighted fusion.

Target emotion classes:

`happy`, `sad`, `angry`, `neutral`, `fear`, `surprise`, `disgust`

## Research Problem Statement

Existing Problem:

- Simple averaging ignores unreliable modalities.
- Missing modalities reduce system usability.
- Different modalities may contradict each other.
- Most MER systems lack explanation.

Proposed Solution:

- Reliability-aware fusion.
- Dynamic modality weighting.
- Missing modality robustness.
- Conflict detection.
- Explainable prediction output.

## System Architecture

```text
Input
  |
  +--> Text Encoder
  +--> Audio Encoder
  +--> Video Encoder
          |
          v
Reliability Estimator
          |
          v
Dynamic Fusion Engine
          |
          v
Conflict Detector
          |
          v
Final Emotion Prediction
          |
          v
Explainability Output
```

## Proposed Research Architecture

```text
MELD Dataset / User Input
        |
        v
Text Branch       Audio Branch        Video Branch
BERT              BiLSTM              CNN
text features     speech features     facial/visual features
        \             |              /
         \            |             /
          v           v            v
              Fusion Layer
                  |
                  v
        Reliability Scoring Module
        - text reliability
        - audio reliability
        - video reliability
                  |
                  v
        Adaptive Weighted Fusion
                  |
                  v
        Final Emotion Prediction
```

## Current Working Prototype Pipeline

- **Prototype Mode** uses fallback/demo logic.
- **Pretrained Model Mode** uses Hugging Face models where available.
- Backend status indicators show whether each modality used a real model or fallback logic.
- Uploaded MP4/AVI/MOV videos are processed as videos: frames are sampled for the
  visual branch, and the audio track is extracted for the audio branch when no
  separate audio upload is provided.
- BiLSTM and custom CNN are proposed research components; current implementation
  uses pretrained/fallback models unless custom trained checkpoints are added.

## Implemented Research Features

1. **Reliability-Aware Fusion**
   Each modality has a predicted emotion, confidence score, reliability score,
   and final contribution weight. Fusion uses:

   ```text
   final_score = sum(modality_score x reliability_weight)
   ```

2. **Missing Modality Handling**
   The app works with text only, audio only, video only, any pair, or all three.
   Missing modalities receive fusion weight `0`, and available modalities are
   re-weighted.

3. **Explainability Panel**
   The app explains text, audio, video, and fusion decisions in plain language.

4. **Emotion Conflict Detector**
   If modalities disagree, the app shows a warning describing the conflict and
   possible emotional masking or modality disagreement.

5. **Live Input Support**
   The UI supports file upload, webcam frame capture through `st.camera_input`,
   and microphone recording through Streamlit audio input when available.

6. **Reliability Visualization**
   The UI shows progress bars for reliability, contribution weight, and
   confidence for text/audio/video.

7. **Research Metrics Page**
   The app includes demo/example metrics: accuracy, precision, recall, F1-score,
   and confusion matrix. Real MELD/project metrics can be added through a
   manifest later.

8. **Model Diagnostics / Test Page**
   After prediction, the app shows sampled video frames, detected face count,
   extracted audio waveform, and model confidence scores.

## Algorithms Used

- **Text: BERT / Transformer text classifier**
  Pretrained mode uses a Hugging Face Transformer classifier. Prototype mode uses
  a local lexical fallback for offline demos.
- **Audio: Wav2Vec2 audio emotion classifier OR BiLSTM placeholder**
  Current implementation can use a Hugging Face audio classifier over uploaded
  audio or audio extracted from video; custom BiLSTM checkpoints are not yet included.
- **Video: CNN-based facial emotion classifier OR image-classification model**
  Current implementation detects faces in uploaded images or sampled video
  frames, runs a pretrained image-classification model when available, and
  aggregates frame-level video predictions.
- **Reliability Scoring**
  Confidence and quality checking for every modality.
- **Fusion: Reliability-aware adaptive weighted fusion**
  Final prediction uses reliability-weighted modality scores.

## How to Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

The app runs at:

```text
http://localhost:8501
```

## Testing

```powershell
.\.venv\Scripts\python.exe -m pytest tests
.\.venv\Scripts\python.exe -m compileall src app tests app.py
```

## Sample Inputs

- Happy: `I am really excited and proud of the progress we made today!`
- Sad: `I feel lonely and disappointed because the result did not work.`
- Mixed/contrastive: `I am happy but feeling depressed`
- Angry: `This is unfair and I am extremely frustrated about what happened.`
- Fear: `I am nervous and worried that something unsafe might happen.`
- Surprise: `Wow, I did not expect this result at all!`
- Disgust: `That was gross, awful, and completely unacceptable.`
- Neutral: `The meeting is scheduled for tomorrow and the report is ready.`

## Video and Audio Upload Behavior

- Image uploads are processed as single facial/visual frames.
- Video uploads are not opened as images. The app samples frames with OpenCV,
  detects faces, predicts emotion per detected face frame, and aggregates the
  frame probabilities.
- If a video contains audio and no separate audio file is uploaded, the app
  extracts a temporary WAV file from the video and sends it to the audio emotion
  model.
- If a model fails or a face/audio stream cannot be decoded, the output table
  marks the branch as unavailable or fallback instead of silently pretending it
  was a real model result.

## Evaluation Data Format

For MELD-style evaluation, create a CSV manifest:

```csv
label,text,audio_path,video_path
happy,"I am excited about this result",samples/happy.wav,samples/happy_frame.jpg
sad,"I feel low today",samples/sad.wav,samples/sad_frame.jpg
```

Run:

```powershell
python -m src.evaluate --manifest data/eval_manifest.csv --output results/evaluation.json
```

## Repository Structure

```text
app.py
app/
  streamlit_app.py
src/
  common.py
  text_model.py
  audio_model.py
  face_model.py
  reliability.py
  fusion.py
  conflict_detector.py
  explainability.py
  evaluation.py
  evaluate.py
tests/
  test_imports.py
  test_fusion.py
  test_research_modules.py
requirements.txt
report.md
```
