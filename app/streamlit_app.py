"""Professional Streamlit demo for reliability-aware multimodal emotion recognition."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_model import AudioEmotionRecognizer
from src.common import EMOTION_CLASSES, FusedPrediction, ModalityPrediction
from src.conflict_detector import ConflictResult, detect_emotion_conflict
from src.evaluation import demo_metrics
from src.explainability import build_explainability_panel
from src.face_model import FacialEmotionRecognizer
from src.fusion import ReliabilityAwareFusion
from src.media_utils import audio_waveform, extract_audio_from_video, is_video_file, sample_video_frames, save_preview_frames
from src.reliability import ReliabilityResult, ReliabilityScorer
from src.text_model import TextEmotionRecognizer


SAMPLE_TEXTS = {
    "Happy": "I am really excited and proud of the progress we made today!",
    "Sad": "I feel lonely and disappointed because the result did not work.",
    "Angry": "This is unfair and I am extremely frustrated about what happened.",
    "Fear": "I am nervous and worried that something unsafe might happen.",
    "Surprise": "Wow, I did not expect this result at all!",
    "Disgust": "That was gross, awful, and completely unacceptable.",
    "Neutral": "The meeting is scheduled for tomorrow and the report is ready.",
}

TEXT_TIMEOUT_SECONDS = 15
AUDIO_TIMEOUT_SECONDS = 20
VIDEO_TIMEOUT_SECONDS = 20
MAX_AUDIO_SECONDS = 10.0
MAX_VIDEO_FRAMES = 5

st.set_page_config(
    page_title="Reliability-Aware MER",
    page_icon=":material/psychology:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_text_model(use_pretrained: bool) -> TextEmotionRecognizer:
    return TextEmotionRecognizer(use_transformer=use_pretrained)


@st.cache_resource
def get_audio_model(use_pretrained: bool) -> AudioEmotionRecognizer:
    return AudioEmotionRecognizer(use_pretrained=use_pretrained)


@st.cache_resource
def get_video_model(use_pretrained: bool) -> FacialEmotionRecognizer:
    return FacialEmotionRecognizer(use_pretrained=use_pretrained)


@st.cache_resource
def get_reliability_scorer() -> ReliabilityScorer:
    return ReliabilityScorer()


@st.cache_resource
def get_fusion() -> ReliabilityAwareFusion:
    return ReliabilityAwareFusion()


@st.cache_resource
def get_prediction_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=6, thread_name_prefix="mer-predict")


def main() -> None:
    _inject_css()
    use_pretrained = _render_sidebar()
    _render_header(use_pretrained)

    text, audio_file, video_file = _render_inputs()

    if st.button("Run Reliability-Aware Prediction", type="primary", use_container_width=True):
        run_prediction(text, audio_file, video_file, use_pretrained)
    else:
        _render_sample_cases()


def _render_sidebar() -> bool:
    with st.sidebar:
        st.title("Demo Controls")
        use_pretrained = st.toggle(
            "Use pretrained Hugging Face models",
            value=True,
            help="On by default for pretrained model mode. Backend status shows when a branch uses a real model or fallback.",
        )
        st.markdown("**Target classes**")
        st.write(", ".join(EMOTION_CLASSES))
        st.markdown("**Reliability signals**")
        st.write("Confidence, entropy, text length, audio quality, video/visual quality, missing modality handling.")
        st.markdown("**Demo guarantee**")
        st.info(
            "The app stays runnable with missing modalities. In pretrained mode, backend status indicators show "
            "which branches used real Hugging Face models and which branches used transparent fallback logic."
        )
    return use_pretrained


def _render_header(use_pretrained: bool) -> None:
    st.markdown(
        """
        <section class="hero">
          <div>
            <p class="eyebrow">NIT Goa Research Internship Prototype</p>
            <h1>Reliability-Aware Multimodal Emotion Recognition</h1>
            <p class="subhead">
              A working MER demo that combines text, audio, and video/facial expression
              by estimating how reliable each modality is before fusion.
            </p>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    _render_mode_status_badge(use_pretrained)

    st.markdown("### Proposed Research Architecture")
    _render_architecture_flow()

    st.markdown("### Current Working Prototype Pipeline")
    _render_current_pipeline(use_pretrained)

    st.markdown("### Project overview")
    st.markdown(
        """
        <div class="workflow-grid" aria-label="Reliability-aware MER workflow">
          <article class="workflow-card workflow-card-inputs">
            <div class="workflow-icon" aria-hidden="true">
              <svg viewBox="0 0 24 24" role="img">
                <path d="M4 5.5A2.5 2.5 0 0 1 6.5 3h7A2.5 2.5 0 0 1 16 5.5v2A2.5 2.5 0 0 1 13.5 10h-7A2.5 2.5 0 0 1 4 7.5v-2Zm2.5-.7a.7.7 0 0 0-.7.7v2c0 .39.31.7.7.7h7a.7.7 0 0 0 .7-.7v-2a.7.7 0 0 0-.7-.7h-7ZM3 15.5A2.5 2.5 0 0 1 5.5 13h4a2.5 2.5 0 0 1 2.5 2.5v3A2.5 2.5 0 0 1 9.5 21h-4A2.5 2.5 0 0 1 3 18.5v-3Zm2.5-.7a.7.7 0 0 0-.7.7v3c0 .39.31.7.7.7h4a.7.7 0 0 0 .7-.7v-3a.7.7 0 0 0-.7-.7h-4ZM15 13h4.5a1 1 0 0 1 0 2H15a1 1 0 1 1 0-2Zm0 4h5.5a1 1 0 1 1 0 2H15a1 1 0 1 1 0-2Z"/>
              </svg>
            </div>
            <h3>Multimodal Inputs</h3>
            <p>MELD Dataset or user input is split into text, audio and video branches for independent analysis.</p>
          </article>
          <article class="workflow-card workflow-card-reliability">
            <div class="workflow-icon" aria-hidden="true">
              <svg viewBox="0 0 24 24" role="img">
                <path d="M12 2.75 4.75 5.9v5.55c0 4.47 2.86 8.47 7.25 9.8 4.39-1.33 7.25-5.33 7.25-9.8V5.9L12 2.75Zm0 2.18 5.25 2.28v4.24c0 3.34-2.02 6.4-5.25 7.68-3.23-1.28-5.25-4.34-5.25-7.68V7.21L12 4.93Zm3.92 5.16a1 1 0 0 1 0 1.41l-4.24 4.24a1 1 0 0 1-1.42 0l-2.12-2.12a1 1 0 1 1 1.42-1.41l1.41 1.41 3.54-3.53a1 1 0 0 1 1.41 0Z"/>
              </svg>
            </div>
            <h3>Reliability Estimation</h3>
            <p>Confidence, quality metrics and missing modality checks are used to estimate reliability for each modality.</p>
          </article>
          <article class="workflow-card workflow-card-fusion">
            <div class="workflow-icon" aria-hidden="true">
              <svg viewBox="0 0 24 24" role="img">
                <path d="M7 3a4 4 0 0 1 3.87 3H13a4 4 0 1 1 0 2h-2.13A4 4 0 0 1 8 10.87V13a4 4 0 1 1-2 0v-2.13A4 4 0 0 1 7 3Zm0 2a2 2 0 1 0 0 4 2 2 0 0 0 0-4Zm10 1a2 2 0 1 0 0 4 2 2 0 0 0 0-4ZM7 15a2 2 0 1 0 0 4 2 2 0 0 0 0-4Zm8.1.5a1 1 0 0 1 1.4-.2l3.1 2.3a1 1 0 1 1-1.2 1.6l-3.1-2.3a1 1 0 0 1-.2-1.4Zm-2.2 0a1 1 0 0 1-.2 1.4l-3.1 2.3a1 1 0 1 1-1.2-1.6l3.1-2.3a1 1 0 0 1 1.4.2Z"/>
              </svg>
            </div>
            <h3>Reliability-Aware Fusion</h3>
            <p>Reliable modalities receive higher weights during fusion to generate the final emotion prediction.</p>
          </article>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write(
        "Most multimodal emotion systems treat all streams equally. This project separates the proposed "
        "research architecture from the current working prototype, then keeps reliability scoring and "
        "adaptive weighted fusion visible for every prediction."
    )

    _render_research_problem_statement()
    _render_algorithms_used()
    st.info(
        "BiLSTM and custom CNN are part of the proposed research architecture. Current implementation uses "
        "pretrained/fallback models for prototype testing unless custom trained checkpoints are added."
    )


def _render_mode_status_badge(use_pretrained: bool) -> None:
    if use_pretrained:
        mode = "Pretrained Model Mode"
        description = "uses Hugging Face models where available"
        css_class = "status-pretrained"
    else:
        mode = "Prototype Mode"
        description = "uses fallback/demo logic"
        css_class = "status-prototype"

    st.markdown(
        f"""
        <div class="mode-status {css_class}">
          <span class="mode-status-label">{mode}</span>
          <span class="mode-status-description">{description}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_architecture_flow() -> None:
    st.markdown(
        """
        <div class="architecture-flow">
          <div class="arch-row arch-row-inputs">
            <div class="arch-node arch-source">MELD Dataset / User Input</div>
            <div class="arch-arrow">&rarr;</div>
            <div class="arch-branches">
              <div class="arch-node arch-text">Text Branch<br><span>BERT feature extraction</span></div>
              <div class="arch-node arch-audio">Audio Branch<br><span>BiLSTM speech features</span></div>
              <div class="arch-node arch-video">Video Branch<br><span>CNN visual features</span></div>
            </div>
          </div>
          <div class="arch-row arch-row-processing">
            <div class="arch-node arch-fusion">Fusion Layer<br><span>combines modality features</span></div>
            <div class="arch-arrow">&rarr;</div>
            <div class="arch-node arch-reliability">Reliability Scoring<br><span>text + audio + video confidence</span></div>
            <div class="arch-arrow">&rarr;</div>
            <div class="arch-node arch-output">Adaptive Weighted Fusion<br><span>final emotion prediction</span></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_current_pipeline(use_pretrained: bool) -> None:
    text_backend = "Hugging Face BERT/Transformer if load succeeds" if use_pretrained else "local lexical fallback"
    audio_backend = "Hugging Face Wav2Vec2-style audio classifier if load succeeds" if use_pretrained else "audio quality fallback"
    video_backend = "Hugging Face image-classification model if load succeeds" if use_pretrained else "visual quality fallback"
    st.markdown(
        f"""
        <div class="current-pipeline-grid">
          <div class="current-card"><strong>Text branch</strong><span>{text_backend}</span></div>
          <div class="current-card"><strong>Audio branch</strong><span>{audio_backend}</span></div>
          <div class="current-card"><strong>Video branch</strong><span>{video_backend}</span></div>
          <div class="current-card current-card-fusion"><strong>Fusion</strong><span>Reliability-aware adaptive weighted fusion is implemented and active.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_research_problem_statement() -> None:
    st.markdown("### Research problem statement")
    st.markdown(
        """
        <div class="problem-solution-grid">
          <div class="research-panel problem-panel">
            <h3>Existing Problem</h3>
            <ul>
              <li>Simple averaging ignores unreliable modalities.</li>
              <li>Missing modalities reduce system usability.</li>
              <li>Different modalities may contradict each other.</li>
              <li>Most MER systems lack explanation.</li>
            </ul>
          </div>
          <div class="research-panel solution-panel">
            <h3>Proposed Solution</h3>
            <ul>
              <li>Reliability-aware fusion.</li>
              <li>Dynamic modality weighting.</li>
              <li>Missing modality robustness.</li>
              <li>Conflict detection and explainable prediction output.</li>
            </ul>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_algorithms_used() -> None:
    st.markdown("### Algorithms Used")
    st.markdown(
        """
        <div class="algorithm-grid">
          <div class="algorithm-card"><strong>Text: BERT / Transformer text classifier</strong><span>Pretrained mode attempts a Hugging Face Transformer; prototype mode uses lexical fallback.</span></div>
          <div class="algorithm-card"><strong>Audio: Wav2Vec2 audio emotion classifier OR BiLSTM placeholder</strong><span>Current implementation can use a Hugging Face audio classifier; custom BiLSTM checkpoints are not yet included.</span></div>
          <div class="algorithm-card"><strong>Video: CNN-based facial emotion classifier OR image-classification model</strong><span>Current implementation can use a pretrained image-classification model or visual quality fallback.</span></div>
          <div class="algorithm-card"><strong>Reliability Scoring</strong><span>Confidence and quality checking for every modality.</span></div>
          <div class="algorithm-card"><strong>Fusion: Reliability-aware adaptive weighted fusion</strong><span>Final prediction uses reliability-weighted modality scores.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_inputs() -> tuple[str, object, object]:
    st.markdown("### Input")
    sample_name = st.selectbox("Load a sample text case", ["Custom"] + list(SAMPLE_TEXTS))
    default_text = "" if sample_name == "Custom" else SAMPLE_TEXTS[sample_name]

    left, right = st.columns([1.2, 0.8])
    with left:
        text = st.text_area(
            "Text input",
            value=default_text,
            height=140,
            placeholder="Type an emotional utterance here...",
        )
    with right:
        st.markdown("#### Video / Frame Input")
        video_mode = st.radio("Video input mode", ["Upload file", "Use webcam"])
        if video_mode == "Upload file":
            video_file = st.file_uploader("Upload video/frame", type=["jpg", "png", "mp4", "avi", "mov"])
        else:
            camera_image = st.camera_input("Capture image from webcam")
            if camera_image is not None:
                st.image(camera_image, caption="Captured webcam frame")
            else:
                st.info("Please allow camera access in your browser.")
            video_file = camera_image

        st.markdown("#### Audio Input")
        audio_mode = st.radio("Audio input mode", ["Upload audio", "Record microphone"])
        if audio_mode == "Upload audio":
            audio_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "flac", "ogg", "m4a"])
        elif hasattr(st, "audio_input"):
            audio_file = st.audio_input("Record from microphone")
            if audio_file is not None:
                st.audio(audio_file)
        else:
            audio_file = None
            st.warning(
                "Microphone recording is not supported in this Streamlit version. "
                "Upgrade Streamlit inside this environment with: python -m pip install --upgrade streamlit"
            )

        st.caption("Audio and video are optional. Missing modalities receive zero fusion weight.")

    return text, audio_file, video_file


def run_prediction(text: str, audio_file: object, video_file: object, use_pretrained: bool = False) -> None:
    scorer = get_reliability_scorer()
    predictions: Dict[str, ModalityPrediction] = {}
    reliabilities: Dict[str, ReliabilityResult] = {}
    diagnostics: Dict[str, object] = {}
    progress_logs: List[Dict[str, str]] = []
    progress_table = st.empty()
    current_step = st.empty()
    started_at = time.perf_counter()

    st.markdown("### Prediction progress")
    _append_progress(progress_logs, progress_table, "system", "started", "Preparing inputs and cached model resources.")

    if text.strip():
        text_model = get_text_model(use_pretrained)
        text_fallback_model = get_text_model(False)
        current_step.info("Currently running: text emotion prediction")
        _append_progress(progress_logs, progress_table, "text", "running", "Running BERT/Transformer text branch.")
        text_prediction = _predict_with_timeout(
            modality="text",
            real_fn=lambda: text_model.predict(text),
            fallback_fn=lambda reason: _text_fallback_prediction(text_fallback_model, text, reason),
            timeout_seconds=TEXT_TIMEOUT_SECONDS,
            use_pretrained=use_pretrained,
        )
        predictions["text"] = text_prediction
        reliabilities["text"] = scorer.score_text(text_prediction, text)
        _append_prediction_progress(progress_logs, progress_table, "text", text_prediction)
    else:
        _append_progress(progress_logs, progress_table, "text", "skipped", "No text input provided.")

    video_path = ""
    if video_file is not None:
        video_path = _save_upload(video_file)
        diagnostics["video_path"] = video_path
        if str(getattr(video_file, "type", "")).startswith("image"):
            st.image(video_file, caption="Uploaded video frame / visual input", width=320)
        else:
            st.video(video_file)

        current_step.info(f"Currently running: video/facial prediction, sampling up to {MAX_VIDEO_FRAMES} frames")
        video_model = get_video_model(use_pretrained)
        video_fallback_model = get_video_model(False)
        _append_progress(
            progress_logs,
            progress_table,
            "video",
            "running",
            f"Sampling up to {MAX_VIDEO_FRAMES} frames and running face/video branch.",
        )
        video_prediction = _predict_with_timeout(
            modality="video",
            real_fn=lambda: video_model.predict(image_path=video_path),
            fallback_fn=lambda reason: _video_fallback_prediction(video_fallback_model, video_path, reason),
            timeout_seconds=VIDEO_TIMEOUT_SECONDS,
            use_pretrained=use_pretrained,
        )
        predictions["video"] = video_prediction
        reliabilities["video"] = scorer.score(video_prediction)
        _append_prediction_progress(progress_logs, progress_table, "video", video_prediction)
    else:
        _append_progress(progress_logs, progress_table, "video", "skipped", "No video/frame input provided.")

    if audio_file is not None:
        audio_path = _save_upload(audio_file)
        diagnostics["audio_path"] = audio_path
        st.audio(audio_file)
        current_step.info(f"Currently running: audio prediction on first {MAX_AUDIO_SECONDS:.0f} seconds")
        audio_model = get_audio_model(use_pretrained)
        audio_fallback_model = get_audio_model(False)
        _append_progress(
            progress_logs,
            progress_table,
            "audio",
            "running",
            f"Running audio branch on the first {MAX_AUDIO_SECONDS:.0f} seconds.",
        )
        audio_prediction = _predict_with_timeout(
            modality="audio",
            real_fn=lambda: audio_model.predict(audio_path=audio_path),
            fallback_fn=lambda reason: _audio_fallback_prediction(audio_fallback_model, audio_path, reason),
            timeout_seconds=AUDIO_TIMEOUT_SECONDS,
            use_pretrained=use_pretrained,
        )
        predictions["audio"] = audio_prediction
        reliabilities["audio"] = scorer.score_audio(audio_prediction)
        _append_prediction_progress(progress_logs, progress_table, "audio", audio_prediction)
    elif video_path and is_video_file(video_path):
        current_step.info(f"Currently running: extracting first {MAX_AUDIO_SECONDS:.0f} seconds of audio from video")
        _append_progress(
            progress_logs,
            progress_table,
            "audio",
            "running",
            "Extracting video audio track for the audio emotion branch.",
        )
        try:
            extracted_audio_path = extract_audio_from_video(video_path, max_duration_seconds=MAX_AUDIO_SECONDS)
            diagnostics["audio_path"] = extracted_audio_path
            diagnostics["audio_extracted_from_video"] = True
            audio_model = get_audio_model(use_pretrained)
            audio_fallback_model = get_audio_model(False)
            audio_prediction = _predict_with_timeout(
                modality="audio",
                real_fn=lambda: audio_model.predict(audio_path=extracted_audio_path),
                fallback_fn=lambda reason: _audio_fallback_prediction(audio_fallback_model, extracted_audio_path, reason),
                timeout_seconds=AUDIO_TIMEOUT_SECONDS,
                use_pretrained=use_pretrained,
            )
            predictions["audio"] = audio_prediction
            reliabilities["audio"] = scorer.score_audio(audio_prediction)
            st.info("Audio track extracted from uploaded video and sent to the audio emotion model.")
            _append_prediction_progress(progress_logs, progress_table, "audio", audio_prediction)
        except Exception as exc:
            diagnostics["audio_extraction_error"] = str(exc)
            _append_progress(progress_logs, progress_table, "audio", "fallback/unavailable", f"Could not extract audio: {exc}")
            st.warning(f"Could not extract audio from video: {exc}")
    else:
        _append_progress(progress_logs, progress_table, "audio", "skipped", "No audio input provided.")

    elapsed = time.perf_counter() - started_at
    current_step.success(f"Prediction pipeline finished in {elapsed:.1f} seconds.")
    _append_progress(progress_logs, progress_table, "system", "finished", f"Fusion inputs ready in {elapsed:.1f} seconds.")

    if not predictions:
        st.warning("Please provide text, audio, or video/frame input.")
        return

    _render_missing_modality_messages(predictions)
    fused = get_fusion().fuse(predictions, reliabilities)
    conflict = detect_emotion_conflict(predictions)
    explanation = build_explainability_panel(predictions, reliabilities, fused, conflict)

    st.markdown("### Modality predictions")
    _render_modality_table(predictions, reliabilities, fused)

    st.markdown("### Reliability visualization")
    _render_reliability_visualization(predictions, reliabilities, fused)

    st.markdown("### Final weighted fusion result")
    _render_fusion_view(fused)

    st.markdown("### Emotion conflict detector")
    _render_conflict_panel(conflict)

    st.markdown("### Model diagnostics / test page")
    _render_model_diagnostics(predictions, diagnostics)

    st.markdown("### Final result")
    _render_final_result(fused)
    _render_modality_mode_notice(fused)

    st.markdown("### Explainability panel")
    _render_explainability_panel(explanation)

    st.markdown("### Additional fusion notes")
    for line in explain_result(fused):
        st.write(f"- {line}")

    _render_realtime_mode()
    _render_metrics_section()


def _predict_with_timeout(
    modality: str,
    real_fn: Callable[[], ModalityPrediction],
    fallback_fn: Callable[[str], ModalityPrediction],
    timeout_seconds: int,
    use_pretrained: bool,
) -> ModalityPrediction:
    if not use_pretrained:
        return fallback_fn("Prototype mode selected.")

    future = get_prediction_executor().submit(real_fn)
    try:
        prediction = future.result(timeout=timeout_seconds)
        if _should_fallback_from_prediction(prediction):
            reason = prediction.error or "Pretrained model did not return an available prediction."
            return fallback_fn(reason)
        return prediction
    except TimeoutError:
        future.cancel()
        return fallback_fn(f"{modality.title()} model exceeded {timeout_seconds} seconds, so fallback was used.")
    except Exception as exc:
        return fallback_fn(f"{modality.title()} model failed: {exc}")


def _should_fallback_from_prediction(prediction: ModalityPrediction) -> bool:
    if prediction.is_available:
        return False
    backend_status = str(prediction.metadata.get("backend_status", "")).lower()
    error = str(prediction.error or "").lower()
    return "model failed" in backend_status or "transformer model failed" in error or "classifier" in error


def _text_fallback_prediction(
    fallback_model: TextEmotionRecognizer,
    text: str,
    reason: str,
) -> ModalityPrediction:
    prediction = fallback_model.predict(text)
    prediction.metadata["backend_status"] = "fallback model"
    prediction.metadata["fallback_used"] = True
    prediction.metadata["fallback_reason"] = reason
    return prediction


def _audio_fallback_prediction(
    fallback_model: AudioEmotionRecognizer,
    audio_path: str,
    reason: str,
) -> ModalityPrediction:
    prediction = fallback_model.predict(audio_path=audio_path)
    prediction.metadata["backend_status"] = "fallback model"
    prediction.metadata["fallback_used"] = True
    prediction.metadata["fallback_reason"] = reason
    return prediction


def _video_fallback_prediction(
    fallback_model: FacialEmotionRecognizer,
    video_path: str,
    reason: str,
) -> ModalityPrediction:
    prediction = fallback_model.predict(image_path=video_path)
    prediction.metadata["backend_status"] = "fallback model"
    prediction.metadata["fallback_used"] = True
    prediction.metadata["fallback_reason"] = reason
    return prediction


def _append_progress(
    logs: List[Dict[str, str]],
    placeholder: object,
    modality: str,
    status: str,
    message: str,
) -> None:
    logs.append(
        {
            "time": time.strftime("%H:%M:%S"),
            "modality": modality,
            "status": status,
            "message": message,
        }
    )
    placeholder.dataframe(pd.DataFrame(logs), use_container_width=True, hide_index=True)


def _append_prediction_progress(
    logs: List[Dict[str, str]],
    placeholder: object,
    modality: str,
    prediction: ModalityPrediction,
) -> None:
    backend_status = _backend_status(prediction)
    if prediction.is_available:
        message = f"{prediction.label} ({prediction.confidence:.2f}); backend: {backend_status}"
        status = "done"
    else:
        message = f"Unavailable: {prediction.error or 'No prediction'}; backend: {backend_status}"
        status = "unavailable"
    _append_progress(logs, placeholder, modality, status, message)


def _render_missing_modality_messages(predictions: Dict[str, ModalityPrediction]) -> None:
    for modality in ("text", "audio", "video"):
        if modality not in predictions:
            st.info(f"{modality.title()} not provided. Re-weighting available modalities.")


def _render_modality_table(
    predictions: Dict[str, ModalityPrediction],
    reliabilities: Dict[str, ReliabilityResult],
    fused: FusedPrediction,
) -> None:
    rows = []
    for modality in ("text", "audio", "video"):
        prediction = predictions.get(modality)
        reliability = reliabilities.get(modality)
        if prediction is None:
            rows.append(
                {
                    "modality": modality,
                    "branch algorithm": _algorithm_for_modality(modality),
                    "model_used": "none",
                    "backend_status": "missing modality",
                    "fallback_used": "yes",
                    "emotion": "missing",
                    "confidence": 0.0,
                    "reliability": 0.0,
                    "fusion_weight": 0.0,
                    "engine/status": "not provided",
                }
            )
            continue
        rows.append(
            {
                "modality": modality,
                "branch algorithm": prediction.metadata.get("algorithm", _algorithm_for_modality(modality)),
                "model_used": _model_used(prediction),
                "backend_status": _backend_status(prediction),
                "fallback_used": "yes" if _fallback_used(prediction) else "no",
                "emotion": prediction.label,
                "confidence": round(prediction.confidence, 3),
                "reliability": round(reliability.score if reliability else 0.0, 3),
                "fusion_weight": round(fused.weights.get(modality, 0.0), 3),
                "engine/status": prediction.metadata.get("engine", "model") if prediction.is_available else prediction.error,
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("Reliability component details"):
        for modality, reliability in reliabilities.items():
            st.write(f"**{modality.title()} reliability = {reliability.score:.3f}**")
            st.json(reliability.components)


def _render_reliability_visualization(
    predictions: Dict[str, ModalityPrediction],
    reliabilities: Dict[str, ReliabilityResult],
    fused: FusedPrediction,
) -> None:
    cols = st.columns(3)
    for idx, modality in enumerate(("text", "audio", "video")):
        with cols[idx]:
            reliability = reliabilities.get(modality)
            prediction = predictions.get(modality)
            reliability_score = reliability.score if reliability else 0.0
            weight = fused.weights.get(modality, 0.0)
            confidence = prediction.confidence if prediction else 0.0
            st.markdown(f"**{modality.title()}**")
            st.caption(f"Reliability: {reliability_score:.2f}")
            st.progress(reliability_score)
            st.caption(f"Contribution weight: {weight:.2f}")
            st.progress(weight)
            st.caption(f"Prediction confidence: {confidence:.2f}")
            st.progress(confidence)


def _algorithm_for_modality(modality: str) -> str:
    return {
        "text": "BERT text branch",
        "audio": "BiLSTM audio branch",
        "video": "CNN video branch",
    }.get(modality, "unknown")


def _model_used(prediction: ModalityPrediction) -> str:
    if not prediction.is_available:
        return "none"
    engine = str(prediction.metadata.get("engine", ""))
    if "fallback" in engine:
        if engine == "lexical_fallback":
            return "local lexical fallback"
        if engine == "quality_fallback":
            return "quality-based fallback"
        if engine == "file_metadata_fallback":
            return "file metadata fallback"
        return "fallback/demo logic"
    return prediction.metadata.get("model_name", "pretrained model")


def _backend_status(prediction: ModalityPrediction) -> str:
    if not prediction.is_available:
        return prediction.error or "unavailable"
    engine = str(prediction.metadata.get("engine", "model"))
    if engine == "transformer":
        return "pretrained Hugging Face text model"
    if engine == "audio_transformer":
        return "pretrained Hugging Face audio model"
    if engine == "image_transformer":
        return "pretrained Hugging Face image model"
    if engine == "video_frame_aggregation":
        return prediction.metadata.get("backend_status", "real model")
    if engine == "model_error":
        return "real model failed"
    if engine == "no_face_detected":
        return "no face detected"
    if engine == "quality_fallback":
        return "quality-based fallback"
    if engine == "lexical_fallback":
        return "lexical fallback"
    if engine == "file_metadata_fallback":
        return "file metadata fallback"
    if "fallback" in engine:
        return engine.replace("_", " ")
    return engine.replace("_", " ")


def _fallback_used(prediction: ModalityPrediction) -> bool:
    if not prediction.is_available:
        return True
    if bool(prediction.metadata.get("fallback_used", False)):
        return True
    engine = str(prediction.metadata.get("engine", ""))
    return "fallback" in engine


def _render_fusion_view(fused: FusedPrediction) -> None:
    weights = pd.Series({key: round(value, 3) for key, value in fused.weights.items()}, name="fusion weight")
    probabilities = pd.Series(
        {emotion: round(fused.probabilities.get(emotion, 0.0), 3) for emotion in EMOTION_CLASSES},
        name="final probability",
    )
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Adaptive modality weights")
        st.bar_chart(weights)
    with c2:
        st.caption("Final emotion probability distribution")
        st.bar_chart(probabilities)


def _render_conflict_panel(conflict: ConflictResult) -> None:
    if conflict.conflict_detected:
        st.warning(conflict.message)
    else:
        st.success(conflict.message)
    if conflict.agreeing_modalities:
        rows = [
            {"emotion": emotion, "modalities": ", ".join(modalities)}
            for emotion, modalities in conflict.agreeing_modalities.items()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_model_diagnostics(predictions: Dict[str, ModalityPrediction], diagnostics: Dict[str, object]) -> None:
    tabs = st.tabs(["Video frames", "Audio waveform", "Model confidence"])

    with tabs[0]:
        video_prediction = predictions.get("video")
        video_path = diagnostics.get("video_path")
        detected_faces = 0
        sampled_frames = 0
        if video_prediction:
            detected_faces = int(video_prediction.metadata.get("detected_faces", video_prediction.metadata.get("face_detected", 0)) or 0)
            sampled_frames = int(video_prediction.metadata.get("sampled_frames", 1 if video_prediction.is_available else 0) or 0)

        c1, c2 = st.columns(2)
        c1.metric("Extracted/sampled frames", sampled_frames)
        c2.metric("Detected face count", detected_faces)

        if video_path and is_video_file(str(video_path)):
            try:
                frames = sample_video_frames(str(video_path), max_frames=MAX_VIDEO_FRAMES)
                preview_paths = save_preview_frames(frames, max_frames=4)
                if preview_paths:
                    st.image(preview_paths, caption=[f"Sampled frame {idx + 1}" for idx in range(len(preview_paths))], width=180)
                else:
                    st.info("No frames could be extracted from the uploaded video.")
            except Exception as exc:
                st.warning(f"Frame diagnostics unavailable: {exc}")
        elif video_path:
            st.info("Image/frame input provided. Video frame sampling is only shown for video files.")
        else:
            st.info("No video/frame input provided.")

        if video_prediction and video_prediction.metadata.get("frame_predictions"):
            st.dataframe(pd.DataFrame(video_prediction.metadata["frame_predictions"]), use_container_width=True)

    with tabs[1]:
        audio_path = diagnostics.get("audio_path")
        if diagnostics.get("audio_extracted_from_video"):
            st.success("Audio waveform is from the uploaded video's extracted audio track.")
        if diagnostics.get("audio_extraction_error"):
            st.warning(f"Audio extraction failed: {diagnostics['audio_extraction_error']}")
        if audio_path:
            try:
                waveform = audio_waveform(str(audio_path))
                if waveform["time"]:
                    waveform_df = pd.DataFrame(waveform).set_index("time")
                    st.line_chart(waveform_df)
                else:
                    st.info("Audio waveform is empty.")
            except Exception as exc:
                st.warning(f"Audio waveform unavailable: {exc}")
        else:
            st.info("No audio input or extracted video audio available.")

    with tabs[2]:
        rows = []
        for modality, prediction in predictions.items():
            for emotion, probability in prediction.probabilities.items():
                rows.append(
                    {
                        "modality": modality,
                        "emotion": emotion,
                        "probability": round(probability, 4),
                        "backend_status": _backend_status(prediction),
                    }
                )
        if rows:
            confidence_df = pd.DataFrame(rows)
            st.dataframe(confidence_df, use_container_width=True, hide_index=True)
            chart_df = confidence_df.pivot(index="emotion", columns="modality", values="probability").fillna(0.0)
            st.bar_chart(chart_df)
        else:
            st.info("No model confidence scores available yet.")


def _render_final_result(fused: FusedPrediction) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Final emotion", fused.label.title())
    c2.metric("Final confidence", f"{fused.confidence:.2f}")
    c3.metric("Active modalities", str(sum(weight > 0 for weight in fused.weights.values())))

    sorted_probs = sorted(fused.probabilities.items(), key=lambda item: item[1], reverse=True)
    st.write(
        f"The fused model predicts **{fused.label}**. The next strongest class is "
        f"**{sorted_probs[1][0]}** with probability `{sorted_probs[1][1]:.2f}`."
    )


def _render_modality_mode_notice(fused: FusedPrediction) -> None:
    provided = [name for name in ("text", "audio", "video") if name in fused.modality_predictions]
    count = len(provided)

    if count <= 1:
        mode = "Single-modality mode"
        tone = "single"
        message = (
            "Only one modality was provided. This prediction is based on limited information. "
            "Add text, audio, and video inputs together for a more reliable multimodal emotion prediction. "
            "Single text input can contain sarcasm, contrast, or mixed emotional cues, so multimodal input is recommended."
        )
    elif count == 2:
        mode = "Bi-modal mode"
        tone = "bi"
        message = (
            "Two modalities were provided. Prediction reliability is better than single-modality input, "
            "but using all three modalities can improve robustness."
        )
    else:
        mode = "Full multimodal mode"
        tone = "full"
        message = (
            "All three modalities were provided. Reliability-aware fusion is active and the final prediction "
            "uses text, audio, and video signals."
        )

    missing = [name for name in ("text", "audio", "video") if name not in provided]
    missing_text = ", ".join(missing) if missing else "none"
    provided_text = ", ".join(provided) if provided else "none"

    st.markdown(
        f"""
        <div class="mode-notice mode-notice-{tone}">
          <div class="mode-notice-header">
            <span class="mode-pill">{mode}</span>
            <span class="mode-count">{count}/3 modalities provided</span>
          </div>
          <p>{message}</p>
          <p class="mode-detail">
            Provided: <strong>{provided_text}</strong>. Missing: <strong>{missing_text}</strong>.
            Reliability-aware MER down-weights missing modalities instead of failing.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_explainability_panel(explanation: Dict[str, str | List[str]]) -> None:
    for reason in explanation["modality_reasons"]:
        st.write(f"- {reason}")
    st.write(f"- {explanation['fusion_reason']}")
    st.write(f"- {explanation['conflict_reason']}")


def _render_realtime_mode() -> None:
    st.markdown("### Live input support")
    with st.expander("Webcam and microphone capture status"):
        st.info(
            "Live webcam capture uses Streamlit's camera widget and sends the captured frame to the same "
            "video/facial emotion branch. Microphone recording uses Streamlit audio input when the installed "
            "Streamlit version exposes it; otherwise the app shows an in-page fallback message."
        )
        c1, c2 = st.columns(2)
        c1.success("Webcam frame input: available through the Video / Frame Input selector.")
        c2.success("Microphone recording: available through the Audio Input selector when supported.")
        st.caption("Uploaded files and live captures are both handled by reliability-aware fusion.")


def _render_metrics_section() -> None:
    st.markdown("### Research metrics page")
    metrics = demo_metrics()
    st.caption("Demo/example metrics. Replace with MELD or project test samples for real evaluation.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']:.2f}")
    c2.metric("Precision", f"{metrics['precision']:.2f}")
    c3.metric("Recall", f"{metrics['recall']:.2f}")
    c4.metric("F1-score", f"{metrics['f1_score']:.2f}")

    matrix_df = pd.DataFrame(
        metrics["confusion_matrix"],
        index=metrics["labels"],
        columns=metrics["labels"],
    )
    st.caption("Demo confusion matrix")
    st.dataframe(matrix_df, use_container_width=True)


def explain_result(fused: FusedPrediction) -> List[str]:
    active = [
        (name, weight, fused.modality_predictions[name], fused.reliability_scores.get(name, 0.0))
        for name, weight in fused.weights.items()
        if weight > 0 and name in fused.modality_predictions
    ]
    active.sort(key=lambda item: item[1], reverse=True)

    if not active:
        return ["No reliable modality was available, so the system returned a uniform fallback distribution."]

    lines = []
    leader_name, leader_weight, leader_prediction, leader_reliability = active[0]
    lines.append(
        f"{leader_name.title()} had the largest fusion weight ({leader_weight:.2f}) because its reliability score was {leader_reliability:.2f}."
    )
    lines.append(
        f"The strongest {leader_name} prediction was {leader_prediction.label} with confidence {leader_prediction.confidence:.2f}."
    )

    text_prediction = fused.modality_predictions.get("text")
    if text_prediction and text_prediction.metadata.get("mixed_emotional_cues"):
        lines.append(
            "Mixed emotional cues detected in the text; confidence was reduced because text alone may be ambiguous."
        )
    if text_prediction and text_prediction.metadata.get("contrast_detected"):
        contrast_word = text_prediction.metadata.get("contrast_word") or "contrast"
        lines.append(
            f"The text contains a contrast cue ('{contrast_word}'), so the phrase after it was weighted more strongly."
        )

    missing = [name for name in ("text", "audio", "video") if name not in fused.modality_predictions]
    if missing:
        lines.append(f"Missing modalities ({', '.join(missing)}) were assigned zero weight instead of weakening the result.")
        if set(missing) == {"audio", "video"} and text_prediction:
            lines.append(
                "For text-only input, audio tone and video/facial expression are recommended because they can help resolve sarcasm, contrast, and mixed emotions."
            )

    fallback_modalities = [
        name
        for name, prediction in fused.modality_predictions.items()
        if "fallback" in str(prediction.metadata.get("engine", ""))
    ]
    if fallback_modalities:
        lines.append(
            f"{', '.join(name.title() for name in fallback_modalities)} used transparent fallback logic, so reliability limits its influence."
        )

    lines.append("The final probabilities are a reliability-weighted average, not a simple majority vote.")
    return lines


def _render_sample_cases() -> None:
    st.markdown("### Sample examples/test cases")
    st.dataframe(
        pd.DataFrame(
            [
                {"case": name, "sample_text": text}
                for name, text in SAMPLE_TEXTS.items()
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )


def _save_upload(uploaded_file: object) -> str:
    suffix = Path(uploaded_file.name).suffix if getattr(uploaded_file, "name", None) else ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            max-width: 1180px;
        }
        .hero {
            padding: 2rem 2.25rem;
            border-radius: 8px;
            background: linear-gradient(135deg, #11324d 0%, #226f54 52%, #c98b2c 100%);
            color: white;
            margin-bottom: 1.5rem;
        }
        .hero h1 {
            font-size: 2.35rem;
            line-height: 1.1;
            margin: 0.15rem 0 0.75rem 0;
            letter-spacing: 0;
        }
        .hero .subhead {
            font-size: 1.03rem;
            max-width: 760px;
            margin-bottom: 0;
        }
        .eyebrow {
            text-transform: uppercase;
            letter-spacing: .08rem;
            font-weight: 700;
            font-size: .78rem;
            opacity: .86;
            margin: 0;
        }
        [data-testid="stCameraInput"] a,
        div[data-testid="stCameraInput"] a,
        section[data-testid="stCameraInput"] a {
            display: none !important;
            pointer-events: none !important;
        }
        .mode-status {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: .75rem;
            margin: 0 0 1.25rem;
            padding: .85rem 1rem;
            border-radius: 8px;
            border: 1px solid #dbe4ea;
            box-shadow: 0 10px 24px rgba(15, 23, 42, .08);
        }
        .mode-status-label {
            display: inline-flex;
            align-items: center;
            min-height: 32px;
            padding: .28rem .75rem;
            border-radius: 999px;
            color: #ffffff;
            font-weight: 800;
            font-size: .86rem;
        }
        .mode-status-description {
            color: #334155;
            font-weight: 700;
            font-size: .94rem;
        }
        .status-pretrained {
            background: #effaf4;
            border-left: 6px solid #226f54;
        }
        .status-pretrained .mode-status-label {
            background: #226f54;
        }
        .status-prototype {
            background: #fff8ed;
            border-left: 6px solid #c47b16;
        }
        .status-prototype .mode-status-label {
            background: #b76b00;
        }
        .current-pipeline-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: .85rem;
            margin: .8rem 0 1.25rem;
        }
        .current-card {
            min-height: 116px;
            padding: .95rem;
            border-radius: 8px;
            border: 1px solid #dbe4ea;
            background: #ffffff;
            box-shadow: 0 8px 20px rgba(15, 23, 42, .06);
        }
        .current-card strong {
            display: block;
            color: #0f172a;
            font-size: .96rem;
            margin-bottom: .42rem;
        }
        .current-card span {
            display: block;
            color: #475569;
            font-size: .86rem;
            line-height: 1.45;
        }
        .current-card-fusion {
            background: #f8fafc;
            border-top: 5px solid #7c3aed;
        }
        .problem-solution-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 1rem;
            margin: .8rem 0 1.3rem;
        }
        .research-panel {
            padding: 1rem 1.1rem;
            border-radius: 8px;
            border: 1px solid #dbe4ea;
            background: #ffffff;
            box-shadow: 0 10px 24px rgba(15, 23, 42, .07);
        }
        .research-panel h3 {
            margin: 0 0 .6rem;
            color: #0f172a;
            font-size: 1.02rem;
        }
        .research-panel ul {
            margin: 0;
            padding-left: 1.2rem;
            color: #475569;
            line-height: 1.55;
            font-size: .93rem;
        }
        .problem-panel {
            border-top: 5px solid #be123c;
        }
        .solution-panel {
            border-top: 5px solid #226f54;
        }
        .architecture-flow {
            display: flex;
            flex-direction: column;
            gap: .9rem;
            margin: .8rem 0 1.35rem;
        }
        .arch-row {
            display: grid;
            align-items: stretch;
            gap: .8rem;
        }
        .arch-row-inputs {
            grid-template-columns: minmax(230px, .85fr) 42px minmax(620px, 2.3fr);
        }
        .arch-row-processing {
            grid-template-columns: minmax(260px, 1fr) 42px minmax(260px, 1fr) 42px minmax(280px, 1.05fr);
        }
        .arch-node {
            min-height: 88px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            padding: .95rem 1rem;
            border-radius: 8px;
            border: 1px solid #dbe4ea;
            background: #ffffff;
            color: #0f172a;
            font-weight: 800;
            line-height: 1.25;
            box-shadow: 0 10px 22px rgba(15, 23, 42, .08);
            text-align: center;
            white-space: normal;
            word-break: normal;
            overflow-wrap: normal;
        }
        .arch-node span {
            color: #64748b;
            font-weight: 600;
            font-size: .84rem;
            margin-top: .28rem;
            white-space: normal;
        }
        .arch-arrow {
            color: #64748b;
            font-size: 1.35rem;
            font-weight: 900;
            text-align: center;
        }
        .arch-branches {
            display: grid;
            grid-template-columns: repeat(3, minmax(170px, 1fr));
            gap: .8rem;
        }
        .arch-source { border-top: 5px solid #475569; }
        .arch-text { border-top: 5px solid #1f6feb; }
        .arch-audio { border-top: 5px solid #226f54; }
        .arch-video { border-top: 5px solid #c47b16; }
        .arch-fusion { border-top: 5px solid #7c3aed; }
        .arch-reliability { border-top: 5px solid #0f766e; }
        .arch-output { border-top: 5px solid #be123c; }
        .algorithm-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: .8rem;
            margin: .8rem 0 1.35rem;
        }
        .algorithm-card {
            min-height: 118px;
            padding: .9rem;
            border-radius: 8px;
            border: 1px solid #dbe4ea;
            background: #f8fafc;
            box-shadow: 0 8px 20px rgba(15, 23, 42, .06);
        }
        .algorithm-card strong {
            display: block;
            color: #0f172a;
            font-size: .94rem;
            line-height: 1.25;
            margin-bottom: .45rem;
        }
        .algorithm-card span {
            color: #475569;
            font-size: .84rem;
            line-height: 1.45;
        }
        .workflow-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1rem;
            margin: .85rem 0 1.2rem;
        }
        .workflow-card {
            position: relative;
            min-height: 190px;
            padding: 1.15rem 1.15rem 1.05rem;
            border: 1px solid #dbe4ea;
            border-radius: 8px;
            background: #ffffff;
            box-shadow: 0 12px 28px rgba(15, 23, 42, .08);
            transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
            overflow: hidden;
        }
        .workflow-card::before {
            content: "";
            position: absolute;
            inset: 0 0 auto 0;
            height: 5px;
            background: #226f54;
        }
        .workflow-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 18px 38px rgba(15, 23, 42, .14);
            border-color: #9fb6c5;
        }
        .workflow-card-inputs::before { background: #1f6feb; }
        .workflow-card-reliability::before { background: #226f54; }
        .workflow-card-fusion::before { background: #c47b16; }
        .workflow-icon {
            width: 48px;
            height: 48px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 8px;
            margin-bottom: .85rem;
        }
        .workflow-card-inputs .workflow-icon {
            color: #1f6feb;
            background: #eaf2ff;
        }
        .workflow-card-reliability .workflow-icon {
            color: #226f54;
            background: #eaf7f1;
        }
        .workflow-card-fusion .workflow-icon {
            color: #b76b00;
            background: #fff4df;
        }
        .workflow-icon svg {
            width: 28px;
            height: 28px;
            fill: currentColor;
        }
        .workflow-card h3 {
            color: #0f172a;
            font-size: 1.05rem;
            line-height: 1.25;
            margin: 0 0 .45rem;
            letter-spacing: 0;
        }
        .workflow-card p {
            color: #475569;
            font-size: .94rem;
            line-height: 1.55;
            margin: 0;
        }
        .mode-notice {
            margin: 1rem 0 1.25rem;
            padding: 1rem 1.1rem;
            border-radius: 8px;
            border: 1px solid #d7e1ea;
            background: #f8fafc;
            box-shadow: 0 10px 24px rgba(15, 23, 42, .07);
        }
        .mode-notice-single {
            border-left: 6px solid #c47b16;
            background: #fff8ed;
        }
        .mode-notice-bi {
            border-left: 6px solid #1f6feb;
            background: #eff6ff;
        }
        .mode-notice-full {
            border-left: 6px solid #226f54;
            background: #effaf4;
        }
        .mode-notice-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: .75rem;
            flex-wrap: wrap;
            margin-bottom: .45rem;
        }
        .mode-pill {
            display: inline-flex;
            align-items: center;
            min-height: 30px;
            padding: .25rem .65rem;
            border-radius: 999px;
            background: #0f172a;
            color: #ffffff;
            font-size: .82rem;
            font-weight: 700;
            letter-spacing: 0;
        }
        .mode-count {
            color: #475569;
            font-size: .88rem;
            font-weight: 700;
        }
        .mode-notice p {
            color: #334155;
            font-size: .95rem;
            line-height: 1.55;
            margin: .3rem 0 0;
        }
        .mode-notice .mode-detail {
            color: #64748b;
            font-size: .88rem;
        }
        @media (max-width: 900px) {
            .mode-status {
                align-items: flex-start;
                flex-direction: column;
            }
            .current-pipeline-grid {
                grid-template-columns: 1fr;
            }
            .problem-solution-grid {
                grid-template-columns: 1fr;
            }
            .arch-row,
            .arch-row-inputs,
            .arch-row-processing {
                grid-template-columns: 1fr;
            }
            .arch-arrow {
                transform: rotate(90deg);
            }
            .arch-branches,
            .algorithm-grid {
                grid-template-columns: 1fr;
            }
            .workflow-grid {
                grid-template-columns: 1fr;
            }
            .workflow-card {
                min-height: 0;
            }
            .mode-notice {
                padding: .95rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
