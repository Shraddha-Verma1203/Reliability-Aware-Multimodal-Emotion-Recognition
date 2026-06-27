"""Utilities for video audio extraction, frame sampling, and diagnostics."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def is_video_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def extract_audio_from_video(video_path: str | Path, max_duration_seconds: float = 10.0) -> str:
    """Extract audio from a video file into a temporary WAV file.

    Raises an exception when moviepy/ffmpeg cannot decode the file. The caller
    decides whether that becomes an unavailable modality or a fallback.
    """

    output = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    output.close()
    output_path = output.name

    try:
        try:
            from moviepy import VideoFileClip
        except Exception:
            from moviepy.editor import VideoFileClip

        clip = VideoFileClip(str(video_path))
        working_clip = clip
        duration = float(getattr(clip, "duration", 0.0) or 0.0)
        if duration > max_duration_seconds:
            if hasattr(clip, "subclipped"):
                working_clip = clip.subclipped(0, max_duration_seconds)
            else:
                working_clip = clip.subclip(0, max_duration_seconds)
        if working_clip.audio is None:
            raise ValueError("Video file does not contain an audio track.")
        working_clip.audio.write_audiofile(output_path, logger=None)
        if working_clip is not clip:
            working_clip.close()
        clip.close()
        return output_path
    except Exception:
        Path(output_path).unlink(missing_ok=True)
        raise


def sample_video_frames(video_path: str | Path, max_frames: int = 5) -> List[Any]:
    """Sample frames from a video using OpenCV."""

    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return []

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        positions = list(range(max_frames))
    else:
        step = max(frame_count // max_frames, 1)
        positions = list(range(0, frame_count, step))[:max_frames]

    frames = []
    for position in positions:
        capture.set(cv2.CAP_PROP_POS_FRAMES, position)
        success, frame = capture.read()
        if success and frame is not None:
            frames.append(frame)
    capture.release()
    return frames


def save_preview_frames(frames: List[Any], max_frames: int = 4) -> List[str]:
    """Save sampled OpenCV BGR frames as temporary JPEG previews."""

    import cv2

    paths: List[str] = []
    for frame in frames[:max_frames]:
        output = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        output.close()
        cv2.imwrite(output.name, frame)
        paths.append(output.name)
    return paths


def audio_waveform(
    audio_path: str | Path,
    max_points: int = 500,
    max_duration_seconds: float = 10.0,
) -> Dict[str, List[float]]:
    """Return downsampled audio waveform points for Streamlit line charts."""

    import librosa
    import numpy as np

    signal, sample_rate = librosa.load(audio_path, sr=16000, mono=True, duration=max_duration_seconds)
    if signal.size == 0:
        return {"time": [], "amplitude": []}

    if signal.size > max_points:
        indices = np.linspace(0, signal.size - 1, max_points).astype(int)
        sampled = signal[indices]
    else:
        indices = np.arange(signal.size)
        sampled = signal

    times = indices / sample_rate
    return {
        "time": [float(value) for value in times],
        "amplitude": [float(value) for value in sampled],
    }
