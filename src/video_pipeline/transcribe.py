"""Local transcription via faster-whisper, producing a timestamped .srt.

The whisper model is cached per (model, device, compute_type) so the
continuously-running watcher only pays the model-load cost once, not per
job.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from .config import Config
from .subtitles import Word, cues_to_srt, words_to_cues


class TranscriptionError(RuntimeError):
    pass


_model_cache: dict[tuple, object] = {}
_model_lock = threading.Lock()


def get_model(config: Config):
    """Lazily imports faster_whisper so environments that only need the
    non-transcription parts of the pipeline (e.g. running media.py tests)
    don't require the (large) dependency to be installed."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise TranscriptionError(
            "faster-whisper is not installed. Run: pip install -r requirements.txt"
        ) from e

    key = (config.whisper.model, config.whisper.device, config.whisper.compute_type)
    with _model_lock:
        if key not in _model_cache:
            _model_cache[key] = WhisperModel(
                config.whisper.model,
                device=config.whisper.device,
                compute_type=config.whisper.compute_type,
            )
        return _model_cache[key]


def transcribe_to_srt(
    audio_path: Path,
    out_srt_path: Path,
    config: Config,
    logger: logging.Logger | None = None,
) -> Path:
    model = get_model(config)

    if logger:
        logger.info(
            "Transcribing %s with whisper model=%s device=%s compute_type=%s",
            audio_path,
            config.whisper.model,
            config.whisper.device,
            config.whisper.compute_type,
        )

    segments, info = model.transcribe(
        str(audio_path),
        language=config.whisper.language,
        beam_size=config.whisper.beam_size,
        vad_filter=config.whisper.vad_filter,
        word_timestamps=True,
    )

    words: list[Word] = []
    segment_count = 0
    for seg in segments:
        segment_count += 1
        seg_words = getattr(seg, "words", None)
        if seg_words:
            for w in seg_words:
                text = (w.word or "").strip()
                if text:
                    words.append(Word(start=w.start, end=w.end, text=text))
        else:
            # Fallback if word-level timestamps aren't available for a
            # segment: treat the whole segment as one cue.
            text = (seg.text or "").strip()
            if text:
                words.append(Word(start=seg.start, end=seg.end, text=text))

    if logger:
        detected_lang = getattr(info, "language", "unknown")
        logger.info(
            "Transcription complete: %d segments, %d words, detected language=%s",
            segment_count,
            len(words),
            detected_lang,
        )

    if not words:
        raise TranscriptionError(
            f"Transcription produced no text for {audio_path}. "
            "The speech audio may be silent, too quiet, or unintelligible."
        )

    cues = words_to_cues(
        words,
        max_chars_per_line=config.subtitles.max_chars_per_line,
        max_lines_per_cue=config.subtitles.max_lines_per_cue,
        max_cue_duration_seconds=config.subtitles.max_cue_duration_seconds,
    )
    srt_text = cues_to_srt(cues)

    out_srt_path.parent.mkdir(parents=True, exist_ok=True)
    out_srt_path.write_text(srt_text, encoding="utf-8")

    if logger:
        logger.info("Wrote SRT (%d cues) to %s", len(cues), out_srt_path)

    return out_srt_path
