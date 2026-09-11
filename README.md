# video-pipeline

Fully automated local video production: drop a speech/commentary recording
into `speech/`, drop racing gameplay footage into `racing/`, and the app
combines them into a subtitled, upload-ready MP4 — no manual editing, no
DaVinci Resolve.

```
DROP SPEECH VIDEO
       +
DROP RACING VIDEO
       ↓
     WAIT
       ↓
FINAL VIDEO APPEARS in output/
```

Everything runs locally: ffmpeg for media processing, faster-whisper for
local transcription. No cloud APIs, no uploads.

## How it works

1. You drop a speech/commentary video into `video-pipeline/speech/` and a
   racing gameplay video into `video-pipeline/racing/`.
2. The app watches both folders continuously (via `watchdog`, with a
   periodic poll as a fallback). Once a file has stopped growing for a few
   seconds (so a still-copying file is never grabbed mid-write), it's
   considered ready.
3. As soon as **both** sides have at least one ready, unclaimed file, they're
   paired FIFO (oldest speech + oldest racing) into a job and moved into
   `processing/`.
4. The job runs:
   - Extract the speech file's audio (its video is discarded).
   - Measure the exact audio duration.
   - Prepare the racing video to cover that duration: if it's already long
     enough it's used as-is (zero extra work); if it's shorter it's looped
     via ffmpeg's concat demuxer with stream copy (fast, lossless, no
     re-encode). The racing file's own audio is discarded entirely.
   - Transcribe the speech audio locally with faster-whisper, producing a
     timestamped `.srt`.
   - One ffmpeg pass combines the prepared racing video + speech audio,
     burns in the subtitles, trims to the exact speech duration, and
     encodes to the configured codec/resolution/fps — this is the only
     re-encode in the whole pipeline.
5. The final `.mp4` (and a copy of the `.srt`) land in `output/`. The job's
   working files are cleaned up.
6. If anything fails, the job (with its inputs and a detailed log) is moved
   to `failed/` instead, and the app keeps running — one bad file never
   takes down the watcher.

A SQLite database (`video-pipeline/state.db`) tracks every claimed file and
every job's status, so restarting the app never reprocesses or loses a job:
on startup, any job left mid-flight by a previous crash is automatically
moved to `failed/` with an explanation rather than silently retried.

## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) (provides both `ffmpeg` and `ffprobe`) on
  your PATH
- ~2-6 GB free disk for the Whisper model (downloaded automatically on
  first transcription)

The app inspects your machine at startup (OS, CPU, RAM, GPU/CUDA/VRAM,
ffmpeg encoders) and picks sensible defaults:

| Hardware | Whisper model | Video encoder |
|---|---|---|
| NVIDIA GPU, ≥10GB VRAM | `large-v3` | `h264_nvenc` |
| NVIDIA GPU, 6-10GB VRAM | `medium` | `h264_nvenc` |
| NVIDIA GPU, <6GB VRAM | `small`/`base` | `h264_nvenc` |
| CPU only, ≥15GB RAM | `small` | `libx264` |
| CPU only, ≥7GB RAM | `base` | `libx264` |
| CPU only, <7GB RAM | `tiny` | `libx264` |

Run `python3 scripts/inspect_env.py` any time to see exactly what was
detected and what it recommends on your machine. Every recommendation can
be overridden in `config.yaml`.

## Setup

### Linux / macOS

```bash
cd video-pipeline-app
./scripts/start.sh
```

### Windows

```bat
cd video-pipeline-app
scripts\start.bat
```

Either script will, on first run:
1. Create a virtualenv (`.venv/`) and install dependencies.
2. Run the environment check and print hardware/whisper/codec recommendations.
3. Copy `config.example.yaml` to `config.yaml` if you don't have one yet.
4. Start the watcher.

You'll see the folders it's watching printed on startup:

```
video-pipeline/
    speech/       <- drop your commentary/narration recordings here
    racing/       <- drop your racing gameplay recordings here
    output/       <- finished .mp4 + .srt appear here
    failed/       <- jobs that errored, with a full error log
    processing/   <- in-flight jobs (you shouldn't need to touch this)
```

Drop a video into `speech/` and one into `racing/`, then watch `output/`.

To stop the app, press Ctrl+C — it shuts down cleanly and any in-flight job
picks up correctly (moved to `failed/` with an explanation) if you restart
mid-job.

## Configuration

Copy `config.example.yaml` to `config.yaml` and edit it. Every setting has
an inline comment explaining what it does. Highlights:

- `paths.root` — where the speech/racing/output/failed/processing folders
  live. Defaults to `./video-pipeline` next to the app.
- `whisper.model` / `whisper.device` / `whisper.compute_type` — leave as
  `"auto"` to use the hardware-based recommendation, or force a specific
  model (e.g. `"medium"`) if you want to trade speed for accuracy.
- `subtitles.*` — line length, max lines per cue, font, size, color,
  position. `subtitles.burn_in: false` keeps the `.srt` without hardcoding
  it into the video.
- `video.codec` — `"auto"` picks `h264_nvenc` if you have an NVIDIA GPU
  with a compatible ffmpeg build, else `libx264`.
- `video.resolution` / `video.fps` / `audio.bitrate` — standard YouTube
  upload settings, defaults to 1920x1080 @ 60fps, 192k AAC audio.

## Testing

```bash
./run_tests.sh
```

This generates real synthetic test media locally (a plain-color video with
actual synthesized speech via `espeak-ng`, and an ffmpeg `testsrc` pattern
video standing in for gameplay footage — no internet, no sample files
needed) and runs the full test suite against it, including:

- Real `ffmpeg`/`ffprobe` commands (audio extraction, racing-video
  looping/trimming, the final combine+subtitle-burn encode) — nothing is
  mocked here, because the point is to verify the actual commands work.
- A full pipeline run end-to-end (`pipeline.run_job`) asserting the output
  duration matches the speech audio, the racing audio never appears in the
  output, resolution/codec settings are applied, and the `.srt` lands next
  to the `.mp4`.
- Failure handling: a corrupt input file is confirmed to land in `failed/`
  with a readable `ERROR.txt`, without crashing the app.
- Restart safety: a job frozen mid-flight (simulating a crash) is confirmed
  to be recovered into `failed/` on the next startup, never silently
  reprocessed.
- FIFO pairing and file-stability detection with real files and real
  timestamps in temp directories.

**Note on transcription in tests:** the Whisper model itself is mocked in
the automated test suite (`WhisperModel.transcribe`), so `run_tests.sh`
doesn't require downloading a model or network access. This keeps the
suite fast and runnable offline, and still exercises everything else for
real, including the ffmpeg subtitle-burning filter with realistic cue
content. To verify actual transcription end-to-end, just run the app for
real (`./scripts/start.sh`) with a real speech recording — the first run
will download the configured Whisper model from Hugging Face
automatically (this does require normal internet access once).

## Project layout

```
video-pipeline-app/
    src/video_pipeline/
        hw_detect.py      # OS/CPU/GPU/CUDA/ffmpeg inspection -> recommendations
        config.py         # config.yaml loading + defaults
        db.py             # SQLite job/claim tracking (restart-safe)
        media.py          # ffmpeg/ffprobe wrappers: probe, extract, loop/trim, combine
        subtitles.py      # word-timestamp -> readable SRT cue formatting
        transcribe.py     # faster-whisper wrapper
        pipeline.py        # the 11-step job, error handling, recovery
        watcher.py         # folder watching, stability detection, FIFO pairing
        logging_setup.py   # app + per-job logging
        main.py            # entrypoint / main loop
    scripts/
        inspect_env.py     # standalone hardware report
        start.sh / start.bat
    tests/
        generate_sample_media.py
        test_media.py / test_subtitles.py / test_transcribe.py
        test_watcher.py / test_pipeline_e2e.py
    config.example.yaml
    requirements.txt
    run_tests.sh
```

## Design notes

- **Minimal re-encoding.** Burning subtitles requires decoding and
  re-encoding the video no matter what, so the pipeline does exactly one
  real video re-encode: the final combine+subtitle-burn pass. Looping a
  short racing clip is done via ffmpeg's concat demuxer with `-c copy`
  (stream copy, no re-encode), and a racing clip that's already long
  enough is used completely untouched — not even copied.
- **Restart safety, not just crash-avoidance.** Every claimed file and
  every job's status lives in SQLite. A source file is only ever claimed
  once, and is physically moved out of `speech/`/`racing/` the moment it's
  claimed, so the watcher can never see it again. If the app dies mid-job,
  the next startup finds it (`status = "processing"` with no matching
  completion) and moves it to `failed/` with an explanation instead of
  silently retrying or losing it.
- **One failure never stops the app.** `pipeline.run_job` catches every
  exception, writes a full traceback into both the per-job log and
  `failed/<job_id>/ERROR.txt`, and returns control to the main loop, which
  moves on to the next pair.
