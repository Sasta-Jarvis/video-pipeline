# Runs the whole pipeline (ffmpeg + Python + faster-whisper) in a known-good
# Linux environment. This sidesteps Windows-specific issues entirely (e.g.
# ctranslate2/faster-whisper DLL load failures) since everything -- ffmpeg,
# Python, and every native dependency -- is installed fresh inside the image
# rather than relying on whatever is (or isn't) already on your host.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY webapp/ webapp/
COPY config.docker.yaml /app/config.yaml

ENV PYTHONPATH=/app/src
ENV VIDEO_PIPELINE_CONFIG=/app/config.yaml

# Data (speech/racing/output/failed/processing/state.db/logs) lives on a
# mounted volume so it survives container restarts/rebuilds.
VOLUME ["/data"]

EXPOSE 8080

CMD ["python", "webapp/server.py"]
