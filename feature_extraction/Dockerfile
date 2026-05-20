# Feature-extraction worker. Build from repo root:
#   docker build -f feature_extraction/Dockerfile -t fe-worker .

FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

COPY feature_extraction/requirements-worker.txt /tmp/requirements-worker.txt
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r /tmp/requirements-worker.txt

COPY feature_extraction/ /app/feature_extraction/
COPY src/ /app/src/
COPY data_labeling/ /app/data_labeling/
COPY cv-pipeline/calibration/ /app/cv-pipeline/calibration/
COPY cv-pipeline/pose-detection/fetch_s3_clip.py \
     cv-pipeline/pose-detection/pose_side_by_side_video.py \
     /app/cv-pipeline/pose-detection/

COPY yolov8s-pose.pt /app/yolov8s-pose.pt

RUN mkdir -p /app/cv-pipeline/pose-detection/media/clips

ENTRYPOINT ["python"]
CMD ["feature_extraction/job.py", "--help"]
