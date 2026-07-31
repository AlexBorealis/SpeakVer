FROM nvidia/cuda:12.6.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    libsndfile1 \
    ffmpeg \
    gcc \
    git \
    nano \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN python3 -m pip install \
    --no-cache-dir \
    --break-system-packages \
    --ignore-installed \
    -r requirements.txt

COPY src/ ./src/
COPY pretrained_models/ ./pretrained_models/
COPY report.py .
COPY app.py .

RUN mkdir -p \
    /app/datasets \
    /app/reports \
    /app/runs \
    /app/archives

VOLUME ["/app/datasets"]
VOLUME ["/app/reports"]
VOLUME ["/app/runs"]
VOLUME ["/app/archives"]

EXPOSE 7860

ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

CMD ["python", "app.py"]