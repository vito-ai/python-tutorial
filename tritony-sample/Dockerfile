FROM nvcr.io/nvidia/tritonserver:23.08-py3

RUN apt update && apt install -y --no-install-recommends \
    libsndfile1 \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*;

RUN pip install --upgrade pip && \
    pip install torch torchaudio==2.1.2 && \
    pip install pyannote.audio
