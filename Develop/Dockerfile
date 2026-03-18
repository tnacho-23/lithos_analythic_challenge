FROM nvidia/cuda:12.3.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TZ=Etc/UTC

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    wget \
    curl \
    git \
    kagglehub \
    tzdata && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN pip3 install --no-cache-dir \
    timm \
    pillow \
    matplotlib \
    numpy \
    scikit-learn \
    joblib \
    pandas \
    gdown \
    transformers 

WORKDIR /home

CMD ["/bin/bash"]
