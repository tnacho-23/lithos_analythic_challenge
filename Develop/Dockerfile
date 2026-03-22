FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TZ=Etc/UTC

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    wget \
    curl \
    git \
    tzdata \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install --no-cache-dir git+https://github.com/facebookresearch/segment-anything-2.git

RUN pip3 install --no-cache-dir \
    timm \
    pillow \
    matplotlib \
    numpy \
    scikit-learn \
    joblib \
    pandas \
    gdown \
    transformers \
    ultralytics \
    sahi \
    accelerate \
    datasets \
    opencv-python \
    evaluate

WORKDIR /home

RUN mkdir -p checkpoints && \
    wget -P checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt

CMD ["/bin/bash"]