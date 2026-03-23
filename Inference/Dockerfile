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

# Instalación de PyTorch con soporte CUDA 12.1
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Instalación de SAM2
RUN pip3 install --no-cache-dir git+https://github.com/facebookresearch/segment-anything-2.git

# Instalación de dependencias del proyecto
RUN pip3 install --no-cache-dir \
    timm pillow matplotlib numpy scikit-learn \
    joblib pandas gdown transformers ultralytics \
    sahi accelerate datasets opencv-python evaluate tqdm scipy

# --- CONFIGURACIÓN DEL PROYECTO LITHOS ---
WORKDIR /app

# Crear estructura de carpetas para montajes
RUN mkdir -p weights/yolo weights/segformer /data_input /data_output checkpoints

# Descargar Checkpoint de SAM2 por defecto
RUN wget -P checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt

# --- SCRIPT DE VALIDACIÓN ---
# Creamos un pequeño script bash para validar los archivos antes de correr Python
RUN echo '#!/bin/bash\n\
echo "[*] Verificando integridad de pesos..." \n\
MISSING=0 \n\
if [ ! -f "/app/weights/yolo/best.pt" ]; then echo "[!] ERROR: No se encontró weights/yolo/best.pt"; MISSING=1; fi \n\
if [ ! -d "/app/weights/segformer/final" ] && [ ! -f "/app/weights/segformer/final/config.json" ]; then \n\
    echo "[!] ERROR: No se encontró la carpeta weights/segformer/final o está incompleta"; MISSING=1; \n\
fi \n\
if [ $MISSING -eq 1 ]; then \n\
    echo "[!] Abortando: Asegúrate de montar correctamente el volumen de weights."; exit 1; \n\
fi \n\
echo "[+] Pesos validados. Iniciando Inferencia..." \n\
exec python3 /app/Inference/main.py "$@"' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# El ENTRYPOINT ahora apunta al script de validación
ENTRYPOINT ["/app/entrypoint.sh"]

# Comando por defecto
CMD []