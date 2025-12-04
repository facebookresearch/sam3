# Dockerfile pour SAM3 (Segment Anything with Concepts)
# Image de base PyTorch avec CUDA 12.6
FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

# Métadonnées
LABEL maintainer="Antoine TADROS"
LABEL description="SAM3 for Preven6"
LABEL version="0.1.0"

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch

# Répertoire de travail
WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers de configuration d'abord (pour le cache Docker)
COPY pyproject.toml README.md LICENSE ./
COPY sam3/__init__.py sam3/__init__.py
COPY .env ./

# Installer les dépendances Python
RUN pip install --no-cache-dir \
    timm>=1.0.17 \
    "numpy==1.26.*" \
    tqdm \
    ftfy==6.1.1 \
    regex \
    iopath>=0.1.10 \
    typing_extensions \
    huggingface_hub \
    psutil \
    pycocotools \
    decord \
    einops \
    opencv-python

# Copier le reste du code source
COPY . .

# Installer SAM3 en mode éditable
RUN pip install --no-cache-dir -e .

# Créer les répertoires de cache
RUN mkdir -p /app/.cache/huggingface /app/.cache/torch

# Créer un utilisateur non-root pour la sécurité
RUN useradd -m -u 1000 sam3user && \
    chown -R sam3user:sam3user /app
USER sam3user

# Port par défaut (si vous créez un serveur d'inférence)
EXPOSE 8000

# Healthcheck basique
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import sam3; print('OK')" || exit 1

# Point d'entrée par défaut - Python interactif
# Remplacez par votre script de serveur si nécessaire
CMD ["python", "-c", "from sam3 import build_sam3_image_model; print('SAM3 prêt')"]
