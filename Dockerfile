FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf_cache \
    HUGGINGFACE_HUB_CACHE=/opt/hf_cache/hub

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir \
    torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
    numpy \
    scikit-learn \
    matplotlib \
    seaborn \
    biopython

RUN pip install --no-cache-dir \
    git+https://github.com/evolutionaryscale/esm.git@v3.2.3

COPY models--EvolutionaryScale--esmc-600m-2024-12 /opt/hf_cache/hub/models--EvolutionaryScale--esmc-600m-2024-12

WORKDIR /mnt

CMD ["/bin/bash"]
