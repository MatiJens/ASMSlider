FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf_cache \
    HUGGINGFACE_HUB_CACHE=/opt/hf_cache/hub

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir \
    torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir biopython tqdm httpx pandas numpy h5py

RUN pip install --no-cache-dir scikit-learn xgboost lightgbm

RUN pip install --no-cache-dir optuna optuna-integration

RUN pip install --no-cache-dir matplotlib plotly kaleido seaborn

RUN pip install --no-cache-dir pytorch-lightning

RUN pip install --no-cache-dir esm

RUN pip install --no-cache-dir \
    --extra-index-url https://pypi.nvidia.com \
    cudf-cu12 \
    cuml-cu12

ENV TRANSFORMERS_CACHE=/tmp/transformers_cache

COPY version.txt /opt/hf_cache/hub/version.txt
COPY models--EvolutionaryScale--esmc-600m-2024-12 /opt/hf_cache/hub/models--EvolutionaryScale--esmc-600m-2024-12

WORKDIR /mnt

CMD ["/bin/bash"]