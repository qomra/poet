FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

WORKDIR /app

# Install torch with CUDA 12.4 wheels first (large), then the rest
RUN pip install --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.5.1

COPY requirements.txt /app/
RUN pip install -r requirements.txt

COPY server.py /app/

EXPOSE 8502

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8502"]
