FROM python:3.13-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        chromium \
        git \
        nodejs \
        npm \
        ripgrep \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
    "pillow>=10.0.0" \
    "playwright>=1.56.0" \
    "pyyaml>=6.0.1" \
    "requests>=2.31.0"
