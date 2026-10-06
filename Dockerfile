FROM python:3-trixie

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "smolagents[litellm]" Pillow requests

WORKDIR /mnt/colibri
CMD ["python3", "/mnt/colibri/entrypoint.py"]

