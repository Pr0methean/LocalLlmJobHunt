FROM nvidia/cuda:13.4.1-runtime-ubuntu26.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential python3-dev python3-pip python3-venv
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "smolagents[litellm]" Pillow
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /mnt/colibri
CMD ["python3", "/mnt/colibri/entrypoint.py"]

