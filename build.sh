#!/bin/sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[build.sh] Pulling ollama/ollama base image..."
docker pull ollama/ollama

echo "[build.sh] Building jobhunt:0.0.1 image..."
docker build -t jobhunt:0.0.1 "$SCRIPT_DIR"
