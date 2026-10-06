#!/bin/sh
set -e

NETWORK_NAME="colibri-net"
OLLAMA_CONTAINER="ollama"
IMAGE_NAME="jobhunt:0.0.1"

# Create shared network if it doesn't already exist
docker network create "$NETWORK_NAME" 2>/dev/null || true

# Start or ensure ollama container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${OLLAMA_CONTAINER}$"; then
    if docker ps -a --format '{{.Names}}' | grep -q "^${OLLAMA_CONTAINER}$"; then
        echo "[run.sh] Starting existing ${OLLAMA_CONTAINER} container..."
        docker start "$OLLAMA_CONTAINER"
    else
        echo "[run.sh] Creating and starting ${OLLAMA_CONTAINER} container..."
        docker run -d \
            --name "$OLLAMA_CONTAINER" \
            --gpus all \
            -v ollama:/root/.ollama \
            -p 11434:11434 \
            --restart unless-stopped \
            ollama/ollama
    fi
fi

# Ensure ollama is connected to our shared network
docker network connect "$NETWORK_NAME" "$OLLAMA_CONTAINER" 2>/dev/null || true

# Clean up any leftover agent container with the same name
docker rm -f jobhunt-agent 2>/dev/null || true

echo "[run.sh] Running ${IMAGE_NAME} agent container..."
docker run --rm \
    --name jobhunt-agent \
    --network "$NETWORK_NAME" \
    -v /mnt/colibri:/mnt/colibri \
    -e API_BASE="http://${OLLAMA_CONTAINER}:11434" \
    -e OLLAMA_API_BASE="http://${OLLAMA_CONTAINER}:11434" \
    "$IMAGE_NAME"
