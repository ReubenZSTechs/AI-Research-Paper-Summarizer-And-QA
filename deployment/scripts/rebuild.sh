#!/bin/bash

echo "Rebuilding system from scratch..."

docker compose down -v
docker compose build --no-cache

echo "Rebuild complete."