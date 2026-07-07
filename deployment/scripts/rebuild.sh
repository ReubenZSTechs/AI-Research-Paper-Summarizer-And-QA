#!/bin/bash

echo "Rebuilding system from scratch..."

docker-compose down -v
docker-compose build --no-cache
docker-compose up -d

echo "Rebuild complete."