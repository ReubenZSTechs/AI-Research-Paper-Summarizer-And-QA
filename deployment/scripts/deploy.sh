#!/bin/bash

set -e

echo "Deploying system..."

docker compose pull
docker compose build
docker compose --env-file .env up

echo "Deployment complete."