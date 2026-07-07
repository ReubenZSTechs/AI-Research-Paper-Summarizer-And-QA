#!/bin/bash

set -e

echo "Deploying system..."

docker-compose pull
docker-compose build
docker-compose up -d

echo "Deployment complete."