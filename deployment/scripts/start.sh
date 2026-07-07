#!/bin/bash

echo "Starting full LLM system..."

docker-compose up -d --build

echo "System running:"
echo "Frontend: http://localhost:8501"
echo "Backend: http://localhost:8000"
echo "Nginx: http://localhost:80"