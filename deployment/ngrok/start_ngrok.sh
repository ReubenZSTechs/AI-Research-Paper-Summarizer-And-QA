#!/bin/bash

set -e

echo "Starting full LLM system with ngrok tunnel..."

if [ -z "$NGROK_AUTHTOKEN" ] && [ ! -f .env ]; then
  echo "Warning: NGROK_AUTHTOKEN not set and no .env file found."
  echo "Get your token at https://dashboard.ngrok.com/get-started/your-authtoken"
fi

docker-compose up -d --build

echo ""
echo "Waiting for ngrok tunnel to come up..."
sleep 5

echo "System running locally:"
echo "  Frontend: http://localhost:8080"
echo "  Backend:  http://localhost:8000"
echo "  LLM:      http://localhost:15000"
echo ""
echo "Public ngrok URL (check the dashboard for the exact address):"
echo "  http://localhost:4040"