#!/bin/bash

echo "Stopping system..."

docker compose --env-file .env down

echo "All services stopped."