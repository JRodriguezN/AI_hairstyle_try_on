#!/bin/bash
set -e

echo "🔄 Actualizando repositorio..."
cd /app
git pull origin main

echo "🔨 Construyendo imagen Docker..."
docker build -t hair-app:latest .

echo "🛑 Deteniendo contenedor anterior..."
docker stop hair-app || true

echo "🧹 Eliminando contenedor anterior..."
docker rm hair-app || true

echo "🚀 Iniciando nuevo contenedor..."
docker run -d --name hair-app -p 80:8000 --env-file .env hair-app:latest

echo "✅ Deploy completado!"