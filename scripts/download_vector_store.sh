#!/bin/bash
# Download and extract vector store on Render deployment

set -e  # Exit on error

echo "🔍 Checking for vector store..."

if [ -d "data/vector_store" ] && [ "$(ls -A data/vector_store)" ]; then
    echo "✅ Vector store already exists, skipping download"
    exit 0
fi

if [ -z "$VECTOR_STORE_URL" ]; then
    echo "❌ ERROR: VECTOR_STORE_URL environment variable not set!"
    echo "Please add it in Render dashboard environment variables"
    exit 1
fi

echo "📥 Downloading vector store from cloud..."
mkdir -p data/vector_store
curl -L "$VECTOR_STORE_URL" -o vector_store_backup.tar.gz

echo "📦 Extracting vector store..."
tar -xzf vector_store_backup.tar.gz

echo "🧹 Cleaning up..."
rm vector_store_backup.tar.gz

echo "✅ Vector store ready!"
ls -lh data/vector_store/
