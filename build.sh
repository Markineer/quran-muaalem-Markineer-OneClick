#!/usr/bin/env bash
# Build script for Render

set -o errexit

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Build frontend
cd frontend
npm install
npx vite build
cd ..

echo "Build completed successfully!"
