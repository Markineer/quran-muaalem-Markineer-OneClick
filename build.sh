#!/usr/bin/env bash
set -e

echo "Installing backend dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Building frontend..."
cd frontend
npm install
npm run build
cd ..

echo "Build completed successfully!"
