#!/bin/bash
# Quick script to view documentation in browser

cd "$(dirname "$0")"

if [ ! -d "build/html" ]; then
    echo "Documentation not built yet. Building now..."
    python -m sphinx -M html source build
fi

echo "Starting local web server..."
echo "Documentation will be available at: http://localhost:8000"
echo "Press Ctrl+C to stop the server"
echo ""

python -m http.server --directory build/html 8000
