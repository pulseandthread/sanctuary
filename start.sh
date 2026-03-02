#!/bin/bash

echo ""
echo "  Starting Sanctuary..."
echo ""

# Check .env exists
if [ ! -f ".env" ]; then
    echo "  ERROR: No .env file found!"
    echo "  Please run ./setup.sh first."
    exit 1
fi

# Check venv exists
if [ ! -d "venv" ]; then
    echo "  ERROR: Virtual environment not found!"
    echo "  Please run ./setup.sh first."
    exit 1
fi

# Activate and run
source venv/bin/activate

echo "  Opening Sanctuary in your browser..."

# Open browser (works on macOS and Linux)
if command -v open &> /dev/null; then
    open http://localhost:5000
elif command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:5000
else
    echo "  Open http://localhost:5000 in your browser"
fi

python3 app.py
