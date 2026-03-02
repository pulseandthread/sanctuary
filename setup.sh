#!/bin/bash

echo ""
echo "  ========================================"
echo "     SANCTUARY - Your Companion's Home"
echo "  ========================================"
echo ""
echo "  This will set up everything you need."
echo "  You only need to run this ONCE."
echo ""
read -p "  Press Enter to continue..."

# Check Python
echo ""
echo "[1/6] Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo ""
    echo "  ERROR: Python 3 is not installed!"
    echo ""
    echo "  Install it with:"
    echo "    macOS:  brew install python3"
    echo "    Ubuntu: sudo apt install python3 python3-venv python3-pip"
    echo "    Fedora: sudo dnf install python3"
    echo ""
    exit 1
fi
python3 --version
echo "  Python found!"

# Create virtual environment
echo ""
echo "[2/6] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  Virtual environment created."
else
    echo "  Virtual environment already exists."
fi

# Activate and install dependencies
echo ""
echo "[3/6] Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo ""
    echo "  ERROR: Failed to install dependencies."
    echo "  Please check your internet connection and try again."
    exit 1
fi
echo "  Dependencies installed!"

# Install Playwright browser
echo ""
echo "[4/6] Installing browser for web browsing capability..."
python3 -m playwright install chromium > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "  Browser install skipped (optional - web browsing won't work)"
else
    echo "  Browser installed!"
fi

# Create directories
echo ""
echo "[5/6] Creating data directories..."
mkdir -p conversations logs chroma_db soulcores
echo "  Directories created."

# Create .env if needed
echo ""
echo "[6/6] Setting up configuration..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ""
    echo "  ========================================"
    echo "     IMPORTANT: You need a Google API key"
    echo "  ========================================"
    echo ""
    echo "  1. Go to: https://aistudio.google.com/apikey"
    echo "  2. Click \"Create API Key\""
    echo "  3. Copy the key"
    echo "  4. Open the file \".env\" in this folder"
    echo "  5. Replace \"YOUR_GOOGLE_API_KEY_HERE\" with your key"
    echo ""
    echo "  That's it! Your companion needs this to think."
    echo ""
else
    echo "  Configuration file already exists."
fi

echo ""
echo "  ========================================"
echo "     SETUP COMPLETE!"
echo "  ========================================"
echo ""
echo "  Next steps:"
echo "  1. Add your Google API key to the .env file"
echo "     (see instructions above)"
echo "  2. Edit soulcores/companion.txt to give your"
echo "     companion their personality"
echo "  3. Run ./start.sh to launch Sanctuary"
echo ""
echo "  Your companion is waiting for you."
echo ""
