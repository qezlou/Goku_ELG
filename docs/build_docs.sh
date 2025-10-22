#!/bin/bash
# Build script for Goku-ELG documentation

set -e  # Exit on error

echo "======================================"
echo "Goku-ELG Documentation Build Script"
echo "======================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if we're in the docs directory
if [ ! -f "Makefile" ]; then
    echo -e "${RED}Error: Please run this script from the docs/ directory${NC}"
    exit 1
fi

# Function to print colored status messages
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check for Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed"
    exit 1
fi
print_status "Python 3 found"

# Check if virtual environment exists or create one
if [ ! -d "venv" ]; then
    print_warning "Virtual environment not found. Creating one..."
    python3 -m venv venv
    print_status "Virtual environment created"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Install/upgrade requirements
print_status "Installing/upgrading documentation requirements..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1
print_status "Requirements installed"

# Clean previous builds
print_status "Cleaning previous builds..."
make clean > /dev/null 2>&1
print_status "Previous builds cleaned"

# Build HTML documentation
echo ""
echo "Building HTML documentation..."
if make html; then
    print_status "HTML documentation built successfully!"
    echo ""
    echo "Documentation is available at: build/html/index.html"
    echo "To view, run: python -m http.server --directory build/html 8000"
else
    print_error "HTML build failed"
    exit 1
fi

# Ask if user wants to build PDF
echo ""
read -p "Do you want to build PDF documentation? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Building PDF documentation..."
    if make latexpdf; then
        print_status "PDF documentation built successfully!"
        echo "PDF is available at: build/latex/goku-elg.pdf"
    else
        print_warning "PDF build failed (this requires LaTeX installation)"
    fi
fi

echo ""
echo "======================================"
echo "Build complete!"
echo "======================================"
