#!/bin/bash
# SAM3 Installation Script
# Automatically detects platform (x86 or Jetson) and configures accordingly.
#
# Usage:
#   ./install.sh              # Install with notebook dependencies
#   ./install.sh --minimal    # Install without optional dependencies
#   ./install.sh --dev        # Install with development dependencies
#   ./install.sh --all        # Install all optional dependencies

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                   SAM3 Installation                        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Parse arguments
EXTRAS="notebooks"
while [[ $# -gt 0 ]]; do
    case $1 in
        --minimal)
            EXTRAS=""
            shift
            ;;
        --notebooks)
            EXTRAS="notebooks"
            shift
            ;;
        --dev)
            EXTRAS="dev,notebooks"
            shift
            ;;
        --all)
            EXTRAS="dev,notebooks,train"
            shift
            ;;
        --jetson)
            EXTRAS="jetson,notebooks"
            shift
            ;;
        -h|--help)
            echo "Usage: ./install.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --minimal     Install base package only"
            echo "  --notebooks   Install with notebook dependencies (default)"
            echo "  --dev         Install with development dependencies"
            echo "  --all         Install all optional dependencies"
            echo "  --jetson      Install with Jetson-specific dependencies"
            echo "  -h, --help    Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Detect platform
echo -e "${YELLOW}Detecting platform...${NC}"

if [ -f /etc/nv_tegra_release ]; then
    PLATFORM="jetson"
    PYTHON_CMD="python3.10"
    PYTORCH_INDEX="https://pypi.jetson-ai-lab.io/jp6/cu126"
    PYTORCH_VERSION="torch==2.8.0 torchvision==0.23.0"

    # Add jetson extra if not already included
    if [[ ! "$EXTRAS" == *"jetson"* ]]; then
        if [ -n "$EXTRAS" ]; then
            EXTRAS="jetson,$EXTRAS"
        else
            EXTRAS="jetson"
        fi
    fi

    echo -e "${GREEN}✓ Detected NVIDIA Jetson platform${NC}"

    # Show Jetson info
    if [ -f /proc/device-tree/model ]; then
        MODEL=$(cat /proc/device-tree/model | tr -d '\0')
        echo -e "  Device: ${MODEL}"
    fi
    L4T=$(head -1 /etc/nv_tegra_release)
    echo -e "  L4T: ${L4T}"
    echo -e "  Using Python 3.10 (NVIDIA PyTorch requirement)"
else
    PLATFORM="x86"
    PYTHON_CMD="python3.12"
    PYTORCH_INDEX="https://download.pytorch.org/whl/cu126"
    PYTORCH_VERSION="torch==2.7.0 torchvision torchaudio"

    echo -e "${GREEN}✓ Detected x86 platform${NC}"
    echo -e "  Using Python 3.12 (recommended)"
fi

# Check if Python is available
echo -e "\n${YELLOW}Checking Python installation...${NC}"
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo -e "${RED}Error: $PYTHON_CMD not found${NC}"
    echo -e "Please install Python first:"
    if [ "$PLATFORM" == "jetson" ]; then
        echo -e "  sudo apt install python3.10 python3.10-venv"
    else
        echo -e "  sudo apt install python3.12 python3.12-venv"
    fi
    exit 1
fi
echo -e "${GREEN}✓ Found $PYTHON_CMD${NC}"

# Create virtual environment
VENV_DIR=".venv"
echo -e "\n${YELLOW}Creating virtual environment...${NC}"
if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}  Existing .venv found, removing...${NC}"
    rm -rf "$VENV_DIR"
fi
$PYTHON_CMD -m venv "$VENV_DIR"
echo -e "${GREEN}✓ Created virtual environment at .venv${NC}"

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo -e "\n${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip --quiet
echo -e "${GREEN}✓ pip upgraded${NC}"

# Install PyTorch
echo -e "\n${YELLOW}Installing PyTorch from $PYTORCH_INDEX...${NC}"
pip install $PYTORCH_VERSION --index-url=$PYTORCH_INDEX --quiet
echo -e "${GREEN}✓ PyTorch installed${NC}"

# Verify CUDA
echo -e "\n${YELLOW}Verifying CUDA availability...${NC}"
CUDA_CHECK=$(python -c "import torch; print('available' if torch.cuda.is_available() else 'not available')" 2>/dev/null || echo "error")
if [ "$CUDA_CHECK" == "available" ]; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")
    echo -e "${GREEN}✓ CUDA is available (${GPU_NAME})${NC}"
elif [ "$CUDA_CHECK" == "not available" ]; then
    echo -e "${YELLOW}⚠ CUDA not available - running on CPU${NC}"
else
    echo -e "${RED}⚠ Could not verify CUDA${NC}"
fi

# Install SAM3
echo -e "\n${YELLOW}Installing SAM3...${NC}"
if [ -n "$EXTRAS" ]; then
    pip install -e ".[$EXTRAS]" --quiet
    echo -e "${GREEN}✓ SAM3 installed with extras: $EXTRAS${NC}"
else
    pip install -e . --quiet
    echo -e "${GREEN}✓ SAM3 installed (minimal)${NC}"
fi

# Verify installation
echo -e "\n${YELLOW}Verifying SAM3 installation...${NC}"
IMPORT_CHECK=$(python -c "from sam3 import build_sam3_image_model; print('ok')" 2>/dev/null || echo "error")
if [ "$IMPORT_CHECK" == "ok" ]; then
    echo -e "${GREEN}✓ SAM3 imports successfully${NC}"
else
    echo -e "${RED}⚠ SAM3 import failed - check installation${NC}"
fi

# Print summary
echo -e "\n${BLUE}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                 Installation Complete!                      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo -e "To activate the environment:"
echo -e "  ${GREEN}source .venv/bin/activate${NC}"
echo -e ""
echo -e "To download model checkpoints:"
echo -e "  ${GREEN}huggingface-cli login${NC}"
echo -e "  ${GREEN}huggingface-cli download facebook/sam3 --local-dir ./checkpoints${NC}"
echo -e ""
echo -e "To run the example notebooks:"
echo -e "  ${GREEN}jupyter notebook examples/${NC}"
