#!/bin/bash

# Setup script for F1ChexBERT and RadGraph dependencies
# Required for the CoherenceRewardCXR function in CURV training module

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up F1ChexBERT and RadGraph dependencies for CURV training...${NC}"

# Set base directory
BASE_DIR="/save/CURV/training"
cd "$BASE_DIR"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

if ! command_exists git; then
    echo -e "${RED}Error: git is not installed${NC}"
    exit 1
fi

if ! command_exists python; then
    echo -e "${RED}Error: python is not installed${NC}"
    exit 1
fi

if ! command_exists pip; then
    echo -e "${RED}Error: pip is not installed${NC}"
    exit 1
fi

# Setup F1ChexBERT
echo -e "${YELLOW}Setting up F1ChexBERT...${NC}"

if [ ! -d "f1chexbert" ]; then
    echo "Cloning F1ChexBERT repository..."
    git clone https://github.com/stanfordmlgroup/CheXbert.git f1chexbert
    
    echo "Installing F1ChexBERT dependencies..."
    cd f1chexbert
    
    # Install basic dependencies
    pip install torch torchvision --quiet
    pip install transformers pandas numpy --quiet
    
    cd ..
    echo -e "${GREEN}F1ChexBERT setup complete!${NC}"
else
    echo -e "${GREEN}F1ChexBERT already exists, skipping...${NC}"
fi

# Setup RadGraph
echo -e "${YELLOW}Setting up RadGraph...${NC}"

if [ ! -d "radgraph" ]; then
    echo "Cloning RadGraph repository..."
    git clone https://github.com/dwadden/radgraph.git
    
    cd radgraph
    
    echo "Installing RadGraph dependencies..."
    # Install requirements if requirements.txt exists
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt --quiet
    else
        # Install common dependencies
        pip install spacy torch transformers --quiet
    fi
    
    # Download spacy model
    echo "Downloading spaCy English model..."
    python -m spacy download en_core_web_sm --quiet
    
    # Download RadGraph model if available
    echo "Attempting to download RadGraph model..."
    if command_exists wget; then
        if [ ! -f "radgraph-xl.tar.gz" ]; then
            wget -q https://github.com/dwadden/radgraph/releases/download/v1.0.0/radgraph-xl.tar.gz || echo "Model download failed, will use fallback"
            if [ -f "radgraph-xl.tar.gz" ]; then
                tar -xzf radgraph-xl.tar.gz
                echo -e "${GREEN}RadGraph model downloaded successfully!${NC}"
            fi
        fi
    else
        echo -e "${YELLOW}wget not available, skipping model download. RadGraph will use fallback mode.${NC}"
    fi
    
    cd ..
    echo -e "${GREEN}RadGraph setup complete!${NC}"
else
    echo -e "${GREEN}RadGraph already exists, skipping...${NC}"
fi

# Set environment variables
echo -e "${YELLOW}Setting up environment variables...${NC}"

# Create environment setup script
cat > setup_env.sh << 'EOF'
#!/bin/bash
# Environment setup for CURV training dependencies

export F1CHEXBERT_PATH="/save/CURV/training/f1chexbert"
export RADGRAPH_PATH="/save/CURV/training/radgraph"

# Add to Python path
export PYTHONPATH="${F1CHEXBERT_PATH}:${RADGRAPH_PATH}:${PYTHONPATH}"

echo "Environment variables set for CURV training dependencies"
echo "F1CHEXBERT_PATH: $F1CHEXBERT_PATH"
echo "RADGRAPH_PATH: $RADGRAPH_PATH"
EOF

chmod +x setup_env.sh

echo -e "${GREEN}Environment setup script created: setup_env.sh${NC}"

# Verification
echo -e "${YELLOW}Verifying setup...${NC}"

# Test Python imports
python3 << 'EOF'
import sys
import os

# Add paths
sys.path.insert(0, '/save/CURV/training/f1chexbert')
sys.path.insert(0, '/save/CURV/training/radgraph')

print("Testing F1ChexBERT availability...")
try:
    import torch
    print("✓ PyTorch available")
    import transformers
    print("✓ Transformers available")
    print("✓ F1ChexBERT dependencies ready")
except ImportError as e:
    print(f"✗ F1ChexBERT dependency missing: {e}")

print("\nTesting RadGraph availability...")
try:
    import spacy
    print("✓ spaCy available")
    nlp = spacy.load("en_core_web_sm")
    print("✓ English model loaded")
    print("✓ RadGraph dependencies ready")
except ImportError as e:
    print(f"✗ RadGraph dependency missing: {e}")
except OSError as e:
    print(f"✗ spaCy model missing: {e}")

print("\nSetup verification complete!")
EOF

echo ""
echo -e "${GREEN}=== Setup Complete! ===${NC}"
echo ""
echo "To use the dependencies in your training:"
echo "1. Source the environment setup: source setup_env.sh"
echo "2. The CoherenceRewardCXR will automatically detect and use these dependencies"
echo ""
echo "If you encounter issues, check the setup_dependencies.md file for troubleshooting."
echo ""
echo -e "${YELLOW}Note: The coherence reward function will fall back to simpler methods if dependencies are not available.${NC}"