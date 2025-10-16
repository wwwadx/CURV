# Setting up F1ChexBERT and RadGraph Dependencies

This document provides instructions for setting up the F1ChexBERT and RadGraph dependencies required for the coherence reward function in the CURV training module.

## Overview

The `CoherenceRewardCXR` function uses both F1ChexBERT and RadGraph models to extract medical entities and calculate coherence between different sections of medical reports. These dependencies need to be properly installed and configured.

## F1ChexBERT Setup

F1ChexBERT is used for medical entity extraction and classification in chest X-ray reports.

### Installation

1. **Clone the F1ChexBERT repository:**
   ```bash
   cd /save/CURV/training
   git clone https://github.com/stanfordmlgroup/CheXbert.git f1chexbert
   cd f1chexbert
   ```

2. **Install dependencies:**
   ```bash
   pip install torch torchvision
   pip install transformers
   pip install pandas numpy
   ```

3. **Download the pre-trained model:**
   ```bash
   # The model will be automatically downloaded when first used
   # Or manually download from: https://stanfordmedicine.box.com/s/c3stck6w6dol3h36grdc97xoydzxd7w9
   ```

### Configuration

Set the F1ChexBERT model path in your training configuration:

```python
# In your training script or config
import os
f1chexbert_path = "/save/CURV/training/f1chexbert"
os.environ['F1CHEXBERT_PATH'] = f1chexbert_path
```

## RadGraph Setup

RadGraph is used for extracting structured information from radiology reports.

### Installation

1. **Clone the RadGraph repository:**
   ```bash
   cd /save/CURV/training
   git clone https://github.com/dwadden/radgraph.git
   cd radgraph
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install spacy
   python -m spacy download en_core_web_sm
   ```

3. **Download the pre-trained model:**
   ```bash
   # Download the RadGraph model
   wget https://github.com/dwadden/radgraph/releases/download/v1.0.0/radgraph-xl.tar.gz
   tar -xzf radgraph-xl.tar.gz
   ```

### Configuration

Set the RadGraph model path in your training configuration:

```python
# In your training script or config
import os
radgraph_path = "/save/CURV/training/radgraph"
os.environ['RADGRAPH_PATH'] = radgraph_path
```

## Environment Setup Script

Create a setup script to automatically configure both dependencies:

```bash
#!/bin/bash
# setup_dependencies.sh

# Set base directory
BASE_DIR="/save/CURV/training"
cd $BASE_DIR

# Setup F1ChexBERT
echo "Setting up F1ChexBERT..."
if [ ! -d "f1chexbert" ]; then
    git clone https://github.com/stanfordmlgroup/CheXbert.git f1chexbert
fi

# Setup RadGraph
echo "Setting up RadGraph..."
if [ ! -d "radgraph" ]; then
    git clone https://github.com/dwadden/radgraph.git
    cd radgraph
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    
    # Download model if not exists
    if [ ! -f "radgraph-xl.tar.gz" ]; then
        wget https://github.com/dwadden/radgraph/releases/download/v1.0.0/radgraph-xl.tar.gz
        tar -xzf radgraph-xl.tar.gz
    fi
    cd ..
fi

echo "Dependencies setup complete!"
```

## Usage in Training

The coherence reward function will automatically use these dependencies when available:

```python
from CURV.training.reward_functions import CoherenceRewardCXR

# The reward function will automatically detect and use the installed dependencies
coherence_reward = CoherenceRewardCXR()

# Use in training configuration
config.reward_functions = ["format_cxr", "accuracy_cxr", "coherence_cxr"]
config.reward_weights = {
    "format_cxr": 0.33,
    "accuracy_cxr": 0.33,
    "coherence_cxr": 0.34
}
```

## Troubleshooting

### Common Issues

1. **Import errors:** Ensure all dependencies are installed in the correct Python environment
2. **Model download failures:** Check internet connection and disk space
3. **Path issues:** Verify that the model paths are correctly set in environment variables

### Fallback Mode

If the dependencies are not available, the coherence reward function will fall back to a simpler entity extraction method using basic NLP techniques.

### Verification

To verify the setup is working correctly:

```python
# Test F1ChexBERT
try:
    from f1chexbert.src.models.bert_labeler import bert_labeler
    print("F1ChexBERT setup successful")
except ImportError:
    print("F1ChexBERT not available - using fallback")

# Test RadGraph
try:
    import radgraph
    print("RadGraph setup successful")
except ImportError:
    print("RadGraph not available - using fallback")
```

## Performance Notes

- F1ChexBERT and RadGraph models require significant GPU memory
- Consider using CPU-only versions for development/testing
- The coherence calculation may be slower with full models but more accurate

## License and Attribution

- F1ChexBERT: Please refer to the original repository for license information
- RadGraph: Please refer to the original repository for license information
- Ensure proper attribution when using these models in research or production