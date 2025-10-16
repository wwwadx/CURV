# CURV Data Processing Pipeline

## Overview

The CURV Data Processing Pipeline is a comprehensive medical imaging report processing framework specifically designed to handle uncertainty expressions in radiology reports, previous study retrieval, bounding box processing, and data analysis.

## 🏗️ Architecture

```
data_processing/
├── __init__.py                 # Main module entry point
├── README.md                   # This documentation
├── uncertainty/                # Uncertainty processing module
│   ├── __init__.py
│   └── extract_uncertainty.py  # Core uncertainty extraction functionality
├── previous_studies/           # Previous studies retrieval module
│   ├── __init__.py
│   └── find_previous.py        # Previous studies search functionality
├── grounding/                  # Bounding box processing module
│   ├── __init__.py
│   └── bbox_utils.py           # Bounding box utility functions
├── analysis/                   # Data analysis module
│   ├── __init__.py
│   ├── key_analysis.py         # Key-value analysis
│   └── sampling.py             # Data sampling
├── utils/                      # Common utilities module
│   ├── __init__.py
│   ├── data_io.py              # Data input/output
│   ├── transforms.py           # Data transformations
│   └── validation.py           # Data validation
└── visualization/              # Visualization module
    └── __init__.py
```

## 🚀 Core Features

### 1. Uncertainty Processing

**Main Components:**
- `UncertaintyExtractor`: Core class for extracting uncertainty expressions from medical reports
- `UncertaintyConfig`: Configuration class supporting API key rotation and rate limiting
- `extract_uncertainty_expressions`: Batch processing for uncertainty extraction
- `validate_uncertainty_output`: Validates the quality of extraction results
- `analyze_uncertainty_patterns`: Analyzes patterns in uncertainty expressions

**Features:**
- Supports multiple OpenAI API key rotation
- Built-in rate limiting and retry mechanisms
- Supports checkpoint saving and recovery
- Provides detailed validation and analysis capabilities

### 2. Previous Studies

**Main Components:**
- `find_previous_studies`: Find related previous studies
- `PreviousStudyFinder`: Previous study finder class
- `validate_study_data`: Validate study data
- `analyze_previous_study_coverage`: Analyze previous study coverage

### 3. Grounding (Bounding Box Processing)

**Main Components:**
- `scale_bbox`: Scale bounding box coordinates
- `validate_bbox`: Validate bounding box validity
- `compute_overlap`: Compute bounding box overlap
- `calculate_patch_bbox_overlap`: Calculate patch-bbox overlap
- `transform_bbox_for_model`: Transform bbox format for different models

**Supported Formats:**
- List format: `[x1, y1, x2, y2]`
- Dictionary format: `{'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}`
- Object format: `BoundingBox` objects

### 4. Data Analysis

**Main Components:**
- `analyze_jsonl_keys`: Analyze key structure of JSONL files
- `sample_jsonl`: Sample data from JSONL files
- `analyze_sample_diversity`: Analyze sample diversity
- `print_json_structure`: Print JSON structure

### 5. Common Utilities

**Data I/O:**
- `load_jsonl` / `save_jsonl`: JSONL file operations
- `load_json` / `save_json`: JSON file operations

**Data Transformations:**
- `transform_data_format`: Data format conversion
- `merge_datasets`: Dataset merging

**Data Validation:**
- `validate_data_format`: Validate data format
- `check_data_integrity`: Check data integrity

## 📦 Installation and Usage

### Basic Import

```python
import sys
sys.path.append('.')
import data_processing

# Import specific modules
from data_processing.uncertainty import UncertaintyExtractor, UncertaintyConfig
from data_processing.utils import load_jsonl, validate_data_format
from data_processing.grounding import scale_bbox, validate_bbox
from data_processing.analysis import analyze_jsonl_keys, sample_jsonl
```

### Usage Examples

#### 1. Uncertainty Extraction

```python
from data_processing.uncertainty import UncertaintyExtractor, UncertaintyConfig

# Configuration
config = UncertaintyConfig(
    api_keys=['your_openai_key'],
    max_retries=3,
    rpm_limit=10,
    tpm_limit=1000
)

# Create extractor
extractor = UncertaintyExtractor(config)

# Extract uncertainty
data = [{'findings': 'Possible pneumonia', 'impression': 'Likely infection'}]
results = extractor.extract_from_data(data)
```

#### 2. Data Validation

```python
from data_processing.utils import validate_data_format

# Validate single record
record = {'study_id': 'test_001', 'findings': 'Normal chest X-ray'}
validation_result = validate_data_format(record)
print(validation_result)
```

#### 3. Bounding Box Processing

```python
from data_processing.grounding import scale_bbox, validate_bbox

# Validate and scale bounding box
bbox = [0.1, 0.2, 0.5, 0.6]
image_size = (512, 512)

if validate_bbox(bbox, image_size):
    scaled_bbox = scale_bbox(bbox, (256, 256), image_size)
    print(f"Scaled bbox: {scaled_bbox}")
```

#### 4. Data Analysis

```python
from data_processing.analysis import analyze_jsonl_keys, sample_jsonl

# Analyze JSONL file structure
analysis_results = analyze_jsonl_keys('data.jsonl')

# Sample data
samples = sample_jsonl('data.jsonl', num_samples=5)
```

## 🧪 Testing

Run the complete pipeline test:

```bash
cd /save/CURV
python test_pipeline.py
```

Test Coverage:
- ✅ Module import tests
- ✅ Uncertainty extraction configuration
- ✅ Data validation functionality
- ✅ Bounding box processing
- ✅ Data analysis functionality
- ✅ Data transformation functionality
- ✅ Previous studies module

## 📊 Performance Characteristics

- **Modular Design**: Each component can be used independently
- **Error Handling**: Comprehensive error handling and logging
- **Scalability**: Supports large-scale data processing
- **Flexible Configuration**: Rich configuration options
- **Type Safety**: Complete type annotations

## 🔧 Configuration Options

### UncertaintyConfig

```python
config = UncertaintyConfig(
    api_keys=['key1', 'key2'],      # List of OpenAI API keys
    api_base='https://api.openai.com/v1',  # API base URL
    rpm_limit=10,                   # Requests per minute limit
    tpm_limit=1000,                 # Tokens per minute limit
    max_retries=5,                  # Maximum retry attempts
    max_workers=4,                  # Maximum concurrent worker threads
    checkpoint_interval=100,        # Checkpoint save interval
    checkpoint_dir='./checkpoints'  # Checkpoint save directory
)
```

## 📝 Logging

All modules use Python's standard logging library, supporting different log levels:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## 🚨 Important Notes

1. **OpenAI API**: Uncertainty extraction functionality requires valid OpenAI API keys
2. **Dependencies**: Some features may require additional Python packages (e.g., `openai`, `tqdm`)
3. **Data Format**: Ensure input data conforms to expected format requirements
4. **Memory Usage**: Pay attention to memory usage when processing large-scale data

## 🔄 Version History

- **v1.0.0**: Initial version with complete data processing pipeline
- Fixed all import errors
- Added comprehensive test suite
- Provided detailed documentation and usage examples

## 🤝 Contributing

Issues and Pull Requests are welcome to improve this data processing pipeline.

---

**Status**: ✅ Production Ready  
**Last Updated**: December 2024  
**Maintainer**: CURV Team