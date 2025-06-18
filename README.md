# Financial EPS Parser

A robust Python parser for extracting GAAP (Generally Accepted Accounting Principles) earnings per share values from HTML financial documents such as 10-Q and 10-K filings.

## Overview

This parser uses a multi-method approach combining table parsing, regex pattern matching, and Named Entity Recognition (NER) to reliably extract EPS values while filtering out non-GAAP adjustments. The system employs a weighted voting mechanism to select the most accurate result across methods.

## Key Features

- **Multi-Strategy Extraction**: Combines structural table parsing, intelligent regex, and NLP-based entity recognition
- **GAAP Filtering**: Automatically identifies and excludes non-GAAP adjustments and pro forma values
- **Consensus Algorithm**: Weighted voting system prioritizes more reliable extraction methods
- **Financial Intelligence**: Understands financial statement structures, loss reporting conventions, and current period identification
- **Production Ready**: Comprehensive error handling, logging, and batch processing capabilities

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd financial-eps-parser

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

## Usage

### Command Line Interface

```bash
# Basic usage
python eps_parser.py input_folder/ output.csv

# With verbose logging
python eps_parser.py input_folder/ output.csv --verbose

# Custom debug log location
python eps_parser.py input_folder/ output.csv --debug-log custom_log.txt
```

### Python API

```python
from eps_parser import EPSExtractor

# Initialize extractor
extractor = EPSExtractor()

# Parse single file
eps_value = extractor.parse_file("path/to/document.html")

# Batch process directory
from eps_parser import process_directory
process_directory("input_folder/", "results.csv")
```

## Architecture

### Extraction Methods

1. **Table Parser**: Analyzes HTML table structures to find EPS data in financial statements
2. **Regex Engine**: Uses pattern matching with contextual scoring for text-based extraction
3. **NER System**: Leverages spaCy's named entity recognition for monetary value identification

### Scoring Algorithm

Each extraction method assigns scores based on:
- **Label Quality**: "basic earnings per share" > "diluted earnings per share"
- **Context Relevance**: Current period indicators, financial terminology
- **GAAP Compliance**: Penalties for non-GAAP adjustments
- **Data Quality**: Proximity to expected value ranges

### Consensus Mechanism

The final EPS value is selected through weighted voting:
- Table parsing: Weight 3 (highest reliability)
- Regex extraction: Weight 2 
- NER extraction: Weight 1
- Tie-breaking by method priority and absolute value

## Example Output

```csv
filename,EPS
company_q1_2024.html,1.23
company_q2_2024.html,1.45
company_q3_2024.html,-0.67
```

## Configuration

Key parameters can be adjusted in the script:

```python
# Value constraints
MAX_REASONABLE_EPS = 50.0  # Flag unreasonably high values

# Fuzzy matching threshold
FUZZY_MATCH_THRESHOLD = 85  # 0-100 similarity score

# Keywords for filtering
SKIP_KEYWORDS = ["adjusted", "non-gaap", "pro forma", ...]
```

## Logging and Debugging

The parser generates detailed debug logs showing:
- Extraction method results for each file
- Scoring details and candidate selection
- Error handling and edge cases
- Final consensus voting process

## Quantitative Finance Applications

This parser demonstrates several concepts relevant to quantitative finance:

- **Multi-Factor Models**: Combining multiple signals (extraction methods) with different weights
- **Consensus Mechanisms**: Similar to analyst consensus or factor model ensembles
- **Data Quality Scoring**: Systematic approach to evaluating information reliability
- **Systematic Processing**: Automated, repeatable methodology for large datasets

## Performance Considerations

- **Memory Efficient**: Processes files individually to handle large datasets
- **Error Resilient**: Continues processing even if individual files fail
- **Configurable**: Adjustable parameters for different document types
- **Scalable**: Designed for batch processing of hundreds of documents

## Dependencies

- `beautifulsoup4`: HTML parsing and table extraction
- `spacy`: Natural language processing and NER
- `fuzzywuzzy`: Fuzzy string matching for keyword detection
- `python-levenshtein`: Optimized string distance calculations

## License

This code is provided as a demonstration of financial data processing capabilities.

---

*Developed for quantitative finance applications requiring reliable automated extraction of financial metrics from unstructured documents.*
