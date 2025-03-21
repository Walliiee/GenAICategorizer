# GenAI Conversation Analyzer

This tool analyzes and categorizes GenAI conversations using two approaches:
1. **Keyword-based categorization**: Uses predefined categories and keywords 
2. **Topic modeling**: Uses BERTopic to discover natural topics in the data

The system supports multilingual analysis (English and Danish) using the `paraphrase-multilingual-mpnet-base-v2` model.

## Features

- Multilingual support for English and Danish conversations
- Topic discovery using BERTopic with multilingual embeddings
- Keyword-based categorization with predefined categories
- Special handling for voice conversations
- Rich visualizations including:
  - Topic distribution charts
  - Category comparison
  - Topic-to-category mapping Sankey diagrams
  - Interactive dashboards

## System Architecture

The system consists of the following modules:

- `main.py`: Coordinates the workflow and combines results
- `categorization.py`: Implements the keyword-based categorization
- `topic_modeling.py`: Implements BERTopic-based topic modeling
- `visualization.py`: Creates visualizations for the results
- `data_processing.py`: Handles data preprocessing and cleaning
- `embedding.py`: Manages text embedding generation
- `clustering.py`: Implements clustering functionality
- `example.py`: Provides example usage and demonstrations

## Testing

The project includes unit tests for key components:
- `test_topic_modeling.py`: Tests for topic modeling functionality
- `test_categorization.py`: Tests for categorization functionality

Run tests using:
```bash
python -m pytest src/test_*.py
```

## Installation

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`

## Usage

The analysis workflow consists of two steps:

1. **Data Preprocessing** (required first step):
```bash
python src/data_processing.py
```
This processes raw JSON files from `/data/raw` and creates a cleaned CSV file at `/data/processed/cleaned_conversations.csv`.

2. **Main Analysis**:
```bash
# Run the full workflow with default settings
python src/main.py

# Run with custom settings
python src/main.py --input-file path/to/data.csv --min-topic-size 15 --nr-topics 20
```

### Command-line Arguments

- `--input-file`: Path to the cleaned conversation CSV file
- `--min-topic-size`: Minimum size of topics for BERTopic (default: 10)
- `--nr-topics`: Number of topics to generate ("auto" or a number, default: "auto")
- `--output-dir`: Directory to save analysis outputs (default: "outputs/[timestamp]")

## Best Practices

### Managing Outputs

To preserve results from different analysis runs:

```bash
# Save output to a timestamped directory
python src/main.py --output-dir outputs/$(date +%Y%m%d_%H%M%S)

# Save output to a named directory
python src/main.py --output-dir outputs/experiment_name
```

### Configuration Management

Keep your category definitions and other configurations separate from code:

1. Create a config directory:
```bash
mkdir -p config
```

2. Move categories to a JSON file:
```json
# config/categories.json
{
  "Technical Support": {
    "keywords": ["error", "bug", "fix", "issue", "problem"],
    "subcategories": {
      "Installation": ["install", "setup", "download"],
      "Performance": ["slow", "speed", "memory", "cpu"]
    }
  }
}
```

3. Load configuration in code:
```python
import json
with open("config/categories.json", "r") as f:
    CATEGORIES = json.load(f)
```

## Output

The system generates several output files:

- `keyword_categorized_conversations.csv`: Results from keyword-based categorization
- `topic_modeled_conversations.csv`: Results from topic modeling
- `combined_results.csv`: Combined results from both methods
- `keyword_metrics.json`: Metrics from keyword-based categorization
- `topic_data.json`: Detailed information about discovered topics
- `combined_metrics.json`: Metrics from the combined analysis
- Various visualization files in HTML format

## Interpreting Results

### Understanding Topic Modeling

The topic modeling process discovers natural patterns in your conversations, which may not perfectly align with your predefined categories. This can be valuable for:

1. **Discovering unexpected patterns**: Topics might reveal use cases you hadn't considered
2. **Refining categories**: You can update your keyword-based categories based on discovered topics
3. **Understanding multilingual patterns**: The system works across languages (English/Danish)

### Analyzing Agreement

The agreement rate between keyword-based and topic-based categorization is insightful:

- **High agreement** (>70%): Your predefined categories match natural language patterns well
- **Medium agreement** (30-70%): Some categories match well, others may need refinement
- **Low agreement** (<30%): Consider revising your category definitions or adding new categories

### Key Metrics to Monitor

- **Topic coherence**: How semantically coherent are the discovered topics?
- **Category distribution**: Are conversations evenly distributed or concentrated in few categories?
- **Outlier rate**: High percentage of outliers suggests topics aren't capturing conversation patterns well

### Improving Results

To improve categorization quality:

1. **Adjust min_topic_size**: Lower values find more specific topics, higher values find broader patterns
2. **Update keywords**: Add discovered topic words to your category keywords
3. **Add missing categories**: Create new categories based on discovered topics
4. **Iterate**: Run the analysis multiple times, refining after each run

## How It Works

1. **Embedding Generation**: Conversations are encoded using a multilingual SentenceTransformer model
2. **Topic Modeling**: BERTopic is applied to discover natural topics in the data
3. **Topic Analysis**: Topics are analyzed to extract meaningful keywords and representative examples
4. **Categorization**: 
   - Keyword-based categorization assigns categories based on predefined keywords
   - Topics are mapped to categories by comparing topic keywords with category definitions
5. **Visualization**: Interactive visualizations are created to explore the results
6. **Comparison**: The two categorization methods are compared to identify agreements and differences

## Recommended Project Structure

For better organization and reproducibility, it's recommended to organize the project as follows:

```
GenAICategorizer/
├── src/              # Source code only
│   ├── *.py          # Python modules
│   └── tests/        # Unit tests
├── data/             # Data directory (could be outside project)
│   ├── raw/          # Original, immutable data
│   └── processed/    # Cleaned, transformed data
├── outputs/          # Analysis results (separate from source code)
│   ├── models/       # Trained models
│   ├── figures/      # Generated visualizations
│   └── reports/      # Analysis reports and metrics
├── notebooks/        # Jupyter notebooks for exploration
├── config/           # Configuration files
│   └── categories.py # or categories.json
├── requirements.txt  # Dependencies
└── README.md
```

Benefits of this structure:
- Clear separation between code, data, and outputs
- Better version control (avoid storing large data files with code)
- Preserves results from different runs
- Easier collaboration and deployment

## Extending the System

### Adding New Categories

To add new categories to the keyword-based system, edit the `CATEGORIES` dictionary in `categorization.py`:

```python
CATEGORIES = {
    'New Category': {
        'keywords': ['keyword1', 'keyword2', 'keyword3'],
        'subcategories': {
            'Subcategory 1': ['sub-keyword1', 'sub-keyword2'],
            'Subcategory 2': ['sub-keyword3', 'sub-keyword4']
        }
    },
    # ... existing categories ...
}
```