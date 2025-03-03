# Conversation Categorizer

A Python-based tool for analyzing and categorizing conversations from AI assistants (Claude, ChatGPT) to understand usage patterns and topics.

## Features

- Processes conversation data from multiple AI assistant formats (Claude, ChatGPT)
- Supports multilingual analysis (optimized for English and Danish)
- Generates language-agnostic embeddings using SentenceTransformer
- Categorizes conversations into predefined topics with subcategories
- Provides detailed metrics about conversation patterns

## Categories

The system categorizes conversations into various topics including:
- Learning/Education
- Code Development
- Writing Assistance
- Analysis/Research
- Creative & Ideation
- Professional/Business
- Technical Support
- Personal Projects
- Social Media/Marketing
- DALL-E/Image
- Cooking/Food
- Information/Curiosity

## Setup

### Prerequisites

- Python 3.7+
- pip or conda

### Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd [repository-name]
```

2. Install dependencies:

Using pip:
```bash
pip install -r requirements.txt
```

Using conda:
```bash
conda install pandas numpy scikit-learn pytorch -c pytorch -c conda-forge
pip install langdetect sentence-transformers tqdm
```

### Directory Structure

```
.
├── data/
│   ├── raw/         # Place your JSON conversation files here
│   └── processed/   # Output directory for processed data
├── src/
│   ├── data_processing.py
│   ├── embedding.py
│   └── clustering.py
├── outputs/         # Analysis outputs
└── models/         # Model files (if any)
```

## Usage

1. Add your JSON conversation files to the `data/raw` folder

2. Process the raw data:
```bash
cd src
python data_processing.py
```
This will create a CSV file in `data/processed`

3. Generate embeddings:
```bash
python embedding.py
```
Note: This step may take some time depending on the amount of data

4. Run the categorization:
```bash
python clustering.py
```

## Output

The system generates:
- Cleaned conversation data (CSV)
- Language-agnostic embeddings (NumPy array)
- Category distribution metrics (JSON)
- Detailed analysis of conversation patterns

## Recommendations

- Install Rainbow CSV extension in your IDE for better CSV file viewing
- Monitor the memory usage when processing large datasets
- Regularly backup your raw data

## Notes

- The `data/raw` and `data/processed` directories are git-ignored to prevent accidental commit of sensitive conversation data
- Adjust the category keywords in `clustering.py` if you need to customize the categorization