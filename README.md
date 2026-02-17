# GenAI Categorizer

[![CI](https://github.com/Walliiee/GenAICategorizer/actions/workflows/ci.yml/badge.svg)](https://github.com/Walliiee/GenAICategorizer/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A Python tool that categorizes AI assistant conversations (Claude, ChatGPT) by topic using multilingual sentence embeddings and keyword-based classification. Analyze your conversation history to discover usage patterns across 12 topic categories with subcategory granularity.

## Features

- **Interactive web dashboard** — Drag-and-drop JSON or CSV files in your browser to instantly visualize conversation patterns with charts and a searchable data table
- **Multi-format ingestion** — Parses conversation exports from both Claude and ChatGPT, handling nested JSON structures automatically
- **Multilingual NLP** — Generates language-agnostic embeddings with `paraphrase-multilingual-mpnet-base-v2`, supporting English, Danish, and more
- **12 topic categories** — Classifies conversations into categories like Code Development, Learning/Education, and Creative & Ideation, each with fine-grained subcategories
- **GPU acceleration** — Automatic CUDA / Apple Silicon (MPS) detection with dynamic batch sizing and mixed-precision (FP16) inference
- **Performance-first design** — Parallel file I/O, persistent caching, compiled regex patterns, and chunked memory management
- **Built-in benchmarking** — Measure pipeline throughput and resource usage across all stages

## How It Works

```
JSON Exports ──► Data Processing ──► Embedding Generation ──► Categorization ──► Results
(Claude/ChatGPT)  │                   │                        │                  │
                  ├─ Text extraction  ├─ SentenceTransformer   ├─ Regex scoring   ├─ CSV output
                  ├─ Language detect  ├─ GPU/CPU auto-select   ├─ 12 categories   ├─ JSON metrics
                  └─ Parallel I/O    └─ Embedding cache        └─ Subcategories   └─ Confidence
```

**Stage 1 — Data Processing** reads raw JSON conversation exports, extracts text from Claude (`chat_messages`) and ChatGPT (`mapping`) formats, detects language, and writes a cleaned CSV. Uses parallel I/O and language detection caching for speed.

**Stage 2 — Embedding Generation** encodes conversation text into dense vector embeddings using a multilingual SentenceTransformer model. Supports GPU acceleration, dynamic batching, and persistent caching to avoid redundant computation.

**Stage 3 — Categorization** scores each conversation against 12 topic categories using compiled regex keyword patterns. Assigns a primary category, subcategory, confidence score, and voice-conversation flag. Outputs a categorized CSV and JSON metrics summary.

## Categories

| Category | Example Subcategories |
|---|---|
| Learning/Education | Concept Understanding, How-to Learning, Academic Topics |
| Code Development | Bug Fixing, Feature Development, Code Review |
| Writing Assistance | Content Creation, Editing, Format/Style |
| Analysis/Research | Data Analysis, Research Review, Comparative Analysis |
| Creative & Ideation | Idea Generation, Visual Design, Innovation |
| Professional/Business | Strategy, Client/Customer, Business Analysis |
| Technical Support | Troubleshooting, Setup/Installation, Integration Issues |
| Personal Projects | Project Planning, Implementation, Review/Feedback |
| SoMe/Marketing | Content Creation, Event Announcements, Campaign Planning |
| DALL-E/Image | Image Generation, Image Editing, Style Transfer |
| Cooking/Food | Recipe Help, Ingredient Questions, Meal Planning |
| Information/Curiosity | General Knowledge, Cause/Effect, Research Requests |

## Quick Start

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/Walliiee/GenAICategorizer.git
cd GenAICategorizer
pip install -e ".[web]"
```

For GPU acceleration (CUDA):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -e ".[web]"
```

### Web Dashboard (recommended)

The fastest way to explore your conversation data:

```bash
python -m src.app
```

Open **http://localhost:8000** in your browser, then drag-and-drop your JSON or CSV files onto the page. The dashboard shows:

- **Category distribution** — horizontal bar chart of all 12 topic categories
- **Language breakdown** — doughnut chart of detected languages
- **Complexity analysis** — simple / medium / complex split
- **Voice vs text** — voice conversation detection
- **Confidence scores** — histogram of categorization confidence
- **Searchable data table** — filter by category, complexity, or free-text search; sortable columns; click to expand text previews

### CLI Pipeline

For batch processing or scripting, run the three-stage pipeline directly:

1. Place your conversation JSON exports in `data/raw/`.

2. Run each stage:

```bash
cd src

# Stage 1 — Process raw JSON files into cleaned CSV
python data_processing.py

# Stage 2 — Generate sentence embeddings
python embedding.py

# Stage 3 — Categorize conversations
python clustering.py
```

3. Find your results in `data/processed/`:

| File | Description |
|---|---|
| `cleaned_conversations.csv` | Extracted text with language labels |
| `embeddings.npy` | Dense vector embeddings |
| `categorized_conversations.csv` | Final categorized output |
| `conversation_metrics.json` | Distribution and performance metrics |

### Benchmarking

```bash
cd src
python benchmark.py
```

Measures execution time, memory delta, and throughput for each pipeline stage. Results are saved as timestamped JSON files in `outputs/benchmarks/`.

## Project Structure

```
GenAICategorizer/
├── src/
│   ├── __init__.py
│   ├── app.py                   # FastAPI web dashboard (drag-and-drop UI)
│   ├── data_processing.py       # JSON parsing, text extraction, language detection
│   ├── embedding.py             # Sentence embedding generation with caching
│   ├── clustering.py            # Keyword-based topic categorization
│   ├── benchmark.py             # Pipeline performance measurement
│   └── static/
│       └── index.html           # Dashboard SPA (Chart.js, dark theme)
├── tests/
│   ├── test_data_processing.py
│   ├── test_clustering.py
│   └── test_embedding.py
├── data/
│   ├── raw/                     # Input: conversation JSON exports (git-ignored)
│   └── processed/               # Output: CSVs, embeddings, metrics (git-ignored)
├── .github/workflows/ci.yml     # GitHub Actions CI
├── pyproject.toml                # Project metadata and dependencies
├── LICENSE
└── README.md
```

## Development

### Setup

```bash
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
```

### Linting

```bash
ruff check src/ tests/
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
