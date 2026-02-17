"""Web dashboard for the GenAI Categorizer.

Provides a drag-and-drop browser interface for analyzing AI conversation
exports (Claude, ChatGPT). Supports JSON and CSV file uploads with
interactive visualizations.

Run with::

    python -m src.app
"""

import io
import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, List

try:
    import uvicorn
    from fastapi import FastAPI, File, UploadFile
    from fastapi.responses import HTMLResponse
except ImportError:
    print("\n  Web dependencies not installed. Run:")
    print('    pip install "genai-categorizer[web]"')
    print("    # or: pip install fastapi uvicorn python-multipart\n")
    raise SystemExit(1)

import pandas as pd

try:
    from src.clustering import Categorizer
    from src.data_processing import DataProcessor
except ImportError:
    from clustering import Categorizer  # type: ignore[no-redef]
    from data_processing import DataProcessor  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="GenAI Categorizer", version="1.0.0")
STATIC_DIR = Path(__file__).resolve().parent / "static"

_processor = DataProcessor(cache_dir=tempfile.mkdtemp())
_categorizer = Categorizer()

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the main dashboard page."""
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/api/analyze")
async def analyze(files: List[UploadFile] = File(...)) -> Dict:
    """Process uploaded JSON/CSV files and return categorized results."""
    all_rows: List[Dict] = []

    for upload in files:
        raw = await upload.read()
        name = upload.filename or "unknown"

        if name.lower().endswith(".json"):
            all_rows.extend(_extract_from_json(raw, name))
        elif name.lower().endswith(".csv"):
            all_rows.extend(_extract_from_csv(raw, name))

    if not all_rows:
        return {"conversations": [], "metrics": None}

    texts = [r["text"] for r in all_rows]
    results = _categorizer.assign_categories_batch(texts)

    conversations: List[Dict] = []
    for row, cat in zip(all_rows, results):
        length = len(row["text"])
        complexity = "simple" if length <= 100 else ("medium" if length <= 500 else "complex")
        preview = (row["text"][:300] + "…") if len(row["text"]) > 300 else row["text"]

        conversations.append({
            "conversation_id": row["conversation_id"],
            "text_preview": preview,
            "language": row["language"],
            "source_file": row["source_file"],
            "main_category": cat["main_category"],
            "subcategory": cat["subcategory"],
            "is_voice": cat["is_voice"],
            "confidence_score": cat["confidence_score"],
            "char_length": length,
            "complexity": complexity,
        })

    return {"conversations": conversations, "metrics": _compute_metrics(conversations)}


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _extract_from_json(raw: bytes, filename: str) -> List[Dict]:
    """Parse JSON content and extract conversations."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []

    items = data if isinstance(data, list) else [data] if isinstance(data, dict) else []
    rows: List[Dict] = []

    for i, conv in enumerate(items):
        text = _processor.extract_text(conv)
        if not text.strip():
            continue
        text_hash = _processor.get_text_hash(text)
        lang = _processor.detect_language_cached(text_hash, text)
        conv_id = conv.get("uuid", conv.get("title", conv.get("id", f"conv_{i + 1}")))

        rows.append({
            "conversation_id": str(conv_id),
            "text": text,
            "language": lang,
            "source_file": filename,
        })
    return rows


def _extract_from_csv(raw: bytes, filename: str) -> List[Dict]:
    """Parse CSV content and extract conversations."""
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        return []

    if "text" not in df.columns:
        return []

    rows: List[Dict] = []
    for idx, r in df.iterrows():
        text = str(r["text"])
        if not text.strip() or text == "nan":
            continue
        rows.append({
            "conversation_id": str(r.get("conversation_id", f"row_{idx}")),
            "text": text,
            "language": str(r.get("language", "unknown")),
            "source_file": filename,
        })
    return rows


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _compute_metrics(conversations: List[Dict]) -> Dict:
    """Build comprehensive metrics from categorized conversations."""
    n = len(conversations)
    if n == 0:
        return {}

    cats: Dict[str, int] = {}
    subcats: Dict[str, int] = {}
    langs: Dict[str, int] = {}
    complexities: Dict[str, int] = {}
    voice_count = 0
    total_conf = 0
    high_conf = 0
    total_length = 0

    for c in conversations:
        cats[c["main_category"]] = cats.get(c["main_category"], 0) + 1
        subcats[c["subcategory"]] = subcats.get(c["subcategory"], 0) + 1
        langs[c["language"]] = langs.get(c["language"], 0) + 1
        complexities[c["complexity"]] = complexities.get(c["complexity"], 0) + 1
        if c["is_voice"]:
            voice_count += 1
        total_conf += c["confidence_score"]
        if c["confidence_score"] > 5:
            high_conf += 1
        total_length += c["char_length"]

    cats = dict(sorted(cats.items(), key=lambda x: x[1], reverse=True))
    subcats = dict(sorted(subcats.items(), key=lambda x: x[1], reverse=True))
    langs = dict(sorted(langs.items(), key=lambda x: x[1], reverse=True))

    return {
        "total_conversations": n,
        "avg_length": round(total_length / n, 1),
        "category_distribution": cats,
        "subcategory_distribution": subcats,
        "language_distribution": langs,
        "complexity_distribution": complexities,
        "voice_conversations": voice_count,
        "confidence_avg": round(total_conf / n, 1),
        "confidence_high_pct": round(high_conf / n * 100, 1),
        "files_processed": len({c["source_file"] for c in conversations}),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print()
    print("  +--------------------------------------+")
    print("  |     GenAI Categorizer Dashboard      |")
    print("  |                                      |")
    print("  |  -> http://localhost:8000             |")
    print("  |                                      |")
    print("  |  Drop JSON or CSV files to analyze   |")
    print("  +--------------------------------------+")
    print()
    uvicorn.run(app, host="127.0.0.1", port=8000)
