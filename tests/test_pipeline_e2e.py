"""End-to-end pipeline test for processing and categorization."""

import json

import pandas as pd

from clustering import run_clustering
from data_processing import process_raw_files


def test_pipeline_raw_to_categorized_outputs(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    cache_dir = tmp_path / "cache"
    raw_dir.mkdir()
    processed_dir.mkdir()
    cache_dir.mkdir()

    raw_payload = [
        {
            "uuid": "c1",
            "chat_messages": [{"text": "Please help debug my Python function error"}],
        },
        {
            "uuid": "c2",
            "chat_messages": [{"text": "Can you share a pasta dinner recipe?"}],
        },
    ]
    (raw_dir / "sample.json").write_text(json.dumps(raw_payload), encoding="utf-8")

    cleaned_csv = processed_dir / "cleaned.csv"
    categorized_csv = processed_dir / "categorized.csv"
    process_raw_files(
        raw_dir=str(raw_dir),
        output_csv=str(cleaned_csv),
        max_workers=1,
        cache_dir=str(cache_dir),
    )
    assert cleaned_csv.exists()

    run_clustering(str(cleaned_csv), str(categorized_csv), batch_size=10)
    assert categorized_csv.exists()
    assert (processed_dir / "conversation_metrics.json").exists()

    df = pd.read_csv(categorized_csv)
    assert "main_category" in df.columns
    assert "confidence_score" in df.columns
    assert len(df) == 2
