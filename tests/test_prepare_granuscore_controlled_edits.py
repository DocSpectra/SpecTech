import csv
import hashlib
import json
from pathlib import Path

from scripts.prepare_granuscore_controlled_edits import prepare


def _write_edits(path: Path, corpus_id: str) -> None:
    fields = [
        "sent_id", "corpus_id", "sentence_original", "speciteller_score_original",
        "token_count", "edit_type", "sentence_edited", "speciteller_score_edited", "delta",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerow({
            "sent_id": "source-1", "corpus_id": corpus_id,
            "sentence_original": "Restart the service.", "speciteller_score_original": "0.1",
            "token_count": "3", "edit_type": "add_specific",
            "sentence_edited": "Restart nginx 1.24 on host web-03.",
            "speciteller_score_edited": "0.2", "delta": "0.1",
        })


def test_prepare_emits_paired_exact_text_rows(tmp_path: Path) -> None:
    source = tmp_path / "edits.csv"
    _write_edits(source, "ansible_docs")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "schema_version": "round2_granuscore_v1", "outcome_blind_freeze": True,
        "controlled_edits": {
            "directional_expectations": {"add_specific": "negative"},
            "inputs": {"ansible_docs": {"path": str(source), "sha256": digest, "rows": 1}},
        },
    }), encoding="utf-8")
    sentences = tmp_path / "sentences.csv"
    manifest = tmp_path / "manifest.csv"
    metadata = tmp_path / "metadata.json"
    result = prepare(config, sentences, manifest, metadata)
    assert result["case_count"] == 1
    assert result["score_row_count"] == 2
    sentence_rows = list(csv.DictReader(sentences.open(encoding="utf-8")))
    manifest_rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    assert [row["sent_text"] for row in sentence_rows] == [
        "Restart the service.", "Restart nginx 1.24 on host web-03.",
    ]
    assert [row["version"] for row in manifest_rows] == ["original", "edited"]
    assert sentence_rows[0]["sent_id"] == manifest_rows[0]["score_sent_id"]
    saved = json.loads(metadata.read_text(encoding="utf-8"))
    assert saved["sentence_sha256"] == result["sentence_sha256"]
    assert saved["manifest_sha256"] == result["manifest_sha256"]
