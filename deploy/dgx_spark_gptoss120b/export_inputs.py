#!/usr/bin/env python3
"""Export checksum-bound minimum inputs without emitting controlled text."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.human_first_reranking import load_cases, load_protocol
from src.analysis.qwen_rubric import load_project_rows

FORBIDDEN = {
    "sentence_edit", "edited_sentence", "candidate", "score", "rating",
    "annotator", "participant", "note", "answer", "speciteller", "ko",
    "granuscore", "qwen_score", "gemma_score", "gptoss_score",
}


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_primary(config: Path, output: Path) -> dict[str, object]:
    record, config_sha, _ = load_protocol(config)
    cases = load_cases(record)
    cell_names: dict[tuple[str, str], str] = {}
    rows: list[dict[str, object]] = []
    for case in cases:
        key = (case.corpus_id, case.edit_type)
        if key not in cell_names:
            cell_names[key] = f"cell{len(cell_names) + 1:02d}"
        rows.append({
            "case_id": case.case_id,
            "source_position": case.source_position,
            "direction": case.edit_type,
            "cell_id": cell_names[key],
            "source_text": case.sentence_original,
            "source_text_sha256": case.original_sha256,
        })
    fields = ["case_id", "source_position", "direction", "cell_id", "source_text", "source_text_sha256"]
    assert len(rows) == 60 and len({row["case_id"] for row in rows}) == 60
    assert not FORBIDDEN.intersection({field.casefold() for field in fields})
    write_csv(output, rows, fields)
    manifest = {
        "schema_version": "spectech_dgx_primary_input_manifest_v1",
        "rows": 60,
        "ordered_case_sha256": record["inputs"]["ordered_case_sha256"],
        "base_config_sha256": config_sha,
        "input_sha256": file_sha(output),
        "columns": fields,
        "cell_map": [
            {"cell_id": value, "direction": key[1]}
            for key, value in cell_names.items()
        ],
        "forbidden_columns_present": [],
    }
    return manifest


def export_rubric(config: Path, output: Path) -> dict[str, object]:
    qwen = json.loads(config.read_text(encoding="utf-8"))
    rows_in = load_project_rows(qwen)
    rows = [
        {
            "rubric_case_id": sha_text(f"{row['corpus_id']}\0{row['sent_id']}\0{row['pilot_position']}"),
            "rubric_position": index,
            "sentence_text": row["sent_text"],
            "sentence_text_sha256": sha_text(row["sent_text"]),
        }
        for index, row in enumerate(rows_in, 1)
    ]
    fields = ["rubric_case_id", "rubric_position", "sentence_text", "sentence_text_sha256"]
    assert len(rows) == 80 and len({row["rubric_case_id"] for row in rows}) == 80
    write_csv(output, rows, fields)
    return {
        "schema_version": "spectech_dgx_rubric_input_manifest_v1",
        "rows": 80,
        "base_config_sha256": file_sha(config),
        "input_sha256": file_sha(output),
        "columns": fields,
        "forbidden_columns_present": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-config", type=Path, default=Path("configs/round2_human_first_reranking_v1.json"))
    parser.add_argument("--rubric-config", type=Path, default=Path("configs/round2_qwen_rubric_v1.json"))
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    primary = args.output_directory / "primary_inputs.csv"
    rubric = args.output_directory / "rubric_inputs.csv"
    manifest = {
        "primary": export_primary(args.primary_config, primary),
        "rubric": export_rubric(args.rubric_config, rubric),
    }
    (args.output_directory / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"primary_rows": 60, "rubric_rows": 80}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
