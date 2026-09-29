"""Prepare the frozen manual edit pairs for one GranuScore scoring pass."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case_id(corpus_id: str, source_sent_id: str, edit_type: str, position: int) -> str:
    value = f"{corpus_id}\0{source_sent_id}\0{edit_type}\0{position}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def prepare(
    config_path: Path,
    sentence_path: Path,
    manifest_path: Path,
    metadata_path: Path | None = None,
) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["schema_version"] != "round2_granuscore_v1" or not config["outcome_blind_freeze"]:
        raise ValueError("unexpected or unfrozen GranuScore config")
    sentence_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    sentence_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    seen_cases: set[str] = set()
    for corpus_id, entry in config["controlled_edits"]["inputs"].items():
        source = Path(entry["path"])
        if sha256(source) != entry["sha256"]:
            raise ValueError(f"controlled-edit SHA-256 mismatch: {corpus_id}")
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != entry["rows"]:
            raise ValueError(f"controlled-edit row-count mismatch: {corpus_id}")
        for position, row in enumerate(rows, start=1):
            if row["corpus_id"] != corpus_id:
                raise ValueError(f"controlled-edit corpus mismatch: {corpus_id}")
            if row["edit_type"] not in config["controlled_edits"]["directional_expectations"]:
                raise ValueError(f"unexpected edit type: {row['edit_type']}")
            if not row["sentence_original"] or not row["sentence_edited"]:
                raise ValueError("controlled-edit text must be nonempty")
            identifier = case_id(corpus_id, row["sent_id"], row["edit_type"], position)
            if identifier in seen_cases:
                raise ValueError("duplicate controlled-edit case identity")
            seen_cases.add(identifier)
            for version, text in (
                ("original", row["sentence_original"]),
                ("edited", row["sentence_edited"]),
            ):
                score_sent_id = f"{identifier}:{version}"
                sentence_rows.append(
                    {
                        "corpus_id": "controlled_edits",
                        "doc_path": f"{corpus_id}/{row['edit_type']}",
                        "sent_idx": len(sentence_rows),
                        "sent_text": text,
                        "sent_id": score_sent_id,
                    }
                )
                manifest_rows.append(
                    {
                        "case_id": identifier,
                        "source_corpus_id": corpus_id,
                        "source_sent_id": row["sent_id"],
                        "edit_type": row["edit_type"],
                        "version": version,
                        "score_sent_id": score_sent_id,
                    }
                )
    with sentence_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sentence_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(sentence_rows)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest_rows)
    metadata: dict[str, object] = {
        "schema_version": "granuscore_controlled_edit_preparation_v1",
        "config_path": config_path.as_posix(),
        "config_sha256": sha256(config_path),
        "source_inputs": config["controlled_edits"]["inputs"],
        "case_count": len(seen_cases),
        "score_row_count": len(sentence_rows),
        "sentence_path": sentence_path.as_posix(),
        "sentence_sha256": sha256(sentence_path),
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": sha256(manifest_path),
    }
    if metadata_path is not None:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        metadata["metadata_path"] = metadata_path.as_posix()
        metadata["metadata_sha256"] = sha256(metadata_path)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/round2_granuscore_v1.json"))
    parser.add_argument("--sentence-output", type=Path, default=Path("outputs/round2/granuscore/controlled_edits/scoring_input.csv"))
    parser.add_argument("--manifest-output", type=Path, default=Path("outputs/round2/granuscore/controlled_edits/scoring_manifest.csv"))
    parser.add_argument("--metadata-output", type=Path, default=Path("outputs/round2/granuscore/controlled_edits/preparation_metadata.json"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.config, args.sentence_output, args.manifest_output, args.metadata_output), sort_keys=True))


if __name__ == "__main__":
    main()
