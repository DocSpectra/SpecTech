"""Container runner for SpeciTeller scoring."""
from __future__ import annotations

import subprocess
from pathlib import Path

from uuid import uuid4

from src.speciteller.config import SpeciTellerConfig


def preflight_speciteller(config: SpeciTellerConfig) -> None:
    """Fail fast if the containerized scorer is not runnable."""
    command = [
        "docker",
        "run",
        "--rm",
        config.image,
        "python",
        "speciteller.py",
        "--help",
    ]
    subprocess.run(command, check=True)


def run_speciteller(
    config: SpeciTellerConfig,
    input_path: Path,
    output_path: Path,
    batch_size: int = 25000,
) -> None:
    """Invoke SpeciTeller and emit ``sent_id\tscore`` rows.

    Input format:
    - TSV file at ``input_path`` with two columns per line: ``sent_id\ttokenized_text``

    Output format:
    - TSV file at ``output_path`` with two columns per line: ``sent_id\tscore``
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ids: list[str] = []
    sentences: list[str] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            sent_id, sent_text = line.split("\t", 1)
            ids.append(sent_id)
            sentences.append(sent_text)

    if not ids:
        output_path.write_text("", encoding="utf-8")
        return

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    run_id = uuid4().hex[:8]
    all_rows: list[tuple[str, str]] = []

    for batch_idx, start in enumerate(range(0, len(ids), batch_size)):
        end = min(len(ids), start + batch_size)
        batch_ids = ids[start:end]
        batch_sentences = sentences[start:end]

        sentence_file = input_path.parent / f"{input_path.stem}.{run_id}.{batch_idx}.sentences.txt"
        raw_scores_file = input_path.parent / f"{input_path.stem}.{run_id}.{batch_idx}.scores.txt"

        with sentence_file.open("w", encoding="utf-8") as handle:
            for sentence in batch_sentences:
                handle.write(f"{sentence}\n")

        command = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{input_path.parent.resolve()}:/data",
            config.image,
            "python",
            "speciteller.py",
            "--inputfile",
            f"/data/{sentence_file.name}",
            "--outputfile",
            f"/data/{raw_scores_file.name}",
        ]
        try:
            subprocess.run(command, check=True)

            with raw_scores_file.open("r", encoding="utf-8") as handle:
                scores = [line.strip() for line in handle if line.strip()]

            if len(scores) != len(batch_ids):
                raise ValueError(
                    "SpeciTeller output length mismatch: "
                    f"{len(scores)} scores for {len(batch_ids)} inputs in batch {batch_idx}"
                )

            all_rows.extend(zip(batch_ids, scores))
        finally:
            sentence_file.unlink(missing_ok=True)
            raw_scores_file.unlink(missing_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for sent_id, score in all_rows:
            handle.write(f"{sent_id}\t{score}\n")