"""Tests for the exact Ko et al. container adapter contract."""
from __future__ import annotations

import csv
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from src.ko_specificity.config import DEFAULT_KO_CONFIG
from src.ko_specificity.io import (
    KoInputRow,
    parse_prediction_line,
    parse_predictions,
    read_input_csv,
    write_legacy_target_bundle,
    write_model_scores,
)
from src.ko_specificity.runner import build_docker_command, run_ko_specificity


FIXTURES = Path("tests/fixtures")
AUTHORIZED_TEST_CONFIG = replace(DEFAULT_KO_CONFIG, project_scoring_authorized=True)


def _write_runtime_input(path: Path, count: int = 50) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sent_id", "corpus_id", "text"])
        for index in range(count):
            writer.writerow([f"runtime-{index:03d}", "fixture", f"Released fixture row {index}."])
    return path


def test_input_header_and_row_identity_are_strict(tmp_path: Path) -> None:
    rows = read_input_csv(FIXTURES / "ko_input.csv")
    assert [row.sent_id for row in rows] == ["fixture-001", "fixture-002"]

    wrong = tmp_path / "wrong.csv"
    wrong.write_text("corpus_id,sent_id,text\nx,y,z\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly"):
        read_input_csv(wrong)

    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text(
        "sent_id,corpus_id,text\nsame,c,First.\nsame,c,Second.\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Duplicate sent_id"):
        read_input_csv(duplicate)


def test_legacy_bundle_accounts_for_first_ignored_test_row(tmp_path: Path) -> None:
    rows = read_input_csv(FIXTURES / "ko_input.csv")
    write_legacy_target_bundle(rows, tmp_path)
    test_lines = (tmp_path / "twitters.txt").read_text(encoding="utf-8").splitlines()
    unlab_lines = (tmp_path / "twitteru.txt").read_text(encoding="utf-8").splitlines()
    assert test_lines == [rows[0].text, rows[0].text, rows[1].text]
    assert unlab_lines == [rows[0].text, rows[1].text]
    assert len((tmp_path / "twitterl.txt").read_text().splitlines()) == 3
    assert len((tmp_path / "twitterv.txt").read_text().splitlines()) == 3
    with (tmp_path / "row_map.csv").open(newline="", encoding="utf-8") as handle:
        mapping = list(csv.DictReader(handle))
    assert [row["prediction_index"] for row in mapping] == ["0", "1"]
    assert [row["sent_id"] for row in mapping] == ["fixture-001", "fixture-002"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [("tensor(0.1250)", 0.125), ("0.875", 0.875), ("1e-3", 0.001)],
)
def test_output_parser_accepts_legacy_tensor_and_plain_forms(text: str, expected: float) -> None:
    assert parse_prediction_line(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["tensor([0.2])", "nan", "-0.1", "1.1", "garbage"])
def test_output_parser_rejects_malformed_or_out_of_scale_values(text: str) -> None:
    with pytest.raises(ValueError):
        parse_prediction_line(text)


def test_output_parser_enforces_exact_count() -> None:
    assert parse_predictions(FIXTURES / "ko_predictions.txt", 2) == [0.125, 0.875]
    with pytest.raises(ValueError, match="length mismatch"):
        parse_predictions(FIXTURES / "ko_predictions.txt", 3)


def test_model_aware_schema_preserves_order_and_context(tmp_path: Path) -> None:
    rows = read_input_csv(FIXTURES / "ko_input.csv")
    output = tmp_path / "scores.csv"
    write_model_scores(output, rows, [0.125, 0.875], DEFAULT_KO_CONFIG, "fixture-v1")
    with output.open(newline="", encoding="utf-8") as handle:
        result = list(csv.DictReader(handle))
    assert [row["sent_id"] for row in result] == ["fixture-001", "fixture-002"]
    assert {row["model_id"] for row in result} == {"ko2019_se_ad_mean_std"}
    assert {row["adaptation_context"] for row in result} == {"fixture-v1"}
    assert {row["score_direction"] for row in result} == {"higher_is_more_specific"}


def test_docker_command_keeps_artifacts_external(tmp_path: Path) -> None:
    command = build_docker_command(DEFAULT_KO_CONFIG, tmp_path / "bundle", tmp_path / "run")
    assert command[:3] == ["docker", "run", "--rm"]
    assert f"{DEFAULT_KO_CONFIG.glove_volume}:/artifacts:ro" in command
    assert command[-2:] == ["/opt/spectech/run_target.sh", "target"]


def test_runner_parses_container_output_without_reordering(tmp_path: Path) -> None:
    output = tmp_path / "scores.csv"
    runtime_input = _write_runtime_input(tmp_path / "runtime.csv")
    observed: list[str] = []

    def fake_runner(command, check):
        assert check is True
        observed.extend(command)
        mount = next(item for item in command if item.endswith(":/output"))
        container_output = Path(mount[: -len(":/output")])
        container_output.mkdir(parents=True, exist_ok=True)
        (container_output / "predictions.txt").write_text(
            "tensor(0.1250)\n" * 50, encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0)

    run_ko_specificity(
        AUTHORIZED_TEST_CONFIG,
        runtime_input,
        output,
        "fixture-adaptation-v1",
        command_runner=fake_runner,
    )
    assert observed[0] == "docker"
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 50
    assert rows[0]["sent_id"] == "runtime-000"
    assert rows[-1]["sent_id"] == "runtime-049"
    assert {row["score_raw"] for row in rows} == {"0.125"}


def test_runner_propagates_container_failure(tmp_path: Path) -> None:
    runtime_input = _write_runtime_input(tmp_path / "runtime.csv")

    def failing_runner(command, check):
        raise subprocess.CalledProcessError(17, command)

    with pytest.raises(subprocess.CalledProcessError):
        run_ko_specificity(
            AUTHORIZED_TEST_CONFIG,
            runtime_input,
            tmp_path / "scores.csv",
            "fixture-adaptation-v1",
            command_runner=failing_runner,
        )
    assert not (tmp_path / "scores.csv").exists()


def test_runner_rejects_fewer_than_50_adaptation_rows(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 50"):
        run_ko_specificity(
            AUTHORIZED_TEST_CONFIG,
            FIXTURES / "ko_input.csv",
            tmp_path / "scores.csv",
            "fixture-adaptation-v1",
        )


def test_runner_blocks_project_scoring_by_default(tmp_path: Path) -> None:
    runtime_input = _write_runtime_input(tmp_path / "runtime.csv")
    with pytest.raises(RuntimeError, match="project scoring is blocked"):
        run_ko_specificity(
            DEFAULT_KO_CONFIG,
            runtime_input,
            tmp_path / "scores.csv",
            "fixture-adaptation-v1",
        )
    assert not (tmp_path / "scores.csv").exists()


def test_empty_and_embedded_newline_inputs_fail(tmp_path: Path) -> None:
    empty = tmp_path / "empty.csv"
    empty.write_text("sent_id,corpus_id,text\n", encoding="utf-8")
    with pytest.raises(ValueError, match="at least one"):
        read_input_csv(empty)

    rows = [KoInputRow("id", "fixture", "line one\nline two")]
    path = tmp_path / "newline.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sent_id", "corpus_id", "text"])
        writer.writerow([rows[0].sent_id, rows[0].corpus_id, rows[0].text])
    with pytest.raises(ValueError, match="Embedded newline"):
        read_input_csv(path)
