import csv
import json
import subprocess
import zipfile
from pathlib import Path

from src.analysis.pilot_model_human import sha256_file

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "outputs" / "round2" / "human_first_reranking"
COMPACT = ROOT / "analysis" / "round2_human_first_reranking"


def _csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_generation_proxy_and_scorer_coverage_are_complete():
    generation = json.loads((RAW / "generation_metadata.json").read_text(encoding="utf-8"))
    assert generation["generation_complete"] is True
    assert generation["source_cases"] == 60
    assert generation["retained_candidates"] == generation["planned_candidates"] == 180
    assert generation["attempt_rows"] == 182
    assert len(_csv(RAW / "retained_candidates.csv")) == 180
    assert len(_csv(RAW / "automatic_proxy_audit.csv")) == 180
    assert sum(1 for line in (RAW / "scoring/speciteller/scores.tsv").read_text(encoding="utf-8").splitlines() if line) == 240
    assert len(_csv(RAW / "scoring/granuscore/scores.csv")) == 240
    for corpus in ("ansible_docs", "github_docs"):
        for run in ("run01", "run02", "run03"):
            assert len(_csv(RAW / f"scoring/ko/{corpus}/{run}/scores.csv")) == 120


def test_hidden_metrics_policies_packet_and_key_are_bijective():
    candidates = _csv(RAW / "retained_candidates.csv")
    metrics = _csv(RAW / "candidate_metric_deltas.csv")
    policies = _csv(RAW / "policy_selections.csv")
    packet = _csv(RAW / "human_first_review_packet.csv")
    key = _csv(RAW / "private_review_key.csv")
    candidate_ids = {row["candidate_id"] for row in candidates}
    assert len(candidate_ids) == 180
    assert {row["candidate_id"] for row in metrics} == candidate_ids
    assert {row["candidate_id"] for row in key} == candidate_ids
    assert len(policies) == 480
    assert {row["selected_candidate_id"] for row in policies} <= candidate_ids
    assert len(packet) == 180 and len({row["Review ID"] for row in packet}) == 180
    assert set(packet[0]) == {"Review ID", "A", "B", "Score"}
    assert all(row["A"].strip() and row["B"].strip() and row["Score"] == "" for row in packet)
    assert {row["Review ID"] for row in packet} == {row["review_id"] for row in key}


def test_release_manifest_hashes_side_balance_and_workbook_contract():
    manifest = json.loads((RAW / "packet_manifest.json").read_text(encoding="utf-8"))
    assert manifest["release_gate_passed"] is True and manifest["xlsx_pending"] is False
    assert manifest["candidate_side_counts"] == {"A": 90, "B": 90}
    assert manifest["review_rows"] == manifest["candidate_rows"] == manifest["metric_rows"] == manifest["proxy_rows"] == 180
    assert manifest["policy_rows"] == 480 and manifest["score_manifest_rows"] == 240
    assert sha256_file(RAW / "human_first_review_packet.csv") == manifest["review_csv_sha256"]
    assert sha256_file(RAW / "human_first_review_packet.xlsx") == manifest["review_xlsx_sha256"]
    assert sha256_file(RAW / "private_review_key.csv") == manifest["private_key_sha256"]
    with zipfile.ZipFile(RAW / "human_first_review_packet.xlsx") as archive:
        workbook = archive.read("xl/workbook.xml").decode("utf-8")
        review_sheet = archive.read("xl/worksheets/sheet1.xml").decode("utf-8")
    assert "Review" in workbook and "Instructions" in workbook
    assert "dataValidations" in review_sheet and "D2:D181" in review_sheet


def test_compact_release_is_content_free_and_raw_outputs_are_ignored():
    compact_text = "\n".join(path.read_text(encoding="utf-8-sig") for path in COMPACT.iterdir() if path.is_file())
    assert "sentence_original" not in compact_text and "sentence_candidate" not in compact_text
    assert "C:\\Users\\" not in compact_text and "private_key_sha256\"" not in compact_text
    result = subprocess.run(["git", "check-ignore", str(RAW / "human_first_review_packet.xlsx"), str(RAW / "private_review_key.csv")], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0
