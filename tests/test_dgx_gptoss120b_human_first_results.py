import csv
import json
import subprocess
import zipfile
from pathlib import Path

from src.analysis.pilot_model_human import sha256_file


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "outputs" / "round2" / "dgx_gptoss120b_human_first"
COMPACT = ROOT / "analysis" / "round2_dgx_gptoss120b_human_first"


def _csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_population_proxy_and_scorer_coverage_are_complete():
    preparation = json.loads((RAW / "scoring/preparation_metadata.json").read_text(encoding="utf-8"))
    assert preparation["source_cases"] == 60
    assert preparation["candidate_rows"] == 180
    assert preparation["score_rows"] == 240
    assert preparation["proxy_rows"] == 180
    assert len(_csv(RAW / "retained_candidates.csv")) == 180
    assert len(_csv(RAW / "automatic_proxy_audit.csv")) == 180
    assert sum(1 for line in (RAW / "scoring/speciteller/scores.tsv").read_text(encoding="utf-8").splitlines() if line) == 240
    assert len(_csv(RAW / "scoring/granuscore/scores.csv")) == 240
    assert json.loads((RAW / "scoring/granuscore/scores.metadata.json").read_text(encoding="utf-8"))["no_referential_unit_count"] == 0
    for corpus in ("ansible_docs", "github_docs"):
        for run in ("run01", "run02", "run03"):
            assert len(_csv(RAW / f"scoring/ko/{corpus}/{run}/scores.csv")) == 120


def test_metrics_policies_packet_and_key_are_bijective():
    candidates = _csv(RAW / "retained_candidates.csv")
    metrics = _csv(RAW / "candidate_metric_deltas.csv")
    policies = _csv(RAW / "policy_selections.csv")
    packet = _csv(RAW / "dgx_gptoss120b_review_packet.csv")
    key = _csv(RAW / "private_review_key.csv")
    candidate_ids = {row["candidate_id"] for row in candidates}
    assert len(candidate_ids) == 180
    assert {row["candidate_id"] for row in metrics} == candidate_ids
    assert {row["candidate_id"] for row in key} == candidate_ids
    assert len(policies) == 480
    assert {row["selected_candidate_id"] for row in policies} <= candidate_ids
    assert len(packet) == 180
    assert len({row["Review ID"] for row in packet}) == 180
    assert set(packet[0]) == {"Review ID", "A", "B", "Score"}
    assert all(row["Review ID"].startswith("DHF-") and row["A"].strip() and row["B"].strip() and row["Score"] == "" for row in packet)
    assert {row["Review ID"] for row in packet} == {row["review_id"] for row in key}
    policy_counts = {}
    for row in policies:
        policy_counts[row["policy"]] = policy_counts.get(row["policy"], 0) + 1
    assert len(policy_counts) == 8 and set(policy_counts.values()) == {60}


def test_release_manifest_hashes_balance_and_workbook_contract():
    manifest = json.loads((RAW / "packet_manifest.json").read_text(encoding="utf-8"))
    assert manifest["release_gate_passed"] is True and manifest["xlsx_pending"] is False
    assert manifest["candidate_side_counts"] == {"A": 90, "B": 90}
    assert manifest["review_rows"] == manifest["candidate_rows"] == manifest["metric_rows"] == manifest["proxy_rows"] == 180
    assert manifest["policy_rows"] == 480 and manifest["score_manifest_rows"] == 240
    assert all(counts["A"] == counts["B"] for counts in manifest["hidden_cell_side_counts"].values())
    assert sha256_file(RAW / "dgx_gptoss120b_review_packet.csv") == manifest["review_csv_sha256"]
    assert sha256_file(RAW / "dgx_gptoss120b_review_packet.xlsx") == manifest["review_xlsx_sha256"]
    assert (RAW / "dgx_gptoss120b_review_packet.xlsx").stat().st_size == manifest["review_xlsx_bytes"]
    assert sha256_file(RAW / "private_review_key.csv") == manifest["private_key_sha256"]
    verification = json.loads((RAW / "artifact_tool_verification.json").read_text(encoding="utf-8"))
    assert verification["reviewRows"] == 180
    assert verification["scoreCellsBlank"] == 180
    assert verification["roundTripEquality"] is True
    with zipfile.ZipFile(RAW / "dgx_gptoss120b_review_packet.xlsx") as archive:
        names = set(archive.namelist())
        workbook_xml = archive.read("xl/workbook.xml").decode("utf-8")
        review_xml = archive.read("xl/worksheets/sheet1.xml").decode("utf-8")
    assert "Review" in workbook_xml and "Instructions" in workbook_xml
    assert "state=\"hidden\"" not in workbook_xml and "veryHidden" not in workbook_xml
    assert "dataValidations" in review_xml and "D2:D181" in review_xml
    assert not any(name.endswith("vbaProject.bin") for name in names)


def test_compact_release_is_content_free_and_raw_outputs_are_ignored():
    compact_text = "\n".join(path.read_text(encoding="utf-8-sig") for path in COMPACT.iterdir() if path.is_file())
    assert "sentence_original" not in compact_text and "sentence_candidate" not in compact_text
    assert "C:\\Users\\" not in compact_text and "private_key_sha256\"" not in compact_text
    result = subprocess.run(
        ["git", "check-ignore", str(RAW / "dgx_gptoss120b_review_packet.xlsx"), str(RAW / "private_review_key.csv")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
