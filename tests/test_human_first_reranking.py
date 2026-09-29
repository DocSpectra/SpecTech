import json
from pathlib import Path

from src.analysis.human_first_reranking import build_request, candidate_id, derive_seed, integrity_reasons, load_cases, load_protocol

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/round2_human_first_reranking_v1.json"


def test_request_contains_only_original_and_frozen_direction_prompt():
    record, _, _ = load_protocol(CONFIG); case = load_cases(record)[0]
    request = build_request(record, case, 2, 1); rendered = json.dumps(request)
    assert request["model"] == "gemma4:12b" and request["think"] is False
    assert case.sentence_original in rendered
    assert case.case_id not in rendered and case.corpus_id not in rendered and case.source_sent_id not in rendered
    for forbidden in ("speciteller", "granuscore", "ko score", "author edit", "qwen rubric"):
        assert forbidden not in rendered.casefold()
    assert request["options"]["seed"] == derive_seed(record, case.case_id, 2, 1)


def test_candidate_slots_and_seeds_are_unique():
    record, _, _ = load_protocol(CONFIG); case = load_cases(record)[0]
    assert len({candidate_id(case.case_id, slot) for slot in (1,2,3)}) == 3
    assert len({derive_seed(record, case.case_id, slot, attempt) for slot in (1,2,3) for attempt in (1,2,3)}) == 9


def test_integrity_gate_excludes_no_substantive_proxy():
    assert integrity_reasons("Restart the service.", "Restart the service after deployment.") == []
    assert integrity_reasons("Restart the service.", "Restart the service") == []
    assert "unchanged_normalized" in integrity_reasons("Restart the service.", " Restart   the service. ")
    assert "multiple_sentences" in integrity_reasons("Restart it.", "Restart it. Then wait.")
    assert "new_non_latin_script" in integrity_reasons("Restart it.", "Restart it safely 文.")
