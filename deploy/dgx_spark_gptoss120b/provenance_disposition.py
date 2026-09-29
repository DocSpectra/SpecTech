#!/usr/bin/env python3
"""Derive a content-blind DGX return provenance disposition.

This module consumes only the compact return audit and a versioned decision
record.  It never reads candidate text or row-level rubric scores.  A derived
acceptance does not modify, replace, or weaken the original withheld audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from packet_core import validate_schema


WAIVER_DISPOSITION = "accepted_with_author_provenance_waiver"
SUPPLEMENT_DISPOSITION = "accepted_after_supplement"
WITHHELD_DISPOSITION = "scientific_eligibility_withheld"


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch")


def derive_disposition(
    audit: dict[str, Any],
    decision: dict[str, Any],
    *,
    audit_sha256: str,
    supplement: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a derived acceptance or preserve withholding.

    ``supplement`` is intentionally content-free.  It may authorize the
    supplement path only when it explicitly resolves every original blocker.
    Otherwise an exact, scoped author waiver is required.
    """
    bindings = decision["bindings"]
    _require_equal("original audit disposition", audit["audit_disposition"], WITHHELD_DISPOSITION)
    _require_equal("audit sha256", audit_sha256, bindings["original_audit_sha256"])
    _require_equal("run id", audit["run_id"], bindings["run_id"])
    _require_equal("return archive sha256", audit["return_archive_sha256"], bindings["return_archive_sha256"])
    _require_equal("source packet sha256", audit["source_packet_sha256"], bindings["source_packet_sha256"])
    _require_equal("original blockers", audit["blockers"], decision["unresolved_evidence"]["audit_blocker_codes"])

    supplement_passed = bool(
        supplement
        and supplement.get("run_id") == bindings["run_id"]
        and supplement.get("status") == "passed"
        and supplement.get("resolved_blockers") == audit["blockers"]
        and supplement.get("outputs_unchanged") is True
    )
    if supplement_passed:
        disposition = SUPPLEMENT_DISPOSITION
        basis = "content_free_no_rerun_supplement"
    elif decision["authorization"]["explicit_author_waiver"] is True:
        disposition = WAIVER_DISPOSITION
        basis = "explicit_author_provenance_waiver"
    else:
        disposition = WITHHELD_DISPOSITION
        basis = "no_passing_supplement_or_explicit_waiver"

    return {
        "schema_version": "spectech_dgx_provenance_acceptance_summary_v1",
        "decision_id": decision["decision_id"],
        "run_id": bindings["run_id"],
        "disposition": disposition,
        "basis": basis,
        "decision_date": decision["authorization"]["authorization_date"],
        "supplement_status": "passed" if supplement_passed else decision["supplement_status"],
        "original_audit": {
            "sha256": audit_sha256,
            "disposition": audit["audit_disposition"],
            "preserved_unchanged": True,
        },
        "bindings": bindings,
        "positive_evidence": decision["positive_evidence"],
        "unresolved_evidence": decision["unresolved_evidence"],
        "implementation_deviations": decision["implementation_deviations"],
        "interpretation": {
            "evidence_of_tampering": False,
            "audit_completeness": "incomplete" if not supplement_passed else "supplemented",
            "missing_checks_claimed_as_passed": False,
            "outputs_may_be_replaced_or_changed": False,
            "authorized_scope": decision["authorization"]["authorized_scope"],
        },
        "later_supplement_policy": decision["later_supplement_policy"],
    }


def write_outputs(
    output_dir: Path,
    audit_path: Path,
    config_path: Path,
    schema_path: Path,
    supplement_path: Path | None = None,
) -> dict[str, Any]:
    decision = load_json(config_path)
    schema = load_json(schema_path)
    validate_schema(decision, schema)
    audit = load_json(audit_path)
    supplement = load_json(supplement_path) if supplement_path else None
    summary = derive_disposition(audit, decision, audit_sha256=sha_file(audit_path), supplement=supplement)
    if summary["disposition"] == WITHHELD_DISPOSITION:
        raise ValueError("provenance acceptance remains withheld")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "acceptance_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    readme_path = output_dir / "README.md"
    missing = "\n".join(f"- {item}" for item in decision["unresolved_evidence"]["human_readable"])
    deviations = "\n".join(f"- {item}" for item in decision["implementation_deviations"])
    readme_path.write_text(
        "# GPT-OSS-120B run-1 provenance acceptance\n\n"
        f"Run `{summary['run_id']}` is **{summary['disposition']}** for the bounded rubric and "
        "candidate-review analyses. The author explicitly authorized this disposition while a "
        "no-rerun supplement is still being sought.\n\n"
        "The present content, identities, joins, and hashes are internally consistent, and the "
        "reviewed content-blind audit found no evidence of tampering. This is an audit-completeness "
        "waiver, not a claim that omitted records were independently verified.\n\n"
        "## Evidence still missing or incomplete\n\n"
        f"{missing}\n\n"
        "## Preserved implementation deviations\n\n"
        f"{deviations}\n\n"
        "## Scope\n\n"
        "Only the exact returned 180-candidate and 80-rubric-row set is accepted for the bounded "
        "local analyses. No model rerun, candidate replacement, paper claim, or assertion that a "
        "missing check passed is authorized here. A later supplement may strengthen provenance "
        "but cannot change these outputs or replace this chronology.\n",
        encoding="utf-8",
        newline="\n",
    )
    files = []
    for path, role in ((summary_path, "acceptance_summary"), (readme_path, "human_readable_acceptance")):
        files.append({"path": path.name, "role": role, "size_bytes": path.stat().st_size, "sha256": sha_file(path)})
    manifest = {
        "schema_version": "spectech_dgx_provenance_acceptance_manifest_v1",
        "decision_id": decision["decision_id"],
        "run_id": bindings_run_id(summary),
        "disposition": summary["disposition"],
        "source_bindings": {
            "config_sha256": sha_file(config_path),
            "schema_sha256": sha_file(schema_path),
            "original_audit_sha256": sha_file(audit_path),
            "return_archive_sha256": summary["bindings"]["return_archive_sha256"],
            "source_packet_sha256": summary["bindings"]["source_packet_sha256"],
        },
        "files": files,
    }
    manifest_path = output_dir / "acceptance_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return manifest


def bindings_run_id(summary: dict[str, Any]) -> str:
    return str(summary["bindings"]["run_id"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--supplement", type=Path)
    args = parser.parse_args()
    manifest = write_outputs(args.output_dir, args.audit, args.config, args.schema, args.supplement)
    print(json.dumps({"status": "passed", "run_id": manifest["run_id"], "disposition": manifest["disposition"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
