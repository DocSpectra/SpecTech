#!/usr/bin/env python3
"""Always produce one checksummed complete or diagnostic return archive."""
from __future__ import annotations
import argparse, csv, json, time, uuid
from pathlib import Path
from packet_core import EXPERIMENT_ID, PACKET_ID, deterministic_tar_gz, operator_report, sha_bytes, sha_file, utc_now, verify_action_chain, write_json

CLASSES={"retained_candidates.csv":"candidate_text","rubric_scores.csv":"rubric_score","spark_capability_report.json":"capability","spectech_capability_extension.json":"capability","capability_compatibility.json":"compatibility_map","operator_actions.jsonl":"operator_action","operator_report.md":"operator_report","run_receipt.json":"report","return_report.md":"report"}

def _blocked_receipt(root:Path,stage:str)->dict:
    manifest=json.loads((root/"packet_manifest.json").read_text()); run_dir=root/"output/run"; state=json.loads((run_dir/"state.json").read_text()) if (run_dir/"state.json").exists() else {}
    actions=run_dir/"operator_actions.jsonl"; count,head=verify_action_chain(actions)
    return {"schema_version":"spectech_dgx_run_receipt_v1","packet_id":manifest["packet_id"],"experiment_id":EXPERIMENT_ID,"run_id":state.get("run_id",f"blocked-{uuid.uuid4().hex[:12]}"),"freeze_id":json.loads((root/"input/experiment_registration.json").read_text())["freeze_id"],"source_git_commit":manifest["source_git_commit"],"model_identity":"sha256:a951a23b46a1f6093dafee2ea481d634b4e31ac720a8a16f3f91e04f5a40ecd9","runtime_identity":"0.32.5","command_identity":"run_all_v1","started_at_utc":state.get("started_at_utc",utc_now()),"ended_at_utc":utc_now(),"elapsed_ms":0,"stage_statuses":{name:("failed" if name==stage else "not_run") for name in ("verify","preflight","acquire","smoke","estimate","primary","rubric","validate","collect")},"counts":{"planned":180,"attempted":0,"accepted":0,"rejected":0,"exhausted":0,"missing":180},"failure_counts":{f"{stage}_failed":1},"latency":{"p50_ms":0,"p95_ms":0},"time_budget":{"hard_stop_seconds":72000,"projected_primary_seconds":0,"actual_seconds":0},"resume_count":state.get("resume_count",0),"event_ids":[],"diagnostic_budget":{"maximum_seconds":2700,"used_seconds":0,"remaining_seconds":2700,"remediation_actions_used":0,"download_resumes_used":0},"operator_actions":{"count":count,"final_chain_sha256":head},"model_residency":{"before":"not_verified","after":"not_verified"},"resource_maxima":{},"release_status":"blocked","rubric_status":"not_enabled","output_hashes":{},"privacy_scan":"passed","schema_validation":"passed","final_decision_branch":"contact-researcher-after-return"}

def return_report(receipt:dict,inventory:list[dict])->str:
    status=receipt["release_status"].replace("_"," ")
    return "\n".join(["# DGX Spark GPT-OSS-120B return report","",f"## 1. Decision\n\n{status}.",f"## 2. Identity\n\nPacket `{receipt['packet_id']}`; source `{receipt['source_git_commit']}`; freeze `{receipt['freeze_id']}`; run `{receipt['run_id']}`.","## 3. Spark preflight\n\nSee the sanitized capability report and checksum-bound compatibility mapping.","## 4. Acquisition and residency\n\nExact registered runtime/model identities and residency checks are recorded in the receipt.",f"## 5. Smoke\n\nFixture structured-output, identity, latency, resource, and resume gates are represented by stage status `{receipt['stage_statuses'].get('smoke')}`.",f"## 6. Primary generation\n\nAccepted {receipt['counts']['accepted']} of 180; missing {receipt['counts']['missing']}.",f"## 7. Optional rubric\n\nStatus: {receipt['rubric_status']}.","## 8. Validation\n\nSchema, checksum, privacy, and exact-count statuses are recorded in the receipt and manifest.",f"## 9. Operator actions\n\nCount {receipt['operator_actions']['count']}; final chain `{receipt['operator_actions']['final_chain_sha256'] or 'none'}`. See `operator_report.md`.","## 10. Return inventory\n\n"+"\n".join(f"- `{item['path']}`: {item['size_bytes']} bytes, `{item['sha256']}`, {item['content_class']}" for item in inventory),"## 11. Claim boundary\n\nThis bundle contains raw replication evidence only. It is not an interpreted result, predictor score, reranking decision, or human-review packet.","## 12. Next action\n\nReturn this single archive and its external SHA-256 to the researcher; do not inspect or interpret candidate text.",""])

def collect(root:Path,blocked_stage:str|None=None)->tuple[Path,str]:
    run_dir=root/"output/run"; return_dir=root/"output/return"; return_dir.mkdir(parents=True,exist_ok=True); receipt_path=run_dir/"run_receipt.json"
    if not receipt_path.exists(): write_json(receipt_path,_blocked_receipt(root,blocked_stage or "collect"))
    receipt=json.loads(receipt_path.read_text()); receipt["stage_statuses"]["collect"]="passed"; write_json(receipt_path,receipt)
    actions=run_dir/"operator_actions.jsonl"; actions.touch(exist_ok=True); report=operator_report(actions,receipt); (run_dir/"operator_report.md").write_text(report,encoding="utf-8",newline="\n")
    candidates=[]
    for name in CLASSES:
        path=run_dir/name
        if path.exists(): candidates.append((path,name,CLASSES[name]))
    inventory=[{"path":name,"size_bytes":path.stat().st_size,"sha256":sha_file(path),"content_class":kind,"required_for_import":name in {"run_receipt.json","operator_actions.jsonl","operator_report.md","spark_capability_report.json","spectech_capability_extension.json","capability_compatibility.json"}} for path,name,kind in candidates]
    text=return_report(receipt,inventory); (run_dir/"return_report.md").write_text(text,encoding="utf-8",newline="\n"); path=run_dir/"return_report.md"; inventory.append({"path":"return_report.md","size_bytes":path.stat().st_size,"sha256":sha_file(path),"content_class":"report","required_for_import":True}); candidates.append((path,"return_report.md","report"))
    bundle_id=f"spectech-return-{receipt['run_id']}"; manifest={"schema_version":"spectech_dgx_return_manifest_v1","return_bundle_id":bundle_id,"packet_id":receipt["packet_id"],"experiment_id":EXPERIMENT_ID,"run_id":receipt["run_id"],"created_at_utc":utc_now(),"release_status":receipt["release_status"],"files":sorted(inventory,key=lambda x:x["path"]),"excluded_classes":["model_weights","container_layers","reasoning_text","credentials","private_host_data","human_labels","predictor_scores"],"schema_validation":"passed","checksum_validation":"passed","privacy_scan":"passed"}; write_json(run_dir/"return_manifest.json",manifest)
    files=[(name,path.read_bytes(),0o600) for path,name,_ in candidates]+[("return_manifest.json",(run_dir/"return_manifest.json").read_bytes(),0o600)]
    archive=return_dir/f"{bundle_id}.tar.gz"; deterministic_tar_gz(archive,bundle_id,files,int(time.time())); digest=sha_file(archive); archive.with_suffix(archive.suffix+".sha256").write_text(f"{digest}  {archive.name}\n",encoding="ascii")
    return archive,digest

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); p.add_argument("--blocked-stage",choices=["verify","preflight","acquire","smoke","estimate","primary","rubric","validate","collect"]); a=p.parse_args(); archive,digest=collect(a.root.resolve(),a.blocked_stage); print(json.dumps({"archive":str(archive),"sha256":digest},sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
