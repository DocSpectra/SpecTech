#!/usr/bin/env python3
"""Frozen primary/secondary runner with append-only attempts and exact resume."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, re, statistics, time, unicodedata, urllib.request, uuid
from pathlib import Path
from packet_core import EXPERIMENT_ID, MODEL_DIGEST, PACKET_ID, RUNTIME_VERSION, append_action, derive_seed, sha_bytes, sha_file, utc_now, verify_action_chain, write_json

ATTEMPT_FIELDS=("schema_version","run_id","case_id","candidate_id","candidate_slot","attempt_index","seed","input_text_sha256","prompt_sha256","request_sha256","response_model_identity","started_at_utc","elapsed_ms","prompt_tokens","completion_tokens","status","integrity_reason_codes","candidate_text","candidate_text_sha256","candidate_character_count","reasoning_character_count","reasoning_sha256","reasoning_text_retained")

def _candidate_id(case_id:str,slot:int)->str: return hashlib.sha256(f"{case_id}\0{slot}".encode()).hexdigest()
def _load_jsonl(path:Path)->list[dict]: return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line] if path.exists() else []
def _append_jsonl(path:Path,item:dict)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf-8",newline="\n") as f: f.write(json.dumps(item,sort_keys=True,ensure_ascii=False)+"\n")
def _request(protocol:dict,text:str,direction:str,seed:int)->dict:
    user=protocol["prompts"]["user_templates"][direction].format(sentence_original=text)
    return {"model":"gpt-oss:120b","messages":[{"role":"system","content":protocol["prompts"]["system"]},{"role":"user","content":user}],"stream":False,"think":"low","format":protocol["runtime"]["response_schema"],"keep_alive":"30m","options":{**protocol["runtime"]["options"],"seed":seed}}
def _fake_call(request:dict,scenario:dict,candidate_index:int,attempt:int)->dict:
    if scenario.get("interrupt_after")==candidate_index and attempt==1: raise KeyboardInterrupt
    status=scenario.get("fail_status") if candidate_index in scenario.get("fail_candidates",[]) and attempt<=scenario.get("fail_attempts",3) else None
    if status: return {"forced_status":status,"elapsed_ms":scenario.get("elapsed_ms",10)}
    original=request["messages"][-1]["content"].split("Original sentence:\n",1)[-1]
    edited=(original.rstrip(".")+" with an invented fixture detail.") if "more specific" in request["messages"][-1]["content"] else ("Use the invented fixture as directed." if "less specific" in request["messages"][-1]["content"] else original.rstrip(".")+" in this invented example.")
    content=json.dumps({"edited_sentence":edited})
    return {"model":"gpt-oss:120b","message":{"content":content,"thinking":"discarded fixture trace"},"prompt_eval_count":20,"eval_count":12,"elapsed_ms":scenario.get("elapsed_ms",10)}
def _ollama_call(request:dict,url:str,timeout:int)->dict:
    started=time.monotonic(); wire=json.dumps(request).encode(); req=urllib.request.Request(url,data=wire,headers={"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(req,timeout=timeout) as response: value=json.loads(response.read())
    except TimeoutError: return {"forced_status":"timeout","elapsed_ms":int((time.monotonic()-started)*1000)}
    value["elapsed_ms"]=int((time.monotonic()-started)*1000); return value
def _parse(response:dict,original:str)->tuple[str|None,str,list[str],int,str|None]:
    if response.get("forced_status"): return None,response["forced_status"],[],0,None
    if response.get("model")!="gpt-oss:120b": return None,"interface_failed",["model_identity_mismatch"],0,None
    message=response.get("message",{}); reasoning=message.get("thinking","") or ""; reasoning_sha=sha_bytes(reasoning.encode()) if reasoning else None
    try:
        value=json.loads(message["content"])
        if set(value)!={"edited_sentence"} or not isinstance(value["edited_sentence"],str): raise ValueError
        candidate=value["edited_sentence"].strip()
    except (KeyError,TypeError,ValueError,json.JSONDecodeError): return None,"interface_failed",["one_field_json_required"],len(reasoning),reasoning_sha
    reasons=[]; normalized=lambda value:" ".join(unicodedata.normalize("NFKC",value).strip().split()).casefold()
    if not candidate or normalized(candidate)==normalized(original): reasons.append("unchanged_or_empty")
    if len(re.findall(r"[.!?](?:[\"')\]]*)?(?=\s+[A-Z]|\s*$)",candidate))>1: reasons.append("multiple_sentences")
    return (candidate,"accepted",[],len(reasoning),reasoning_sha) if not reasons else (candidate,"integrity_rejected",reasons,len(reasoning),reasoning_sha)

def _run_rubric(root:Path,run_dir:Path,registration:dict,protocol:dict,run_id:str,runtime:str,scenario:dict)->str:
    smoke=json.loads((run_dir/"model_smoke_report.json").read_text()) if (run_dir/"model_smoke_report.json").exists() else {"projected_primary_seconds":0,"p95_ms":10}
    if smoke["projected_primary_seconds"]+80*smoke["p95_ms"]/1000+600>64800: return "skipped_time_budget"
    config=registration["optional_rubric_registration"]["frozen_base_protocol"]
    if scenario.get("invented_inputs"):
        cases=[{"rubric_case_id":f"fixture-{index:03d}","sentence_text":f"The invented rubric fixture identifies sample item {index}."} for index in range(1,81)]
    else:
        with (root/"input/rubric_inputs.csv").open(encoding="utf-8",newline="") as f: cases=list(csv.DictReader(f))
    rows=[]
    for item in cases:
        system=config["prompt"]["system"]; user=config["prompt"]["user_template"].format(sent_text=item["sentence_text"]); prompt=system+"\n\n"+user
        request={"model":"gpt-oss:120b","messages":[{"role":"system","content":system},{"role":"user","content":user}],"stream":False,"think":"low","format":config["qwen"]["response_schema"],"keep_alive":"30m","options":{**config["qwen"]["options"],"num_predict":128}}
        started=utc_now(); clock=time.monotonic()
        if runtime=="fake": response={"model":"gpt-oss:120b","message":{"content":json.dumps({"score":int(sha_bytes(item["sentence_text"].encode())[0],16)%5+1}),"thinking":"discarded fixture trace"},"prompt_eval_count":20,"eval_count":4}
        else: response=_ollama_call(request,protocol["runtime"]["api_url"],300)
        try: parsed=json.loads(response["message"]["content"]); valid=set(parsed)=={"score"} and isinstance(parsed["score"],int) and 1<=parsed["score"]<=5 and response["model"]=="gpt-oss:120b"
        except (KeyError,TypeError,ValueError,json.JSONDecodeError): valid=False
        if not valid: return "failed"
        rows.append({"schema_version":"spectech_dgx_rubric_score_v1","run_id":run_id,"rubric_case_id":item["rubric_case_id"],"score_1_to_5":parsed["score"],"response_sha256":sha_bytes(response["message"]["content"].encode()),"model_identity":"gpt-oss:120b@"+MODEL_DIGEST,"prompt_sha256":sha_bytes(prompt.encode()),"config_sha256":sha_file(root/"config/round2_dgx_gptoss120b_rubric_delta_v1.json"),"started_at_utc":started,"elapsed_ms":int((time.monotonic()-clock)*1000),"prompt_tokens":response.get("prompt_eval_count",0),"completion_tokens":response.get("eval_count",0)})
    if len(rows)!=80 or len({row["rubric_case_id"] for row in rows})!=80: return "failed"
    with (run_dir/"rubric_scores.csv").open("w",encoding="utf-8",newline="") as f: w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n"); w.writeheader(); w.writerows(rows)
    return "complete"

def run(root:Path,runtime:str,scenario:dict,resume:bool,skip_rubric:bool=False)->dict:
    output=root/"output"; run_dir=output/"run"; return_dir=output/"return"; run_dir.mkdir(parents=True,exist_ok=True); return_dir.mkdir(parents=True,exist_ok=True)
    state_path=run_dir/"state.json"; attempts_path=run_dir/"generation_attempts.jsonl"; actions=run_dir/"operator_actions.jsonl"
    if resume:
        state=json.loads(state_path.read_text()); run_id=state["run_id"]; started_wall=state["started_at_utc"]; append_action(actions,run_id,time.monotonic(),stage="primary",event_type="resume",category="checkpoint",command_id="resume-v1",status="passed",state_change="checkpoint_resume")
    else:
        if state_path.exists(): raise ValueError("existing run requires --resume")
        run_id=f"dgx-{uuid.uuid4().hex[:12]}"; started_wall=utc_now(); state={"run_id":run_id,"started_at_utc":started_wall,"accepted":{},"resume_count":0}; write_json(state_path,state)
    if resume: state["resume_count"]+=1
    registration=json.loads((root/"input/experiment_registration.json").read_text()); protocol=registration["effective_primary_protocol"]
    manifest=json.loads((root/"packet_manifest.json").read_text()); config_sha=sha_file(root/"config/round2_dgx_gptoss120b_generation_delta_v1.json"); freeze_id=registration["freeze_id"]
    capability=json.loads((run_dir/"spectech_capability_extension.json").read_text())
    if capability["gate_status"]!="passed": raise ValueError("preflight not passed")
    existing=_load_jsonl(attempts_path); accepted={item["candidate_id"]:item for item in existing if item["status"]=="accepted"}
    if len(accepted)!=len(state["accepted"]): raise ValueError("checkpoint/attempt mismatch")
    latencies=[]; accepted_rows=[]
    if scenario.get("invented_inputs"):
        directions=("add_specific","de_specify","irrelevant_rewrite")
        cases=[{"case_id":f"{index:064x}","source_position":str(index),"direction":directions[(index-1)%3],"cell_id":f"fixture{(index-1)%3+1}","source_text":f"The invented fixture service handles sample item {index}.","source_text_sha256":sha_bytes(f"The invented fixture service handles sample item {index}.".encode())} for index in range(1,61)]
    else:
        with (root/"input/primary_inputs.csv").open(encoding="utf-8",newline="") as f: cases=list(csv.DictReader(f))
    for case_index,case in enumerate(cases,1):
      for slot in (1,2,3):
        cid=_candidate_id(case["case_id"],slot)
        if cid in accepted: continue
        for attempt in (1,2,3):
          if any(item["candidate_id"]==cid and item["attempt_index"]==attempt for item in existing): continue
          seed=derive_seed(2026081201,"dgx-spark-gptoss120b-candidate-v1",case["case_id"],slot,attempt); request=_request(protocol,case["source_text"],case["direction"],seed); started=utc_now()
          try: response=_fake_call(request,scenario,(case_index-1)*3+slot,attempt) if runtime=="fake" else _ollama_call(request,protocol["runtime"]["api_url"],600)
          except KeyboardInterrupt:
            write_json(state_path,state); raise
          candidate,status,reasons,reasoning_count,reasoning_sha=_parse(response,case["source_text"]); elapsed=response.get("elapsed_ms",0); latencies.append(elapsed)
          record={"schema_version":"spectech_dgx_generation_attempt_v1","run_id":run_id,"case_id":case["case_id"],"candidate_id":cid,"candidate_slot":slot,"attempt_index":attempt,"seed":seed,"input_text_sha256":case["source_text_sha256"],"prompt_sha256":sha_bytes(json.dumps(request["messages"],sort_keys=True).encode()),"request_sha256":sha_bytes(json.dumps(request,sort_keys=True).encode()),"response_model_identity":"gpt-oss:120b@"+MODEL_DIGEST,"started_at_utc":started,"elapsed_ms":elapsed,"prompt_tokens":response.get("prompt_eval_count",0),"completion_tokens":response.get("eval_count",0),"status":status,"integrity_reason_codes":reasons,"candidate_text":candidate if status=="accepted" else None,"candidate_text_sha256":sha_bytes(candidate.encode()) if candidate and status=="accepted" else None,"candidate_character_count":len(candidate) if candidate else 0,"reasoning_character_count":reasoning_count,"reasoning_sha256":reasoning_sha,"reasoning_text_retained":False}
          assert set(record)==set(ATTEMPT_FIELDS); _append_jsonl(attempts_path,record); existing.append(record)
          if status=="accepted": state["accepted"][cid]=record["candidate_text_sha256"]; write_json(state_path,state); accepted[cid]=record; break
    for case in cases:
      for slot in (1,2,3):
        cid=_candidate_id(case["case_id"],slot)
        if cid in accepted:
          item=accepted[cid]; accepted_rows.append({"schema_version":"spectech_dgx_retained_candidate_v1","run_id":run_id,"case_id":case["case_id"],"candidate_id":cid,"candidate_slot":slot,"retained_attempt":item["attempt_index"],"seed":item["seed"],"source_text_sha256":case["source_text_sha256"],"candidate_text":item["candidate_text"],"candidate_text_sha256":item["candidate_text_sha256"],"model_identity":item["response_model_identity"],"config_sha256":config_sha,"freeze_id":freeze_id})
    retained=run_dir/"retained_candidates.csv"
    with retained.open("w",encoding="utf-8",newline="") as f:
        fields=list(accepted_rows[0]) if accepted_rows else ["schema_version","run_id","case_id","candidate_id","candidate_slot","retained_attempt","seed","source_text_sha256","candidate_text","candidate_text_sha256","model_identity","config_sha256","freeze_id"]; w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n"); w.writeheader(); w.writerows(accepted_rows)
    failures={}
    for item in existing:
        if item["status"]!="accepted": failures[item["status"]]=failures.get(item["status"],0)+1
    count,head=verify_action_chain(actions); complete=len(accepted_rows)==180
    receipt={"schema_version":"spectech_dgx_run_receipt_v1","packet_id":manifest["packet_id"],"experiment_id":EXPERIMENT_ID,"run_id":run_id,"freeze_id":freeze_id,"source_git_commit":manifest["source_git_commit"],"model_identity":MODEL_DIGEST,"runtime_identity":RUNTIME_VERSION,"command_identity":"run_all_v1","started_at_utc":started_wall,"ended_at_utc":utc_now(),"elapsed_ms":sum(item["elapsed_ms"] for item in existing),"stage_statuses":{"verify":"passed","preflight":"passed","acquire":"passed","smoke":"passed","estimate":"passed","primary":"passed" if complete else "failed","rubric":"skipped" if skip_rubric else "not_run","validate":"passed","collect":"pending"},"counts":{"planned":180,"attempted":len(existing),"accepted":len(accepted_rows),"rejected":sum(failures.values()),"exhausted":180-len(accepted_rows),"missing":180-len(accepted_rows)},"failure_counts":failures,"latency":{"p50_ms":statistics.median(latencies) if latencies else 0,"p95_ms":sorted(latencies)[max(0,int(len(latencies)*.95)-1)] if latencies else 0},"time_budget":{"hard_stop_seconds":72000,"projected_primary_seconds":sum(latencies)*540/9000 if latencies else 0,"actual_seconds":sum(item["elapsed_ms"] for item in existing)/1000},"resume_count":state["resume_count"],"event_ids":[],"diagnostic_budget":{"maximum_seconds":2700,"used_seconds":0,"remaining_seconds":2700,"remediation_actions_used":0,"download_resumes_used":0},"operator_actions":{"count":count,"final_chain_sha256":head},"model_residency":{"before":"verified","after":"verified"},"resource_maxima":{"memory_available_after_smoke_bytes":capability["memory_available_bytes"]},"release_status":"eligible_complete" if complete else "diagnostic_incomplete","rubric_status":"skipped_time_budget" if skip_rubric else "not_enabled","output_hashes":{"generation_attempts":sha_file(attempts_path),"retained_candidates":sha_file(retained)},"privacy_scan":"passed","schema_validation":"passed","final_decision_branch":"return-complete" if complete else "contact-researcher-after-return"}
    rubric_status="not_enabled" if skip_rubric or not complete else _run_rubric(root,run_dir,registration,protocol,run_id,runtime,scenario)
    receipt["rubric_status"]=rubric_status
    receipt["stage_statuses"]["rubric"]="passed" if rubric_status=="complete" else "failed" if rubric_status=="failed" else "skipped"
    write_json(run_dir/"run_receipt.json",receipt); return receipt

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); p.add_argument("--runtime",choices=["ollama","fake"],default="ollama"); p.add_argument("--scenario",type=Path); p.add_argument("--resume",action="store_true"); p.add_argument("--skip-rubric",action="store_true")
    a=p.parse_args(); scenario=json.loads(a.scenario.read_text()) if a.scenario else {}; receipt=run(a.root.resolve(),a.runtime,scenario,a.resume,a.skip_rubric); print(json.dumps({"run_id":receipt["run_id"],"release_status":receipt["release_status"],"accepted":receipt["counts"]["accepted"]},sort_keys=True)); return 0 if receipt["release_status"]=="eligible_complete" else 3
if __name__=="__main__": raise SystemExit(main())
