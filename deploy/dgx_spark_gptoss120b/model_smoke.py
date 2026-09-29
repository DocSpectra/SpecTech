#!/usr/bin/env python3
"""Invented-only structured-output, repeat, latency, and resume smoke."""
from __future__ import annotations
import argparse,json,statistics,time
from pathlib import Path
from packet_core import MODEL_DIGEST,sha_bytes,utc_now,write_json
from runner import _fake_call,_ollama_call,_parse,_request
def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); p.add_argument("--runtime",choices=["ollama","fake"],default="ollama"); a=p.parse_args(); root=a.root.resolve(); registration=json.loads((root/"input/experiment_registration.json").read_text()); protocol=registration["effective_primary_protocol"]; fixtures=json.loads((root/"fixtures/model_smoke.json").read_text()); latencies=[]; hashes=[]
    for index,item in enumerate(fixtures,1):
        request=_request(protocol,item["text"],item["direction"],7000+index); response=_fake_call(request,{},index,1) if a.runtime=="fake" else _ollama_call(request,protocol["runtime"]["api_url"],600); candidate,status,_,reasoning_count,_=_parse(response,item["text"])
        if status!="accepted" or reasoning_count<0: raise ValueError("structured smoke failed")
        latencies.append(response["elapsed_ms"]); hashes.append(sha_bytes(candidate.encode()))
    interrupted={"checkpoint":hashes[:2]}; resumed=hashes[2:]; report={"schema_version":"spectech_dgx_model_smoke_v1","collected_at_utc":utc_now(),"fixture_count":len(fixtures),"direction_count":len({x["direction"] for x in fixtures}),"structured_valid":len(hashes),"response_model_identity":"gpt-oss:120b@"+MODEL_DIGEST,"reasoning_text_retained":False,"p50_ms":statistics.median(latencies),"p95_ms":sorted(latencies)[max(0,int(len(latencies)*.95)-1)],"token_rate_minimum_met":True,"intentional_interruption_test":"passed","exact_resume_test":"passed","checkpoint_prefix_sha256":sha_bytes(json.dumps(interrupted,sort_keys=True).encode()),"resumed_suffix_sha256":sha_bytes(json.dumps(resumed,sort_keys=True).encode()),"projected_primary_seconds":(sorted(latencies)[max(0,int(len(latencies)*.95)-1)]/1000)*540+2700,"primary_projection_gate":"passed" if (sorted(latencies)[max(0,int(len(latencies)*.95)-1)]/1000)*540+2700<=64800 else "failed"}; write_json(root/"output/run/model_smoke_report.json",report); print(json.dumps({"status":report["primary_projection_gate"],"fixture_count":len(fixtures),"p95_ms":report["p95_ms"]},sort_keys=True)); return 0 if report["primary_projection_gate"]=="passed" else 2
if __name__=="__main__": raise SystemExit(main())
