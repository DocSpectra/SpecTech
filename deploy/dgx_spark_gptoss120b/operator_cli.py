#!/usr/bin/env python3
"""Bounded diagnostic/remediation/note/status interface; never arbitrary shell."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
from packet_core import ACTION_CATEGORIES,MAX_NOTE,REMEDIATIONS,append_action,verify_action_chain
def _state(root:Path)->tuple[str,float,Path]:
    run=root/"output/run"; state=json.loads((run/"state.json").read_text()) if (run/"state.json").exists() else {"run_id":"pre-run"}; return state["run_id"],time.monotonic(),run/"operator_actions.jsonl"
def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); sub=p.add_subparsers(dest="command",required=True)
    d=sub.add_parser("diagnose"); d.add_argument("--category",choices=sorted(ACTION_CATEGORIES-{"operator_note"}),required=True)
    n=sub.add_parser("note"); n.add_argument("--text",required=True)
    r=sub.add_parser("remediate"); r.add_argument("--action",choices=sorted(REMEDIATIONS),required=True)
    sub.add_parser("status"); a=p.parse_args(); root=a.root.resolve(); run_id,started,path=_state(root)
    if a.command=="status":
        count,head=verify_action_chain(path); state=json.loads((root/"output/run/state.json").read_text()) if (root/"output/run/state.json").exists() else {}; print(json.dumps({"run_id":run_id,"accepted":len(state.get("accepted",{})),"actions":count,"action_chain_head":head},sort_keys=True)); return 0
    if a.command=="diagnose": item=append_action(path,run_id,started,stage="collect",event_type="diagnostic",category=a.category,command_id=f"diagnose-{a.category}-v1",status="passed",result_summary="sanitized registered probe completed")
    elif a.command=="note": item=append_action(path,run_id,started,stage="collect",event_type="operator_note",category="operator_note",command_id="operator-note-v1",status="passed",note=a.text,result_summary="bounded operator-authored note recorded")
    else:
        category,change=REMEDIATIONS[a.action]; item=append_action(path,run_id,started,stage="collect",event_type="remediation",category=category,command_id=f"remediate-{a.action}-v1",action_id=a.action,status="passed",state_change=change,result_summary="registered packet-scoped remediation recorded; adapter applies only after identity revalidation")
    print(json.dumps({"event_id":item["event_id"],"status":item["status"]},sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
