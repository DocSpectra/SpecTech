#!/usr/bin/env python3
"""Verify a returned bundle before any local scientific review."""
from __future__ import annotations
import argparse, csv, io, json, tarfile
from pathlib import Path
from packet_core import inspect_archive, sha_bytes, sha_file

def verify(path:Path,allow_diagnostic:bool=False)->dict:
    members=inspect_archive(path); root=members[0].name.split("/",1)[0]
    with tarfile.open(path,"r:gz") as archive:
        by_name={m.name.split("/",1)[1]:m for m in members}; manifest=json.loads(archive.extractfile(by_name["return_manifest.json"]).read()); expected={item["path"]:item for item in manifest["files"]}
        if set(expected)!=(set(by_name)-{"return_manifest.json"}): raise ValueError("return inventory mismatch")
        for name,item in expected.items():
            data=archive.extractfile(by_name[name]).read()
            if len(data)!=item["size_bytes"] or sha_bytes(data)!=item["sha256"]: raise ValueError(f"return hash mismatch: {name}")
        if manifest["release_status"]!="eligible_complete" and not allow_diagnostic: raise ValueError("diagnostic return requires --allow-diagnostic")
        if manifest["release_status"]=="eligible_complete":
            rows=list(csv.DictReader(io.StringIO(archive.extractfile(by_name["retained_candidates.csv"]).read().decode())))
            if len(rows)!=180 or len({r["candidate_id"] for r in rows})!=180: raise ValueError("eligible return is not exactly 180 unique candidates")
    return manifest
def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("archive",type=Path); p.add_argument("--sha256",type=Path); p.add_argument("--allow-diagnostic",action="store_true"); a=p.parse_args()
    if a.sha256 and sha_file(a.archive)!=a.sha256.read_text().split()[0]: raise ValueError("external checksum mismatch")
    m=verify(a.archive,a.allow_diagnostic); print(json.dumps({"status":"passed","release_status":m["release_status"],"run_id":m["run_id"]},sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
