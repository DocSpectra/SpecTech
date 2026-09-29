#!/usr/bin/env python3
"""Acquire only the registered ARM64 runtime and model through approved paths."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, urllib.request
from pathlib import Path
RUNTIME_URL="https://github.com/ollama/ollama/releases/download/v0.32.5/ollama-linux-arm64.tar.zst"
RUNTIME_SHA="aa7e06b5683ee66c4a3ec68ea7236db43b5a5d0821f0dfe2c5a215f4462bddf4"

def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); p.add_argument("--allow-approved-download",action="store_true"); p.add_argument("--offline-runtime",type=Path)
    a=p.parse_args(); cache=a.root/"scratch/acquisition"; cache.mkdir(parents=True,exist_ok=True); target=cache/"ollama-linux-arm64.tar.zst"
    if a.offline_runtime: shutil.copyfile(a.offline_runtime,target)
    elif a.allow_approved_download:
        with urllib.request.urlopen(RUNTIME_URL,timeout=120) as source,target.open("wb") as out: shutil.copyfileobj(source,out)
    else: raise SystemExit("acquisition requires --allow-approved-download or --offline-runtime")
    if sha(target)!=RUNTIME_SHA: raise ValueError("runtime archive digest mismatch")
    print(json.dumps({"status":"verified","runtime_version":"0.32.5","runtime_sha256":RUNTIME_SHA},sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
