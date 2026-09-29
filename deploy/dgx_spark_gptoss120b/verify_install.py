#!/usr/bin/env python3
"""Verify an extracted packet without needing the transfer archive."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from packet_core import privacy_scan_bytes, require_under, safe_relative, sha_file
def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); a=p.parse_args(); root=a.root.resolve(); manifest=json.loads((root/"packet_manifest.json").read_text())
    for item in manifest["payload_files"]:
        relative=safe_relative(item["path"]); path=require_under(root,root/Path(*relative.parts))
        if not path.is_file() or path.is_symlink() or path.stat().st_size!=item["size_bytes"] or sha_file(path)!=item["sha256"]: raise ValueError(f"payload mismatch: {relative}")
        if item["content_class"]!="input": privacy_scan_bytes(path.read_bytes(),str(relative))
    print(json.dumps({"status":"passed","packet_id":manifest["packet_id"],"files":len(manifest["payload_files"])},sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
