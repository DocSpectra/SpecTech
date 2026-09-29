#!/usr/bin/env python3
"""Verify source packet structure and every payload digest without extraction."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from packet_core import sha_file, verify_archive

def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("archive", type=Path); parser.add_argument("--sha256", type=Path)
    args=parser.parse_args()
    if args.sha256:
        expected=args.sha256.read_text(encoding="ascii").split()[0]
        if sha_file(args.archive) != expected: raise ValueError("external checksum mismatch")
    manifest=verify_archive(args.archive)
    print(json.dumps({"status":"passed","packet_id":manifest["packet_id"],"files":len(manifest["payload_files"])}, sort_keys=True))
    return 0
if __name__ == "__main__": raise SystemExit(main())
