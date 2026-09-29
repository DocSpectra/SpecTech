"""Download one immutable build input and fail on a SHA-256 mismatch."""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: fetch_verified.py URL OUTPUT EXPECTED_SHA256")
    url, output_raw, expected = sys.argv[1:]
    output = Path(output_raw)
    digest = hashlib.sha256()
    request = urllib.request.Request(url, headers={"User-Agent": "SpecTech-GranuScore/1"})
    with urllib.request.urlopen(request, timeout=120) as response, output.open("wb") as handle:
        while block := response.read(1024 * 1024):
            handle.write(block)
            digest.update(block)
    observed = digest.hexdigest()
    if observed != expected.lower():
        output.unlink(missing_ok=True)
        raise SystemExit(f"SHA-256 mismatch for {url}: {observed}")
    print(f"verified {output}: {observed}")


if __name__ == "__main__":
    main()
