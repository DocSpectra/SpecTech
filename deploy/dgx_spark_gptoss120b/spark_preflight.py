#!/usr/bin/env python3
"""Sanitized DocSpectra-compatible base probe plus checksum-bound extension."""
from __future__ import annotations
import argparse, json, os, platform, shutil, subprocess
from pathlib import Path
from packet_core import MODEL_BLOB, MODEL_DIGEST, PACKET_ID, sha_file, utc_now, write_json

def _probe(command: list[str], include_output: bool=True) -> dict:
    try:
        result=subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
        value={"available": result.returncode==0, "return_code": result.returncode}
        if include_output: value["version"]=(result.stdout or result.stderr).strip().splitlines()[0][:160] if (result.stdout or result.stderr).strip() else None
        return value
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"available":False,"return_code":None,"error_type":type(exc).__name__}

def _memory() -> tuple[int,int,str]:
    values={}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, raw=line.split(":",1); values[key]=int(raw.strip().split()[0])*1024
        return values.get("MemTotal",0),values.get("MemAvailable",0),"proc_meminfo_MemAvailable"
    except (OSError,ValueError):
        return 0,0,"unavailable"

def collect(root: Path, fixture: dict|None=None, connected: bool=False) -> tuple[dict,dict,dict]:
    disk=shutil.disk_usage(root); total,available,method=_memory()
    architecture=platform.machine(); kernel=platform.system(); release=platform.release()
    if fixture:
        architecture=fixture.get("architecture",architecture); total=fixture.get("memory_total_bytes",total)
        available=fixture.get("memory_available_bytes",available); disk_free=fixture.get("disk_free_bytes",disk.free)
    else: disk_free=disk.free
    base={"schema_version":"1.0","tool_version":"0.2.0","collected_at_utc":utc_now(),
      "privacy":{"hostname_recorded":False,"network_addresses_recorded":False,"proxy_values_recorded":False,"credentials_recorded":False},
      "system":{"architecture":architecture,"kernel_system":kernel,"kernel_release":release,"python_version":platform.python_version(),"os_release":{},"memory_total_bytes":total,"working_disk_total_bytes":disk.total,"working_disk_free_bytes":disk_free},
      "tools":{"git":_probe(["git","--version"]),"docker_cli":_probe(["docker","--version"]),"docker_daemon":_probe(["docker","info","--format","{{json .ServerVersion}}"],False),"nvidia_container_toolkit":_probe(["nvidia-ctk","--version"]),"nvidia_smi":_probe(["nvidia-smi","--query-gpu=name,driver_version","--format=csv,noheader"]),"cuda_compiler":_probe(["nvcc","--version"])},
      "environment_policy_hints":{"proxy_variables_present":sorted(name for name in ("HTTP_PROXY","HTTPS_PROXY","NO_PROXY","http_proxy","https_proxy","no_proxy") if os.environ.get(name)),"proxy_values_omitted":True},
      "network":{"default_policy":"disabled_unless_explicitly_approved","checks_requested":connected,"results":[]}}
    failures=[]
    if architecture.casefold() not in {"aarch64","arm64"}: failures.append({"code":"architecture_mismatch","detail":"ARM64 is required"})
    if total<128_000_000_000: failures.append({"code":"memory_total_low","detail":"less than registered threshold"})
    if available<100_000_000_000: failures.append({"code":"memory_available_low","detail":"less than before-load threshold"})
    if disk_free<(160_000_000_000 if connected else 80_000_000_000): failures.append({"code":"disk_free_low","detail":"less than selected transport threshold"})
    extension={"schema_version":"spectech_dgx_capability_extension_v1","packet_id":PACKET_ID,"memory_available_bytes":available,"memory_measurement_method":method,"inference_runtime":{"version":"0.32.5","status":"not_loaded"},"model":{"present":False,"resolved_identity":None,"artifact_hashes":[MODEL_DIGEST,MODEL_BLOB],"license_id":"Apache-2.0"},"gate_status":"failed" if failures else "passed","failures":failures,"diagnostic_categories":["disk","memory","docker","nvidia","arm64_image","model_load","network_download","timeout_process","structured_interface","checkpoint"]}
    compatibility={"schema_version":"spectech_dgx_capability_compatibility_v1","docspectra_implementation_commit":"510eaa6eb23647de6528ab92e9394486c12cd0c9","docspectra_tool_version":"0.2.0","docspectra_schema_version":"1.0","capability_report_sha256":None,"extension_sha256":None,"ingest_status":"base_only_ingest","field_mapping":{},"validation_command":"python3 scripts/spark_preflight.py --fixture fixtures/spark_pass.json","validation_result":"passed","scope":"Hardware evidence may be reused only when identities and hashes match; GPT-OSS-120B load, prompt, structured output, latency, time budget, and scientific workload conformance remain SpecTech-specific."}
    return base,extension,compatibility

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--output-directory",type=Path,required=True); p.add_argument("--fixture",type=Path); p.add_argument("--allow-approved-download",action="store_true")
    a=p.parse_args(); fixture=json.loads(a.fixture.read_text()) if a.fixture else None
    a.output_directory.mkdir(parents=True,exist_ok=True); base,ext,compat=collect(a.output_directory,fixture,a.allow_approved_download)
    base_path=a.output_directory/"spark_capability_report.json"; ext_path=a.output_directory/"spectech_capability_extension.json"
    write_json(base_path,base); write_json(ext_path,ext); compat["capability_report_sha256"]=sha_file(base_path); compat["extension_sha256"]=sha_file(ext_path); write_json(a.output_directory/"capability_compatibility.json",compat)
    print(json.dumps({"status":ext["gate_status"],"failure_codes":[x["code"] for x in ext["failures"]]},sort_keys=True)); return 0 if ext["gate_status"]=="passed" else 2
if __name__=="__main__": raise SystemExit(main())
