# Bounded troubleshooting and return

Every diagnosis/remediation is registered, sanitized, budgeted, and appended to
the hash-chained action log. Free-text notes are operator-authored, at most 500
UTF-8 characters, and must contain no credentials, private paths, sentence text,
candidate text, case results, or raw diagnostics.

```bash
./run_all.sh diagnose --category memory
./run_all.sh diagnose --category model_load
./run_all.sh remediate --action restart-runtime
./run_all.sh note --text "Sanitized infrastructure-only observation"
./run_all.sh --status
./run_all.sh --collect-only
```

Registered diagnostic categories are checksum, archive, disk, memory, docker,
nvidia, arm64_image, model_load, network_download, timeout_process,
structured_interface, and checkpoint. Registered remediations are limited to
replacing the exact sealed transfer, clearing marker-owned packet cache,
restarting the packet-owned runtime, and releasing a verified stale packet lock.

Never delete outside `scratch/`, use sudo, install/update host drivers, change a
firewall or proxy, upload diagnostics, contact an unregistered endpoint, run an
arbitrary command, increase budgets, switch models/runtimes, change scientific
settings, or selectively rerun/repair candidates. At most 45 minutes, three
remediations, and two download resumes are allowed; do not begin diagnosis after
hour 19. Invalid input, EOF, or uncertainty means collect-and-stop.

Return exactly one archive and checksum from `output/return/`. A bundle marked
`diagnostic_incomplete` or `blocked` is evidence for diagnosis only and cannot
be imported as a scientific sample without the verifier's explicit diagnostic
flag.
