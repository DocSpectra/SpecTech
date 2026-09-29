# DGX Spark GPT-OSS-120B operator card

Keep the transferred archive sealed until its external SHA-256 matches. Extract
it on one DGX Spark into a filesystem with at least 160 GB free for the connected
path. Do not inspect files under `input/` or generated candidate text.

Recommended connected one-liner (explicitly permits only the registered GitHub
release and Ollama registry acquisition path):

```bash
./run_all.sh --allow-approved-download
```

For a prepared offline handoff, keep both separately checksummed payloads beside
the extracted source packet and run:

```bash
./run_all.sh --offline-runtime /transport/ollama-linux-arm64.tar.zst --offline-model-store /transport/gptoss120b-model-store.tar.zst
```

The command verifies the sealed packet, runs a sanitized local preflight,
acquires or loads only registered artifacts, runs invented-fixture smoke and the
hard time gate, generates the 180 primary candidates, and always collects one
return archive under `output/return/`. The optional rubric is secondary and is
skipped by this conservative one-day profile to protect primary collection.

If interrupted, rerun the same transport command with `--resume`. Read-only
status is `./run_all.sh --status`; safe final collection is
`./run_all.sh --collect-only`. Return the single `.tar.gz` and matching
`.sha256` file without opening candidate text. The 20-hour machine hard stop
reserves at least four hours for transfer and local verification.

## Troubleshooting

Use only the registered commands in `TROUBLESHOOTING_RETURN.md`. Do not change
prompts, seeds, sampling, inputs, attempts, identities, time budgets, or stop
rules; do not run arbitrary shell repairs. Contact the researcher only after
returning the diagnostic archive.
