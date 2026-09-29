#!/usr/bin/env bash
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_DIR="$ROOT/output/run"
SCRATCH_DIR="$ROOT/scratch"
mkdir -p "$RUN_DIR" "$ROOT/output/return" "$SCRATCH_DIR"

collect_on_exit() {
  code=$?
  trap - EXIT INT TERM
  if [[ $code -ne 0 ]]; then
    "$PYTHON_BIN" "$ROOT/scripts/collect_return.py" --root "$ROOT" --blocked-stage "${ACTIVE_STAGE:-collect}" || true
  fi
  exit "$code"
}

execute_workflow() {
  trap collect_on_exit EXIT INT TERM
  ACTIVE_STAGE=verify
  "$PYTHON_BIN" "$ROOT/scripts/verify_install.py" --root "$ROOT"
  ACTIVE_STAGE=preflight
  preflight_args=(--output-directory "$RUN_DIR")
  if [[ "${FIXTURE_MODE:-0}" == 1 ]]; then preflight_args+=(--fixture "$ROOT/fixtures/spark_pass.json"); fi
  if [[ "${CONNECTED_MODE:-0}" == 1 ]]; then preflight_args+=(--allow-approved-download); fi
  "$PYTHON_BIN" "$ROOT/scripts/spark_preflight.py" "${preflight_args[@]}"
  ACTIVE_STAGE=acquire
  if [[ "${FIXTURE_MODE:-0}" != 1 ]]; then
    acquire_args=(--root "$ROOT")
    if [[ "${CONNECTED_MODE:-0}" == 1 ]]; then acquire_args+=(--allow-approved-download); else acquire_args+=(--offline-runtime "${OFFLINE_RUNTIME:?offline runtime required}"); fi
    "$PYTHON_BIN" "$ROOT/scripts/acquire.py" "${acquire_args[@]}"
    mkdir -p "$SCRATCH_DIR/runtime" "$SCRATCH_DIR/model-store"
    tar --zstd -xf "$SCRATCH_DIR/acquisition/ollama-linux-arm64.tar.zst" -C "$SCRATCH_DIR/runtime"
    OLLAMA_BIN="$SCRATCH_DIR/runtime/bin/ollama"
    [[ -x "$OLLAMA_BIN" ]] || { echo "registered runtime binary not found" >&2; return 2; }
    export OLLAMA_MODELS="$SCRATCH_DIR/model-store" OLLAMA_HOST="127.0.0.1:11434"
    "$OLLAMA_BIN" serve >"$RUN_DIR/runtime_redacted.log" 2>&1 & echo $! >"$SCRATCH_DIR/ollama.pid"
    for _ in $(seq 1 60); do "$OLLAMA_BIN" list >/dev/null 2>&1 && break; sleep 1; done
    if [[ "${CONNECTED_MODE:-0}" == 1 ]]; then "$OLLAMA_BIN" pull gpt-oss:120b; else tar --zstd -xf "${OFFLINE_MODEL_STORE:?offline model store required}" -C "$SCRATCH_DIR/model-store"; fi
    manifest="$SCRATCH_DIR/model-store/manifests/registry.ollama.ai/library/gpt-oss/120b"
    [[ -f "$manifest" ]] || { echo "registered model manifest absent" >&2; return 2; }
    [[ "$(sha256sum "$manifest" | cut -d' ' -f1)" == "a951a23b46a1f6093dafee2ea481d634b4e31ac720a8a16f3f91e04f5a40ecd9" ]] || { echo "model identity mismatch" >&2; return 2; }
  fi
  ACTIVE_STAGE=smoke
  runtime=ollama; [[ "${FIXTURE_MODE:-0}" == 1 ]] && runtime=fake
  "$PYTHON_BIN" "$ROOT/scripts/model_smoke.py" --root "$ROOT" --runtime "$runtime"
  ACTIVE_STAGE=estimate
  "$PYTHON_BIN" -c 'import json,sys; r=json.load(open(sys.argv[1],encoding="utf-8")); raise SystemExit(0 if r["primary_projection_gate"]=="passed" else 2)' "$RUN_DIR/model_smoke_report.json"
  ACTIVE_STAGE=primary
  runner_args=(--root "$ROOT" --runtime "$runtime")
  [[ "${RESUME_MODE:-0}" == 1 ]] && runner_args+=(--resume)
  [[ "${FIXTURE_MODE:-0}" == 1 ]] && runner_args+=(--scenario "$ROOT/fixtures/fake_success.json")
  "$PYTHON_BIN" "$ROOT/scripts/runner.py" "${runner_args[@]}"
  ACTIVE_STAGE=validate
  ACTIVE_STAGE=collect
  "$PYTHON_BIN" "$ROOT/scripts/collect_return.py" --root "$ROOT"
  trap - EXIT INT TERM
}

case "${1:-}" in
  status|--status) exec "$PYTHON_BIN" "$ROOT/scripts/operator_cli.py" --root "$ROOT" status ;;
  --collect-only) exec "$PYTHON_BIN" "$ROOT/scripts/collect_return.py" --root "$ROOT" ;;
  diagnose) shift; exec "$PYTHON_BIN" "$ROOT/scripts/operator_cli.py" --root "$ROOT" diagnose "$@" ;;
  note) shift; exec "$PYTHON_BIN" "$ROOT/scripts/operator_cli.py" --root "$ROOT" note "$@" ;;
  remediate) shift; exec "$PYTHON_BIN" "$ROOT/scripts/operator_cli.py" --root "$ROOT" remediate "$@" ;;
esac
CONNECTED_MODE="${CONNECTED_MODE:-0}" FIXTURE_MODE="${FIXTURE_MODE:-0}" RESUME_MODE="${RESUME_MODE:-0}"
OFFLINE_RUNTIME="${OFFLINE_RUNTIME:-}" OFFLINE_MODEL_STORE="${OFFLINE_MODEL_STORE:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --allow-approved-download) CONNECTED_MODE=1 ;;
    --offline-runtime) OFFLINE_RUNTIME="$2"; shift ;;
    --offline-model-store) OFFLINE_MODEL_STORE="$2"; shift ;;
    --resume) RESUME_MODE=1 ;;
    --fixture-mode) FIXTURE_MODE=1 ;;
    --execute) execute_workflow; exit $? ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done
export CONNECTED_MODE FIXTURE_MODE RESUME_MODE OFFLINE_RUNTIME OFFLINE_MODEL_STORE PYTHON_BIN
exec timeout --foreground --signal=TERM --kill-after=60s 72000 "$0" --execute
