# DGX Spark GPT-OSS-120B experiment packet

## Scientific freeze

The primary experiment is a model/runtime delta over
`round2_human_first_reranking_v1`. It uses the same ordered 60 source
sentences, directions, prompts, response schema, sampling controls, context,
output budget, three attempts per slot, and integrity-only first-pass
admission. It creates three fresh slots per source under a new seed namespace.
Exactly 180/180 retained candidates are required for `eligible_complete`.

Only study/schema/output namespaces, exact GPT-OSS-120B and runtime identity,
Harmony low reasoning, sole-residency/resource controls, the fresh seed
namespace, and remote provenance fields may differ. Equality tests compare the
merged protocol and reject every other difference. No author edit, previous
candidate, scorer value, selector, human label, note, answer key, or corpus
identity enters a model request. Predictor scoring and human evaluation remain
local later work.

The optional rubric is a separately registered secondary delta over the exact
Qwen/local-rubric bases. It runs only after complete primary validation and a
fresh time gate. It preserves 80 rows/order, rubric wording, single attempt,
one-field integer JSON, zero temperature, and blindness.

## Packet state machine

`run_all.sh` verifies the packet, runs a sanitized DocSpectra-compatible
preflight, conditionally acquires or loads the exact artifacts, runs invented
fixtures, applies the conservative projection, generates the primary matrix,
validates it, optionally runs the rubric, and always collects exactly one
return archive and external checksum while the packet filesystem is writable.

Resume revalidates every packet/freeze/input/model/runtime/prompt/schema
identity, the append-only attempt log, atomic state index, and accepted rows.
It skips only a completed slot and resumes the next missing frozen unit with
the original attempt index and seed. It never repairs text, deletes a failure,
replaces an accepted candidate, or selectively reruns a case.

## Diagnostics and decisions

Registered read-only categories are checksum, archive, disk, memory, Docker,
NVIDIA, ARM64 image, model load, approved download, timeout/process,
structured interface, and checkpoint. Registered remediations are limited to
the same sealed transfer/download, marker-owned cache cleanup, packet-owned
runtime restart, verified stale-lock release, pre-project connected/offline
path switch for identical artifacts, and exact checkpoint resume. Budgets are
45 minutes, three remediations, and two download resumptions. Invalid menu
input, EOF, timeout, unregistered commands, case inspection, scientific edits,
artifact substitution after project calls, and failure deletion select
collect-and-stop.

Every action is an append-only hash-chained event. Operator notes are bounded,
sanitized, and scientifically inert. Final branches are `fix-and-resume`,
`collect-and-stop`, or `contact-researcher-after-return`; no branch pauses for
a researcher reply.

## Time and privacy

The primary projection is:

```text
acquisition_if_session + cold_model_load
  + 540 * generation_smoke_p95
  + 1800 seconds validation/resume allowance
  + 900 seconds collection allowance
```

It must be no more than 18 hours. Optional rubric admission adds
`80 * rubric_smoke_p95 + 600 seconds`. Collection begins by hour 19 and the
launcher hard-aborts at hour 20. A stopped run returns auditable incomplete
evidence.

Source and optional rubric sentences are controlled packet inputs and never
appear in general logs. Reasoning text, credentials, environment values,
hostnames, addresses, private paths, human labels, predictor scores, earlier
candidates, model weights, and container layers are excluded from the return
bundle. Raw diagnostics are sanitized into stable codes before collection.

## Eligibility boundary

`eligible_complete` means exact identities, schemas, hashes, privacy, order,
and 180 unique retained rows all pass. Any exhausted or missing slot produces
`diagnostic_incomplete`; identity, freeze, privacy, or unsafe-path failures are
`blocked`. Partial candidates are diagnostic only and local import rejects
them unless an explicit diagnostic flag is provided.
