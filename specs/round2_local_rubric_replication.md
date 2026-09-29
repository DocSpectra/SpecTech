# Round 2 matched local-judge rubric replication

## Purpose

This bounded robustness study applies the already-frozen Qwen zero-shot 1--5
specificity rubric to the same 80 pilot sentences with local `gemma4:12b` and
`gpt-oss:20b`. It tests judge-choice sensitivity. It does not validate either
model, create human labels, train a scorer, judge generated text, or authorize
manuscript changes.

## Matched protocol and interface deltas

`configs/round2_local_rubric_replication_v1.json` checksum-binds the complete
Qwen protocol. The pilot manifest and order, rubric, prompt text, zero-shot
blindness, fresh conversation rule, one-attempt rule, JSON score schema,
determinism fixtures, decoding values, analysis targets, bootstrap method,
seed, and claim boundary are inherited unchanged.

Gemma uses Ollama `think=false`, matching Qwen. GPT-OSS requires the Harmony
reasoning interface and therefore uses the previously established `think=low`.
Its prediction budget is 128 rather than 16 because Harmony reasoning shares
that budget with the structured final response. Reasoning text is never
printed, retained, analyzed, or used for admission; only its character count
and SHA-256 may be recorded as runtime metadata. All other numeric options are
matched.

## Gates

Before project scoring, the protocol, code, and tests must be committed and a
separate freeze record must bind that commit. Installed tag, list ID, manifest
digest, backing blobs, Ollama version, pilot inputs, Qwen compact evidence, and
GranuScore inputs must match. Each judge must be the only Ollama-resident model
during its smoke and complete 80-row matrix. A three-fixture, three-repeat
non-project smoke must be valid and deterministic. Any mismatch, invalid
response, incomplete coverage, or requested scientific change stops the run.

## Analysis and privacy

Within each corpus, joint 10,000-replicate sentence bootstraps use the existing
corpus-specific seed derivation. Each new judge is compared with Ann. A/B/C,
the rowwise pooled human mean, frozen Qwen, SpeciTeller, Ko run01/run02/run03,
and secondary direction-aligned negative GranuScore. The two new judges are
also compared directly. Results are descriptive correspondence and paired
correlation contrasts, never accuracy, expert judgment, or a winner claim.

Ignored raw records and tracked compact scores contain IDs, scores, hashes, and
runtime metadata but no sentence text, labels, participant identities, notes,
reasoning text, or absolute paths.
