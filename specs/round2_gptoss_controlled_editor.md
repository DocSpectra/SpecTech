# Round 2 GPT-OSS controlled-editor comparison

## Purpose and freeze boundary

This fresh comparison asks whether the official local `gpt-oss:20b` tag can
produce a complete 60-row blinded-review packet under the already frozen Gemma
scientific protocol. It changes the model/runtime identity only. It does not
reuse or inspect Gemma outputs, tune from Gemma coverage, score outputs, run a
rubric, expand sources, or edit the manuscript.

Before any project sentence is sent, the delta config, schemas, this
specification, and outcome-blind equality tests must be committed. A second
freeze record binds that commit. The base Gemma config is checksum-pinned; the
runner may apply only the explicitly enumerated JSON-pointer overrides.

## Exact comparison controls

After merging the delta, every scientific value remains identical to the Gemma
protocol: all 60 source originals in their byte-frozen order and directions;
the system and direction prompts; dynamic word bounds; one-field structured
response; temperature, top-k, top-p, repeat penalty, context, prediction limit;
three attempts; master seed and seed derivation; mechanical gates; first-pass
retention; exact 60/60 packet-release prerequisite; blinded questions and
randomization seeds; 30/30 side balance; and human pass thresholds.

Only exact model/blob identity, runtime capability and residency controls,
GPT-OSS reasoning-interface handling, review-ID namespace, schemas, and raw/
compact output namespaces differ. Tests compare the fully merged protocol
against the frozen Gemma base and fail on any unapproved difference.

## Reasoning-interface difference

Official GPT-OSS uses the Harmony reasoning interface. Invented, non-project
smoke showed that `think=false` returns no final content under Ollama 0.32.5,
whereas `think="low"` returns exact structured JSON. The comparison therefore
freezes low reasoning as the minimum supported level. Reasoning text is never
printed, inspected, stored, gated, selected, or passed into later review; only
its character count and SHA-256 are recorded to verify interface behavior.

## Runtime and privacy

The runner stops Qwen, Gemma, and GPT-OSS, warms only the exact GPT-OSS tag on
an invented fixture, and requires exactly one resident model with the pinned
digest before and after generation. Every request and response names that exact
tag. GPT-OSS is unloaded in `finally`; all three installed models are preserved.

Raw originals, attempts, candidate text, reasoning text, packet, answer key,
and responses are excluded from tracked evidence. If fewer than 60 cases pass,
only aggregate non-content counts and hashes are tracked and no partial packet
is emitted. If all 60 pass, packet/key reversibility and exact cell balance are
validated without opening the key or scoring responses.

## Interpretation

Mechanical coverage concerns this frozen prompt--model--proxy interaction. It
does not establish factual correctness, semantic preservation, naturalness,
directional validity, or overall GPT-OSS editing ability. Only a complete
blinded human review can make the prespecified pilot decision.
