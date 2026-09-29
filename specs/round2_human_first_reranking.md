# Human-first controlled-edit reranking pilot

## Purpose

Test whether frozen sentence-level metrics can help choose among independently
generated controlled edits. Human A/B specificity judgments are the only
evaluation outcome. Metric scores and the older automatic proxies are hidden
diagnostics and never admission criteria.

## Frozen design

The exact existing 60 originals and directions are used in their checksum-bound
order. Gemma 4 12B, selected before the earlier model outcomes, generates three
fresh candidate slots per original (180 total). Every slot uses a fresh context,
a new namespace/seed, and a minimal direction prompt. Prior Gemma, GPT-OSS, and
Qwen text is neither read nor reused.

Only response integrity can trigger a retry: exact structured output, one
nonempty single sentence/line, 5--2000 characters, changed after normalization,
and compatible script. The earlier specificity, length, anchor, marker,
polarity, and modality checks are computed afterward for audit but cannot
exclude a row. Missing any slot after three attempts suppresses the packet.

After complete generation, every original and candidate is scored without
retraining by the exact existing SpeciTeller release, all three retained Ko
checkpoints for its corpus, and native GranuScore. Each Ko run is primary; their
mean is secondary. GranuScore is direction-aligned by negating its native
higher-is-coarser score. Frozen policies select among three candidates using
direction-appropriate delta, with lowest slot as the deterministic tie break.
Slot 1 is the unguided baseline. A cross-model rank consensus is secondary.

## Blind packet

The reviewer receives 180 randomly ordered rows with exactly four columns:
`Review ID | A | B | Score`. Source/candidate side is balanced 90/90 overall
and 50/50 within each hidden corpus-by-direction cell. Score is restricted to
1, 2, 3, 4, 5, or X. The packet hides every identifier, condition, role, score,
proxy, and selector other than opaque Review ID and the two sentences.

The private ignored key must prove a bijection among all sources, candidates,
packet rows, scorer rows, proxy rows, and policy mappings. It must not be opened
or exposed after construction until all 180 responses are frozen. No human
outcomes, manuscript claims, or rubric study belong to this sprint.
