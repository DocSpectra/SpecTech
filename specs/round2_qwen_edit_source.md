# Round 2 QE-A Qwen edit-source protocol

## Purpose and frozen boundary

QE-A asks whether scorer responses to the 60 existing author/domain-informed
controlled edits persist when an identified local open-weight model generates
direction-matched counterparts from the same originals. The machine-readable
source of truth is `configs/round2_qwen_edit_source_v1.json`.

The method is frozen before project generation or scorer outcomes. Generation
receives only the original sentence and its pre-existing direction class. It
never receives the author edit, SpeciTeller/Ko/GranuScore/Qwen scores, human
labels, corpus identity, or another case. The author-edited text is read only
after generation closes, for paired scoring. No prompt, threshold, retry,
inclusion, or analysis rule may be tuned after generated text or scores appear.

## Frozen cases and order

The two ignored source CSVs are checksum-addressed in the config. Their 60
originals, stable source IDs, direction classes, and byte-frozen row order are
immutable. Ansible contributes 8 add-specific, 12 de-specify, and 10 neutral
rewrite cases; GitHub contributes 10 of each. QE-A adds no hand-authored cases
and no Wikipedia arm.

Each case ID is the SHA-256 of corpus ID, source sentence ID, direction, and
one-based source position separated by NUL bytes. The ordered digest also
binds each original-text SHA-256. This permits release checks without tracking
source or edit text.

## Generator and attempts

The generator is local `qwen3:14b` under Ollama 0.32.5, model-list ID
`bdbd181c33f2`, blob
`a8cc1361f3145dc01f6d77c6c82c9116b9ffe3c97b34716fe20418455876c40e`.
Thinking is disabled. Every attempt uses a fresh conversation, structured JSON,
the frozen direction prompt, temperature 0.2, top-k 20, top-p 0.9, and a
case/attempt seed deterministically derived from master seed 20260810.

At most three attempts are generated in fixed order. Generation stops at the
first passing attempt. Failed attempts and reason codes remain accounted for;
passing output is retained without manual selection; no later alternative is
generated. A case with no passing attempt is excluded. The identical prompt is
used for every retry and no failed output is shown back to the model.

### Outcome-blind interface correction

The first execution produced no generated text: all 180 calls returned the
same Ollama HTTP 400 grammar-initialization error because Ollama 0.32.5 does not
accept the JSON-Schema `minLength`/`maxLength` keywords for a structured string
field. The complete failed pass is retained under the ignored raw-output
directory. Before any scorer output, those two redundant response-schema
keywords were removed. The independently frozen universal gate still enforces
5--2000 characters. Prompts, decoding, case seeds/order, attempts, gates,
inclusion, scoring, joins, analysis, and claims were unchanged. A non-project
fixture demonstrated the failure and then the corrected interface.

The first GranuScore host invocation also exited at argument parsing before
writing any score row because the wrapper repeated the image's already-frozen
`python /opt/granuscore/score_csv.py` entrypoint. Removing only the duplicated
host arguments made the wrapper invoke that same frozen image entrypoint; no
image, runner, input, scoring option, row, join, or analysis rule changed.

## Automated safeguards

The gates are fixed mechanical and lexical-semantic safeguards. They require
valid single-field JSON, one nonempty changed line, frozen length-ratio and
content-anchor thresholds, unchanged polarity/modal tokens, and
direction-specific concrete-marker behavior. Add-specific edits retain all
original concrete markers and add words or markers; de-specific edits shorten
or reduce markers without adding markers; neutral paraphrases preserve the
concrete-marker multiset.

These tests cannot establish factual correctness, semantic equivalence, or that
specificity alone changed. They are inclusion safeguards, not validation. No
human validation is planned. If total acceptance is below 80% or any
corpus-by-direction cell is below 75%, accepted rows are still scored and all
failures packaged, but paper-facing paired source inference stops without
prompt or gate tuning.

## Scoring and joins

All 60 cases finish generation/gating before any scorer runs. Author edits and
accepted Qwen edits are then scored with frozen SpeciTeller, all three retained
per-corpus Ko checkpoints, and official GranuScore. Ko inference loads retained
checkpoints without retraining and preserves each checkpoint's original target
corpus normalization/vocabulary context. Ko run01/run02/run03 are primary
instances; their arithmetic mean is secondary. GranuScore remains native
higher-is-coarser. Qwen does not judge its own generations.

Scorer outputs remain uninspected comparatively until all required identities,
ranges, coverage counts, and exact joins pass. Each edited value is paired with
the same original under the same model/run. Raw values are never compared across
incompatible scales.

## Frozen estimands

Within each model, corpus, direction, and source arm, report count, mean and
median native delta, mean absolute delta, and directional-response rate for
add/de-specific edits. On accepted matched cases, report paired Qwen-minus-
author differences in mean signed delta, mean absolute delta, and direction
rate plus concordance. Neutral rewrites receive signed and absolute summaries
without an arbitrary success threshold.

Use 10,000 joint case-bootstrap replicates within corpus-by-direction cells,
master seed 20260810, and percentile 95% intervals. Results are descriptive;
there are no p-values, winner/superiority claims, or accuracy conclusions.

## Privacy and release

Tracked compact evidence may contain case IDs, directions, source labels,
scores, aggregates, counts, and hashes. It must not contain originals, author
edits, generated text, labels, participant information, nonredistributable Ko
checkpoints, or host paths. Text-bearing artifacts remain ignored and can be
replayed only with the frozen inputs/model identities.

## Immutable limitations

- Every Qwen edit is synthetic and unverified by humans.
- Automated gates do not validate truth, meaning preservation, or a
  single-factor manipulation.
- The generator is one local model/blob and does not represent all edit sources.
- The exact Ko 2019 reproduction remains failed; QE-A uses only the three
  retained runs of the separately disclosed maintained post-publication code.
- Evidence characterizes scorer response robustness, not latent specificity,
  documentation quality, or downstream utility.
