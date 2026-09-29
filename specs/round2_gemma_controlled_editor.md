# Round 2 Gemma controlled-editor pilot

## Purpose and boundary

This post-final pilot tests whether an independently generated controlled-edit
arm can survive direct author-blinded quality review on the existing 60 source
sentences. It addresses the reviewer concern that the published controlled
edits were author-crafted with domain and hypothesis knowledge. It does not
score sentences, run the Gemma rubric, expand to new sources, or change the
manuscript.

The complete machine-readable contract is
`configs/round2_gemma_controlled_editor_v1.json`. The config, schemas, tests,
and this specification must be committed before any project sentence is sent
to Gemma. A second immutable freeze record binds that commit before generation.

## Outcome isolation

The method uses only the 60 frozen originals and their existing three direction
labels. Prompt and mechanical-gate choices may learn from the documented QE-R
process failures and score-blind positive-control constraints. They may not use
Qwen text, case-level Qwen outcomes, any predictor or rubric score, prior human
labels, or author-edit wording. Each Gemma request contains only one original
sentence and its direction-matched frozen prompt.

The out-of-box smoke is limited to invented non-project fixtures. Smoke output
may establish API, structured-response, thinking-disable, GPU-residency, and
timing functionality only; it cannot select project outcomes.

## Model and residency

The generator is the official Ollama workstation tag `gemma4:12b`, exact tag
digest `4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c`,
under Ollama 0.32.5. The two exact local blobs, sizes, quantization, parameters,
license, and official source are frozen in the config.

Before controlled calls, stop both installed models, warm only the explicit
Gemma tag on a non-project fixture, and require `ollama ps` to show exactly the
pinned Gemma identity. Every request and response is checked for that exact
tag, and `/api/ps` is checked again after generation. Unload Gemma afterward.
The existing installed `qwen3:14b` model is preserved but never resident during
project calls.

## Generation and mechanical gates

Cases remain in the exact source-file order bound by the existing 60-case
digest. For each case, generate at most three fresh seeded attempts and retain
the first attempt passing the frozen mechanical gates. No text repair, ranking,
or manual selection is allowed. The gates protect JSON/line/sentence shape,
minimum integrity, polarity, modality, script, length ratio, lexical anchor,
and transparent direction proxies. They are not validation of fact, meaning,
naturalness, or a single-factor specificity intervention.

All 60 cases and every hidden corpus-by-direction cell must retain exact
coverage. Otherwise stop before releasing a review packet. Do not tune the
method after generation.

## Blinded author review

For each retained case, place the frozen original and Gemma candidate in A/B
order without revealing which is which. Candidate side is exactly balanced
within every hidden corpus-by-direction cell and 30/30 overall. Stable opaque
review IDs and global row order are derived from separate frozen seeds. The
packet reveals no corpus, direction, source position, model, provenance,
expected direction, attempt, gate outcome, or answer.

For all 60 pairs, the author records:

- factual correctness for sentence A and sentence B separately;
- semantic preservation for the pair;
- grammaticality/naturalness for A and B separately;
- whether A, B, or neither/tie is more specific;
- confidence from 1 to 5; and
- optional notes.

The private answer key is not opened until exactly 60 completed unique review
responses are frozen. Integrity checks must reverse every A/B row to its exact
original and retained candidate, verify hashes and a bijection, reconstruct the
six hidden cells, and reproduce the exact side balance.

## Pilot decision gate

A case passes only when the unblinded candidate is judged factually correct,
the pair is judged semantically preserved, the candidate is natural or has at
most a minor issue, and the specificity judgment matches the hidden direction:
candidate more specific for `add_specific`, original more specific for
`de_specify`, and tie for `irrelevant_rewrite`.

The pilot needs at least 48/60 passing cases and the exact 75% minima in all six
hidden cells: 6/8, 9/12, 8/10, 8/10, 8/10, and 8/10. Passing only makes a
separately frozen 180-source expansion eligible for the user's time-based
decision. Failure stops expansion and scoring. Neither outcome authorizes a
rubric study, scorer analysis, or manuscript change.

## Privacy and release boundary

Originals, generated text, attempts, review packet, answer key, and responses
remain under ignored `outputs/`. Tracked artifacts contain only method files,
schemas, tests, aggregate non-content diagnostics, counts, and cryptographic
hashes. No tracked file may contain a sentence, review judgment, host path,
participant information, or hidden A/B answer.
