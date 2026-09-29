# Round 2 Qwen Zero-Shot Rubric Experiment

## Purpose

QR-A tests whether a frozen local instruction-following model applies the same
1--5 specificity rubric as the three existing pilot annotators. It uses exactly
the existing 40 Ansible and 40 GitHub pilot sentences. It does not add human
labels, train a model, score the full corpora, or alter the pilot sample.

The machine-readable contract is `configs/round2_qwen_rubric_v1.json`. That
file, this specification, implementation, tests, and schema must be committed
before any project sentence is sent to Qwen or any Qwen--human outcome is
inspected.

## Rubric and blindness

The prompt preserves the complete reviewed human rubric: definitions for all
five scale points, more/less-specific cues, first-impression instruction, and
the generic service-restart illustration. Zero-shot means there are no labeled
pilot examples or numeric worked input-output examples. Qwen never receives a
human label, SpeciTeller/Ko score, sample bucket, corpus identity, document path,
or neighboring sentence.

Each sentence is an independent fresh conversation. The response schema permits
only `{"score": <integer 1--5>}` and no rationale. The scientific run follows
the frozen manifest order. No outcome-based prompt repair, row deletion, or
manual correction is permitted.

## Reproducible local identity

The candidate is local Ollama `qwen3:14b` with the exact model/blob and runtime
identities recorded in the config. Thinking is disabled. Sampling options are
explicitly overridden because the packaged model defaults are stochastic. A
three-fixture, three-repeat non-project smoke gate must return the same valid
score for every repetition of each fixture before project scoring begins.

## Analysis

Spearman correlation with average ranks is primary. Qwen is compared with each
annotator and the rowwise pooled human label separately in each corpus. Joint
10,000-replicate sentence bootstraps reuse the existing pilot seed/stream rule.
The same draws estimate Qwen-minus-SpeciTeller and Qwen-minus-each-Ko-run human-
correlation contrasts and Qwen--model rank agreement.

Secondary ordinal summaries report mean absolute distance for all human targets
and exact/within-one agreement for individual annotators. Score counts expose
ties or category collapse. Results are descriptive: no winner, multiplicity
claim, or gold-standard accuracy interpretation is permitted.

## Privacy and outputs

Ignored raw responses and tracked compact scores contain stable IDs and Qwen
scores but no sentence text, human labels, participant identities, notes, or
absolute paths. Compact aggregate tables and metadata must be reproducible from
the retained ignored inputs and checksum-addressed protocol.

