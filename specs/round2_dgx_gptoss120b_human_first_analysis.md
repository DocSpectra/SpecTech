# GPT-OSS-120B human-first Phase D analysis freeze

This analysis inherits the reviewed Gemma human-first estimator family without
changing rating orientation, score-3 handling, X handling, policies, strata,
or candidate-level associations. The primary GPT-OSS-120B population is all
180 blinded judgments: three candidates for each of the same 60 source cases.

The primary outcome is direction-validity under the evaluator's specificity
judgment. X counts as a failure in all-case rates and remains separate from
ordinal utilities and correlations. Score 3 is a success only for the neutral
rewrite direction. The analysis reports first/middle/final-60 distributions
and exclude-first-20/30 sensitivities without recalibration or recoding.

Every frozen policy is evaluated: unguided slot 1, SpeciTeller, Ko run01,
run02, and run03, and direction-aligned GranuScore. Exact-decimal Ko mean and
the inherited rank consensus remain secondary. Candidate associations and
automatic-proxy confusion are reported overall, by corpus, and by direction.

The generator comparison is paired by the 60 shared source cases. It compares
per-source valid and X fractions, frozen-policy selected outcomes,
guided-minus-unguided gain differences, candidate-metric associations, and
proxy behavior. All three slots stay clustered. Twenty thousand deterministic
bootstrap replicates resample source cases within corpus-by-direction cells.

The two generators were reviewed by one evaluator in separate blinded
sessions. The comparison does not identify a causal effect of model size,
family, runtime, or quantization and does not establish general editing
superiority. Favorable, null, adverse, heterogeneous, order-sensitive, and
session-sensitive findings are equally eligible for reporting.

Tracked artifacts contain aggregate privacy-safe evidence only. Phase E paper
editing remains locked until a separate primary review passes Phase D.
