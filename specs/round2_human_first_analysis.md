# Human-first controlled-edit reranking analysis protocol

## Gate and purpose

Analyze the completed 180-row blinded review packet only after its file hash,
row structure, immutable Review ID/A/B values, and allowed Score values pass.
The hidden key, metric values, proxy decisions, and policy mappings remain
sealed until the protocol commit is complete.

The study asks whether already-frozen sentence metrics choose candidates whose
specificity direction agrees more often with one evaluator's blinded ratings
than the frozen unguided slot-1 baseline. It does not establish edit quality,
semantic validity, model accuracy, truth, or a population-level human judgment.
The 180 rows are three candidates for each of 60 source cases.

## Annotation rule and orientation

The evaluator used `3` whenever A and B had about the same specificity, even
for a small number of rows with subtle meaning changes or alternative
interpretations. `X` was reserved for rows where no usable meaning could be
derived and specificity did not matter. These scores are not recoded. Tracked
evidence describes this caveat conceptually and never quotes the private text.

After unblinding side assignment, map the rating to candidate-minus-original
specificity on `{-2,-1,0,1,2}`. For add-specific edits, a positive delta is
direction-valid; for de-specific edits, a negative delta is direction-valid;
for neutral rewrites, zero is direction-valid. `3` is therefore non-success for
the directional arms and success only for the neutral arm. `X` counts as
failure in the all-case success rate, remains absent from ordinal utilities and
correlations, and is always reported separately. Direction-valid means
specificity agreement under this operational rule, not semantic preservation.

## Order drift

The primary analysis uses all 180 ratings unchanged. Report blinded score
distributions for review positions 1--60, 61--120, and 121--180. Repeat primary
policy contrasts and proxy diagnostics after excluding positions 1--20 and
1--30. Do not rescale, recenter, calibrate, or otherwise adjust early ratings.
No clerical correction is currently reported.

## Policy analysis

Preserve the frozen policy identities. Primary policies are unguided slot 1,
SpeciTeller, Ko run01, Ko run02, Ko run03, and direction-aligned GranuScore.
The Ko three-run arithmetic mean and the already-frozen rank consensus are
secondary.

For each policy, report all-case direction-validity rate, X rate, and
comparable-only mean direction-aligned ordinal utility. Compare every guided
policy with unguided using paired percentage-point success differences and
ordinal win/loss/tie counts. Report overall, corpus, and direction views. In
early-position sensitivities, retain a source in a guided-versus-unguided
contrast only if both selected review rows survive, and report the paired
source count.

## Predictor and proxy diagnostics

For each predictor delta, report case-bootstrap Spearman association with the
human candidate-minus-original delta among non-X candidates. Keep all three Ko
runs separate and primary; label the Ko mean secondary. Cross-model raw scales
are never compared as interchangeable values.

Compare the old automatic proxy pass/fail decision against operational human
direction-validity. Report true accept, false accept, false reject, and true
reject counts plus false-accept and false-reject rates overall, by corpus, and
by edit direction. `X` is a human-negative row in the primary confusion table
and a separately visible component. Repeat this diagnostic after both early-
position exclusions.

## Uncertainty, privacy, and reporting

Use 20,000 deterministic percentile bootstrap replicates. The resampling unit
is the 60-source case, never an individual candidate. Preserve frozen
corpus-by-direction composition for overall draws and reuse draws within paired
policy contrasts. If a small subset is degenerate, report its point estimate,
denominator, and interval limitation rather than substituting a method after
outcome inspection.

Raw joined rows remain ignored. Tracked evidence contains aggregates, method
labels, hashes, and provenance only: no sentence text, opaque review IDs,
source/candidate IDs, selector mappings, or row-level ratings. Report every
result, favorable or unfavorable. Manuscript integration requires a separate
primary review and authorization.
