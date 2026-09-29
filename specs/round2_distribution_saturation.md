# Round 2 distribution and saturation analysis

## Scope and immutable inputs

This release analyzes complete valid full-corpus outputs for the frozen
SpeciTeller baseline, Ko official-release run01/run02/run03, their secondary
row-wise arithmetic mean, and native GranuScore. Qwen rubric scores are
pilot-only, the Qwen edit arm failed its gate, and VAGO failed its release
gate; none enters this analysis. The three Ko runs are the primary stochastic
instances. Their mean is not a fourth independent model.

`configs/round2_distribution_saturation_v1.json` freezes every source path,
SHA-256, model identity, native bound and direction, statistic, figure bin,
bootstrap rule, comparison, claim rule, and stop condition. It was committed
before any new distribution summary or partial outcome was inspected. The
analysis must first reproduce the 1,295,205-row ordered joins, the Round 1
SpeciTeller baseline, the strict-subset counts, every prior score identity,
and all finite-scale bounds. No partial result may be written before all gates
pass.

## Native scales and subsets

- SpeciTeller and each Ko posterior use supported finite bounds `[0, 1]` and
  are higher-is-more-specific.
- Native GranuScore uses its official percentile-like `[0, 100]` output and is
  higher-is-coarser/more-abstract.
- Raw values are compared only within a model instance. No raw cross-model
  scale comparison or latent-scale equivalence is allowed.
- Every primary table has `original` and `strict` rows. GranuScore additionally
  repeats summaries and boundary statistics after removing flagged official
  `NO_FACT_SCORE` rows. The all-row analysis remains primary.

## Frozen summaries

For each model instance, corpus, and subset, report `n`, minimum, p01, p05,
p10, p25, p50, p75, p90, p95, p99, maximum, IQR, p95-p05, p99-p01,
occupied range, occupied-range utilization, effective dynamic range
`(p95-p05)/(upper-lower)`, secondary `(p99-p01)/(upper-lower)`, exact unique
count, exact tie rate, largest exact mass-point share, and secondary mean and
population standard deviation. NumPy's linear quantile definition is fixed.

Boundary concentration is scale-defined, not empirical-tail membership. For
fraction `a`, the lower region is `score <= lower + a*(upper-lower)` and the
upper region is `score >= upper - a*(upper-lower)`. Five percent is primary;
one and ten percent are sensitivities. Lower and upper rates stay separate.

## Wikipedia comparisons and uncertainty

Every technical corpus is compared with Wikipedia within each model instance
and subset. Primary lower/upper five-percent boundary-rate differences use
1,000 nonparametric document-cluster bootstrap replicates, master seed
`20260810`, and independent resampling within each corpus. The estimator is the
sentence-weighted proportion after sampling whole `doc_path` clusters.

Exact corpus p95-p05 differences and technical/Wikipedia ratios are reported
as point diagnostics. A complementary, computationally transparent spread
uncertainty analysis first computes p95-p05 within every document having at
least 20 subset rows, then bootstraps the equal-document mean. Fewer than 20
eligible documents, or any nonpositive Wikipedia denominator in the point or
bootstrap ratio, stops the run. This document-average estimand must not be
misreported as a bootstrap interval for the pooled sentence-level quantile.

## Figure and claim audit

The primary figure is a 5-by-4 faceted 50-bin proportion histogram: frozen
SpeciTeller, Ko run01/run02/run03, and native GranuScore by the four corpora.
Original proportions use filled bars and strict proportions use an outline.
Probability rows have `[0,1]` axes; GranuScore has `[0,100]`. Y axes are shared
only within a model row. The secondary Ko mean is tabulated but omitted from
the primary figure.

Frozen diagnostic categories describe observable output shape only. Effective
dynamic range below 0.40 is `compressed`, 0.40--0.70 is `intermediate`, and at
least 0.70 is `broad`. A top exact mass of at least 0.10 is dominant; either
outer-five-percent share of at least 0.20 is boundary-concentrated; any frozen
two-percent histogram bin of at least 0.20 is histogram-concentrated. The term
"smooth" is never asserted mathematically. A broad/graded appearance requires
all concentration flags to be absent and broad effective range.

Technical/Wikipedia p95-p05 ratios at most 0.80 are narrower, ratios at least
1.25 are broader, and intervening ratios are in a similarity band. Boundary
differences of at least +0.05 or at most -0.05 are more or less concentrated;
smaller differences are in a similarity band. A cross-model direction is
called consistent only if SpeciTeller, all three Ko runs, and native
GranuScore share the same category. Otherwise it is heterogeneous. These are
prespecified descriptive aids, not hypothesis tests or accuracy judgments.

## Reproduction

From the SpecTech repository root, after restoring the checksum-addressed
ignored inputs documented by the preceding sprints:

```powershell
python scripts/distribution_saturation_analysis.py
```

Full aggregate outputs are written under ignored
`outputs/round2/distribution_saturation/`. The compact checksum-addressed pack
under `analysis/round2_distribution_saturation/` contains no sentence text,
human labels, participant data, or private path. Metadata follows
`schemas/round2_distribution_saturation_run_metadata.schema.json`.

The results characterize model behavior under domain shift. They do not
measure latent specificity truth, establish comparative accuracy, evaluate
documentation quality, or identify a causal domain effect.
