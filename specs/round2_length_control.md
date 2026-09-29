# Round 2 Length-Controlled Analysis

## Purpose

This workflow answers whether the observed Wikipedia-to-technical
SpeciTeller score gaps remain after controlling for observed sentence length.
It runs independently on all canonical rows and on rows retained by
`strict_natural_language_v1`. Results describe model-score behavior; they are
not causal effects or measurements of true sentence specificity.

## Frozen method

`configs/round2_length_control_v1.json` was frozen after inspecting only
sentence/document identifiers, token and character counts, and Pair 1 keep
status. SpeciTeller scores and score-by-length relationships were not inspected
before the freeze.

The primary estimand is a directly standardized mean over exact token counts.
For each variant, an exact count is eligible only when every corpus supplies at
least 100 rows from at least 20 documents. Each corpus distribution is
normalized on that support, and the reference distribution is the unweighted
mean of the four distributions. Row weights reproduce one shared target
distribution. The run fails if any weight exceeds 10; it does not cap weights
or silently change the estimand.

Uncertainty uses 1,000 nonparametric document-cluster bootstrap replicates per
corpus with master seed `20260809`; the observed reference distribution remains
fixed. The regression sensitivity uses corpus indicators and a five-knot
restricted cubic spline of `log(token_count)` on the same support, with CR1
document-cluster-robust covariance. A pre-specified interaction model exports
length-specific contrasts. Character count is not simultaneously controlled.

## Inputs

All paths are relative to the repository root and may be relocated together.

| Artifact | Required columns/fields |
|---|---|
| `outputs/sentences/<corpus>.csv` | `corpus_id`, `doc_path`, `sent_id` |
| `outputs/speciteller/<corpus>_scores.tsv` | headerless `sent_id`, `score` |
| `outputs/features/<corpus>_features.csv` | `corpus_id`, `sent_id`, `token_count`, `char_count` |
| `outputs/round2/preprocessing_ablation/strict_natural_language_v1_manifest.csv` | `corpus_id`, `sent_id`, `keep`, `rule_version` |

The loader streams the four files in canonical order and fails on a duplicate,
missing, extra, reordered, or mismatched `sent_id`. The 12 canonical artifact
checksums, Pair 1 manifest checksum, and published Round 1 baseline must all
pass before inference.

## Outputs

The complete rerun pack is written to ignored
`outputs/round2/length_controlled/`. The same small paper-facing tables, plot,
README, and portable metadata are intentionally frozen under
`analysis/round2_length_control/`.

| File | Row meaning |
|---|---|
| `baseline_validation.csv` | one row per corpus |
| `length_distributions.csv` | token/character distribution summary per variant/corpus |
| `common_support_diagnostics.csv` | support loss, weights, documents, and ESS per variant/corpus |
| `common_support_reference.csv` | exact support count and target probability per variant |
| `length_strata.csv` | every observed exact count per variant/corpus, including eligibility and score summaries |
| `standardized_means.csv` | raw, support-unweighted, and directly standardized corpus means |
| `gap_estimates.csv` | raw and standardized Wikipedia-to-technical gaps, bootstrap intervals, and categories |
| `regression_contrasts.csv` | additive regression-standardized means and gaps with cluster-robust intervals |
| `regression_diagnostics.csv` | ranks, fit, condition numbers, knots, and interaction partial R-squared |
| `length_specific_contrasts.csv` | interaction-model gaps at frozen reference quantiles |
| `paper_length_control_table.csv` | compact paper-facing primary table |
| `length_specific_gap_plot.svg` | compact plot justified when interaction heterogeneity is material |
| `run_metadata.json` | method, provenance, checksums, seeds, environment, and result categories |

The metadata contract is documented by
`schemas/round2_length_control_run_metadata.schema.json`.

## Rerun

From the SpecTech repository root:

```powershell
python scripts/length_controlled_analysis.py
```

Validation:

```powershell
python -m pytest tests/test_length_controlled.py -q --basetemp=.pytest_tmp_pair2a
python -m pytest -q --basetemp=.pytest_tmp_pair2a_full
python -m py_compile src/analysis/length_controlled.py scripts/length_controlled_analysis.py
git diff --check
```

The CLI exposes relocation arguments, but changing the frozen seed or replicate
count is rejected. Inputs still must match the committed checksum contracts.
