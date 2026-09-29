# SpecTech

Deterministic pipeline for evaluating sentence specificity model behavior under domain
transfer to technical and procedural documentation.

## Public source snapshot

This repository contains release code, versioned configurations and schemas,
tests, and public method documentation. It does not contain raw corpora,
generated analysis outputs, completed human-review workbooks, model weights,
or internal sprint handoffs. Obtain inputs from their pinned public sources and
run the documented workflows to reconstruct outputs. Tests that inspect
generated artifacts require those artifacts to be reconstructed first.

The public repository begins with one root commit. Historical commit hashes in
frozen experiment records identify the original execution state; they are
provenance values, not ancestors of this public commit.

## Requirements
- Python 3.12
- Dependencies listed in `requirements.txt`

## Virtual Environment (recommended)
```bash
python -m venv .venv
```

Activate the venv:

- **Windows (PowerShell):**
  ```bash
  .\.venv\Scripts\Activate.ps1
  ```
- **Windows (cmd.exe):**
  ```bash
  .\.venv\Scripts\activate.bat
  ```
- **macOS/Linux:**
  ```bash
  source .venv/bin/activate
  ```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Tests
Run the source-only smoke tests:

```bash
python -m pytest -q tests/test_sentences.py tests/test_text_filters.py
```

The full test suite also includes checks against generated research artifacts
that are not distributed in this source snapshot. Run those checks after
reconstructing the required inputs and outputs.

## Blocked Ko et al. (2019) reproduction gate

The pinned Ko, Durrett, and Li (2019) diagnostic is isolated in Python 3.6 and
PyTorch 1.0.0 CPU. The official CPU-capable state is post-publication and its
released-Twitter full run did not reproduce the published metrics; paper-era
source has no matching official CPU patch or checkpoint. Exact reproduction is
therefore blocked, the adapter refuses project scoring, and Sprints 3B/3C are
unauthorized. SpecTech does not alter SpeciTeller.

The upstream repository supplies no checkpoint and no license. SpecTech fetches
the exact commit during a local build and does not vendor or redistribute the
source, released data, built image, embeddings, or trained model. See
`specs/ko_2019_reproduction.md` for pinned checksums, build/smoke commands,
adapter schema, frozen failure evidence, and repeatable unblock conditions.

## Quick Start (pilot)
```bash
python run_pipeline.py --mode pilot
```

## Download corpora (initial setup)
Download selected corpora into `data/corpora/` (gitignored):

```bash
python run_pipeline.py --download --corpus-ids github_docs ansible_docs
```

Download everything in the manifest and validate:

```bash
python run_pipeline.py --download-all
```

By default, already-downloaded corpora are skipped. To force re-downloads:

```bash
python run_pipeline.py --download-all --force-download
```

The `--force-download` flag applies to both `--download` (with `--corpus-ids`) and
`--download-all`.

Download the Python docs HTML zip:

```bash
python run_pipeline.py --download --corpus-ids python_312_html
```

Wikipedia dumps are not automated yet; place extracted `.txt` files in
`data/corpora/wikipedia_en/`.

### Reproducible Wikipedia build from official dump shard (recommended)

For scientific reproducibility, prefer building `wikipedia_en` from a pinned Wikimedia
`pages-articles-multistream` shard and recording provenance metadata.

1) Place one or more dump shards (`.bz2`) under:

- `data/corpora/wikipedia_en_dumps/`

2) Build deterministic plaintext article files (example target: 2000):

```bash
python scripts/build_wikipedia_from_dump.py --dump-glob data/corpora/wikipedia_en_dumps/*.bz2 --target-count 2000 --output-dir data/corpora/wikipedia_en --reset --provenance-path data/corpora/wikipedia_en/PROVENANCE.json
```

3) Re-run pipeline for Wikipedia:

```bash
python run_pipeline.py --mode full --corpus-ids wikipedia_en
```

`PROVENANCE.json` captures reproducibility-critical metadata:
- extraction workflow identifier
- input dump glob
- build parameters (`target_count`, `reset`, `written_count`)
- exact input shard path(s)
- input shard byte size(s)
- input shard SHA256 hash(es)

This lets other researchers verify they used the same source artifact and rerun the same
deterministic corpus build.

### Corpus provenance for all manifest corpora

To standardize reproducibility metadata across **all** corpora, generate per-corpus
provenance files:

```bash
python scripts/generate_corpus_provenance.py --overwrite
```

This writes:

- `data/corpora/github_docs/PROVENANCE.json`
- `data/corpora/ansible_docs/PROVENANCE.json`
- `data/corpora/python_312_html/PROVENANCE.json`
- `data/corpora/wikipedia_en/PROVENANCE.json`

Each file includes manifest source metadata (`source_ref`, `snapshot`, include/exclude globs)
plus source-specific details:

- `git` corpora: resolved `git_head` commit
- `html_zip` corpora: zip artifact path(s), size(s), SHA256 hash(es), html file count
- `wiki_dump` corpora: text file count and sample filenames (and if built with the dump
  workflow, detailed dump artifact hashes in `wikipedia_en/PROVENANCE.json`)

After downloads, validate discovery:

```bash
python run_pipeline.py --list-docs --list-limit 5
```

## Manifest
Corpora are defined in `data/manifests/manifest.yml`. Download helpers are included in
`src/ingestion/downloader.py`, but the pipeline itself operates offline once corpora are
present in `data/corpora/`.

## Phase Module Rename Map (2026-02-15)

To improve long-term readability while preserving phase ordering semantics, analysis modules
were renamed as follows:

- `src/analysis/phase7.py` → `src/analysis/phase07_analysis_outputs.py`
- `src/analysis/phase75.py` → `src/analysis/phase075_controlled_edit_evaluation.py`

Backward-compatible shim modules remain at the old paths and re-export from the new modules
for one transition cycle.

## Outputs by Pipeline Stage (Detailed Output Map)

This section is a stage-by-stage map of artifacts written under `outputs/`.
Unless otherwise noted, files are generated per corpus as `<corpus_id>`-scoped artifacts.

### Stage A — Sentence Extraction

Directory:
- `outputs/sentences/`

Primary file:
- `outputs/sentences/<corpus_id>.csv`

Columns:
- `corpus_id`
- `doc_path`
- `sent_idx`
- `sent_text`
- `sent_id` (stable hash key used for joins)

Purpose:
- Canonical sentence-level table for downstream scoring and analysis.

---

### Stage B — SpeciTeller Input/Output

Directory:
- `outputs/speciteller/`

Primary files:
- `outputs/speciteller/<corpus_id>.tsv` (tokenized scorer input; `sent_id<TAB>tokenized_text`)
- `outputs/speciteller/<corpus_id>_scores.tsv` (scorer output; `sent_id<TAB>score`)

Notes:
- Scores are cached/reused when compatible outputs already exist.
- Any transient intermediate batch files are cleaned up by runner logic.

Purpose:
- Deterministic sentence-level specificity scores.

---

### Stage C — Proxy Features

Directory:
- `outputs/features/`

Primary file:
- `outputs/features/<corpus_id>_features.csv`

Columns:
- `corpus_id`
- `sent_id`
- `tfidf_mean_nonzero`
- `tfidf_max`
- `technical_token_ratio`
- `identifier_density`
- `command_path_flag_density`
- `version_numeric_density`
- `assignment_parameter_density`
- `token_shape_complexity_mean`
- `token_count`
- `char_count`

Purpose:
- Interpretable proxy features for descriptive analysis.

---

### Stage D — Analysis (Phase 7)

Directories:
- `outputs/analysis/tables/`
- `outputs/analysis/figures/`
- `outputs/analysis/samples/`

Tables:
- `outputs/analysis/tables/<corpus_id>_corpus_stats.csv`
- `outputs/analysis/tables/<corpus_id>_score_stats.csv`
- `outputs/analysis/tables/<corpus_id>_score_feature_spearman.csv`
- `outputs/analysis/tables/corpus_score_stats_all.csv`

Optional QA note files (written only when score-table validation fails):
- `outputs/analysis/tables/<corpus_id>_score_stats_qa.txt`

Figures (deterministic SVG):
- `outputs/analysis/figures/<corpus_id>_score_distribution.svg`
- `outputs/analysis/figures/<corpus_id>_score_vs_tfidf_mean.svg`
- `outputs/analysis/figures/<corpus_id>_score_vs_tfidf_max.svg`
- `outputs/analysis/figures/<corpus_id>_score_vs_technical_ratio.svg`

Qualitative samples:
- `outputs/analysis/samples/<corpus_id>_divergence_samples.csv`

Purpose:
- Corpus-level descriptive stats, feature correlations (including `token_count`),
  reproducible score summaries (mean/median/std/quartiles/IQR), distributional plots,
  and divergence-oriented qualitative samples.

How to read each figure:

1) `..._score_distribution.svg`
- What it shows: histogram of SpeciTeller specificity scores for all sentences in a corpus.
- X-axis: specificity score (0 to 1).
- Y-axis: number of sentences in each score bin.
- Interpretation: shape indicates where the corpus concentrates (lower vs higher specificity).

2) `..._score_vs_tfidf_mean.svg`
- What it shows: relationship between specificity and mean non-zero TF-IDF.
- X-axis: TF-IDF mean (non-zero) feature value.
- Y-axis: specificity score.
- Interpretation: upward trend suggests sentences with more distinctive lexical content
  tend to receive higher specificity scores.

3) `..._score_vs_tfidf_max.svg`
- What it shows: relationship between specificity and strongest TF-IDF token per sentence.
- X-axis: TF-IDF max feature value.
- Y-axis: specificity score.
- Interpretation: helps detect whether a single high-information token aligns with
  higher specificity.

4) `..._score_vs_technical_ratio.svg`
- What it shows: relationship between specificity and fraction of technical-looking tokens.
- X-axis: technical token ratio.
- Y-axis: specificity score.
- Interpretation: tests whether code-like/technical token density tracks specificity.

Important plotting note (applies to all `score_vs_*` curves):
- Each dot is a **binned mean**, not an individual sentence.
- The pipeline sorts pairs and partitions into deterministic equal-sized bins, then plots
  mean(feature) vs mean(score) per bin.
- Repeated x-values (e.g., multiple points near `0.00`) can be expected when many sentences
  share the same feature value (common for `technical_token_ratio == 0`).

---

### Stage E — Spot-Check Verification Pack (Phase 4.5)

Directory:
- `outputs/spot_checks/`

Primary file pattern:
- `outputs/spot_checks/<corpus_id>_spot_check_edge<edge_count>_avg<average_count>.csv`

Default/common run artifacts:
- `outputs/spot_checks/github_docs_spot_check_edge20_avg20.csv`
- `outputs/spot_checks/ansible_docs_spot_check_edge20_avg20.csv`

Purpose:
- Deterministic manual verification samples with annotation-ready columns.

---

### Stage F — Controlled Edit Behavioral Evaluation (Phase 7.5)

Directory:
- `outputs/controlled_edits/`

Template generation outputs:
- `<corpus_id>_controlled_edit_template.csv`
  - includes `sent_id`, `corpus_id`, `sentence_original`,
    `speciteller_score_original`, `token_count`, `edit_type`, `sentence_edited`

Post-edit scoring outputs:
- `<template_stem>_scored.csv` (includes edited score + `delta`)
- `<template_stem>_stats.csv` (mean/median delta, directional stats, abs-delta)

Purpose:
- Deterministic behavioral sensitivity evaluation under manual controlled rewrites.

---

### Stage G — Corpus Provenance (Reproducibility Metadata)

Directories:
- `data/corpora/<corpus_id>/PROVENANCE.json` (source provenance per corpus)
- `data/corpora/wikipedia_en/PROVENANCE.json` (dump-build provenance details)

Purpose:
- Reproducibility metadata: source refs, extraction/build parameters,
  and artifact integrity hashes (where applicable).

---

### Stage H — Sentence Artifact QA (Standalone, Read-Only)

Directory:
- `outputs/qa/`
- `outputs/qa/examples/`

Purpose:
- Quantify potential non-natural-language artifacts in canonical sentence tables
  without changing extraction, scoring, or analysis stages.

Run (per corpus):

```bash
python scripts/qa_sentence_artifacts.py --corpus-id github_docs
python scripts/qa_sentence_artifacts.py --corpus-id ansible_docs
```

Input (default):
- `outputs/sentences/<corpus_id>.csv`

Summary output:
- `outputs/qa/<corpus_id>_qa_summary.json`

Summary fields include:
- `total_sentences`
- `pct_short` (`token_count < 5`, whitespace split)
- `pct_template` (contains `{%`, `%}`, `{{`, `}}`)
- `pct_table` (contains `| ---` or `|` count >= 3)
- `pct_code_fence` (contains code-fence artifacts)
- `pct_nonalpha_heavy` (alphabetic ratio over non-space chars < 0.6)
- `pct_flagged_any` (union of all above)

Deterministic example outputs (first 10 matches each):
- `outputs/qa/examples/<corpus_id>_short_examples.csv`
- `outputs/qa/examples/<corpus_id>_template_examples.csv`
- `outputs/qa/examples/<corpus_id>_table_examples.csv`
- `outputs/qa/examples/<corpus_id>_code_fence_examples.csv`
- `outputs/qa/examples/<corpus_id>_nonalpha_heavy_examples.csv`
- `outputs/qa/examples/<corpus_id>_flagged_any_examples.csv`

Important scope note:
- This QA script is read-only over sentence CSVs and emits QA artifacts only.
- It does **not** modify sentence extraction outputs or regenerate scorer outputs.

---

### Stage I — Paper Pack (assembled artifacts for drafting/review)

Directory:
- `outputs/paper_pack/`

Subdirectories:
- `outputs/paper_pack/figures/`
- `outputs/paper_pack/tables/`
- `outputs/paper_pack/samples/`

Core score-summary tables (mirrored from analysis):
- `outputs/paper_pack/tables/<corpus_id>_score_stats.csv`
- `outputs/paper_pack/tables/corpus_score_stats_all.csv`

Core analysis tables/figures mirrored into `paper_pack` may include:
- `<corpus_id>_corpus_stats.csv`
- `<corpus_id>_score_feature_spearman.csv`
- `<corpus_id>_score_distribution.svg`
- `<corpus_id>_score_vs_tfidf_mean.svg`
- `<corpus_id>_score_vs_tfidf_max.svg`
- `<corpus_id>_score_vs_technical_ratio.svg`

Controlled-edit packaged artifacts (`scripts/package_controlled_edit_results.py`):
- `outputs/paper_pack/controlled_edit_summary_by_corpus.csv`
- `outputs/paper_pack/controlled_edit_summary_table.tex`
- `outputs/paper_pack/controlled_edit_delta_distributions.svg`
- `outputs/paper_pack/controlled_edit_outlier_rows.csv` (optional)

Human pilot packaged artifacts (`scripts/package_human_pilot_results.py`):
- `outputs/paper_pack/human_pilot_summary_by_corpus.csv`
- `outputs/paper_pack/human_pilot_label_distributions.csv`
- `outputs/paper_pack/human_pilot_agreement_pairwise.csv`
- `outputs/paper_pack/human_pilot_model_alignment_by_annotator.csv`
- `outputs/paper_pack/human_pilot_merged_sentences.csv`
- `outputs/paper_pack/human_pilot_score_mean_by_pooled_label.csv`
- `outputs/paper_pack/human_pilot_score_by_label.svg`

Optional score-stats QA note files (written on validation failures):
- `outputs/paper_pack/tables/<corpus_id>_score_stats_qa.txt`

---

### Join Integrity Expectations (Quick Audit Rules)

For each corpus after a successful run:
- row count in `outputs/sentences/<corpus_id>.csv` (excluding header)
- equals non-empty line count in `outputs/speciteller/<corpus_id>_scores.tsv`
- equals row count in `outputs/features/<corpus_id>_features.csv` (excluding header)

If those are equal and Stage D artifacts exist, the main pipeline outputs are complete
for that corpus.

## Notes
- `data/corpora/` and `outputs/` are gitignored.
- SpeciTeller will be invoked as a containerized, pinned dependency (see specs).

## SpeciTeller (Phase 4)
SpeciTeller is treated as a containerized, pinned Python 2.7 dependency. The pipeline
now performs live scoring and writes outputs to `outputs/speciteller/`.

SpeciTeller expects **word-tokenized** sentences; we use NLTK's
`TreebankWordTokenizer` to prepare its input deterministically.

Pinned references (2026-02-07):
- SpeciTeller repo commit: `218cb5a389b3e51d393a76e72714ff65e7e81f47`
- speciteller_data.tar.gz SHA256: `dd70443a77b6fc576e3fff88ddc09e6e0777bea3537465a11f339b9de8105fbe`
- liblinear repo commit: `491c9f1188b97ba70847c70a68be363d186ddf9d`

### Reading SpeciTeller scoring outputs
After running (doc list is examples):

```bash
python run_pipeline.py --mode pilot --corpus-ids github_docs ansible_docs
```

you will see two key output types per corpus:

1. Sentence table:
   - `outputs/sentences/<corpus_id>.csv`
   - Includes sentence metadata and `sent_id`.

2. Score table:
   - `outputs/speciteller/<corpus_id>_scores.tsv`
   - Format: `sent_id<TAB>score`

3. Feature table:
   - `outputs/features/<corpus_id>_features.csv`
   - Columns:
     - `corpus_id`, `sent_id`
     - `tfidf_mean_nonzero`, `tfidf_max`
     - `technical_token_ratio`
     - `identifier_density`, `command_path_flag_density`
     - `version_numeric_density`, `assignment_parameter_density`
     - `token_shape_complexity_mean`
     - `token_count`, `char_count`

4. Analysis outputs (Phase 7):
   - Tables:
     - `outputs/analysis/tables/<corpus_id>_corpus_stats.csv`
     - `outputs/analysis/tables/<corpus_id>_score_stats.csv`
     - `outputs/analysis/tables/corpus_score_stats_all.csv`
     - `outputs/analysis/tables/<corpus_id>_score_feature_spearman.csv`
   - Figures (deterministic SVG):
     - `outputs/analysis/figures/<corpus_id>_score_distribution.svg`
     - `outputs/analysis/figures/<corpus_id>_score_vs_tfidf_mean.svg`
     - `outputs/analysis/figures/<corpus_id>_score_vs_tfidf_max.svg`
     - `outputs/analysis/figures/<corpus_id>_score_vs_technical_ratio.svg`
   - Qualitative samples:
     - `outputs/analysis/samples/<corpus_id>_divergence_samples.csv`

Interpretation notes:
- `sent_id` is the stable join key back to the sentence CSV.
- One score row should exist for each sentence row (excluding CSV header).
- A larger score indicates greater predicted specificity.

Quick checks (PowerShell):

```powershell
(Get-Content outputs/speciteller/github_docs_scores.tsv | Measure-Object -Line).Lines
(Get-Content outputs/speciteller/ansible_docs_scores.tsv | Measure-Object -Line).Lines
(Import-Csv outputs/features/github_docs_features.csv | Measure-Object).Count
(Import-Csv outputs/features/ansible_docs_features.csv | Measure-Object).Count
(Import-Csv outputs/analysis/tables/github_docs_score_feature_spearman.csv | Measure-Object).Count
(Import-Csv outputs/analysis/samples/github_docs_divergence_samples.csv | Measure-Object).Count
```

### Spot-check sample generation (for human verification)
For manual QA/annotation prep, generate a deterministic 40-row sample per corpus:
- 20 edge cases total (`edge_low` + `edge_high`)
- 20 average cases (`average`, near median score)
- sampling candidate filter defaults: `min_tokens=8`, `max_tokens=40`

```bash
python scripts/generate_spot_check.py --corpus-id github_docs --edge-count 20 --average-count 20
python scripts/generate_spot_check.py --corpus-id ansible_docs --edge-count 20 --average-count 20
```

Override token-length bounds when needed:

```bash
python scripts/generate_spot_check.py --corpus-id github_docs --edge-count 20 --average-count 20 --min-tokens 10 --max-tokens 35
python scripts/generate_spot_check.py --corpus-id ansible_docs --edge-count 20 --average-count 20 --min-tokens 10 --max-tokens 35
```

Outputs:
- `outputs/spot_checks/github_docs_spot_check_edge20_avg20.csv`
- `outputs/spot_checks/ansible_docs_spot_check_edge20_avg20.csv`

Each row includes `bucket`, sentence metadata, score, and blank `human_label` / `human_notes`
fields for annotation.

Recommendation on timing:
- **Do spot-check now** for pipeline sanity (format/join/noise).
- Do a second spot-check after feature-analysis/plots are generated, to verify interpretation
  consistency before broader human annotation.

## Controlled Edit Behavioral Evaluation (Phase 7.5)

Phase 7.5 uses a deterministic **controlled edit template** workflow to behaviorally probe
SpeciTeller without retraining or supervised labels.

### What the controlled edit template is
The template is a CSV of sampled, already-scored sentences with an assigned `edit_type` and a
blank `sentence_edited` field for manual completion.

Sampling defaults (deterministic):
- 10 from bottom decile specificity
- 10 from median band
- 10 from top decile specificity
- optional length-controlled band (default off)
- sampling candidate filter defaults: `min_tokens=8`, `max_tokens=40`

Template columns:
- `sent_id`, `corpus_id`
- `sentence_original`, `speciteller_score_original`
- `token_count`
- `edit_type` (`de_specify`, `add_specific`, `irrelevant_rewrite`)
- `sentence_edited` (blank for manual edits)

### Step 1: Generate template

```bash
python scripts/generate_controlled_edit_template.py --corpus-id github_docs
python scripts/generate_controlled_edit_template.py --corpus-id ansible_docs
```

Override token-length bounds when needed:

```bash
python scripts/generate_controlled_edit_template.py --corpus-id github_docs --min-tokens 10 --max-tokens 35
python scripts/generate_controlled_edit_template.py --corpus-id ansible_docs --min-tokens 10 --max-tokens 35
```

Default outputs:
- `outputs/controlled_edits/github_docs_controlled_edit_template.csv`
- `outputs/controlled_edits/ansible_docs_controlled_edit_template.csv`

### Step 2: Manually fill edited sentences

Open each template CSV and fill `sentence_edited` according to `edit_type`:
- `de_specify`: make sentence less specific
- `add_specific`: make sentence more specific
- `irrelevant_rewrite`: rewrite while preserving rough specificity intent

### Step 3: Re-score completed templates and compute deltas

```bash
python scripts/score_controlled_edits.py --template-path outputs/controlled_edits/github_docs_controlled_edit_template.csv
python scripts/score_controlled_edits.py --template-path outputs/controlled_edits/ansible_docs_controlled_edit_template.csv
```

This writes:
- `<template_stem>_scored.csv` (original score, edited score, delta)
- `<template_stem>_stats.csv` (mean/median delta; directional stats per edit type; mean absolute delta)

under:
- `outputs/controlled_edits/`

Notes:
- Scripts are runnable directly from the `SpecTech/` repo root.
- Scoring uses the same containerized SpeciTeller integration as the main pipeline.

## Paper Pack (Phase 7.5 results packaging)

Package existing controlled-edit scored outputs into paper-ready artifacts:

```bash
python scripts/package_controlled_edit_results.py
```

This writes to `outputs/paper_pack/`, including:
- `outputs/paper_pack/controlled_edit_summary_by_corpus.csv`
- `outputs/paper_pack/controlled_edit_delta_distributions.svg`

Optional appendix/debug rows with large deltas are written when present:
- `outputs/paper_pack/controlled_edit_outlier_rows.csv`

## Reproducible Score Summary Stats (Corpus-level)

Generate deterministic corpus-level specificity score summaries from canonical
`outputs/speciteller/*_scores.tsv`:

```bash
python scripts/generate_score_stats.py --self-check
```

This writes both per-corpus and aggregate files to:
- `outputs/analysis/tables/`
- `outputs/paper_pack/tables/`

Files written:
- `{corpus_id}_score_stats.csv` (single-row per corpus)
- `corpus_score_stats_all.csv` (one row per corpus)

Columns:
- `corpus_id`, `score_count`, `score_mean`, `score_median`, `score_std`,
  `score_min`, `score_max`, `score_q1`, `score_q3`, `score_iqr`

## Paper Pack: Human Pilot

Package annotated spot-check pilot results into deterministic descriptive artifacts:

```bash
python scripts/package_human_pilot_results.py
```

Key outputs:
- `outputs/paper_pack/human_pilot_summary_by_corpus.csv`
- `outputs/paper_pack/human_pilot_agreement_pairwise.csv`
- `outputs/paper_pack/human_pilot_score_by_label.svg`

### Round 2 existing-pilot four-score comparison

Sprint 3D reuses the exact existing 40 GitHub and 40 Ansible pilot rows and
their three complete label sets. It joins already-saved scores for frozen
SpeciTeller and Ko run01/run02/run03; the Ko rowwise mean is a labeled
secondary sensitivity, not an independent fifth model. No annotation,
relabeling, retraining, or rescoring is performed.

```bash
python scripts/pilot_model_human_comparison.py
```

The outcome-blind contract and exact ID manifest are
`configs/round2_pilot_model_human_comparison_v1.json` and
`configs/round2_pilot_sent_ids_v1.csv`. The command verifies every frozen
input/checkpoint hash and exact 40/40 joins before computing separate-corpus
Spearman estimates, jointly paired sentence-bootstrap intervals and contrasts,
and pairwise model agreement. Raw anonymized local labels and row-level joins
remain ignored under `outputs/`; the privacy-filtered compact evidence pack is
`analysis/round2_pilot_model_human_comparison/`.

## Round 2 preprocessing-sensitivity analysis

`scripts/preprocessing_ablation.py` is the release-quality entry point for the
`strict_natural_language_v1` row-selection sensitivity analysis. It first
enforces the frozen Round 1 input checksums and published baseline gate, then
applies one corpus-invariant, score-blind predicate to the existing scored
rows. It does not alter, resplit, or rescore sentence text.

### Scientific contracts

- `configs/strict_natural_language_v1.json` freezes the reason codes,
  thresholds, command-starter vocabulary, and corpus-invariant scope.
- `configs/round1_speciteller_baseline.csv` freezes the published sentence
  counts and rounded mean, median, standard deviation, and IQR values.
- `configs/round1_artifact_checksums.csv` freezes byte sizes and SHA-256 hashes
  for the 12 canonical sentence, score, and feature files.
- `configs/round1_source_provenance.json` freezes public corpus sources,
  commits/archive hashes, Wikipedia extraction parameters, and SpeciTeller
  model provenance.
- The bootstrap uses document clusters (`doc_path`), 1,000 replicates, master
  seed `20260808`, percentile 95% intervals, and paired raw/filtered resampling
  within each corpus.

### Required input schemas

The ignored `outputs/` tree must contain the exact files named in
`configs/round1_artifact_checksums.csv`.

- Sentence CSV: header fields `corpus_id`, `doc_path`, `sent_idx`, `sent_text`,
  and `sent_id`.
- SpeciTeller score TSV: no header; exactly `sent_id<TAB>score` per row.
- Feature CSV: at least `corpus_id` and `sent_id`; additional frozen proxy
  columns are preserved and join-validated but are not used for filtering.

All three files for a corpus must contain the same unique `sent_id` values in
the same deterministic row order. The CLI rejects missing, extra, duplicate,
malformed, non-finite, or out-of-range score rows before producing evidence.

### Restore or regenerate the canonical inputs

Create the pinned Python environment first:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

On POSIX systems, use `.venv/bin/python` in place of
`.venv/Scripts/python`. Restore the two Git corpora at their frozen commits:

```bash
git clone https://github.com/github/docs data/corpora/github_docs
git -C data/corpora/github_docs checkout 5b8a8c96f169e50278e1c6bdc573495173f7b339
git clone https://github.com/ansible/ansible-documentation data/corpora/ansible_docs
git -C data/corpora/ansible_docs checkout 83bbdd618bb67097f92f9282a8ca7ae12fe6ed4f
```

Download `python-3.12-docs-html.zip` from the Python 3.12 documentation
download page, verify SHA-256
`9e98419a01ef306c5f1471126a54b33f4393636c0a827b4bc59061e9d16cd2af`,
and extract it under `data/corpora/python_312_html/`. Download the pinned
Wikipedia shard named in `configs/round1_source_provenance.json`, verify its
SHA-256, place it under `data/corpora/wikipedia_en_dumps/`, and build exactly
2,000 plaintext files:

```bash
.venv/Scripts/python scripts/build_wikipedia_from_dump.py --dump-glob "data/corpora/wikipedia_en_dumps/*.bz2" --target-count 2000 --output-dir data/corpora/wikipedia_en --reset --provenance-path data/corpora/wikipedia_en/PROVENANCE.json
```

Build the pinned SpeciTeller image and regenerate all row-level artifacts:

```bash
docker build -t speciteller:py27 speciteller_docker
.venv/Scripts/python run_pipeline.py --mode full --corpus-ids wikipedia_en github_docs ansible_docs python_312_html
```

The next command is both the checksum/baseline verification gate and the
analysis run. It stops rather than silently accepting regenerated artifacts
that differ from the frozen Round 1 evidence:

```bash
.venv/Scripts/python scripts/preprocessing_ablation.py
```

### Output schemas and regeneration

The complete ignored audit pack is written to
`outputs/round2/preprocessing_ablation/`:

- `strict_natural_language_v1_manifest.csv`: `corpus_id`, `sent_id`, `keep`
  (`0`/`1`), semicolon-delimited `reason_codes`, and `rule_version`.
- `baseline_validation.csv`: expected/actual baseline values and per-field
  pass flags.
- `score_summaries.csv`: original/filtered count, mean, median, population
  standard deviation, IQR, min/max, and fixed quantiles.
- `corpus_ordering.csv`: descending raw/filtered mean and median ordering.
- `wikipedia_technical_gaps.csv`: raw/filtered Wikipedia-minus-technical
  mean/median gaps, changes, document-cluster bootstrap intervals, and the
  survival/attenuation/reversal/ambiguity category.
- `rule_prevalence.csv`, `rule_overlap_counts.csv`, and
  `reason_examples.csv`: auditable exclusions and deterministic score-free
  examples.
- `run_metadata.json`: portable config hashes, verified input hashes, source
  and model provenance, environment versions, join/reconciliation gates,
  bootstrap configuration, and row-manifest checksum.

Small intentional paper-facing copies of the summary tables, examples, and
portable run metadata are committed under
`analysis/round2_preprocessing_ablation/`. Every file there is regenerated by
the same CLI; the 1.3-million-row keep/drop manifest remains ignored.

## Round 2 length-controlled analysis

After the preprocessing run has produced and checksum-frozen the Pair 1 row
manifest, run the outcome-blind frozen length-control workflow:

```bash
.venv/Scripts/python scripts/length_controlled_analysis.py
```

The primary analysis directly standardizes exact token-count strata on
variant-specific four-corpus common support; a five-knot restricted cubic
spline regression with document-cluster-robust covariance is a sensitivity
analysis. The full rerun pack is ignored under
`outputs/round2/length_controlled/`, while the compact evidence pack is frozen
under `analysis/round2_length_control/`. The estimand, support/weight gates,
schemas, uncertainty, interaction diagnostic, categories, and all table fields
are documented in `specs/round2_length_control.md` and
`schemas/round2_length_control_run_metadata.schema.json`.

## Round 2 official-repository Ko comparator

The failed exact-2019 reproduction evidence remains immutable under
`analysis/round2_ko_reproduction/`. The separately approved comparator uses
the authors' official post-publication CPU repository state pinned at
`36f8e835e9dc6087d5b6763accf302db175947b1`; it is not described as a
reproduced paper checkpoint.

The outcome-blind protocol, canonical input order, three-run design,
per-corpus adaptation boundary, scale rules, bootstrap units, and reuse of the
Pair 1/2 methods are frozen in
`configs/ko_official_release_comparator_v1.json` and
`configs/ko_official_release_inputs_v1.csv`. Build and run instructions,
licensing limits, artifact policy, and interpretation boundaries are in
`specs/ko_official_release_comparator.md`.
