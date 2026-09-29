# SpecTech

Research code and frozen protocols for *Sentence Specificity Scores for
Collaborative Technical Documentation: A Domain-Transfer Study*, accepted at
IEEE CIC 2026. The study examines how sentence-specificity scores behave on
technical documentation and in one bounded candidate-selection decision.

## Citation

Please cite the accepted paper using [CITATION.cff](CITATION.cff). The citation
is marked in press; no DOI or proceedings page numbers have been assigned here.

## Repository contents

- `src/` and `scripts/`: corpus processing, scoring adapters, and analysis
  commands.
- `configs/` and `schemas/`: frozen inputs, checksums, prompts, seeds, and
  validation contracts.
- `specs/`: method details, prerequisites, and interpretation boundaries for
  individual analyses.
- `tests/`: source checks and tests that require reconstructed research data.

This source snapshot excludes raw corpora, generated outputs, completed human
review workbooks, model weights, and internal handoffs. The pinned corpus and
SpeciTeller sources are recorded in
[`configs/round1_source_provenance.json`](configs/round1_source_provenance.json).
The expected canonical artifact hashes are in
[`configs/round1_artifact_checksums.csv`](configs/round1_artifact_checksums.csv).
Historical experiment commit hashes are provenance identifiers, not ancestors
of this snapshot's Git history.

## Setup and source checks

Use a Python 3.12 virtual environment. Docker is needed for the pinned legacy
scorer environments.

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests/test_sentences.py tests/test_text_filters.py
```

Generated-artifact tests require the corresponding inputs and outputs, which
are not included here.

## Reproducing the analyses

The paper's canonical sentence, SpeciTeller-score, and feature tables are
expected under `outputs/sentences/`, `outputs/speciteller/`, and
`outputs/features/`. These files are excluded from this snapshot. Restore the
four corpora at the exact revisions and hashes in
[`configs/round1_source_provenance.json`](configs/round1_source_provenance.json)
before running the root pipeline. Its download helper does not itself enforce
the pinned Git commits or archive checksums.

Build the pinned SpeciTeller image, then generate the canonical tables:

```bash
docker build -t speciteller:py27 speciteller_docker
python run_pipeline.py --mode full --corpus-ids wikipedia_en github_docs ansible_docs python_312_html
```

Run the preprocessing gate before later analyses. It checks the generated
tables against the frozen Round 1 baseline and artifact hashes; stop if the
gate fails.

```bash
python scripts/preprocessing_ablation.py
python scripts/length_controlled_analysis.py
```

The first command validates the Round 1 baseline before selecting the strict
natural-language subset. The second uses that subset for exact token-length
standardization. Further comparator, distribution, granularity, pilot, and
candidate-selection protocols are documented in [`specs/`](specs/) with their
corresponding scripts and frozen configurations. Commands that depend on
excluded model artifacts or review records require those inputs to be restored
separately. In particular, this source snapshot cannot regenerate the exact
human-pilot and candidate-selection results without the excluded completed
review records.

The Ko et al. (2019) paper-era system was not reproduced. The evaluated
comparator is the separate pinned post-publication author-repository
implementation. Its objective change, failed released-domain checks, and
licensing boundary are documented in
[`specs/ko_official_release_comparator.md`](specs/ko_official_release_comparator.md).
