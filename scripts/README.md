# Scripts

Use this directory for project automation and reproducible command wrappers.

## Blocked Ko et al. (2019) diagnostic

`ko_fetch_glove.ps1` restores and checksum-verifies the external GloVe
840B/300d artifact in an isolated Docker volume. `ko_build.ps1` builds the
checksum-pinned Python 3.6/PyTorch 1.0 CPU image from the official pinned
upstream commit. `ko_released_smoke.ps1` runs the released Twitter
infrastructure smoke twice and checks the frozen tolerance.

`ko_specificity.py` is the model-aware adapter CLI, but the default config now
refuses all project scoring because the released-data reproduction gate failed.
Do not override that guard. Full commands, frozen evidence, licensing limits,
schemas, and unblock conditions are in `specs/ko_2019_reproduction.md`.

## `length_controlled_analysis.py`

Validates the frozen Round 1 artifacts and Pair 1 manifest, then performs exact
token-count common-support standardization and document-cluster-robust
regression sensitivity analyses on original and strict rows. The outcome-blind
method is frozen in `configs/round2_length_control_v1.json`; schemas and rerun
details are in `specs/round2_length_control.md`.

Default run from the repository root:

```bash
python scripts/length_controlled_analysis.py
```

## `preprocessing_ablation.py`

Validates the exact Round 1 sentence/score/feature artifacts and published
SpeciTeller baseline, applies `strict_natural_language_v1` identically across
all four corpora, and writes the full audit plus compact paper-facing packs.

Default run from the repository root:

```bash
python scripts/preprocessing_ablation.py
```

Important defaults:

- inputs: `outputs/{sentences,speciteller,features}/`
- full output: `outputs/round2/preprocessing_ablation/`
- small frozen output: `analysis/round2_preprocessing_ablation/`
- rule config: `configs/strict_natural_language_v1.json`
- baseline config: `configs/round1_speciteller_baseline.csv`
- artifact hashes: `configs/round1_artifact_checksums.csv`
- public/model provenance: `configs/round1_source_provenance.json`
- bootstrap: 1,000 document-cluster replicates, seed `20260808`

Run `python scripts/preprocessing_ablation.py --help` for relocation and test
fixture options. Input/output schemas and full restoration commands are in the
root `README.md` under “Round 2 preprocessing-sensitivity analysis.”

