# Ko Official-Repository Post-Publication Comparator

## Identity and scope

`ko_official_release_comparator_v1` is the user-approved Pair 3 comparator. It
is the CPU implementation in the authors' official repository pinned at
`36f8e835e9dc6087d5b6763accf302db175947b1`. It is a post-publication
continuation of the repository, not a reproduced 2019 checkpoint and not an
author-designated replacement checkpoint.

The immutable exact-reproduction failure remains under
`configs/ko_2019_reproduction_v1.json` and
`analysis/round2_ko_reproduction/`. In particular, the pinned release changes
distribution regularization from the source batch to the unlabeled target
batch, has no released checkpoint, and missed the published Twitter Table 2
result. The upstream repository has no license file at the pinned commit.
Upstream code, released data, built images, and trained checkpoints therefore
remain local and non-redistributable absent permission.

## Outcome-blind protocol

The complete scientific contract is frozen in
`configs/ko_official_release_comparator_v1.json`. The key choices are:

- Adapt and train independently for each of the four target corpora.
- Preserve the byte-frozen canonical sentence order. All canonical rows pass
  the unchanged legacy one-physical-line contract; coverage must be 100%.
- Retain the order because all target rows affect shallow-feature
  normalization and only the first 50 enter consistency batches in upstream
  `train.py`.
- Run three separate Docker processes per corpus. Upstream fixes NumPy and
  Torch to 1234 but leaves Python `random` unseeded; this official augmentation
  variability is recorded rather than repaired.
- Treat the arithmetic mean of the three native posteriors as the primary Ko
  row score while retaining every run and checkpoint hash.
- Join `strict_natural_language_v1` after scoring. Do not retrain on that
  subset.
- Reuse the exact-token common-support, weighting, regression, interaction,
  and document-bootstrap definitions from `round2_length_control_v1`.
- Preserve native scales. Cross-model comparisons use within-corpus Spearman,
  percentile ranks, or within-model pooled standardization. Raw SpeciTeller
  and Ko scores are not interchangeable.

The frozen input manifest is
`configs/ko_official_release_inputs_v1.csv`. It records canonical and prepared
input hashes, ordered-ID hashes, first-50 ID/text hashes, counts, and adaptation
contexts. Prepared text files are large ignored artifacts under
`outputs/round2/ko_official_release/inputs/`.

## Build and run

No Ko dependency is installed on the host. Build the dedicated tag from the
pinned Docker recipe and verify that the external checksum-addressed GloVe
volume already exists:

```powershell
.\scripts\ko_build_official_release.ps1
docker volume inspect spectech-ko-glove-840b-v1
python scripts\ko_official_release.py prepare
```

The twelve frozen runs are:

```powershell
python scripts\ko_official_release.py run --corpus-id wikipedia_en --run-id run01
python scripts\ko_official_release.py run --corpus-id wikipedia_en --run-id run02
python scripts\ko_official_release.py run --corpus-id wikipedia_en --run-id run03
python scripts\ko_official_release.py run --corpus-id github_docs --run-id run01
python scripts\ko_official_release.py run --corpus-id github_docs --run-id run02
python scripts\ko_official_release.py run --corpus-id github_docs --run-id run03
python scripts\ko_official_release.py run --corpus-id ansible_docs --run-id run01
python scripts\ko_official_release.py run --corpus-id ansible_docs --run-id run02
python scripts\ko_official_release.py run --corpus-id ansible_docs --run-id run03
python scripts\ko_official_release.py run --corpus-id python_312_html --run-id run01
python scripts\ko_official_release.py run --corpus-id python_312_html --run-id run02
python scripts\ko_official_release.py run --corpus-id python_312_html --run-id run03
```

Each ignored run directory contains the enriched score table, run metadata,
raw predictions, logs, and checkpoint plus checksums. Every row records model,
adaptation, training-run, config, input-manifest, native-scale, and upstream
identities.

The host finalizer is idempotent: if a completed score/checkpoint/evidence set
already exists, rerunning the same frozen corpus/run command verifies and
rehashes that evidence without retraining. Run metadata checksum-addresses the
raw predictions, checkpoint, train/test logs, container metadata, and model
hash record.

After all run coverage gates pass:

```powershell
python scripts\ko_model_comparison.py
```

This produces the full ignored pack under
`outputs/round2/ko_official_release/analysis/` and a compact checksum-addressed
paper-facing pack under `analysis/round2_ko_official_release_comparison/`.

Implementation correction record: the first completed Ansible run exposed a
Windows-only host metadata `relative_to` defect after the container had already
finished and written all raw evidence. The finalizer was corrected to resolve
the path first and to recover that already-completed run without retraining.
No score was inspected and no scientific choice, target row, or container
execution changed.

## Completed execution

All twelve runs passed complete ordered coverage. The primary score-table
SHA-256 hashes are:

| Corpus | run01 | run02 | run03 |
| --- | --- | --- | --- |
| Wikipedia | `e2968613ea148af3f40e85c16b8c8f905bc2bdf58716c6c7f83137a5404eda1f` | `3db5c3989dcaf255754dda46c26a4a6c31703f21d52bbf47275a07c7c16ba921` | `060737ae851a73643507bd978baba94d08cb120b5c1dfa32eb80eefb6af6125a` |
| GitHub | `221dd63491abdfc6e232a758244736cf77e4c2d8973b417d5da32164fde4ece6` | `3b514937aaf58f2f20473588d3df2b80ae8f03e7c6448c56d09d5d323828aedc` | `72a4ba84c7a478a6b288ae816f8b523d2b67dc3b5272be85aed22aecd3ba8f5d` |
| Ansible | `ff522001181be873d21a5713215aa78f6ca61f385263b6c73b0066afb6c21bc3` | `c9b7befa9f8ee97b0637de93f7d6dfcd20e1d634ef651d1b669ceffbde534841` | `d19196e6c6bba0495f84d79bf6b6462b5d422dec179ab01371270eaeced99e12` |
| Python | `2325e8ccf222d83b50d52bb08bc417eacefc8cf2f969329f271770161d1370b1` | `e655f279f6b1d501046edb077c84979800882b2d0b04d5055e7421233ac7df17` | `80278af2010f917186cacf3226a6d46e67951518381089204ccbfa2c4f0f6a03` |

The frozen analysis completed thirteen tables with 100% coverage and ordered
join gates. SpeciTeller and the three-run Ko aggregate yield different corpus
orders and heterogeneous within-corpus agreement. Python also shows material
Ko training variability. The compact interpretation and exact estimates are
in `analysis/round2_ko_official_release_comparison/README.md`; table hashes and
run provenance are in that directory's `run_metadata.json`.

Focused validation is 26 passed. The full suite is 93 passed and five known
legacy failures: four controlled-edit tests construct stale `AnalysisRow`
fixtures without the later five proxy fields, and one feature test retains an
older `--help` tokenization expectation. None touches the comparator or its
analysis.

Post-outcome release correction record: the first completed paper-facing
metadata pack inherited absolute host mount strings from the ignored run
commands. The analysis writer was corrected to omit only those host-specific
command arrays while retaining portable scientific identities, artifact
paths, and hashes. Tables, scores, statistical choices, and results did not
change. A focused test now enforces that boundary.

## Interpretation boundary

The comparison characterizes two predictors' corpus-conditioned outputs. The
Ko scores arise from fresh target-adapted training, whereas SpeciTeller is
fixed. Agreement does not establish true specificity, and smaller Ko corpus
gaps do not establish greater accuracy without labels. Controlled-edit
expansion and a new general-domain edit arm remain out of scope.
