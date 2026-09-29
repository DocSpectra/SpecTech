# Ko et al. (2019) Reproduction Blocker Contract

## Scope and gate

This workflow tests the official implementation associated with Ko,
Durrett, and Li, *Domain Agnostic Real-Valued Specificity Prediction* (AAAI
2019). It may train on the official binary-labeled news source data and adapt
to unlabeled target sentences. It must not add target
labels, alter SpeciTeller, fine-tune on documentation labels, replace the
model, or silently modernize its runtime.

The feasibility gate is **blocked**. The official CPU-capable repository state
is post-publication, includes a material adaptation-objective change, and its
full released-Twitter run does not reproduce Table 2. The paper-era source has
no matching official CPU patch or released checkpoint. This container is a
provenance-pinned diagnostic candidate, not an exact comparator, and its
adapter refuses project scoring by default.

Sprint 3A is outcome blind for the four project corpora. Its only model run is
against official released Twitter data. Project-corpus scoring is forbidden
until the repeatable unblock conditions below are met. Sprints 3B and 3C are
not authorized.

## Pinned diagnostic identity

- Paper: <https://arxiv.org/abs/1811.05085> (v2 / camera ready)
- Official repository:
  <https://github.com/wjko2/Domain-Agnostic-Sentence-Specificity-Prediction>
- Pinned CPU-capable commit: `36f8e835e9dc6087d5b6763accf302db175947b1`
- Last commit before arXiv v2: `bee414ad5994f960d445cb5454c66199aa3920c6`
- Post-publication objective change:
  `cb6243c174ce9283e1155c7a97ef55d14f587bc0`
- Official CPU patch: the commit's `no_cuda.zip`, SHA-256
  `623fd4d762d5eaf45c034a4c10759babeebd374575d3d1ff53fcfe050d3604f5`
- Runtime: Python 3.6.15, PyTorch 1.0.0 CPU, NumPy 1.16.4, SciPy 1.2.1
- External embeddings: GloVe Common Crawl 840B/300d, verified in
  `ko_container/artifact_manifest.json`
- Model: self-ensembling plus adaptation and mean/std posterior-distribution
  regularization (`SE+AD mean-std`)
- Native output: teacher-network positive-class posterior in `[0,1]`, where a
  higher value means more specific

The code uses 30 epochs, batch size 32, Adam at `0.0001`, teacher EMA decay
`0.999`, consistency weight `1000`, distribution weight `100`, word-embedding
noise `0.1`, shallow-feature noise `0.2`, and the official augmentation
branches. The paper gives reference mean/std `0.417/0.227`; the code rounds
these to `0.42/0.23`. Reproduction follows the code and records the difference.

The official repository does **not** provide a pretrained checkpoint. The
diagnostic execution therefore trains a target-adapted model from the released source
data on every run. It also does not contain a license file at the pinned
commit. The Dockerfile fetches the source at build time; SpecTech does not
vendor it. The built image, upstream data, and trained checkpoints are local
research artifacts and must not be redistributed without permission.

The paper-era `train.py` applies mean/std regularization to the labeled source
batch. The May 2019 change applies it to the unlabeled target batch. The only
official CPU patch arrived in April 2020 and matches the later code. The exact
diff and commit dates are frozen under `analysis/round2_ko_reproduction/`.

## Container build and artifacts

From the SpecTech repository root:

```powershell
.\scripts\ko_fetch_glove.ps1
.\scripts\ko_build.ps1
```

The GloVe script downloads into the named Docker volume
`spectech-ko-glove-840b-v1`, verifies the archive checksum, expands it, and
verifies the text checksum. The 2+ GB archive, 5+ GB text file, upstream source,
image layers, models, and caches are not tracked.

The Dockerfile pins the Linux/amd64 base-image digest, the exact upstream
archive and CPU-patch checksums, and all Python wheel versions/checksums. A
local image is tagged
`spectech-ko-specificity:36f8e835-py36-torch100-cpu`.

## Released-data validation

Run two reduced infrastructure smokes:

```powershell
.\scripts\ko_released_smoke.ps1
```

Each smoke uses the official released Twitter files but deliberately reduces
training to one epoch, 32 source rows, and 32 unlabeled rows. It must emit 983
predictions because the official loader reserves the first of 984 test rows.
This smoke validates dependencies, GloVe loading, feature extraction, source
training, target adaptation, checkpoint loading, output parsing, and repeated
run tolerance. It does not reproduce or claim the paper's Table 2 values.

The frozen tolerance is mean absolute prediction difference at most `0.02`
and maximum absolute difference at most `0.10`. Exact byte determinism is not
expected: the official implementation seeds NumPy and PyTorch but omits
`random.seed`, while Python's `random` module controls augmentation branches.
SpecTech records this behavior rather than changing the recipe.

For the full official released-data gate, use:

```powershell
$out = [System.IO.Path]::GetFullPath("outputs/round2/ko_reproduction/released_full")
New-Item -ItemType Directory -Force -Path $out | Out-Null
docker run --rm `
  --volume "spectech-ko-glove-840b-v1:/artifacts:ro" `
  --volume "${out}:/output" `
  spectech-ko-specificity:36f8e835-py36-torch100-cpu `
  /opt/spectech/run_target.sh released-full
```

The completed full run emitted 983 predictions and produced Spearman
`0.46860126636291055`, Kendall `0.32187806029556726`, and MAE
`0.14544283049944556`. The paper reports `0.676±0.004`, `0.487±0.005`, and
`0.113±0.001`; the discrepancies are respectively about 51.85, 33.02, and
32.44 reported standard deviations. Prediction and checkpoint hashes, runtime
identity, and the summary distribution are frozen in
`analysis/round2_ko_reproduction/`.

## Adapter contract

The host input is CSV with exactly `sent_id,corpus_id,text`. IDs must be unique;
fields must be non-empty; each sentence must occupy one physical legacy line.
The official test loader withholds row zero (`tv=1`). The adapter prepends a
duplicate of the first real sentence as the held-out row, writes each actual
sentence exactly once to the unlabeled adaptation file, and maps the resulting
N predictions back to the original N IDs in order. The legacy code also insists
on `twitterl.txt` and `twitterv.txt`; the adapter supplies non-semantic interface
fillers that are loaded but not used to produce predictions. They are not
documentation annotations.

The score output contract is defined by
`schemas/ko_specificity_scores.schema.json`. Every row carries model identity,
native bounds/direction, upstream commit, and an explicit adaptation context.
The CLI contract is retained for testing, but its default configuration raises
a blocker before reading an input. It must not be used on project corpora:

```powershell
python scripts/ko_specificity.py `
  --input path/to/frozen_input.csv `
  --output path/to/ko_scores.csv `
  --adaptation-context frozen-context-id
```

Do not bypass the guard by constructing an authorized config outside tests.

## Known exact-implementation behaviors

- `test.py` emits strings such as `tensor(0.1234)` under PyTorch 1.0. The
  adapter accepts that form and plain numeric lines, enforces finite `[0,1]`
  values, and fails on count mismatch.
- The test loader excludes the first test row. The released Twitter run thus
  emits 983 scores for 984 released labeled sentences.
- Target adaptation is corpus-conditioned. A score is not solely a function of
  one sentence; the adaptation context is required model metadata.
- The exact `train.py` permutes `uss=5000` indices and then applies modulo 50.
  Consequently only the first 50 unlabeled target rows enter consistency-loss
  batches, although all target rows affect shallow-feature normalization. Exact
  execution requires at least 50 target rows and is target-order sensitive. The
  adapter fails early below that minimum; Sprint 3B must freeze row order.
- The source repository has no checkpoint and no license. Neither is inferred,
  replaced, or approximated.
- PyTorch 1.7.1 is explicitly reported by the official README to produce
  incorrect results. The container pins 1.0.0.

## Repeatable unblock conditions

Sprints 3B and 3C remain unauthorized unless an authoritative artifact closes
the identity gap: either (1) the authors provide the exact paper-era checkpoint
with checksum, runtime/preprocessing provenance, and usable permission; or (2)
the authors provide an exact paper-era CPU implementation/patch, or an accessible
matching CUDA environment and pinned recipe. That artifact must reproduce the
released Table 2 result under a predeclared three-run tolerance, pass the adapter
and provenance gates, and receive primary review before any project input is
scored or any project comparison outcome is viewed. Sprint 3B must then freeze
adaptation boundaries, row order, seeds, coverage/failure gates, and the
model-aware analysis contract outcome-blind.
