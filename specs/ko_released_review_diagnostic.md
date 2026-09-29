# Ko released Yelp/Movie reproduction diagnostic

## Scope and immutable boundary

This is a new supplemental diagnostic of the Yelp-review and movie-review
conditions released by Ko, Durrett, and Li. It uses the same pinned
post-publication official CPU implementation, source training data, runtime,
SE+AD mean--standard-deviation objective, preprocessing, metrics, and three-run
interpretation used for the closed Twitter reproduction gate. The only
scientific substitution is the upstream-released domain selected by
`--test_data`.

The failed Twitter gate remains closed and byte-frozen under
`configs/ko_2019_reproduction_v1.json` and
`analysis/round2_ko_reproduction/`. This diagnostic neither reopens that gate
nor changes the approved comparator identity.

## Authoritative released mapping

The official README says that `*s.txt` contains annotated sentences, `*v.txt`
contains mean Turker specificity labels in matching order, `*u.txt` contains
unlabeled target sentences, and `*l.txt` contains unused binary labels. The
pinned `data2.py` maps `yelp` to `yelps/yelpv/yelpu/yelpl` and `movie` to
`movies/moviev/movieu/moviel`. It also sets `tv=1`, so the first annotated row
is withheld and predictions cover 844 Yelp rows and 919 Movie rows.

The paper reports 845 annotated and approximately 95K unlabeled Yelp sentences,
and 920 annotated and approximately 12K unlabeled Movie sentences. Exact
released counts are 95,650 and 11,855 unlabeled rows. Table 2's SE+AD
mean--standard-deviation targets are frozen in the config.

The Linux image executes LF-normalized bytes from the checksum-pinned GitHub
archive. The CRLF form reproduces the previously frozen source/Twitter hashes;
the config records both LF execution hashes and CRLF-equivalent released hashes
for Yelp and Movie. No text or label value was inspected during this identity
audit.

## Frozen execution

The complete machine-readable contract is
`configs/ko_released_review_diagnostic_v1.json`. Run exactly three separate
Docker processes for each domain:

```powershell
python scripts/ko_released_review_diagnostic.py run --domain yelp --run-id run01 --attempt-id attempt02
python scripts/ko_released_review_diagnostic.py run --domain yelp --run-id run02 --attempt-id attempt01
python scripts/ko_released_review_diagnostic.py run --domain yelp --run-id run03 --attempt-id attempt01
python scripts/ko_released_review_diagnostic.py run --domain movie --run-id run01 --attempt-id attempt02
python scripts/ko_released_review_diagnostic.py run --domain movie --run-id run02 --attempt-id attempt01
python scripts/ko_released_review_diagnostic.py run --domain movie --run-id run03 --attempt-id attempt01
python scripts/ko_released_review_diagnostic.py summarize
```

NumPy and Torch use upstream seed 1234. Python `random`, which controls word
augmentation branches, remains unseeded. Each run trains for the unchanged 30
epochs, evaluates after withholding the first annotated row, transforms the
released 1--5 label as `(label - 1) / 4`, and reports Spearman, Kendall's tau,
and MAE.

The observed standard deviation is the population standard deviation across
the three run-level metrics. The paper does not report its `ddof` convention;
population SD is frozen because it is NumPy's default and matches the existing
Ko variability evidence. Exact run values and both raw and paper-SD-scaled
deltas remain visible. A condition is a match only when both its mean and
dispersion meet the frozen tolerances. A clear mean displacement is a shift;
the gray zone and dispersion-only disagreements are indeterminate, and a
dispersion mismatch alone can never create a shift. A domain matches only if
all three metrics match, shifts if at least one metric shifts (with direction
reported), and is otherwise indeterminate. These are descriptive reproduction
classes, not hypothesis tests.

Every operational attempt receives a unique, non-overwriting `attempt_id`
beneath its scientific `run_id`. Exactly one successful attempt must exist for
each of `run01`--`run03`; otherwise the domain and all its metrics are packaged
as indeterminate without a fabricated three-run summary. The compact evidence
inventories successful, failed, and incomplete attempts. `attempt.json` is
always required; predictions, metrics, checkpoint, logs, and `run_metadata.json`
are required only for a completed attempt.

## Evidence boundary

Raw predictions, checkpoints, logs, and released inputs remain ignored under
`outputs/round2/ko_released_review_diagnostic/`. The compact tracked pack under
`analysis/round2_ko_released_review_diagnostic/` contains only metrics, counts,
hashes, classifications, environment identities, validation, and safe wording.
Do not redistribute upstream source, data, checkpoints, the built image, or
GloVe absent permission.

## Completed result

All six scientific runs completed with exact coverage: 844 predictions per
Yelp run and 919 per Movie run. The frozen classification marks all six
domain-by-metric conditions as shifts. Yelp averages are Spearman `0.6142 +/-
0.0140`, Kendall tau `0.4442 +/- 0.0124`, and MAE `0.1388 +/- 0.0056`; Movie
averages are `0.2616 +/- 0.0268`, `0.1839 +/- 0.0188`, and `0.1642 +/- 0.0009`.
Exact run values, paper comparisons, distances, directions, attempt inventory,
and artifact hashes are under `analysis/round2_ko_released_review_diagnostic/`.

The result extends the released-data reproduction discrepancy beyond Twitter,
but does not reopen the Twitter gate or change the approved post-publication
comparator's separate identity.
