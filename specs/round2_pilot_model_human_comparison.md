# Round 2 Existing-Pilot Model--Human Comparison

## Frozen scope

This outcome-blind protocol reuses the exact existing 40 GitHub and 40 Ansible
pilot sentences, the three complete existing ordinal label sets, the frozen
SpeciTeller scores, and the already-saved Ko run01/run02/run03 scores. It does
not authorize annotation, relabeling, retraining, rescoring, controlled-edit
changes, or new sentence selection.

The four primary score instances are frozen SpeciTeller and Ko run01, run02,
and run03. The Ko rows are three stochastic instances of one maintained
post-publication implementation, not three architectures. Their rowwise
arithmetic mean is a secondary sensitivity and is not an independent fifth
model.

The machine-readable contract is
`configs/round2_pilot_model_human_comparison_v1.json`; the exact ordered IDs
are in `configs/round2_pilot_sent_ids_v1.csv`. Both predate inspection of any
new Ko--human outcome in freeze commit
`345458f09bda6ba673aa4dcaf43b47e64ed9b70b`.

## Identity, privacy, and joins

All six raw annotation files are retained only in the ignored local output
tree. Their SHA-256 identities are frozen under anonymous IDs `ann_a`,
`ann_b`, and `ann_c`, mapped to the already-published Table values. Tracked
evidence must never contain participant names, raw label rows, notes, sentence
text, or absolute legacy paths.

Each corpus must contain exactly 40 unique manifest IDs. Every annotator and
every primary score instance must join one-to-one and cover all 40 rows. The
label fields embedded in the raw pilot files are authoritative; text, bucket,
and the old SpeciTeller score field are identity/reconciliation fields only.
Any missing label or row stops the analysis. The existing pooled label is the
rowwise arithmetic mean of all three complete 1--5 ordinal labels.

## Frozen analysis

Spearman rank correlation is principal. Average ranks handle ties; values are
never jittered. Results are separate for GitHub and Ansible. The analysis
reports each primary score instance against each annotator and pooled label,
all pairwise primary model--model rank agreements, and the Ko mean only as a
secondary sensitivity.

Uncertainty uses 10,000 paired sentence bootstrap replicates with master seed
`20260809`. Within each corpus, every replicate applies one sampled index
vector jointly to every model and human column. Percentile 95% intervals are
reported. All unordered model--human correlation contrasts use the same draws.
Degenerate replicates are excluded and counted; any output with fewer than
95% valid replicates stops the run.

The many estimates are descriptive. No p-value or interval is used to select a
winner, and no multiplicity-adjusted hypothesis-testing family is claimed.
Human correspondence and pairwise model agreement are not comparative
accuracy; the pilot labels are ordinary non-expert ordinal judgments, not a
gold standard.

## Outcome-blind audit completed before freeze

- The six raw label files each contain 40 unique IDs and 40 complete labels.
- Their exact ID sets and order agree within corpus and with the frozen
  manifest.
- The six relevant Ko checkpoints and six full score files independently
  reproduce their Pair 3B metadata hashes.
- Canonical sentence and SpeciTeller score hashes reproduce the retained
  Round 1 artifacts.
- No checkpoint was loaded and no inference was run.
- Only already-published SpeciTeller pilot values were used to preserve the
  anonymous Ann. A/B/C mapping. No Ko score was joined to a label before this
  freeze.

## Stop boundary

Any hash, coverage, ordering, label completeness, range, or bootstrap-validity
failure stops the analysis without repair, row dropping, rescoring, or a
post-outcome method change. Compact tracked evidence is checksum-addressed and
privacy-filtered; full row-level joins remain ignored.

Reporting-only correction: after the first outcome run, the manually entered
`frozen_at_utc` value was corrected to the actual immutable Git commit time and
the full freeze commit ID was added. The original commit already contains the
complete scientific contract. No ID, input hash, join, statistic, bootstrap,
contrast, stop, or claim rule changed.
