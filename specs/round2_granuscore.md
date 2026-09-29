# Round 2 GranuScore experiment

This sprint evaluates the official GranuScore 1.0.1 release as a related but
distinct construct. GranuScore measures semantic referential granularity:
higher percentiles mean coarser, more abstract references. The existing human
rubric and specificity predictors run in the opposite conceptual direction.
Negative association can therefore be agreement in the expected direction.

The official five-score notebook example must reproduce before any project
text is scored. The package wheel, upstream repository, hierarchy-transformer
revision, model artifacts, spaCy model, CUDA base image, and resolved Python
dependencies are pinned. Runtime scoring is offline and uses the exact official
default HiT/random-anchor pipeline. No model is trained or tuned.

The primary scope is all 1,295,205 canonical sentences because the frozen
public benchmark passed the predeclared throughput gate. The same scorer is
also applied to the existing 80-row human pilot and the 60 existing manual
controlled-edit pairs. Inputs are joined by exact IDs and hashes. Rows for
which the official splitter finds no referential unit are retained with the
upstream fallback score and flagged; key results are repeated without them.

Primary reports always preserve native GranuScore direction. A negated-rank
view is allowed only as a clearly marked secondary aid. Cross-model raw score
differences are forbidden because the scales have different meanings. Full
corpus uncertainty uses 1,000 document-cluster bootstrap replicates; pilot and
edit probes use 10,000 paired sentence/pair replicates. These analyses measure
association and behavioral sensitivity, not comparative accuracy.

The complete frozen contract is
`configs/round2_granuscore_v1.json`. Container code belongs in
`granuscore_container/`; raw scores stay under ignored `outputs/round2/`, while
only compact, privacy-safe evidence is eligible for `analysis/round2_granuscore/`.
