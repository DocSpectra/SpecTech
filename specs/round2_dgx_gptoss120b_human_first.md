# GPT-OSS-120B human-first candidate packet

## Purpose and gate

Score the exact 180 returned GPT-OSS-120B candidates with the frozen local
sentence measures and release a new blinded 180-row review packet. The return
is used under the reviewed `accepted_with_author_provenance_waiver`
disposition. This phase creates no result about candidate quality and performs
no model call, human-outcome analysis, generator comparison, or manuscript
edit.

## Frozen population and identities

The population is the exact 60 originals and three returned candidate slots
per source from DGX run `dgx-5411bb15be9f`. The archive, candidate member,
source hashes, candidate hashes, remote IDs, case/slot order, generation
config, generation freeze, model identity, and accepted provenance disposition
are checksum-bound. A new deterministic local alias preserves the remote ID in
the ignored key and is a proven 180-row bijection; it does not change text,
source, direction, or slot.

## Scoring and policies

All 240 sentences receive frozen Round 1 SpeciTeller, each of the three
approved Ko official-release instances, and native GranuScore. Each primary
score must be unique, finite, in range, and complete. The Ko mean uses exact
decimal arithmetic and GranuScore is direction-aligned by negating its native
higher-is-coarser delta. The prior automatic surface proxy is audit-only.

Before human outcomes, freeze unguided slot 1, each primary scorer policy,
secondary Ko mean, direction-aligned GranuScore, and the inherited rank
consensus. Add-specific chooses the greatest aligned delta, de-specific the
smallest, and neutral the smallest absolute delta; every tie uses the lowest
slot.

## Blind review packet

The packet has exactly `Review ID | A | B | Score`. Fresh namespaces and seeds
create opaque IDs, 90/90 overall source/candidate side balance, exact 50/50
balance in all six hidden corpus-by-direction cells, and an independent row
shuffle. The reviewer sees no condition, role, model, slot, metric, proxy,
policy, or prior outcome. Scores are restricted to 1--5 or X under the same
definitions as the reviewed Gemma packet.

Text-bearing files, raw scores, mappings, and the reversible key remain
ignored. Tracked evidence contains only portable code, method bindings,
content-free counts and hashes, tests, and handoffs. Phase D remains locked
until the user returns the completed packet and it is frozen before unblinding.
