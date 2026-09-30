# M8 — Backward codecs: read what older Lucene wrote

> **Goal:** the Rust engine opens, searches and merges segments written by
> every Lucene version an OpenSearch 3.x cluster can hold -- Lucene 9.x and
> 10.x -- instead of requiring a reindex or a force-merge first.

| | |
|---|---|
| **Effort** | XL |
| **Depends on** | [M7](m7-core-complete.md) (per-field formats, the inventory gate) |
| **Unblocks** | adopting the Rust engine on existing indices without a rewrite |
| **Status** | not started |

---

## Why this milestone exists

Today only 10.5.0's `Lucene104` codec is read. A segment with any other codec
name or format version fails its header check, and the plugin falls back to
Lucene for the whole index (`parity.md`, "Deliberately out of scope"). An
OpenSearch 3.x cluster upgraded from 2.x holds Lucene 9 segments until they
are merged away, so the Rust engine cannot serve it without the
reindex-and-alias-swap in [`operations.md`](../operations.md) -- the single
largest adoption cost the rollback work measured (33.8 s for a small index,
proportional to data for a real one).

`lucene-backward-codecs` is 184 files, 42k lines of code. Almost all of it is
*read* code for retired formats: the same decoders this port already has,
at earlier versions.

---

## Scope

### In scope

- Read every codec OpenSearch 3.x can open: `Lucene90`…`Lucene103` and the
  retired postings (`Lucene90`/`Lucene99` postings, `Lucene90BlockTree`),
  points (BKD versions before 10), norms, doc values, stored fields, term
  vectors, vectors (`Lucene91`…`Lucene95` HNSW) and segment infos.
- Merging old segments into `Lucene104` ones (a merge reads old, writes new),
  so an index converges by ordinary merging.
- The OpenSearch plugin serving such indices natively instead of falling back.

### Out of scope

- Writing retired formats. Lucene's backward-codecs write them only in tests.
- Lucene 8 and earlier: OpenSearch 3.x cannot open those either.

---

## Tasks

### T8.1 — Fixture corpus from every supported version

A generator run against each Lucene release OpenSearch 2.x and 3.x shipped
(9.x through 10.4), producing one index per version with every field type;
checked in like today's fixtures, so the differential tests need no network.

### T8.2 — Version-dispatching readers

`CodecUtil` header dispatch per format to the matching decoder generation;
each retired format ported as close to Java as possible, reusing the current
decoder wherever the wire format did not change.

### T8.3 — Old BKD, old HNSW, old postings
### T8.4 — Merge old into new
### T8.5 — Plugin: drop the `postings_format` fallback for supported versions

---

## Acceptance criteria

- [ ] Every fixture index from T8.1 opens, passes this port's `CheckIndex`,
      and returns the same hits and scores as the Lucene version that wrote it.
- [ ] Merging a mixed-version index yields `Lucene104` segments that real
      Lucene 10.5.0 reads and `CheckIndex` passes.
- [ ] A cluster upgraded from OpenSearch 2.x serves its old index natively,
      verified by `verify-opensearch.sh` against a snapshot restored from 2.x.
- [ ] Reading an old format is no slower than Lucene reading it.

## Risks and unknowns

- **Fixture provenance.** Each version's fixtures must come from that
  version's jars; a generator that silently runs 10.5.0 against all of them
  proves nothing. `gen-fixtures.sh --check` must pin the jar per version.
- **Scope creep toward writing.** Only reading is needed; resist porting the
  writers "for symmetry".

## Exit artifacts

- `fixtures/data/backward/<version>/` indices and their generator
- `docs/parity.md` rows for every `lucene-backward-codecs` class
- The plugin's fallback table in `feature-matrix.md` updated
