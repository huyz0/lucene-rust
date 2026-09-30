# M8 — Backward codecs: read what older Lucene wrote

> **Goal:** the Rust engine opens, searches and merges segments written by
> every Lucene version an OpenSearch 3.x cluster can hold -- Lucene 9.x and
> 10.x -- instead of requiring a reindex or a force-merge first.

| | |
|---|---|
| **Effort** | XL |
| **Depends on** | [M7](m7-core-complete.md) (per-field formats, the inventory gate) |
| **Unblocks** | adopting the Rust engine on existing indices without a rewrite |
| **Status** | in progress: T8.1 fixture corpus delivered |

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

### T8.1 — Fixture corpus from every supported version · delivered 2026-09-30

`scripts/gen-bwc-fixtures.sh --only <version>` compiles
`fixtures/bwc/BwcWrite.java` against that release's own `lucene-core` jar
and writes `fixtures/data/bwc/<version>/`: two segments (3,000 and 400
documents, deletions in both) with every field kind -- text with positions,
offsets and payloads, term vectors, `DOCS`/`DOCS_AND_FREQS` fields, all five
doc-values types, norms, 1-D int/long and 2-D int points, float vectors, and
byte vectors from 9.5. `fixtures/bwc/BwcDump.java`, on 10.5.0 with
backward-codecs, then writes `expected.txt`: segment and field infos in full
(codec names, per-field format attributes, file lists), and every kind of
content as an FNV-1a digest over a canonical byte stream with its counts
(postings with positions/offsets/payloads, norms, doc values, points, stored
fields, term vectors, vectors), plus a 10-nearest KNN result per vector
field. The Rust side reproduces each digest from its own readers.

One version per codec era: 9.0.0 (`Lucene90`), 9.1.0 (`Lucene91`), 9.3.0
(`Lucene92`), 9.4.2 (`Lucene94`), 9.8.0 (`Lucene95`), 9.11.1 (`Lucene99`),
9.12.2 (`Lucene912`), 10.0.0 (`Lucene100`), 10.2.2 (`Lucene101`) and 10.4.0
(already `Lucene104`: a cross-check of the current reader). Postings formats
across them: `Lucene90`, `Lucene99`, `Lucene912`, `Lucene101`; vectors:
`Lucene90`/`91`/`92`/`94`/`95`/`99` HNSW -- 9.2 and 9.3 name their vector
files `lucene92HnswVectorsFormat` (lower-case `l`), a quirk the file
resolution must reproduce. Content digests are reproducible on
regeneration; only segment ids and file names change.

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
