# M8 — Backward codecs: read what older Lucene wrote

> **Goal:** the Rust engine opens, searches and merges segments written by
> every Lucene version an OpenSearch 3.x cluster can hold -- Lucene 9.x and
> 10.x -- instead of requiring a reindex or a force-merge first.

| | |
|---|---|
| **Effort** | XL |
| **Depends on** | [M7](m7-core-complete.md) (per-field formats, the inventory gate) |
| **Unblocks** | adopting the Rust engine on existing indices without a rewrite |
| **Status** | in progress: T8.1 delivered; T8.2 and T8.3 delivered for every non-vector format |

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

### T8.2 — Version-dispatching readers · delivered 2026-09-30 (non-vector)

`CodecUtil` header dispatch per format to the matching decoder generation;
each retired format ported as close to Java as possible, reusing the current
decoder wherever the wire format did not change.

Delivered. `crates/lucene-codecs/src/codecs.rs` records what each codec
(`Lucene90`..`Lucene104`) composes; the one component whose header cannot
tell its generations apart, `.si` (`Lucene90SegmentInfoFormat` has no
`hasBlocks` byte), is chosen by codec name at every production read site
(`segment_info::parse_for_codec`). Everything else dispatches on the file:
`.fnm` on its codec name (`Lucene90FieldInfos`), `.kdm` on its BKD version,
`.doc`/`.pos`/`.pay` on `<name>PostingsWriter*` (`postings::PostingsFormat`),
the term dictionary on the postings terms header inside `.tmd`
(`Lucene90PostingsWriterTerms` = `Lucene90BlockTreeTermsReader`), and
`decodeTerm` on each field's `PerFieldPostingsFormat.format`. The per-class
record is `docs/inventory/lucene-backward-codecs.tsv`
(`check-port-inventory.py --module backward-codecs`, in the gate and CI).

**Verification** (`crates/lucene-search/tests/bwc_fixtures.rs`):

- `bwc_fixtures_match_lucene` reproduces every line of every
  `expected.txt` from this port's readers. 10.4.0 passed before any M8 change
  (the current readers were already exact); every other version passes in
  full since the retired HNSW readers landed (T8.3), so the test's
  `EXPECTED_FAILURES` table is empty.
- `every_version_searches_like_the_current_codec` opens every fixture with
  `DirectoryReader` and runs 34 queries (term, prefix, wildcard, regexp,
  term-in-set, exact and sloppy phrase on positions/offsets/payloads fields,
  points range, dismax, boolean with filter and exclusion), top 20 exact and
  block-max pruned: every version returns 10.4.0's hits, scores (bit for bit)
  and totals.
- `every_version_passes_check_index`: this port's `CheckIndex` finds nothing
  on any fixture, and its vector (and, on `_0`, graph) families must have
  run.
- Seen to fail: shifting the retired postings' first-document delta base by
  one fails all three.

### T8.3 — Old BKD, old HNSW, old postings · delivered 2026-09-30 (quantized vectors open)

Old BKD: version 9 (every index before 10.2) and its scalar `BPV_24` doc-id
layout. Old postings: `Lucene90`, `Lucene99`, `Lucene912`, `Lucene101` (and
`Lucene103`) `.doc` framing and `ForUtil`/`PForUtil` generations
(`backward_codecs/{postings,for_util}.rs`), and `Lucene90BlockTreeTermsReader`,
whose FST index is converted at open into the trie the current dictionary
navigates (`backward_codecs/blocktree.rs`; the `.tim` blocks are unchanged).

Old HNSW: `Lucene90`, `Lucene91`, `lucene92`, `Lucene94` and `Lucene95`
(`backward_codecs/hnsw_vectors.rs::RetiredHnswVectorsReader`): each `.vem`
entry becomes the current flat reader's `FlatFieldEntry` plus a graph view
(`Lucene90`'s single level, the fixed-slot graphs of 9.1-9.4, and 9.5's, which
is `Lucene99` version 0). Search is each reader's own: the current
`HnswGraphSearcher` without an exhaustive branch for 9.1-9.8, and 9.0's
random-entry-point search seeded from the `.vex` checksum through a
bit-exact `SplittableRandom.nextInt(bound)`. KNN hits match Java's exactly,
including 10.5.0's `Lucene90` quirk of reporting vector ordinals as doc ids.
The KNN query layer (`vector_query::GraphReader`) and `CheckIndex` serve all
five.

**Open, precisely:**

- `Lucene99ScalarQuantizedVectorsFormat`/`Lucene99HnswScalarQuantizedVectorsFormat`
  and `Lucene102(Hnsw)BinaryQuantizedVectorsFormat`: not ported (inventory
  `deferred:M8`). No default codec wrote them -- they are per-field opt-ins --
  and no fixture holds one yet.
- Performance (port-workflow stages 2-3): a retired-format term is decoded
  whole when its cursor opens and served through the tail-block path, with no
  block skipping and no impacts (the skip data, trailing or inline, is
  stepped over); the FST-to-trie conversion is an open-time pass over each
  field's index. No benchmark against Lucene exists yet for either.
### T8.4 — Merge old into new · delivered 2026-09-30

This port's `IndexWriter` merges segments any 9.0-10.4 release wrote into
`Lucene104` ones, by `force_merge` and by an ordinary policy merge at commit,
and applies buffered deletes to them. What had to change was only where the
writer finds an old segment's files: its postings are named (and their headers
suffixed) after the segment's own format (`_0_Lucene90_0.tim`), read off the
segment's `.tim` (`index_writer.rs::postings_file_base`), and its vectors sit
in a retired format's `.vem`/`.vec`/`.vex` triple
(`index_writer.rs::retired_vector_files`), served by
`RetiredHnswVectorsReader`. The merged graph is built from scratch over a
retired source's vectors, as 10.5.0's `IncrementalHnswGraphMerger` does for a
reader that is not an `HnswGraphProvider`.

**Verification.** `scripts/verify-bwc-merge.sh` (Java 10.5.0 +
backward-codecs; 20 of 20 pass): for every fixture version, (1) the old index
alone through `force_merge(1)`, and (2) the old index plus three segments Lucene
10.5.0 appends to it (`fixtures/bwc/BwcAppend.java`, with deletes against the
old segments) through an ordinary `TieredMergePolicy` merge that takes old and
new segments together. `fixtures/bwc/BwcMergeCheck.java` then requires
`CheckIndex` clean, every segment the merge wrote to be `Lucene104` (postings
`Lucene104`, vectors `Lucene99HnswVectorsFormat`), and every live input
document in the output with all its content: per-field SHA-256 digests over
postings (positions, offsets, payloads), norms, the five doc-values types,
points, stored fields, term vectors and vectors, with documents matched by
their stored fields (a merge may order its sources either way). Seen to fail:
one extra delete, and one changed numeric doc value, in a merged copy. On the
Rust side, `bwc_fixtures.rs::every_version_force_merges_into_lucene104`
deletes a document of an old segment by term, force-merges, and requires one
`Lucene104` segment, a clean `CheckIndex` and every query of the 34-query set
matching as many documents as before; with the delete path reverted to
`Lucene104` file names it fails (`NotFound`).

**Divergence found, not fixed here:** `IndexWriter::force_merge` merges the
smallest segments first and concatenates them in that order, where 10.5.0's
`TieredMergePolicy.findForcedMerges` merges everything in size-descending
order in its single-segment case -- so the merged segment numbers its
documents in a different order than Lucene's would (contents equal).
### T8.5 — Plugin: drop the `postings_format` fallback for supported versions

---

## Acceptance criteria

- [ ] Every fixture index from T8.1 opens, passes this port's `CheckIndex`,
      and returns the same hits and scores as the Lucene version that wrote it.
- [x] Merging a mixed-version index yields `Lucene104` segments that real
      Lucene 10.5.0 reads and `CheckIndex` passes. (`scripts/verify-bwc-merge.sh`,
      T8.4.)
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
