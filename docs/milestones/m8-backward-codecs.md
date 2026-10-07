# M8 — Backward codecs: read what older Lucene wrote

> **Goal:** the Rust engine opens, searches and merges segments written by
> every Lucene version an OpenSearch 3.x cluster can hold -- Lucene 9.x and
> 10.x -- instead of requiring a reindex or a force-merge first.

| | |
|---|---|
| **Effort** | XL |
| **Depends on** | [M7](m7-core-complete.md) (per-field formats, the inventory gate) |
| **Unblocks** | adopting the Rust engine on existing indices without a rewrite |
| **Status** | Delivered 2026-09-30: T8.1-T8.5 (the quantized vector formats and `IndexUpgrader` included, no `deferred:M8` row left in either inventory; T8.5's JVM half and a real 2.19-to-3.8.0 upgrade verified, the latter in CI); acceptance criteria 1-4 met (4 since the retired terms index is read in place) |

---

## Why this milestone exists

Today only 10.5.0's `Lucene104` codec is read. A segment with any other codec
name or format version fails its header check, and the plugin falls back to
Lucene for the whole index (`parity/matrix.md`, "Deliberately out of scope"). An
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

Query-level lines (M8 close-out): for every vector field `BwcDump` also runs,
through an `IndexSearcher` over the one segment, a filtered KNN query
(`knnf`, `doc % 3 == 0`), one whose filter is small enough to go exact
(`knne`, `doc % 89 == 0`, k = 20), `PatienceKnnVectorQuery` (saturation 0.5,
patience 2) and `*VectorSimilarityQuery` unfiltered and filtered (`vsim`,
`vsimf`, the fifth KNN hit's score as threshold). `gen-bwc-fixtures.sh --dump
<dir>` rewrote only `expected.txt` for every version (the indexes are
untouched, every earlier line came back byte for byte).

Quantized vectors (M8 close-out): `fixtures/bwc/BwcQuantized.java` writes
`fixtures/data/bwc-quantized/<version>/` with each release's own jar
(`gen-bwc-fixtures.sh --quantized <version>`), two segments of 1,000 and 200
documents, every 13th deleted, per-field formats through a `FilterCodec`
under the default codec's name: 9.9.2 (`Lucene99ScalarQuantizedVectorsFormat`
metadata version 0, 7 bits, every similarity, plus a `BYTE` field), 9.12.2
(version 1: 7 and 4 bits, 4-bit compressed, dynamic confidence interval, the
flat format per field) and 10.2.2 (all of that plus
`Lucene102(Hnsw)BinaryQuantizedVectorsFormat`, every similarity, dense and
sparse). 5.1 MB in all.

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
whose FST index the current dictionary's enum walks in place (`backward_codecs/blocktree.rs`;
the `.tim` blocks are unchanged; see the close-out note below).

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

**Quantized vectors (M8 close-out).** `Lucene99ScalarQuantizedVectorsFormat`
(`backward_codecs/scalar_quantized_vectors.rs`: both `.vemq` versions, 7/4-bit
and compressed codes, the legacy `ScalarQuantizer`, and both scorers core
ships for it) and `Lucene102BinaryQuantizedVectorsFormat`
(`backward_codecs/binary_quantized_vectors.rs`: one-bit codes, the four-bit
query, its own score clamps), with their HNSW wrappers, as one per-field
reader (`backward_codecs/quantized_vectors.rs`) that the KNN layer serves as
`vector_query::GraphReader::Quantized`: float queries are scored on the codes
wherever Java scores them (graph walk, exhaustive branch, exact fallback,
vector similarity), and each format searches its own way -- the flat scalar
format collects nothing (`FlatVectorsReader.search`), the flat binary one
scores every ordinal. Every quantized fixture line (`vec`, `knn`, `knnf`,
`knne`, `patience`, `vsim`, `vsimf`, with score bits) matches Lucene 10.5.0 +
backward-codecs (`bwc_fixtures.rs`). Seen to fail: dropping the scalar
format's per-vector correction constant fails every dot-product/cosine/MIP
line; a `1/16` query scale for the binary format fails every binary line. Not
caught by the fixtures: the position of the binary format's Euclidean clamp
(its scores never go negative there).

**Retired graphs take any collector (M8 close-out).** The 9.x readers' search
takes a `KnnCollect` (`RetiredHnswVectorsReader::search_with`), so
`AbstractVectorSimilarityQuery` walks a retired graph with its own
`VectorSimilarityCollector` (it used to refuse with `InvalidKnnQuery`), and
`PatienceKnnVectorQuery`'s saturation collector and a deadline wrap a retired
leaf's collector as they wrap a `Lucene99` one (they used to be skipped). On
9.0 the similarity query inherits `Lucene90`'s ordinals-as-documents quirk,
and an ordinal that names a deleted document is dropped at collection, after
the top `k` was cut, as `IndexSearcher` drops it (Java returns nine hits for
9.0's `_1` patience query). The `knnf`/`knne`/`patience`/`vsim`/`vsimf` lines
of every bwc version match. Seen to fail: skipping the collector wrap fails
`patience` on 9.1-9.8; translating `Lucene90` ordinals fails 9.0's `vsim`.

**Performance (port-workflow stages 2-3).** The lazy cursors
(`postings.rs::{LazyDocsCursor,PositionsCursor}`) now serve every retired
generation block by block, as they serve `Lucene104`, at the generation's own
128-document block: `Lucene912`/`Lucene101`/`Lucene103` through their inline
level-0 and level-1 (every 4,096 documents) headers, and `Lucene90`/`Lucene99`
through the trailing multi-level skip list
(`backward_codecs/skip_list.rs::SkipList`, a port of
`MultiLevelSkipListReader` + `Lucene90ScoreSkipReader`), whose level-0 and
level-1 entries give `advance_shallow` each block's extent and impacts without
decoding it. The first cut had decoded a whole term at open and stepped over
its skip data.

Benchmark: `benchmarks/queries.tsv` (87 queries) over a 1M-document index
written by `GenCorpus` compiled against Lucene 9.0.0 and against 9.12.2 (one
segment each), with the M1 bench-runner against Java's `BenchRunner` on 10.5.0
plus backward-codecs, on a 4-core host other jobs kept at load 8-12. Every
query returns Java's hits, top set and top score on both corpora.

| | queries slower than Java, 9.0 corpus | 9.12 corpus |
|---|---|---|
| whole-term decode (first cut) | 47 of 87 (term query q01 at 0.33x) | 42 of 87 |
| skip data + impacts | 2 (q80, q89) | 7 (q34, q58, q59, q60, q77, q80, q89) |
| the losers again, 1 s warm-up, 3 s measured, three rounds | q89 in 3 of 3 (0.68-0.94x) | none in a majority of rounds |
| `for_decode` in one read, masks by multiply (M8 close-out); all 87, 1 s warm-up, 2 s measured, one pass at load 6 | none (lowest q47 1.00x, q89 1.19x) | not re-run |

Single runs of the whole set move by up to 2x on this host (Java's q80 read
311 qps in one run and 1,087 in the next), which is why the losers were run
again rather than taken as measured.

**q89 on the 9.0 corpus, closed (M8 close-out).** q89 is a `t1` term query
sorted by a keyword doc value then score. Profiled (`perf record`, the bench
runner's release build, q89 alone over the 9.0 corpus rebuilt by `GenCorpus`
against `lucene-core-9.0.0`): 20% of the time was
`backward_codecs::for_util::for_decode`, the `decodeSlow` port every retired
generation's blocks decode through (here the `t1` term's `Lucene90` doc
blocks, under the scorer's `advance`) -- which read
the packed block one long at a time and rebuilt every lane mask lane by lane
inside its tail loop. One `read_bytes` of the block and a multiply by a
per-primitive replication constant bring it to 7% (plus 3% zeroing its
scratch), same bits. The rest of the profile is format-independent (the
postings cursor's `advance`, the keyword comparator's `quick_reject`/
`quick_compare_bottom`, the collection loop). Measured on the same 4-core
host at load 4-9, five interleaved rounds of 1 s warm-up and 3 s measured:
q89 88-102 qps against 75-94 before the change and 64-79 for Java (every
round above Java; best of the day's runs 101.9 against 92.8, 1.10x). A
second attempt, decoding in place into the output block to save the zeroing,
measured slower (`for_decode` back to 20% of the profile) and was dropped. `Lucene912`'s blocks decode through the same
function, so the 9.12 corpus (where q89 already won) runs the same code; it was not
rebuilt and re-measured here.

Skip data is verified on `fixtures/data/bwc-big/<version>/`
(`fixtures/bwc/BwcBig.java`, `gen-bwc-fixtures.sh --big`): one 20,000-document
segment per generation whose terms reach every level of both kinds of skip
data. `tests/bwc_postings.rs::*_skip_data_at_every_level` checks the lazy
cursor's `next_doc` and `advance` at strides from 1 to 8,193 against the eager
decode, and that the level-0 and level-1 impacts `advance_shallow` exposes
bound every frequency in their span. `skip_list.rs`'s unit tests check
`skip_to` against a port of the writer at up to four levels, and truncated or
bit-flipped skip data. Seen to fail: `SkipList::level1` handing back the
level-0 entry's impacts fails both unit tests and the `Lucene90`/`Lucene99`
fixture tests (on the `peak` term, whose one high-frequency document in 1,500
is what separates the levels; the first version of the fixture, without it,
passed that mutation). Not caught: a norm-side error in an impact, since the
tests check frequencies only.

**FST-to-trie conversion, deferred to each field's first use (M8
close-out).** `examples/reader_open_profile` over the same corpora gave
`blocktree::open_shared` a minimum of 6.0 ms on the 9.0 segment (88 KB `.tip`)
and 3.3 ms on the 9.12 one (77 KB), against 1 us for the current format's
trie, which is read in place; Lucene opens a `Lucene90BlockTreeTermsReader`
FST in place (`FieldReader`: `new OffHeapFSTStore(indexIn, ...)`). Now `open`
reads each field's `.tmd` record and FST metadata (checking the body's extent
in `.tip`) and keeps a `backward_codecs/blocktree.rs::FstIndex`; the field's
first walk converts it (`blocktree.rs::TermsIndex::Deferred`, a `OnceLock`
shared by clones; a failed conversion fails every later walk with the same
message instead of retrying). A corrupt FST body is therefore found by the
field's first lookup rather than at open -- as Lucene's off-heap FST finds it
at its first seek. Unit tests: `the_fst_index_is_converted_on_first_use_not_at_open`,
`a_corrupt_fst_body_fails_the_first_walk_and_every_later_one`; the
differential suites (`bwc_fixtures.rs`, 5 of 5; `lucene-codecs` with
`bwc_postings`, all; `verify-bwc-merge.sh`, 39 of 39) pass unchanged.

Measured on the 4-core host at load 5-6 (other jobs running), minimum of five
interleaved rounds of 30 repetitions each, before and after binaries of
`reader_open_profile` side by side; Java is Lucene 10.5.0 + backward-codecs,
minimum of 300 `DirectoryReader.open`s after 300 of warm-up, on the same
directories (1M-document corpora rebuilt by `GenCorpus` against
`lucene-core-9.0.0`/`9.12.2`, force-merged to one segment):

| | 9.0 corpus | 9.12 corpus | `bwc-big/9.0.0` | `bwc-big/9.12.2` |
|---|---|---|---|---|
| `blocktree::open_shared`, before | 3,418 us | 3,422 us | 1.9 us | 1.9 us |
| `blocktree::open_shared`, after | 1.6 us | 1.6 us | 1.2 us | 1.2 us |
| `DirectoryReader::open`, before | 3,605 us | 3,585 us | 68 us | 68 us |
| `DirectoryReader::open`, after | 106 us | 106 us | 66 us | 66 us |
| Java `DirectoryReader.open` | 652 us | 716 us | 459 us | 515 us |
| after: `open_shared` + the first lookup in every field | 3,322 us | 3,335 us | 4.1 us | 4.1 us |
| Java: open + the first `seekExact` in every field, minus the open | 42 us | 60 us | 104 us | 84 us |

So opening a retired segment is now cheaper than Lucene's (0.11 ms against
0.65-0.72 ms on the 1M-document segments). What is left is the conversion
itself, moved from the open to the first lookup in each field: a query's first
visit to a field of a retired segment pays that field's share of the 3.3 ms
the 1M-document segment's fields cost together, once per segment for the life
of the reader (the plugin's reader reuse across refreshes keeps it), where
Lucene's first seek costs tens of microseconds. Removing it means walking the
FST in place (a `SegmentTermsEnum` over FST arcs, floor data read from the
arc outputs instead of from the trie); not done. The profile of the conversion
is flat -- the FST enumeration and one allocation per key, output and floor
record -- so a cheaper conversion would not close a 50x gap either.
**Terms index read in place (M8 close-out, superseding the conversion
above).** No path needed the trie: `crate::blocktree`'s `SegmentTermsEnum`
(seeks, `next`, the automaton intersection) now steps through the retired
FST itself (`backward_codecs/blocktree.rs::FstTermsIndex`), as `Lucene90`'s
`SegmentTermsEnum` does -- an arc per target byte (`FST.findTargetArc`), the
arcs' outputs accumulated, a frame pushed on every final arc with
`output + nextFinalOutput` decoded into block fp, `hasTerms` and floor data
(`pushFrame(arc, frameData, length)`). A frame keeps that floor data in its
own buffer (`Frame::floor_buf`) where a trie frame points into `.tip`; the
root frame reads `rootCode`. The conversion (`FstIndex::to_trie`,
`TermsIndex::Deferred`) is gone. Tests: `every_term_is_found_by_walking_the_fst`
(every term of the 9.0 fixture by `seek_exact`, `seek_ceil` on itself and just
past it), `a_corrupt_fst_body_is_met_by_the_lookups_that_reach_it` (three fill
patterns: errors naming the terms index or misses, no panic); `bwc_fixtures.rs`
5 of 5, `lucene-codecs` (with `bwc_postings`) all, `verify-bwc-merge.sh` and
`gen-bwc-fixtures.sh --check` unchanged. Seen to fail: ignoring the floor flag,
or dropping the arcs' outputs from the accumulated frame data, fails four of
`bwc_fixtures.rs`'s five tests (the digest test enumerates, so it does not
notice), and the latter fails the new unit test.

Measured on the same 4-core host at load 7-9 (other jobs running), five
interleaved rounds; each figure the best round. Rust: `reader_open_profile`,
minimum of 30 repetitions, before (the deferred conversion) and after binaries
side by side. Java: Lucene 10.5.0 + backward-codecs, minimum of 300
`DirectoryReader.open`s, and of 300 opens followed by `seekExact` of each
field's max term, after 300 of each as warm-up; the first-lookup figure is the
difference of those two minima, so it is noisy (median over rounds in
brackets).

| | 9.0 corpus | 9.12 corpus | `bwc-big/9.0.0` | `bwc-big/9.12.2` |
|---|---|---|---|---|
| first lookup in every field, before (the conversion) | 3,411 us | 3,478 us | 3.0 us | 3.0 us |
| first lookup in every field, after (in place) | 7.8 us | 7.8 us | 1.7 us | 1.7 us |
| Java, first `seekExact` in every field | 32 us (44) | 7 us (37) | ~0 us (22) | 38 us (44) |
| `DirectoryReader::open`, after | 105 us | 106 us | 69 us | 69 us |
| Java `DirectoryReader.open` | 391 us | 426 us | 285 us | 286 us |

`blocktree::open_shared` is unchanged at 1.2-1.6 us.

**Queries after the change.** The 87-query set over both 1M-document corpora
(1 s warm-up, 2 s measured, load 5-7), every query with Java's hits, top set
and top score: on 9.12 (its first full pass since T8.3) one query below Java,
q80 at 0.98x; on 9.0 one, q17 at 0.82x. Both again, three interleaved rounds
of 1 s warm-up and 3 s measured against the before binary and Java: q17 on
9.0 wins every round (10.0-10.5 qps against 8.9-9.5); q80 on 9.12 wins two
of three (1,014/880/969 qps against 1,013/981/928, 0.90-1.04x; the before
binary 826/930/937). q80 (a `t1` term sorted by a long doc value) spends its
time in the competitive points iterator and the retired `Lucene912` postings,
not the terms index. Two changes came out of profiling it: `for_decode` is
instantiated per (width, primitive, word) as Java's generated `ForUtil` has a
`decodeN` per width (the decode went from 10.0% of q80's profile plus 4.8%
zeroing to 6.6% plus 1.7%), and BKD version 9's scalar `BPV_24` doc ids read
straight from the mapped bytes. What remains of q80's cost is
format-independent: `CompetitiveVisitor::upgrade` allocates a `maxDoc`-bit set
per query, ~27 page faults a query on this segment -- the old conversion's
large free had hidden it by raising glibc's mmap threshold (957 faults per run
against 165,908 without it). Left as a follow-up (reuse the set across
queries).

### T8.4 — Merge old into new · delivered 2026-09-30

**Quantized groups and `IndexUpgrader` (M8 close-out).** A segment with
per-field quantized formats holds one vector file set per
`PerFieldKnnVectorsFormat` suffix; the writer finds each by its
`.vemq`/`.vemb` (`index_writer.rs::quantized_vector_files`) and
`merge::SourceVectors` carries every group of a source, each field served by
the group that has it; the merged field is plain `Lucene99HnswVectorsFormat`
over the raw vectors. `CheckIndex` opens every quantized group
(`vectors.quantized:<field>`: checksum, one code per raw vector, same
documents). `UpgradeIndexMergePolicy` (only segments whose version is not
10.5.0 are offered to the wrapped policy, the rest it leaves out merged into
one more) and `IndexUpgrader` are ported; the upgrader's tests found three
read paths that parsed an old `.si` without its codec
(`index_file_deleter::list_commits`, the pluggable-policy segment view,
`TemporalMergePolicy`'s date ranges), now fixed. `verify-bwc-merge.sh` covers
every bwc and bwc-quantized version three ways -- force merge, ordinary
merge with three Lucene 10.5.0 segments appended, and `IndexUpgrader` over
that mixed index: 39 of 39 pass.

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
### T8.5 — Plugin: drop the `postings_format` fallback for supported versions · delivered 2026-09-30

`NativeReaders` falls back with `postings_format` only for a postings format
outside `SUPPORTED_POSTINGS_FORMATS` (`Lucene90`, `Lucene99`, `Lucene912`,
`Lucene101`, `Lucene103`, `Lucene104`) -- a `completion` field's
`Completion104`, say -- instead of for anything but `Lucene104`.
`feature-matrix.md` and `opensearch-native-queries.md` say so.

**Verified:** `lucene-ffi`'s
`jvm_reader.rs::every_bwc_version_is_served_natively_like_the_current_codec`
opens every T8.1 fixture version (9.0.0 to 10.2.2) through the plugin's own
entry point (`ffi_open_jvm_reader`, with `segments_N` bytes, `maxDoc`s and
live-docs words as the JVM passes them) and requires term and boolean queries
to return the 10.4.0 fixture's hits, scores and totals.

**JVM half, verified (M8 close-out).** `scripts/opensearch-dist.sh && gradle
-p opensearch-plugin check` (JDK 25, Gradle 8.14.3) passes: `NativeSelfTest`
260,316 checks and 0 failures as it stood, 451,812 checks and 0 failures with
the change below (every compared score bit-exact); `EngineWriterDiffTest`
2,708 checks, 0 failures. `NativeSelfTest` used to take only the fixture
directories one level under `fixtures/data`, so no old-format index reached the
JVM path; it now also takes `bwc/<version>`, `bwc-big/<version>` and
`bwc-quantized/<version>` (17 indexes) and requires every one to open natively
through `NativeReaders` and answer its term, boolean, sorted, aggregated,
`terminate_after` and `min_score` searches as Lucene 10.5.0 + backward-codecs'
`IndexSearcher` does: 17 of 17. Seen to fail: dropping `Lucene90` from
`SUPPORTED_POSTINGS_FORMATS` fails the six 9.0-9.8 indexes (11 of 17).

**A real upgrade, verified (M8 close-out).** `scripts/verify-opensearch-upgrade.sh`
(`opensearch-plugin/e2e/verify_upgrade.py`) -- acceptance criterion 3 below.

---

## Acceptance criteria

- [x] Every fixture index from T8.1 opens, passes this port's `CheckIndex`,
      and returns the same hits and scores as the Lucene version that wrote it.
      (`bwc_fixtures.rs`, no expected failures left, now over the 10 bwc and
      3 bwc-quantized fixtures and the query-level `knnf`/`knne`/`patience`/
      `vsim`/`vsimf` lines too; the reference is Lucene 10.5.0 +
      backward-codecs reading each index, `BwcDump`.)
- [x] Merging a mixed-version index yields `Lucene104` segments that real
      Lucene 10.5.0 reads and `CheckIndex` passes. (`scripts/verify-bwc-merge.sh`,
      T8.4: 39 of 39, quantized fixtures and `IndexUpgrader` included.)
- [x] A cluster upgraded from OpenSearch 2.x serves its old index natively.
      `scripts/verify-opensearch-upgrade.sh`, run twice on 2026-09-30 (Docker
      29.3.1; 5 min 14 s, and 3 min 42 s with T8.3's deferred FST conversion in
      the plugin's library, same results): an OpenSearch 2.19.6 node (Lucene 9.12.3) indexes
      `single` (1 shard) and `multi` (3 shards) -- `verify_opensearch.py`'s
      text, keyword, numeric, date and multi-valued fields with deletes and
      updates, plus a `knn_vector` field (the k-NN plugin's, `index.knn` off)
      -- `nest` (nested documents) and `knn` (`index.knn: true`, a Lucene HNSW
      field), 22,756 documents in all, snapshots them to an fs repository and
      flushes. The node stops; OpenSearch 3.8.0 with the plugin opens its data
      directory in place, and a stock 3.8.0 node (same image, no plugin) opens
      a copy. Every index is green on both, every segment still `9.12.3`, and
      `verify_opensearch.py`'s 198-request matrix (aggregations, sorts, paging, highlighting,
      fallback rows) on `single`/`multi`, its
      19 nested-sort rows on `nest`, and vector requests (`knn_score`
      scripts, `knn` queries) give the plugin node the stock node's hits,
      scores (1e-5), totals and aggregations, with the plugin's counters
      showing every native row native on every shard: 723 shard queries over
      the 9.12 segments. The same over the 2.19 snapshot restored into both
      nodes (723), and after `_forcemerge?max_num_segments=1` on both nodes
      rewrote every segment as `10.5.0` (707; after the merge the nested sorts
      may fall back as `sort_nested`, by design, once no deletions remain).
      3,879 checks, 0 failures, `native_errors` 0. One request the stock node
      itself cannot answer after the merge (`nested avg desc`, an
      `unsupported_operation_exception` from OpenSearch) is reported, not
      compared. Seen to fail: without `Lucene912` in
      `SUPPORTED_POSTINGS_FORMATS` the run reports 760 failures (0 shard
      queries native before the merge). **Not native, by design, and outside
      M8:** the `knn` index -- the k-NN plugin's `index.knn: true` segments
      name its own codec (`KNN9120Codec`), which the native reader does not
      open (`native_open_failed`, once per reader, answers still equal). The
      3.8.0 nodes keep the bundled k-NN plugin (the mapping needs it) and drop
      the other bundled plugins, as the 2.19 node does. In CI since the M8
      close-out (`ci.yml`'s `opensearch-upgrade` job).
- [x] Reading an old format is no slower than Lucene reading it. Measured
      on the 1M-document segments `GenCorpus` writes with Lucene 9.0.0 and
      9.12.2 (T8.3), 4-core host at load 5-9, against Lucene 10.5.0 +
      backward-codecs. **Opening:** `DirectoryReader::open` 105/106 us
      against Java's 391/426 us. **First lookup in each field** (the terms
      index now read in place, T8.3's close-out): every field of the segment
      together 7.8 us against Java's 7-70 us (median 37-44 us); it was
      3.4 ms while the FST was converted to a trie. **Queries:** all 87, each
      with Java's hits, top set and top score; on 9.0 none slower than Java
      in a majority of rounds (q17, 0.82x in one full pass, wins 3 of 3
      re-runs), on 9.12 likewise (q80, 0.98x in the full pass, 0.90-1.04x
      in three re-runs, above Java in two) -- q80's remaining cost is a
      format-independent bit-set allocation, noted in T8.3.

## Risks and unknowns

- **Fixture provenance.** Each version's fixtures must come from that
  version's jars; a generator that silently runs 10.5.0 against all of them
  proves nothing. `gen-fixtures.sh --check` must pin the jar per version.
- **Scope creep toward writing.** Only reading is needed; resist porting the
  writers "for symmetry".

## Exit artifacts

- `fixtures/data/bwc/<version>/` and `fixtures/data/bwc-big/<version>/` indices and their generators
- `docs/parity.md` rows for every `lucene-backward-codecs` class
- The plugin's fallback table in `feature-matrix.md` updated

## Ledger evidence

The fixtures and benchmarks behind the parity ledger's rows for this
milestone, by ledger file (current facts; the rows carry the status).

### [backward-codecs.md](../parity/backward-codecs.md): lucene-codecs -- backward codecs (M8)

Reading (and merging away) every format Lucene
9.0-10.4 wrote; Lucene 8 formats are rejected. Per-class status:
`docs/inventory/lucene-backward-codecs.tsv`.

Tests: `crates/lucene-search/tests/bwc_fixtures.rs` reads every
`fixtures/data/bwc/<version>/` index (T8.1: one per codec era, 9.0.0-10.4.0,
written by that release's jars) and reproduces every line of the
`expected.txt` Lucene 10.5.0 + backward-codecs wrote for it (infos in full;
FNV-1a digests over postings with positions/offsets/payloads, norms, doc
values, points, stored fields, term vectors, vectors; KNN hits) --
`EXPECTED_FAILURES` is empty. It also runs 34 queries through
`DirectoryReader` per version (hits, scores, totals equal to the 10.4.0
fixture's bit for bit) and this port's `CheckIndex` (vector and graph families
required). `scripts/verify-bwc-merge.sh` covers merging and upgrading.
