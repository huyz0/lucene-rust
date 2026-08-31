# fixtures

Java programs, pinned to Lucene 10.5.0 (OpenSearch's current pin — see
`gradle/libs.versions.toml` in the OpenSearch checkout), that generate byte-level
fixtures for differential testing. Rust `tests/` in each crate read `data/*.bin` and
compare decoded values against `data/*.expected` / manifests — no JVM needed at Rust
test time, only to regenerate fixtures after a Lucene version bump.

## Regenerating

**Regenerate one generator at a time.**

```sh
scripts/gen-fixtures.sh --list             # the generator names
scripts/gen-fixtures.sh --only GenNorms    # regenerate just norms_index
```

`--only` runs that generator (the `Gen` prefix is optional) and then all the
`Append*Manifest` programs, and touches nothing else: after `--only GenNorms`
the other 60-odd fixture directories are byte-for-byte what they were.

A bare `scripts/gen-fixtures.sh` **refuses to run** and prints the above. That
refusal is deliberate. A full regeneration does not refresh the fixtures, it
*replaces* them: Lucene stamps a fresh random segment ID
(`StringHelper.randomId()`) into every index it writes, so 366 of the 406
generated files change on every run, the suite stays green over evidence that is
no longer the evidence the findings were written against, and the diff is
hundreds of binary files no reviewer can read. Batch c29 triggered exactly this
and had to revert 366 tracked files by hand — and several fixture directories
are not tracked yet, so `git checkout` would not have restored those at all.

If you really do mean all of them — a Lucene version bump is the usual reason —
pass `--all`. Expect `segment-ids.txt` (below) to change; that diff is the
readable record that every index was replaced, and it belongs in the commit
message.

A run compiles every program in `src/` and runs the selected `Gen*` generators
followed by all `Append*Manifest` programs, in that order. The ordering is not
incidental: the appenders open an already-generated index read-only and append
cross-engine ground truth to its `manifest.properties` **without** regenerating
the index (regenerating would perturb the segment ID that committed bytes depend
on). Running only the generators leaves `blocktree_index/manifest.properties`
with 239 keys where the committed fixture has 468. `--only` therefore always
runs the appenders afterwards; each strips its own key prefix before
re-appending, so re-running one over an index the invocation did not touch
rewrites the same bytes.

`--out DIR` writes somewhere else entirely and needs no flag — it cannot clobber
the evidence base.

The script resolves `lucene-core`, `lucene-analysis-common` and `lucene-queries`
10.5.0 from `--jars`, then the Gradle cache, then Maven Central, so it also
works on a machine with no `~/.gradle`. `lucene-queries` is required because
`GenBlockTree` uses `org.apache.lucene.queries.spans`; `lucene-analysis-common`
because `GenAnalysis` exercises real `StandardAnalyzer`/`StopFilter`.

`GenRegexp.java` is the odd one out among the generators: it writes no index at
all. It runs real `RegExp` + `Operations.determinize` + `ByteRunAutomaton` over a
pattern/term matrix and records the accept/reject decision as two plain text
files (`regexp/terms.txt`, `regexp/cases.tsv`), which is what
`crates/lucene-codecs/tests/regexp_fixtures.rs` compares this port's hand-written
`RegExp` parser against. Lucene's regexp grammar is not PCRE, and the failure
mode a fixture has to catch there is not "fails to parse" but "parses and
quietly means something else".

`data/` is checked in so `cargo test` works without Java installed; regenerate
and re-commit whenever the pinned Lucene version changes. Note that "checked in"
is not the same as "reproducible" — see below.

### Checking the committed fixtures

```sh
scripts/gen-fixtures.sh --check
```

Verifies that `data/` is still what Lucene 10.5.0 actually produces, without
touching the tree. It cannot simply diff everything: Lucene stamps a random
segment ID (`StringHelper.randomId()`) into every index header, so 629 of the
675 generated files differ on every run **by design**. So it runs five checks:

1. generates twice, and calls a file *deterministic* only if the two runs agree;
2. asserts the 46 deterministic files match the committed bytes exactly — this
   is what catches a hand-edit;
3. compares the full file tree, so a generator that silently stops emitting a
   file is caught even where the bytes cannot be compared;
4. compares every manifest's **key set** against a full generate-then-append
   run. `blocktree_index/manifest.properties` is itself non-deterministic, so
   check 2 is blind to it — a tree regenerated with the `Gen*` programs alone
   loses 229 keys there and passes every byte comparison. This check names the
   dropped keys.
5. re-derives every index's segment ID and diffs it against the committed
   `segment-ids.txt`. This is the only check that can see an index having been
   *regenerated*: fresh bytes are indistinguishable from correct bytes, same
   generator, same Lucene, only a new `randomId()`.

`segment-ids.txt` is produced by `scripts/fixture-segment-ids.py` (which parses
the id straight out of each `.si`/`segments_N` `CodecUtil.writeIndexHeader`
prologue) and is refreshed automatically by any write-mode run into `data/`.
It matters beyond `--check`: `crates/lucene-ffi/src/segment.rs` hardcodes the
committed `blocktree_index` segment ID, so a regeneration breaks those tests,
and the baseline is where that coupling is now written down.

CI runs `--check` on every change (`.github/workflows/ci.yml`, job `fixtures`).

## Verifying the write path (reverse direction)

Every generator above is Java-writes-Rust-reads. The write path (PLAN.md Phase 5)
needs the opposite: Rust writes real bytes, and a Java program confirms real Lucene
can open and read them back. `VerifyStoredFields.java`, `VerifyFieldInfos.java`,
`VerifySegmentInfo.java`, `VerifySegmentInfos.java`, `VerifyPoints.java`,
`VerifyTermVectors.java`, `VerifyDocValues.java`,
`VerifySparseNumericDocValues.java`, `VerifyNorms.java`,
`VerifyLiveDocs.java`, `VerifyCompoundFormat.java`, `VerifyFst.java`,
`VerifyVectors.java`, `VerifyFullSegment.java`, `VerifyMergedSegment.java`,
`VerifyVectorSegment.java`, `VerifyBlockSegment.java`,
`VerifyDocValuesUpdates.java`, `VerifySortedSegment.java` (used twice -- once
over a flushed segment and once over a merged one),
`VerifyPositionsSegment.java` and `VerifyMergedMetadata.java` are these
verifiers so far:

```sh
scripts/verify-write-path.sh
```

That runs all 23 Rust `write_*_fixture` examples into a temp directory and
checks each with its verifier, resolving the Lucene jars the same way
`gen-fixtures.sh` does. Pass `--keep` to retain the generated fixtures for
inspection. CI runs it on every change (`.github/workflows/ci.yml`, job
`write-path`).

**One case reads a committed fixture rather than writing everything itself.**
`write_merged_metadata_fixture` copies `fixtures/data/merge_metadata/` (three
segments a real `IndexWriter` wrote, whose `.si` files were then rewritten
through the codec with differing `minVersion`s) into its output directory and
merges them. It has to: the two facts it checks -- `SegmentMerger`'s
`minVersion` fold and `IndexWriter.mergeMiddle`'s `hasBlocks` disjunction --
are only observable when the sources disagree with the merging writer, which
segments this port wrote itself never do. Regenerate its sources with
`scripts/gen-fixtures.sh --only GenMergeMetadata`.

`VerifyStoredFields.java` opens each `.fdt`/`.fdx`/`.fdm` triple directly through
`Lucene90StoredFieldsFormat.fieldsReader`, using a hand-built `SegmentInfo`/
`FieldInfos` rather than also requiring Rust to write `.si`/`.fnm` -- this keeps
each write-path slice scoped to exactly the one format it's verifying, the same
way the read-path fixtures below call one codec-level `open`/`document` directly
rather than going through a full `IndexReader`. The Rust example writes **two**
segments: `_0` via `stored_fields::write_best_speed` (LZ4, `Mode.BEST_SPEED`)
and `_1` via `stored_fields::write_best_compression` (DEFLATE,
`Mode.BEST_COMPRESSION`, with one field repeating a phrase ~2000 times so the
dictionary + multi-sub-block DEFLATE framing is actually exercised, not just a
single trivial unit) -- the manifest's `segments=_0,_1` key and per-segment
`<seg>.mode` attribute let one verifier loop over both. `VerifyFieldInfos.java` follows
the same pattern for `.fnm`: it opens the file directly through
`Lucene94FieldInfosFormat.read` with a hand-built `SegmentInfo` (no `.si` writer
needed), then checks every field's properties against `manifest.properties`.
`VerifySegmentInfo.java` verifies the `.si` format itself: since `.si` *is* the
`SegmentInfo` serialization, no hand-built `SegmentInfo` is needed -- it opens
each `<name>.si` written by
`crates/lucene-index/examples/write_segment_info_fixture.rs` directly through
`Lucene99SegmentInfoFormat.read` and checks version, minVersion, doc count,
compound-file flag, diagnostics, files, and attributes against that segment's
`<name>.manifest.properties`.

`VerifySegmentInfos.java` is the first verifier in this reverse direction that
does *not* touch any codec class directly: it opens the whole fixture written
by `crates/lucene-index/examples/write_segment_infos_fixture.rs` (a complete
single-segment index -- `.fdt`/`.fdx`/`.fdm` + `.fnm` + `.si` + `segments_N`)
via real, high-level `DirectoryReader.open(FSDirectory.open(path))`, then
checks doc count and stored field values through ordinary
`IndexReader`/`StoredFields` calls, the way a real application reads an
index. Succeeding here is the actual milestone this slice was building
toward: proof that a Rust-written index is openable by unmodified Lucene
application code, not just by hand-built codec-level access. The fixture's
fields are deliberately stored-only (no postings/doc values/term
vectors/points/vectors), since this port has no write path yet for any of
the other per-field formats. The same verifier, unmodified, also checks
`crates/lucene-index/examples/write_multi_segment_commit_fixture.rs`'s
output -- a *real* multi-segment commit (two independently-flushed segments,
`_0` and `_1`, described by one `segments_N`, built via
`lucene_index::segment_writer::flush_stored_only_segment` called twice) --
because `VerifySegmentInfos.java` only ever reads `manifest.properties` and
calls `DirectoryReader.open` + `StoredFields.document(docId)` across the
whole reader, with no assumption about how many segments back it. Passing
here is proof that real Lucene's `DirectoryReader` federates two
Rust-written segments into one coherent 5-doc space.

those formats -- `SegmentCoreReaders` only opens a postings `FieldsProducer`
when `FieldInfos.hasPostings()` is true, so a segment with zero indexed
fields needs none of those files to exist. See `docs/parity.md`'s
`SegmentInfos.write` row for what a fully-indexed segment would still need.

`VerifyPoints.java` verifies `points::write` (`crates/lucene-codecs/src/points.rs`),
one dimension (`LongPoint`-style), any number of leaves: it opens **two**
`.kdm`/`.kdi`/`.kdd` triples (`_0`, single-leaf; `_1`, `maxPointsInLeafNode = 8`
forcing a multi-level packed-index tree) directly through
`Lucene90PointsFormat.fieldsReader` with a hand-built `SegmentInfo`/`FieldInfos`
(no `.si`/`.fnm` writer needed, same division of labor as
`VerifyStoredFields.java`), then uses real `PointValues.intersect` with an
always-`CELL_CROSSES_QUERY` visitor (the same technique `GenPoints.java` uses
on the read side) to force a full decode of every point and diff `(docID,
value)` pairs against `manifest.properties` for both segments. Multi-dimension
points are out of scope for this writer -- see `docs/parity.md`'s
points/BKD-tree row.

`VerifyTermVectors.java` verifies `term_vectors::write_best_speed`
(`crates/lucene-codecs/src/term_vectors.rs`), scoped to positions only (no
offsets/payloads/prefix-sharing, single chunk): it opens the `.tvd`/`.tvx`/
`.tvm` triple directly through `Lucene90TermVectorsFormat.vectorsReader` with a
hand-built `SegmentInfo`/`FieldInfos`, then checks every doc's term
text/freq/positions via real `Terms`/`TermsEnum`/`PostingsEnum` against
`manifest.properties` (same technique `GenTermVectors.java` uses on the read
side). The Rust example writes **two** segments, `_0` and `_1`: `_0` is the
primary multi-field-number fixture, and `_1` is a regression case where every
field across every doc has `field_number == 0` -- a review pass before this
writer's commit caught that a chunk shaped that way previously encoded
`bits_per_field_num` as 0, which this port's own (more permissive) reader
tolerates but real Lucene's reader does not (it unconditionally indexes
`packedBulkOps[bitsPerValue - 1]`, throwing `ArrayIndexOutOfBoundsException`
on a 0-bit width) -- `_0` alone can never exercise this since it always mixes
field numbers 0 and 1. Also worth naming since it's easy to miss: the `.tvm`
meta stream's `packedIntsVersion` field must be written as `2`
(`PackedInts.VERSION_CURRENT`/`VERSION_MONOTONIC_WITHOUT_ZIGZAG`) -- this
port's own reader never validates that field, but real Lucene's
`BlockPackedReaderIterator` does, so a wrong or placeholder value there would
pass every purely-Rust round-trip test while still failing to open in real
Lucene.

`VerifyDocValues.java` verifies all five of this port's dense, single-field
doc-values writers in `crates/lucene-codecs/src/doc_values.rs`:
`write_single_dense_numeric_field`, `write_single_dense_binary_field`,
`write_single_dense_sorted_numeric_field`, `write_single_dense_sorted_field`,
and `write_single_dense_sorted_set_field`. Each is scoped to exactly one
shape: dense (every doc has a value, or for the multi-valued types, at least
one), plain delta-compressed encoding for the numeric-shaped parts (no
table/GCD compression, no sparse `IndexedDISI`, no varying-bpv blocks). It
opens each `.dvm`/`.dvd`/`.dvs` triple directly through real
`Lucene90DocValuesFormat.fieldsProducer` with a hand-built
`SegmentInfo`/`FieldInfos` (same division of labor as `VerifyPoints.java`),
reading a `<segment>.type` manifest key to pick the matching
production-facing read API -- `NumericDocValues`, `BinaryDocValues`,
`SortedNumericDocValues`, `SortedDocValues`, or `SortedSetDocValues`, never a
codec-internal decode -- and diffs every doc's value(s) against
`manifest.properties`. `.dvs` (the per-field doc-values skip index file) is
always header+footer only in this slice's scope, but still must exist and
pass its own header/footer check: `Lucene90DocValuesProducer`'s constructor
unconditionally opens `.dvs` once the format version is `>=
VERSION_SKIPPER_SEPARATE_FILE`, which this port's `VERSION_CURRENT` always
is, regardless of whether any field actually has a skip index. Sparse
fields, GCD/table compression, the varying-bits-per-value block split,
per-field doc-values skip indexes, and multiple fields in one triple are all
out of scope for these writers -- see `docs/parity.md`'s doc-values row.

The Rust example writes **ten** segments. NUMERIC: `_0` (mixed
small/large/negative values, `min <= 0` throughout), `_1` (every value has
`min > 0` where `unsignedBitsRequired(max) == unsignedBitsRequired(max-min)`,
forcing the min-shift-drop optimization -- `_0` never has `min > 0`, so it
can't reach this branch), and `_2` (all-equal values, forcing the
`bitsPerValue == 0` constant encoding). BINARY: `_3` (every value the same
length, direct `ordinal * length` addressing) and `_4` (varying lengths
including an empty value, the `DirectMonotonicReader` address-block path).
SORTED_NUMERIC: `_5` (every doc exactly one value, the case where real
Lucene collapses the address array away entirely since `numDocsWithField ==
numeric.numValues`) and `_6` (1-3 values per doc, forcing the real
address-range array). SORTED: `_7` (five docs with repeated values over a
3-term dictionary, exercising the ordinal decode and the terms-dict decode
together). SORTED_SET: `_8` (every doc exactly one distinct value, the
`multiValued = false` collapse to the same shape SORTED uses) and `_9` (1-2
distinct values per doc sharing a dictionary, including a doc whose raw
values repeat and dedup down to one ordinal). All ten verify against real
Lucene. **Scope note**: `_7`'s and `_8`/`_9`'s dictionaries are deliberately
small (3 terms), so this fixture only exercises `write_terms_dict`'s
single-64-term-block path -- it does not force real Lucene to open a
multi-LZ4-block/multi-1024-ordinal-reverse-index-sample dictionary this
port wrote; that boundary is covered only by unit tests against this
port's own reader (see `docs/parity.md`'s doc-values row).

`VerifySparseNumericDocValues.java` verifies
`write_single_sparse_numeric_field` (`crates/lucene-codecs/src/doc_values.rs`),
which had previously only been checked against this port's own reader
(`write_single_sparse_numeric_field_round_trips_through_own_reader`'s unit
test in that file) -- never against real Lucene. It opens the
`.dvm`/`.dvd`/`.dvs` triple written by
`crates/lucene-codecs/examples/write_sparse_numeric_doc_values_fixture.rs`
directly through real `Lucene90DocValuesFormat.fieldsProducer`, with a
hand-built `SegmentInfo`/`FieldInfos` (same division of labor as
`VerifyDocValues.java`). Unlike the dense verifier, it does not just walk
present docs via `nextDoc()` -- it calls real `NumericDocValues.advanceExact`
for every doc id from `0` to `max_doc - 1`, confirming docs with a value
return `true` and the correct value via `longValue()`, and docs without a
value correctly report `false`, which is the property that actually matters
for a sparse field's `IndexedDISI`-backed presence check. The Rust example
writes two segments: `_0` (20 docs, missing values interspersed throughout --
not just trailing) and `_1` (200,000 docs, 1 of every 3 present, forcing
`IndexedDISI`'s DENSE-bitset per-block shape, the same shape the pure-Rust
unit test already covers, now also checked against real Lucene). Both passed
on the first run.

`VerifyNorms.java` verifies `norms::write_single_dense_field`
(`crates/lucene-codecs/src/norms.rs`), scoped to exactly one shape: a single
norms field, dense (every doc has a value), at most 1 byte per doc (`bytesPerNorm
0` for the all-equal constant case, or `1` otherwise -- 2/4/8-byte widths, sparse
`IndexedDISI` fields, and multiple fields in one `.nvm`/`.nvd` pair are all out of
scope, see `docs/parity.md`'s norms row). It opens the pair directly through real
`Lucene90NormsFormat.normsProducer` with a hand-built `SegmentInfo`/`FieldInfos`
(same division of labor as `VerifyDocValues.java`), then iterates the field via
real `NumericDocValues.nextDoc`/`longValue` (the same API `NormsProducer.getNorms`
returns) and diffs every doc's value against `manifest.properties`. The Rust
example writes two segments: `_0` (varying small signed values, the real
`bytesPerNorm == 1` path) and `_1` (all-equal values, the `bytesPerNorm == 0`
constant path) -- following the doc-values write-side review's finding directly,
both branches are verified against real Lucene from the start rather than only
this port's own reader.

`VerifyCompoundFormat.java` verifies `compound_format::write`
(`crates/lucene-codecs/src/compound_format.rs`), which packs already-written
sub-files (each a complete standalone codec file: its own header/footer)
into a `.cfs`/`.cfe` pair. The Rust example packs four distinct sub-files --
a `.fnm` (`field_infos::write`) and a `.fdt`/`.fdx`/`.fdm` triple
(`stored_fields::write_best_speed`) -- so the entries table's offset/length
bookkeeping and the smallest-first packing order both get exercised, not
just a single-file passthrough. The Java verifier opens the pair through
real `Lucene90CompoundFormat.getCompoundReader` with a hand-built
`SegmentInfo`, confirms the sub-file list and lengths match, then goes a
step further than the other write-path verifiers: it re-decodes the packed
`.fnm` through real `Lucene94FieldInfosFormat` and the packed
`.fdt`/`.fdx`/`.fdm` through real `Lucene90StoredFieldsFormat`, both reading
*through* the compound reader rather than the raw sub-file bytes directly --
this is what would catch a byte-offset bug that still left the entries
table looking correct. See `docs/parity.md`'s compound-format row for what
Java's writer does beyond a bare concatenation (smallest-first ordering,
64-byte alignment, per-sub-file header/footer verification) and why this
port's simpler "validate then copy verbatim" approach is byte-identical to
it.

`VerifyFst.java` verifies `fst::build_fst`/`fst::write_fst`
(`crates/lucene-codecs/src/fst.rs`), the from-scratch, simplified FST
construction path that (unlike everything else in this list) has no real
`FSTCompiler` counterpart to fall back on for the write side. The Rust example
(`write_fst_fixture.rs`) builds the same 7-key set `GenFst.java` uses
(`app`/`apple`/`application`, `banana`/`band`/`bandana`, `z`) via `build_fst`
and writes the bytes with `write_fst`, and the Java verifier opens the result
through real `FST.read(Path, ByteSequenceOutputs)` and looks up all 7 present
and 8 deliberately-absent keys via real `Util.get(FST, BytesRef)`. A second,
larger fixture (`large/`, 200 keys forcing multi-byte `vlong` node-address
targets -- the same shape `build_fst_many_keys_forces_multi_byte_vlong_targets`
self-round-trips in `fst.rs`'s own unit tests, never previously checked
against a real Lucene reader) is written and verified the same way. This is
the reverse of `GenFst.java`/`fst_fixtures.rs` (which is Java-writes/Rust-reads):
here Rust writes and real Lucene reads. Both fixtures passed on the first run
(`VerifyFst OK (<dir>): 7 present keys resolved, 8 absent keys rejected` and
`VerifyFst OK (large): 200 present keys resolved, 3 absent keys rejected`), a
genuine, non-obvious result worth stating plainly: `build_fst`'s simplified
construction skips real `FSTCompiler`'s suffix sharing/minimization, output
pushing, and fixed-length-arc node compaction, so it was not a given that real
Lucene's reader -- written against `FSTCompiler`'s actual output shapes --
would accept a non-minimal, always-list-encoded, always-explicit-`vlong`-
target FST without complaint. It does: nothing in `FST.java`'s read path
(`readArc`, `findTargetArc`, `seekToNextNode`) assumes minimality or
fixed-length arcs are present, only that whichever encoding a node actually
uses is self-consistent, so a structurally simpler but format-valid FST is
read identically to one `FSTCompiler` would have produced. See
`docs/parity.md`'s FST row for the full detail.

`VerifyVectors.java` verifies `vectors::write_flat_vectors` and
`hnsw_vectors::write_hnsw_vectors` (`crates/lucene-codecs/src/vectors.rs`,
`hnsw.rs`, `hnsw_vectors.rs`) -- the `.vec`/`.vemf` flat store *and* the
`.vem`/`.vex` HNSW graph. The Rust example (`write_vectors_fixture.rs`) writes
four fields covering the dense, sparse, BYTE and no-graph cases, and Lucene
opens all four through its own `Lucene99HnswVectorsFormat` with a hand-built
`SegmentInfo`/`FieldInfos` (same division of labour as `VerifyPoints.java`).

Four things are checked, because a vector segment can be wrong in four
independent ways: every ordinal's components (an order-sensitive hash over raw
float bits, so float summation order cannot hide a difference); every
ordinal's document id (where a mis-written `IndexedDISI` bitset or
`DirectMonotonicWriter` block shows up); the graph's level count, entry node,
max conn and a per-level arc hash (which is what proves the `.vex` node offsets
and group-varint neighbour deltas are what *Lucene* expects, not merely what
this port's own reader expects); and finally a real `TopKnnCollector` search
over the Rust-built graph, whose recall against the exact top-k must clear a
floor. That last one is the only check a graph which decodes cleanly but is
*built* wrong will fail -- it passes all three structural checks. Observed:
recall@10 of 0.91-1.00 across the four fields.

`VerifyDocValuesUpdates.java` verifies the doc-values **field update** path --
`IndexWriter.updateNumericDocValue`/`updateBinaryDocValue`, written by
`crates/lucene-index/src/field_updates.rs`. Unlike the codec-level verifiers
above it opens two whole Rust-written indices through `DirectoryReader`, because
what has to be right is not one file but four things agreeing: the
generation-suffixed `.dvm`/`.dvd`/`.dvs` (the updated field's whole rewritten
column), the `FieldInfos` generation recording that field's
`FieldInfo.docValuesGen`, and `segments_N`'s `docValuesGen` +
`dvUpdatesFiles`. Every way of getting one of them wrong -- a suffix that
disagrees between the file name and the file's own index header, a
`fieldInfosGen` written but not recorded, a `dvUpdatesFiles` entry that
accumulates instead of replacing, a merged column that dropped the documents
the update did not touch -- reads back fine through this port's own reader.

So the verifier reads *every* document's value back and compares it to the
value the last update round set (the numeric index is updated three times, so a
generation is itself read back as the base of the next one), asserts the
superseded generations were reclaimed from the directory, and runs `CheckIndex`
at `MIN_LEVEL_FOR_SLOW_CHECKS`. Before this existed, an index carrying a
doc-values update used a delta format of this port's own invention and could
not be opened by real Lucene at all -- and no verifier case wrote one, so
nothing said so.

`VerifySortedSegment.java` verifies the **index-sorted flush** --
`IndexWriter::set_index_sort`, written by
`crates/lucene-index/src/index_writer.rs`. An index sort is unlike every other
property checked here, because violating it leaves every file *valid*: correct
checksums, in-range doc ids, a decodable term dictionary. Only the association
between the files is wrong. So the verifier re-derives the expected
permutation itself, with its own comparator, from the fixture's generator
functions, and then checks -- per doc id -- the stored `id`, both NUMERIC
doc-values columns (including `rank`'s *absence* on the documents that have
none), the postings term unique to that document, the norm, and every
component of the vector, plus `LeafMetaData.sort()` tier for tier and a real
`KnnFloatVectorQuery` over the Rust-built graph. Then `CheckIndex` at
`MIN_LEVEL_FOR_SLOW_CHECKS`, which runs Lucene's own `testSort`.

Two negative controls were run by hand while it was written, and both are the
reason it asserts on the association rather than on `CheckIndex` alone: a
flush that permutes the documents but *not* the vectors attaches every vector
to the wrong document and is **`CheckIndex`-clean**, and a missing-value
comparator that disagrees with the sort it wrote fails Lucene's `testSort`
outright.

## Generators

- `GenPrimitives.java` — vint/vlong/zlong/group-varint wire encodings.
- `GenCodecUtil.java` — codec header/index-header/footer framing (magic, version,
  object id, suffix, CRC-32 footer), plus a corrupted-checksum fixture.
- `GenSegmentInfo.java` — real `.si` files (`Lucene99SegmentInfoFormat`) written via
  the actual codec, with and without a `minVersion`, round-tripped through Java
  Lucene before being shipped as a fixture.
- `GenSegmentInfos.java` — a real two-commit `IndexWriter` session (`segments_index/`
  subdirectory: full index dir + `segments_2.raw` copy + manifest), exercising real
  segment names/generations/counters/user-data rather than hand-built bytes.
- `GenMergeMetadata.java` — three segments (`merge_metadata/` subdirectory)
  written by a real `IndexWriter` under `NoMergePolicy`, then given **differing
  `minVersion`s** (10.2.0, 10.0.0, 10.1.0 — oldest in the middle) by rewriting
  each `.si` through `codec.segmentInfoFormat().write` with everything else
  carried across unchanged. `hasBlocks` is not synthesised: segment `_1` is
  built with `addDocuments`, which is what makes Lucene set it. This is the
  only fixture whose *sources* exist so a **merge** can be checked: it feeds
  `write_merged_metadata_fixture` (see "Verifying the write path" above), which
  merges the three through this port and lets real Lucene read the merged
  `minVersion`/`hasBlocks` back off `LeafMetaData`. A merge of segments this
  port wrote itself cannot see either field, because they would agree with the
  merging writer's own version by construction.
- `GenLiveDocs.java` — a real single-segment `IndexWriter` session with 2 of 5 docs
  deleted by term after the first commit (`live_docs_index/` subdirectory:
  `NoMergePolicy` keeps the segment from being merged away, so the fixture's `.liv`
  file is a real post-deletion commit, not hand-built bits).
- `GenMultiSegmentScoring.java` — the only genuinely **two-segment** scoring
  fixture in this tree (`multi_segment_scoring_index/`): `NoMergePolicy` plus a
  `commit()` between two deliberately lopsided batches, so segment 0 holds four
  1–3-term documents and segment 1 four 40-term ones. Their own `avgdl` values
  are 1.75 and 40.0 against a reader-wide 20.875, and `fox` has `docFreq`
  1-of-4 in one leaf and 3-of-4 in the other. That spread is the point: real
  Lucene's `IndexSearcher` computes `TermStats`/`FieldStats` once for the whole
  reader, so a port that derives either per leaf scores the same document
  differently — and every *other* scoring fixture here is one segment, where the
  two are the same number by construction and the divergence is invisible. The
  manifest records real `IndexSearcher` `TopDocs` as `Float.floatToIntBits` in
  **global** doc-id space, plus each leaf's counters and the reader-wide sums, so
  the Rust side can assert both the scores and the statistics they came from.
  Consumed by `crates/lucene-search/tests/multi_segment_scoring_fixtures.rs`
  (bit-for-bit, with a negative control asserting per-leaf `avgdl` does *not*
  reproduce them). The index bytes carry a random segment id and are therefore
  non-deterministic, but `manifest.properties` is byte-stable, so `--check`
  compares it.
- `GenMergePolicy.java` — cross-engine ground truth for the `TieredMergePolicy`
  port (`merge_policy/merge_policy.manifest.properties`). Emits **no index**: a
  merge policy is a pure function of segment *statistics*, so the fixture is a
  manifest of 33 `(config, segments, currently-merging set) -> chosen merge
  groups` scenarios decided by running real `findMerges`,
  `findForcedMerges(n)` and `findForcedDeletesMerges` over hand-built
  `SegmentInfos`. Sizes are real (one file per segment of exactly the requested
  byte length, in a `ByteBuffersDirectory`, so `SegmentCommitInfo.sizeInBytes()`
  and `MergePolicy.size()`'s deletion pro-rating are genuinely exercised) and
  every size converts to `setMaxMergedSegmentMB`/`setFloorSegmentMB`'s MB unit
  by dividing by 2^20, which is exact in binary floating point. Deterministic
  (fixed segment ids, no `IndexWriter`), so `--check` compares it byte for byte.
  Consumed by `crates/lucene-index/tests/merge_policy_fixtures.rs`, which
  asserts the identical grouping **in the identical order**.
- `GenFieldInfos.java` — a real two-doc `IndexWriter` session (`field_infos_index/`
  subdirectory) with fields of every notable shape (plain indexed, term vectors,
  numeric/sorted doc values, a point field, a KNN vector field) plus a
  soft-deletes field introduced via a genuine `updateDocValues` call after the
  first commit — this is the mechanism that makes the field live in a
  generation-suffixed `.fnm` file rather than the segment's original one, and
  the fixture exercises reading that generation correctly
  (`SegmentCommitInfo.getFieldInfosGen()` → base-36 suffix).
- `GenNorms.java` — a real single-segment `IndexWriter` session (`norms_index/`
  subdirectory) with a dense norms field ("body", every doc, deliberately
  varying token counts so values aren't all identical) and a sparse one
  ("sparse_body", present on only 3 of 5 docs — Lucene only picks the
  `IndexedDISI`-backed sparse encoding when a field is missing from some docs
  entirely, so that's what actually triggers it). Expected values come from
  reading them back through Lucene's own `NormsProducer`, not our own
  arithmetic on token counts.
- `GenDocValues.java` — a real single-segment `IndexWriter` session
  (`doc_values_index/` subdirectory) with numeric fields ("varying":
  arbitrary signed values, plain delta compression; "gcd": values sharing a
  large common divisor, GCD compression; "sparse": present on only 3 of 5
  docs, `IndexedDISI` path — same mechanism as `GenNorms.java`'s sparse
  field) and binary fields ("bin_fixed": every value the same length,
  direct addressing; "bin_var": varying lengths, `DirectMonotonicReader`
  address block; "bin_sparse": varying lengths + `IndexedDISI` together).
  Also dumps the segment's `.fnm` since parsing `.dvm` requires the field
  infos to check each field's doc-values-skip-index configuration.
  Expected values come from reading them back through Lucene's own
  `Lucene90DocValuesProducer.getNumeric`/`getBinary`, not our own
  arithmetic.
- `GenDocValuesUpdates.java` — a real single-segment `IndexWriter` session
  (`doc_values_updates_index/` subdirectory) whose doc-values are then
  **updated in place** across three rounds
  (`updateNumericDocValue`/`updateBinaryDocValue`). Lucene answers a
  doc-values update by rewriting the updated field's *whole column* into a
  new generation of ordinary `Lucene90DocValuesFormat` files
  (`_0_<base36 gen>_Lucene90_0.dvm/.dvd/.dvs`), plus a `FieldInfos`
  generation (`_0_<base36 gen>.fnm`) recording that field's
  `FieldInfo.docValuesGen`, plus `docValuesGen` and a per-field
  `dvUpdatesFiles` map in `segments_N`. Three doc-values fields, in the three
  states a reader has to tell apart: `val` (NUMERIC, updated twice — so an
  earlier generation is itself read back as the base of a later one, and the
  superseded generation's files are gone), `tag` (BINARY, updated once, at a
  *different* generation number than `val`), and `keep` (NUMERIC, never
  updated — still on the base column at generation -1, the case a reader gets
  wrong by resolving every field to `SegmentCommitInfo.docValuesGen`).
  Expected values are every document's value as Lucene's own
  `DirectoryReader` reads it back, so the Rust test asserts against Lucene's
  answers rather than a second derivation of the format.
- `GenSortedIndex.java` — a real **index-sorted** `IndexWriter` session
  (`sorted_index/` subdirectory) configured with
  `IndexWriterConfig.setIndexSort(new Sort(rank DESC missingValue=Long.MAX_VALUE,
  tie ASC missingValue=Long.MIN_VALUE))`, two commits force-merged into one
  segment. Two things it pins that nothing else does: the `.si`'s
  `numSortFields`/`SortFieldProvider` block for a sort a *real* `IndexWriter`
  chose (rather than one handed to `SegmentInfo` directly, which is what
  `GenSegmentInfo.genSorted` covers), and — more importantly — **what the sort
  means**. A missing value is an ordinary sentinel inside Lucene's comparator,
  so `reverse` applies to it too and the six documents with no `rank` are
  Lucene's *first* six under a missing-**last** descending sort. The manifest
  records the physical order Lucene produced plus both doc-values columns as a
  reader sees them, so `crates/lucene-index/tests/index_sort_fixtures.rs`
  checks this port's own comparator against Lucene's behaviour rather than
  against a reading of its source.
- `GenCompoundFormat.java` — a real single-segment `IndexWriter` session
  (`compound_index/` subdirectory) with `useCompoundFile=true` forced on the
  writer config, so the segment's sub-files (`.fnm`, `.fdt`/`.fdx`/`.fdm`,
  `.dvd`/`.dvm`/`.dvs`, term dictionary files) get packed into one `.cfs`/
  `.cfe` pair instead of written loose. The manifest's sub-file list and
  lengths come from reading the pair back through Lucene's own
  `Lucene90CompoundFormat.getCompoundReader`, not re-derived from the raw
  bytes.
- `GenStoredFields.java` — a real single-segment `IndexWriter` session
  (`stored_fields_index/` subdirectory), `Mode.BEST_SPEED` (the default),
  with 6 documents each carrying one field of every stored-field type
  (string, binary, int, long, float, double) and a string field whose
  length grows per doc, so the chunk uses the bulk (`StoredFieldsInts`)
  multi-doc framing rather than the single-doc shortcut. Expected values
  come from a custom `StoredFieldVisitor` reading them back through
  Lucene's own `Lucene90CompressingStoredFieldsReader`, not our own
  arithmetic.
- `GenStoredFieldsBestCompression.java` — the same document shape as
  `GenStoredFields.java`, but forced onto `Lucene104Codec.Mode.
  BEST_COMPRESSION` (DEFLATE with a preset dictionary, `.fdt` data codec
  `Lucene90StoredFieldsHighData`) with one field repeating a long sentence
  so the DEFLATE dictionary + multi-sub-block decode path actually gets
  exercised, not just a trivial single unit. This fixture caught a real
  bug: DEFLATE's per-unit compressed-length vint sits immediately before
  its own compressed bytes, unlike LZ4's, which are all batched up front --
  getting that backwards (by over-generalizing from the already-working
  LZ4 code) produced a `MalformedVarint` against these real bytes, caught
  and fixed before commit.
- `GenSortedDocValues.java` — a real single-segment `IndexWriter` session
  (`sorted_dv_index/` subdirectory) with a single-valued SORTED field over
  5 docs with repeated values ("banana", "apple", "cherry", "apple",
  "banana"), so the terms dictionary has 3 unique alphabetically-ordered
  terms and the ordinal array has repeats — exercising the terms
  dictionary decode and the ordinal (NUMERIC-shaped) decode together.
  Expected ordinals and terms come from reading them back through
  Lucene's own `SortedDocValues.ordValue`/`lookupOrd`, not our own
  arithmetic.
- `GenMultiValuedDocValues.java` — a real single-segment `IndexWriter`
  session (`multi_valued_dv_index/` subdirectory) with a SORTED_NUMERIC
  field ("nums", 0-3 values/doc) and a SORTED_SET field ("tags", 0-2
  values/doc sharing a 3-term dictionary) across 5 docs, so some docs have
  zero values (the `IndexedDISI`-sparse path, since not every doc has the
  field at all) and others have more than one (the `DirectMonotonicReader`
  address-range path) — both exercised together. Expected values/ordinals
  come from reading them back through Lucene's own
  `SortedNumericDocValues`/`SortedSetDocValues`, not our own arithmetic.
- `GenTermVectors.java` — a real single-segment `IndexWriter` session
  (`term_vectors_index/` subdirectory) using a hand-built `TokenStream`
  (not a real analyzer) so every term's position, offset, and payload is
  known exactly: doc 0 has one field with a repeated term ("cat" twice,
  "car" once) and payloads on some occurrences but not others, exercising
  same-term multi-occurrence delta chains; doc 1 has two fields ("text",
  "title"), exercising the distinct-field-numbers array and multi-field
  bookkeeping; doc 2 has no term-vector field at all. Expected
  positions/offsets/payloads come from reading the segment back through
  Lucene's own `TermVectorsReader`/`TermsEnum`/`PostingsEnum`, not our own
  arithmetic. This fixture is what caught a real decode bug in the first
  version of the port: the LZ4 unit's term-suffix and payload bytes are
  interleaved **per document**, not laid out as two global regions — a
  hand-built single-doc unit test couldn't have caught it since a single
  document's own bytes are contiguous either way.
- `GenPoints.java` — a real single-segment `IndexWriter` session
  (`points_index/` subdirectory) with 2000 docs, a single-dimension
  `LongPoint` field ("val") on two-thirds of them (every third doc skips
  it), spread across a wide positive/negative range — enough points to
  force several leaves past the default 512-point-per-leaf threshold, and
  gaps so a leaf's doc ids aren't trivially continuous. Expected
  (docID, value) pairs come from `PointValues.intersect` with a visitor
  whose `compare` always returns `CELL_CROSSES_QUERY`, forcing Lucene's
  own reader to fully decode every point rather than taking a
  bounding-box shortcut, not our own arithmetic.

  Every doc also gets a second, 2-dimension `IntPoint` field ("multi"):
  dim0 is `i` run through an odd multiplicative hash (bijective mod 2^32,
  spreading doc ids across the full 32-bit range) and dim1 is `i % 4`
  (only 4 distinct values). That shape is deliberate: it's what makes
  real `BKDWriter`'s per-leaf `sortedDim` selection (lowest in-leaf
  cardinality wins) reliably pick dim1, not dim0, so every leaf is
  written with a nonzero `compressedDim` — exercising the leaf decoder's
  `compressed_dim * bytes_per_dim + ...` offset math at a real
  dimension index instead of only ever at 0 (which is all the
  single-dimension "val" field can produce). A naive sequential dim0
  (just `i`) doesn't work here: `BKDWriter`'s recursive range-narrowing
  squeezes such a narrow-range dimension's in-leaf cardinality down to
  1-2 distinct bytes, which ties or beats dim1's fixed cardinality and
  keeps `compressedDim` at 0 in every leaf — confirmed by instrumenting
  `read_leaf_block` with a temporary debug print, not just by reasoning
  about it. The generator mechanically double-checks this at generation
  time via `CompressedDimSpy.java` (see that file), which independently
  re-reads the raw per-leaf `compressedDim` byte straight out of the
  written `.kdd`/`.kdi` bytes without going through this port's own
  decoder, and fails the build if no leaf ever has `compressedDim >= 1`;
  the observed value is also recorded in the `multi_leaf_compressed_dims`
  manifest key so the Rust differential test can assert on it directly.
- `GenVectors.java` — a real single-segment `IndexWriter` session
  (`vectors_index/` subdirectory) with 4000 documents and **five**
  `Lucene99HnswVectorsFormat` fields, chosen so that every branch a
  `Lucene99FlatVectorsReader`/`Lucene99HnswVectorsReader` has is on the
  disk somewhere: a *dense* FLOAT32/EUCLIDEAN field (every document has a
  value, so `OrdToDocDISIReaderConfiguration` writes the `-1` marker and
  no ord↔doc structures at all), two *sparse* FLOAT32 fields
  (COSINE and MAXIMUM_INNER_PRODUCT — an `IndexedDISI` bitset plus a
  `DirectMonotonicWriter` mapping appended to `.vec`), a BYTE/DOT_PRODUCT
  field (4-byte alignment instead of 64, and Java's *different*
  `dotProductScore` transform), and a 5-vector field. Two of the five are
  small enough that Lucene skips graph construction entirely
  (`numLevels == 0`, a zero-length `.vex` region) — the branch a reader is
  most likely to get wrong precisely because no ordinary fixture reaches
  it.

  The manifest records three independent layers of ground truth: per-field
  metadata and spot ordinals as raw float bits; per graph level, the node
  count, node ids, eight neighbour samples and an **order-sensitive hash
  over every node's neighbour list** (so a mis-decoded `.vex` node offset
  or a dropped group-varint group cannot pass); and, for twenty queries per
  field, both Lucene's own HNSW top-10 (through a real
  `KnnFloatVectorQuery`/`KnnByteVectorQuery`, i.e. the actual
  `Lucene99HnswVectorsReader.search`) and the exact brute-force top-10 over
  the same vectors. The first is what a faithful port must reproduce
  doc-for-doc; the second is the denominator of the recall figure both
  engines are measured on in `docs/sweep/m2/c5-vectors.md`.

  It also records the first ten `new SplittableRandom(42).nextDouble()`
  draws. HNSW level assignment is `(int)(-ln(U) * ml)` over exactly that
  stream, so a port whose generator drifts builds a differently-shaped
  graph for a reason that would otherwise look like an algorithmic
  difference — and with the stream matched, this port picks the same entry
  node as Lucene does (171 here, 46601 at 50k x 128).
- `GenVectorsMulti.java` — the **multi-segment** counterpart
  (`vectors_multi_index/` subdirectory): the same four vector field shapes
  over 4000 documents split across four deliberately *unequal* segments
  (2000/1000/960/40), plus two `StringField`s used as KNN filters. It exists
  because three things about `AbstractKnnVectorQuery.rewrite` are invisible
  to a single-segment fixture:

  1. **Per-leaf `k` is pro-rata, not `k`.** `TopKnnCollectorManager` is
     optimistic, so each leaf is searched with a collector of
     `perLeafTopKCalculation(k, leafMaxDoc/indexMaxDoc)` — 24, 30, 23 and 5
     for these four leaves at `k = 10`. Unequal segments put four different
     collector sizes into one query.
  2. **The optimistic re-entry pass.** A leaf whose worst phase-1 hit is
     still at or above the merged top-`k`'s worst is searched again with a
     full-`k` collector. That needs a leaf whose `perLeafTopK` is *below*
     `k` and whose vectors are genuinely competitive, so the 40-document
     segment's `dense_f32` vectors are a tight cluster near the origin and
     every fifth query target is pulled toward it. Five of the twenty dense
     queries then return more than five hits from that leaf — which only a
     second pass can produce.
  3. **Filtered KNN.** A *selective* filter (`bucket:b0`, 20 documents in
     the whole index, fewer per leaf than `perLeafTopK`, so every leaf takes
     `exactSearch`) and a *permissive* one (`group:g0`, a quarter of the
     index, so the graph is walked with `acceptOrds` and
     `visitedLimit = cost + 1`). The accepted **local** doc ids are recorded
     per leaf straight out of Lucene's own postings, so the Rust side is
     checked on the KNN policy rather than on its own term-query resolution.

  The 40-document segment carries no HNSW graph at all, so the fan-out also
  has to merge one exact leaf with three approximate ones. Consumed by
  `crates/lucene-search/tests/vector_query_fixtures.rs`.
- `GenVectorsSeeded.java` — the one thing `GenVectorsMulti`'s index cannot
  reach (`vectors_seeded_index/` subdirectory): the optimistic re-entry pass
  firing on a leaf that **has a graph**, so that
  `ReentrantKnnCollectorManager`'s `KnnSearchStrategy.Seeded` and therefore
  `SeededHnswGraphSearcher` are actually exercised. Two constraints have to
  hold at once and 4000 documents cannot satisfy both: a leaf is only
  re-enterable when `perLeafTopK < k` (`k*p + 16*sqrt(k*p*(1-p)) < k`, i.e.
  `p < 0.039` at `k = 10`, under ~156 documents of a 4000-document index),
  while `shouldCreateGraph` needs about 660 vectors. They are compatible at a
  larger `k`, so this index is **1400/700/700/40** and is queried at
  `k = 100`, where the four `perLeafTopK` values are 129, 93, 93 and 20 — and
  the second 700-document segment holds a tight cluster every query target
  sits next to, which is what makes its 93 phase-1 hits dominate the merged
  top 100 and its re-entry condition true. `k = 10` over the same index is
  recorded as the no-re-entry control, and the per-leaf `perLeafTopK` values
  are recorded so the Rust test can assert the fixture still has the shape it
  was built for. `GenVectorsMulti`'s index *does* reach the re-entry pass, but
  only on its 40-document segment, which is below `shouldCreateGraph` and so
  takes the exhaustive branch — where Java ignores the search strategy
  entirely. Vector generation is `GenVectorsMulti`'s, reused rather than
  copied.
- `GenVectorsFiltered.java` — a **single-segment** index that also carries a
  term dictionary (`vectors_filter_index/` subdirectory): 1200 documents with
  a FLOAT32 and a BYTE vector field plus `bucket`/`group` `StringField`s. It
  exists for the C ABI, which opens one segment's vector files per handle and
  therefore needs ground truth from a *one-leaf* index — where
  `leafProportion == 1` makes `perLeafTopK == k` and no re-entry pass runs.
  Running the same query against one leaf of the four-leaf index is a
  different search (that leaf's collector is pro-rata sized), so
  `vectors_multi_index`'s recorded results are not usable as single-segment
  ground truth. Both of Java's filtered branches are reached: `bucket:b0`
  accepts 6 documents against `k = 10` (`cost <= perLeafTopK`, so
  `exactSearch`) and `group:g0` accepts a quarter of the index (the graph walk
  with `acceptOrds` and `visitedLimit = cost + 1`). The accepted local doc ids
  are recorded straight out of Lucene's own postings, so a test can either
  supply them directly or — as `crates/lucene-ffi/src/vectors.rs` does —
  resolve the same term through this port's block-tree reader and check it
  arrives at the same set. Consumed by
  `crates/lucene-search/tests/vector_query_fixtures.rs` and
  `crates/lucene-ffi/src/vectors.rs`.
- `GenFst.java` — a real `FST<BytesRef>` (`fst/` subdirectory) built via
  real `FSTCompiler` with `ByteSequenceOutputs` (the output type real
  Lucene's term index FST uses) and `allowFixedLengthArcs(false)` (so it
  never emits the fixed-length-arc node encodings -- see
  `GenFstBinarySearch.java` below for those). 7 keys sharing prefixes/suffixes
  (`app`/`apple`/`application`, `banana`/`band`/`bandana`, `z`) exercise
  real arc sharing; the manifest also lists 8 keys deliberately absent
  from the FST (proper prefixes, over-extensions past an accepting node,
  a disjoint key, the empty string) so the differential test checks
  correct rejection, not just correct acceptance.
- `GenFstBinarySearch.java` — a real `FST<BytesRef>` (`fst_binary_search/`
  subdirectory) built via real `FSTCompiler` with
  `allowFixedLengthArcs(true)` and 7 single-byte root labels spread widely
  (1, 40, 80, 120, 160, 200, 240) specifically to make `FSTCompiler`'s own
  cost heuristic pick `ARCS_FOR_BINARY_SEARCH` encoding for the root node
  (confirmed via a self-check that the debug arc dump contains `"(bs)"`,
  not just assumed) -- this port's reader supports that encoding, but still
  rejects `ARCS_FOR_CONTINUOUS` outright, so this fixture deliberately stays
  small/sparse enough to land on binary search rather than direct addressing
  or continuous. The manifest's 8 absent keys are chosen in the gaps between
  and around the present labels (e.g. 60 between 40/80), not just
  far-outside values, so the differential test exercises the binary
  search's boundary behavior, not just "obviously not present."
- `GenFstDirectAddressing.java` — the direct-addressing counterpart to
  `GenFstBinarySearch.java` above: a real `FST<BytesRef>`
  (`fst_direct_addressing/` subdirectory) built via real `FSTCompiler` with
  `allowFixedLengthArcs(true)` and 7 single-byte root labels chosen dense but
  not fully contiguous (`a`-`f` plus `h`, skipping `g`) -- dense enough that
  `FSTCompiler`'s cost heuristic prefers direct addressing's small presence
  bitset over binary search's larger sparse array, but with one gap so the
  label range doesn't qualify for the (even cheaper) `ARCS_FOR_CONTINUOUS`
  encoding, which `FSTCompiler` always picks instead once every label in the
  range is present -- see `GenFstContinuous.java` below for that encoding's
  own fixture. Confirmed via a self-check that the debug arc dump contains
  `"(da)"`, not just assumed. The manifest's 6 absent keys specifically
  include `g` -- the one gap *inside* the label range (present bit clear, not
  merely out of range) -- alongside just-outside-the-range and
  clearly-disjoint values, so the differential test exercises the
  presence-bitset rejection path, not just the range-bounds check.
- `GenFstContinuous.java` — the continuous-range counterpart to
  `GenFstDirectAddressing.java` above: a real `FST<BytesRef>`
  (`fst_continuous/` subdirectory) built via real `FSTCompiler` with
  `allowFixedLengthArcs(true)` and 7 single-byte root labels that are
  *fully* contiguous (`a`-`g`, no gaps at all) -- once a label range has zero
  gaps, `FSTCompiler`'s cost heuristic always prefers `ARCS_FOR_CONTINUOUS`
  over both direct addressing and binary search, since no presence bitset is
  needed at all. Confirmed via a self-check that the debug arc dump contains
  `"(cs)"`, not just assumed. The manifest's 6 absent keys are all strictly
  outside the label range (there is no in-range gap to test, unlike direct
  addressing), so the differential test exercises the before/after-range
  bounds check specifically.
- `GenFstSeekNonRootArrayNode.java` — a real `FST<BytesRef>`
  (`fst_seek_non_root_array_node/` subdirectory) whose root stays
  list-encoded (only 3 arcs: `'B'`, `'C'`, `'D'`) while each of the three
  fixed-length-arc encodings sits one level *below* the root, under a shared
  prefix byte: `'B'` groups widely-spaced labels forced into
  `ARCS_FOR_BINARY_SEARCH`, `'D'` groups `a`-`f`,`h` (gap at `g`) forced into
  `ARCS_FOR_DIRECT_ADDRESSING`, `'C'` groups fully contiguous `a`-`g` forced
  into `ARCS_FOR_CONTINUOUS`. Every prior `GenFst*` fixture above puts its
  array-encoded node at the root, so seeking across them never recurses past
  a non-root array node; this fixture specifically exercises that
  backtracking path (`read_last_target_arc`'s array branch,
  `find_next_floor_arc_binary_search`/`_direct_addressing`/`_continuous`).
  Confirmed via a self-check that each depth-1 node's debug arc dump contains
  the expected `"(bs)"`/`"(da)"`/`"(cs)"` marker and that the root itself has
  `bytesPerArc() == 0` (list-encoded), not just assumed.
- `GenFstSeekBacktrackFloorArc.java` — three real `FST<BytesRef>`s
  (`fst_seek_floor_backtrack_binary_search/`, `_direct_addressing/`,
  `_continuous/` subdirectories), one per fixed-length-arc encoding, where the
  *root itself* is array-encoded (reusing each sibling fixture's own label
  set to force that) *and* one root label additionally has its own
  `ARCS_FOR_CONTINUOUS` child (a fully contiguous `a`-`g` two-byte
  extension). `seek_floor`'s `find_next_floor_arc_binary_search`/
  `_direct_addressing`/`_continuous` are only ever reached from
  `backtrack_to_floor_arc` re-reading a *parent* node that is itself
  array-encoded -- every other `GenFst*` fixture's array nodes sit at or
  below a list-encoded root, so backtracking from them never exercises this
  path. Confirmed via a self-check that both the root and the extended
  label's child node contain their expected debug-arc-dump marker.
- `GenFstDeepTrie.java` — a real `FST<BytesRef>` (`fst_deep_trie/`
  subdirectory) whose 9 keys (`abcaa`/`abcab`/`abcz`/`abda`/`abdz`/`acaa`/
  `aczz`/`baaa`/`bzzz`) share prefixes deeply enough that, with
  `allowFixedLengthArcs(false)` (list-encoded nodes, same scope as
  `GenFst.java`), the path to `"abcaa"` crosses 5 distinct trie levels --
  confirmed by a manual `readFirstTargetArc`/`readNextArc` walk before the
  fixture is written, not assumed. Every prior `GenFstSeek*` fixture's
  interesting structure sits at or one level below the root; this one
  forces `seekCeil`/`seekFloor`/`seekExact` to backtrack across 2-4 levels
  to resolve absent targets correctly. The manifest's 17 seek targets and
  their expected ceil/floor/exact results come from real Lucene's own
  `BytesRefFSTEnum.seekCeil`/`seekFloor`/`seekExact` against the reloaded
  FST, not hand-derived.
- `GenFstWideInputTypes.java` — two real `FST<BytesRef>`s (`fst_byte2/`,
  `fst_byte4/` subdirectories) whose `FST.INPUT_TYPE` is `BYTE2` and `BYTE4`
  respectively, unlike every other `GenFst*` generator here (`BYTE1`).
  `FSTCompiler.Builder`'s `INPUT_TYPE` parameter and `FSTCompiler.add(IntsRef,
  T)` are both public API, so no non-public hook is needed to build a
  genuinely wider-than-byte alphabet: each key is an explicit `int` label
  sequence (not derived from a `BytesRef`), including values well past 255
  (UTF-16-range code units for `BYTE2`, up to `0xFFFF`; full Unicode code
  points for `BYTE4`, up to `0x10FFFF`). `allowFixedLengthArcs(false)` keeps
  both to list-encoded nodes, matching `GenFst.java`'s own scope -- this
  fixture is about the arc *label* width, an orthogonal axis from node
  encoding. Exercised by
  `crates/lucene-codecs/tests/fst_wide_input_types_fixtures.rs` against
  `Fst::get_labels`/`Fst::iter_labels`/`FstEnum::seek_*_labels`.
- `GenBlockTree.java` — a real `IndexWriter` session (`blocktree_index/`
  subdirectory) producing `.tim`/`.tip`/`.tmd` (`Lucene103BlockTreeTermsWriter`,
  via `Lucene104PostingsFormat`), plus the `.fnm`/`.si` this port's readers
  need to open them. Two fields, both small enough to stay a single
  non-floor leaf block: "body" (`IndexOptions.DOCS_AND_FREQS`, five docs
  with repeated terms of known per-term frequencies, one doc missing the
  field) and "id" (`IndexOptions.DOCS`, one distinct token per doc,
  exercising the DOCS-only sumDocFreq/sumTotalTermFreq aliasing path). The
  manifest's per-term lookups (including deliberately-absent terms) are
  read back through real Lucene's own `TermsEnum.seekExact`/`docFreq`/
  `totalTermFreq`, not hand-computed, so the differential test checks
  against ground truth. Later slices added more fields to the same
  generator: "big" ("everywhere" in 300 docs, multi-block `.doc`), "pos"
  (positions/offsets/payloads), "many" (400 terms, multi-block/floor-split
  trie), and "l1" ("l1term" in 8250 docs, past `LEVEL1_NUM_DOCS` = 8192 so
  the `.doc` stream carries one inline level-1 skip entry + a span of 32
  full blocks + a remainder, exercising the level-1 decode/skip path). The
  manifest also dumps real `PostingsEnum.advance(target)` ground truth
  (including at the exact level-1 span boundary for "l1") and
  `TermsEnum.next()`/`seekCeil()` output.
- `GenPostingsSkip.java` — a real `IndexWriter` session
  (`postings_skip_index/` subdirectory) whose one field ("pskip",
  `DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS` with payloads) holds
  `skipterm` in **all 8 500 documents** (25 500 occurrences) and `gapterm`
  in 3 400 of them. 8 500 is past `LEVEL1_NUM_DOCS` (8 192), so real
  `Lucene104PostingsWriter` emits one level-1 skip entry, 33 level-0 block
  headers and a group-varint tail -- and, because the field indexes
  positions, **every one of those skip records carries the `.pos`/`.pay`
  file pointer and in-block offset its documents' occurrences start at**. No
  other fixture here contains those sub-fields at all: `blocktree_index`'s
  "pos" field has `docFreq = 3` and lives entirely in the vint tail, and its
  "l1" field (which does have skip data) indexes no positions.

  Two properties are load-bearing and were got wrong in this generator's
  first revision, so they are asserted by the test rather than assumed:

  - Per-document frequencies cycle 1..5, a period **coprime with 256**, so
    no `.pos` block boundary lines up with a `.doc` block boundary and
    `posBufferUpto` is non-zero in nearly every record — including the
    level-1 one, which is 253. With the period-4 cycle this generator
    started with, `sum(1 + d % 4)` over the first 8 192 documents was
    exactly 80 whole `.pos` blocks, so *every* level-1 `posBufferUpto` was
    `0` and a reader that never read the field would have passed. The
    manifest carries `level1_pos_buffer_upto` so the Rust test can pin it.
  - `gapterm` exists because `skipterm` is in every document, which makes
    all 33 of its level-0 blocks take Lucene's degenerate
    `docRange == BLOCK_SIZE` doc-delta encoding. `gapterm`'s blocks are
    packed-FOR or unary bit sets, and most sampled document ids are not in
    it, so it is what covers a skip-driven walk that must actually
    bit-unpack a block, and an `advance(doc)` for a document the term does
    not contain.

  Payload lengths vary including zero. Ground truth is Java's own
  `PostingsEnum.advance(doc)` +
  `nextPosition()`/`startOffset()`/`endOffset()`/`getPayload()` taken with a
  **fresh enum per sampled document**, so every sample is reached through
  the skip data rather than by sequential iteration -- the exact shape
  `postings::read_occurrences_for_doc` implements. Documents are sampled to
  bracket every structural boundary (either side of the level-1 span end and
  of each level-0 block end, the first document of the tail, first and
  last), plus an irregular stride. Exercised by
  `crates/lucene-codecs/tests/postings_skip_fixture.rs`.
- `GenAnalysis.java` — runs real `StandardAnalyzer` (`StandardTokenizer` +
  `LowerCaseFilter` + `StopFilter`) with a real stopword set (`the`, `a`,
  `of`) over six strings (`analysis/` subdirectory, no `IndexWriter`
  involved -- pure analysis, no index): a stopword mid-sentence, one at the
  very start, one at the very end, three consecutive stopwords in a row, an
  all-stopwords string, and a mixed-case/punctuation sentence with none
  removed. Records each surviving token's term, position increment, and
  char offsets via real `CharTermAttribute`/`PositionIncrementAttribute`/
  `OffsetAttribute`, which is what `lucene-analysis`'s `StopFilter`
  position-increment-preservation rule is checked against. Also includes
  `fold_only`/`fold_then_lower` (task #64, real `ASCIIFoldingFilter`) and
  five `uax29_*` cases (task #207: bare `StandardTokenizer`, no filters,
  over combining-mark, CJK-ideograph, precomposed-Hangul,
  conjoining-Jamo-Hangul, and mixed-CJK/Latin text) checking
  `lucene-analysis`'s `unicode-segmentation`-backed `tokenize()`. Batch c33
  added twelve `utf16_*` cases recording the **offset unit**: every string
  mixes an ASCII word, a Latin-1 accented letter (1 `char`, 2 UTF-8 bytes), a
  CJK ideograph (1, 3), a decomposed combining mark (2, 3) and/or a
  supplementary-plane character (2 `char`s, 1 Unicode scalar, 4 bytes), so
  UTF-8 byte offsets, Unicode scalar counts and Java `char` indices all
  disagree. They run real `StandardTokenizer`, `KeywordAnalyzer`,
  `ASCIIFoldingFilter`, `PorterStemFilter`, `NGramTokenFilter`,
  `EdgeNGramTokenFilter` and `SynonymGraphFilter`, and are what pins
  `OffsetAttribute`'s unit on the **write** side (the offsets `IndexWriter`
  puts in `.pos`/`.pay`/`.tvd`). All are plain-text manifest keys, so this
  generator is fully deterministic: `scripts/gen-fixtures.sh --only
  GenAnalysis --out <scratch>` reproduces `analysis/manifest.properties`
  byte for byte.
- `GenDisiJumpTable.java` — the only Java-written `IndexedDISI` **block jump
  table** in this tree (`disi_jump_table_index/`). `IndexedDISI.writeBitSet`
  emits `jumpTableEntryCount = 0` below two logical 65 536-document blocks, and
  every other Java-written sparse fixture here has five documents, so until
  `c43-final-cleanup` the *read* side of the table had never run over bytes
  Lucene wrote — the one direction of that format not covered by real bytes,
  and this sweep has twice found a writer and a reader agreeing on a shared
  mistake exactly there. 200 000 documents, no indexed field at all (an `id`
  term would add a 900 KB term dictionary to a fixture whose subject is the
  `.dvd`): `sparse` on every third document gives four DENSE blocks, and
  `very_sparse` on every 20 000th gives SPARSE blocks with the last logical
  block empty, which is `flushBlockJumps`' empty-block fill. The manifest
  records a **sampled** ground truth — 41 ascending probes, several blocks
  apart, so each cold lookup is the `advanceBlock` call that consults the table
  — plus each column's full-scan cardinality and value checksum. Consumed by
  `crates/lucene-codecs/tests/disi_jump_table_fixtures.rs`, which also perturbs
  each half of a table entry independently and requires the answer to change.
- `GenFullyDeletedDrop.java` — cross-engine ground truth for
  `IndexWriter.finishApply`'s **100%-deleted segment drop**
  (`fully_deleted_drop/`), recorded as an *outcome* rather than as bytes: the
  index of a writer that dropped a segment is indistinguishable from one that
  never had it, so there is nothing to diff. Four scripts run through a real
  `IndexWriter` in a `ByteBuffersDirectory` — `drop` (the older of two segments
  fully deleted), `partial` (the control: one of its two documents deleted, so
  it must survive), `all` (every segment emptied, so the commit is empty) and
  `block` (`updateDocuments` replacing a whole block) — each recording the
  committed segment count, every segment's `(maxDoc, delCount)` and the visible
  ids. Consumed by `index_writer::tests::
  a_fully_deleted_segment_is_dropped_exactly_where_real_lucene_drops_it`.

## Manifest appenders

`Append*Manifest` programs open an already-generated index **read-only** and
append cross-engine ground truth to its `manifest.properties`, stripping their
own key prefix first so a re-run rewrites the same bytes. They never regenerate
an index, so the committed segment ids do not move — which is why
`scripts/gen-fixtures.sh --append-only` is the safe way to add ground truth to a
committed fixture. Two were added by `c43-final-cleanup`:

- `AppendSpanExtentManifest.java` — real `SpanWeight.getSpans(ctx,
  Postings.POSITIONS)` walked to `NO_MORE_DOCS`/`NO_MORE_POSITIONS`: every
  `(startPosition(), endPosition())` pair, per document, per leaf, for 23
  `SpanQuery` shapes over `multi_segment_scoring_index`. That index is used
  because it is the only Java-written one whose position lists are rich enough
  to separate `NearSpans*`'s forward-only *walk* from a cartesian product —
  `GenMultiSegmentScoring.longBody` puts up to twenty occurrences of one term
  in a document, where `blocktree_index`'s `pos` field has two and the two
  algorithms agree on every query. Each case's query is written in a tiny
  S-expression (`t(field,term)`, `n(slop,inOrder,child,…)`, `o(child,…)`) that
  the Rust test parses with a twin parser, so the recorded query and the tested
  one cannot drift.
- `AppendMultiSegmentFuzzyManifest.java` — `FuzzyQuery` across two segments:
  the rewritten query's own selected terms and boosts (walked out of the
  `BooleanQuery` `BlendedTermQuery.BOOLEAN_REWRITE` produces), each selected
  term's **reader-wide** `docFreq`, and real `IndexSearcher` `TopDocs` as raw
  float bits. Every part of `TopTermsBlendedFreqScoringRewrite` is reader-wide,
  and all of it is invisible on a single segment — which is why every other
  fuzzy fixture here (all over `blocktree_index`) agrees with a per-segment
  implementation.
