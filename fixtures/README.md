# fixtures

Java programs, pinned to Lucene 10.5.0 (OpenSearch's current pin — see
`gradle/libs.versions.toml` in the OpenSearch checkout), that generate byte-level
fixtures for differential testing. Rust `tests/` in each crate read `data/*.bin` and
compare decoded values against `data/*.expected` / manifests — no JVM needed at Rust
test time, only to regenerate fixtures after a Lucene version bump.

## Regenerating

```sh
JAR=$(find ~/.gradle/caches/modules-2/files-2.1/org.apache.lucene/lucene-core/10.5.0 \
  -name 'lucene-core-10.5.0.jar' ! -name '*sources*' ! -name '*javadoc*')
mkdir -p classes data
javac -nowarn -cp "$JAR" -d classes src/*.java
for cls in GenPrimitives GenCodecUtil GenSegmentInfo GenSegmentInfos GenLiveDocs GenFieldInfos GenNorms GenDocValues GenCompoundFormat GenStoredFields GenStoredFieldsBestCompression GenSortedDocValues GenMultiValuedDocValues GenTermVectors GenPoints GenFst GenBlockTree GenBlockTreeCompressed GenFstBinarySearch GenFstDirectAddressing GenFstContinuous GenFstSeekNonRootArrayNode GenFstSeekBacktrackFloorArc; do
  java -cp "classes:$JAR" $cls data
done
```

`GenAnalysis.java` additionally needs `lucene-analysis-common` on the classpath
(it exercises real `StandardAnalyzer`/`StopFilter`, not `lucene-core` alone):

```sh
ANALYSIS_JAR=$(find ~/.gradle/caches/modules-2/files-2.1/org.apache.lucene/lucene-analysis-common/10.5.0 \
  -name '*.jar' ! -name '*sources*' ! -name '*javadoc*')
javac -nowarn -cp "$JAR:$ANALYSIS_JAR" -d classes src/GenAnalysis.java
java -cp "classes:$JAR:$ANALYSIS_JAR" GenAnalysis data
```

`data/` is checked in (small, deterministic) so `cargo test` works without Java
installed; regenerate and re-commit whenever the pinned Lucene version changes.

## Verifying the write path (reverse direction)

Every generator above is Java-writes-Rust-reads. The write path (PLAN.md Phase 5)
needs the opposite: Rust writes real bytes, and a Java program confirms real Lucene
can open and read them back. `VerifyStoredFields.java`, `VerifyFieldInfos.java`,
`VerifySegmentInfo.java`, `VerifySegmentInfos.java`, `VerifyPoints.java`,
`VerifyTermVectors.java`, `VerifyDocValues.java`, `VerifyNorms.java`,
`VerifyLiveDocs.java`, `VerifyCompoundFormat.java`, and `VerifyFst.java` are these
verifiers so far:

```sh
cargo run -p lucene-codecs --example write_stored_fields_fixture -- /tmp/rust-stored-fields
cargo run -p lucene-codecs --example write_field_infos_fixture -- /tmp/rust-field-infos
cargo run -p lucene-index --example write_segment_info_fixture -- /tmp/rust-segment-info
cargo run -p lucene-index --example write_segment_infos_fixture -- /tmp/rust-segment-infos
cargo run -p lucene-index --example write_multi_segment_commit_fixture -- /tmp/rust-multi-segment
cargo run -p lucene-codecs --example write_points_fixture -- /tmp/rust-points
cargo run -p lucene-codecs --example write_term_vectors_fixture -- /tmp/rust-term-vectors
cargo run -p lucene-codecs --example write_doc_values_fixture -- /tmp/rust-doc-values
cargo run -p lucene-codecs --example write_norms_fixture -- /tmp/rust-norms
cargo run -p lucene-codecs --example write_live_docs_fixture -- /tmp/rust-live-docs
cargo run -p lucene-codecs --example write_compound_format_fixture -- /tmp/rust-compound-format
cargo run -p lucene-codecs --example write_fst_fixture -- /tmp/rust-fst
JAR=$(find ~/.gradle/caches/modules-2/files-2.1/org.apache.lucene/lucene-core/10.5.0 \
  -name 'lucene-core-10.5.0.jar' ! -name '*sources*' ! -name '*javadoc*')
javac -nowarn -cp "$JAR" -d classes src/VerifyStoredFields.java src/VerifyFieldInfos.java src/VerifySegmentInfo.java src/VerifySegmentInfos.java src/VerifyPoints.java src/VerifyTermVectors.java src/VerifyDocValues.java src/VerifyNorms.java src/VerifyLiveDocs.java src/VerifyCompoundFormat.java src/VerifyFst.java
java -cp "classes:$JAR" VerifyStoredFields /tmp/rust-stored-fields
java -cp "classes:$JAR" VerifyFieldInfos /tmp/rust-field-infos
java -cp "classes:$JAR" VerifySegmentInfo /tmp/rust-segment-info
java -cp "classes:$JAR" VerifySegmentInfos /tmp/rust-segment-infos
java -cp "classes:$JAR" VerifySegmentInfos /tmp/rust-multi-segment
java -cp "classes:$JAR" VerifyPoints /tmp/rust-points
java -cp "classes:$JAR" VerifyTermVectors /tmp/rust-term-vectors
java -cp "classes:$JAR" VerifyDocValues /tmp/rust-doc-values
java -cp "classes:$JAR" VerifyNorms /tmp/rust-norms
java -cp "classes:$JAR" VerifyLiveDocs /tmp/rust-live-docs
java -cp "classes:$JAR" VerifyCompoundFormat /tmp/rust-compound-format
java -cp "classes:$JAR" VerifyFst /tmp/rust-fst
```

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
- `GenLiveDocs.java` — a real single-segment `IndexWriter` session with 2 of 5 docs
  deleted by term after the first commit (`live_docs_index/` subdirectory:
  `NoMergePolicy` keeps the segment from being merged away, so the fixture's `.liv`
  file is a real post-deletion commit, not hand-built bits).
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
- `GenAnalysis.java` — runs real `StandardAnalyzer` (`StandardTokenizer` +
  `LowerCaseFilter` + `StopFilter`) with a real stopword set (`the`, `a`,
  `of`) over six strings (`analysis/` subdirectory, no `IndexWriter`
  involved -- pure analysis, no index): a stopword mid-sentence, one at the
  very start, one at the very end, three consecutive stopwords in a row, an
  all-stopwords string, and a mixed-case/punctuation sentence with none
  removed. Records each surviving token's term, position increment, and
  char offsets via real `CharTermAttribute`/`PositionIncrementAttribute`/
  `OffsetAttribute`, which is what `lucene-analysis`'s `StopFilter`
  position-increment-preservation rule is checked against.
