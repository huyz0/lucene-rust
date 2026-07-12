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
for cls in GenPrimitives GenCodecUtil GenSegmentInfo GenSegmentInfos GenLiveDocs GenFieldInfos GenNorms GenDocValues GenCompoundFormat GenStoredFields GenStoredFieldsBestCompression GenSortedDocValues GenMultiValuedDocValues GenTermVectors GenPoints; do
  java -cp "classes:$JAR" $cls data
done
```

`data/` is checked in (small, deterministic) so `cargo test` works without Java
installed; regenerate and re-commit whenever the pinned Lucene version changes.

## Verifying the write path (reverse direction)

Every generator above is Java-writes-Rust-reads. The write path (PLAN.md Phase 5)
needs the opposite: Rust writes real bytes, and a Java program confirms real Lucene
can open and read them back. `VerifyStoredFields.java` and `VerifyFieldInfos.java`
are these verifiers so far:

```sh
cargo run -p lucene-codecs --example write_stored_fields_fixture -- /tmp/rust-stored-fields
cargo run -p lucene-codecs --example write_field_infos_fixture -- /tmp/rust-field-infos
JAR=$(find ~/.gradle/caches/modules-2/files-2.1/org.apache.lucene/lucene-core/10.5.0 \
  -name 'lucene-core-10.5.0.jar' ! -name '*sources*' ! -name '*javadoc*')
javac -nowarn -cp "$JAR" -d classes src/VerifyStoredFields.java src/VerifyFieldInfos.java
java -cp "classes:$JAR" VerifyStoredFields /tmp/rust-stored-fields
java -cp "classes:$JAR" VerifyFieldInfos /tmp/rust-field-infos
```

`VerifyStoredFields.java` opens the `.fdt`/`.fdx`/`.fdm` triple directly through
`Lucene90StoredFieldsFormat.fieldsReader`, using a hand-built `SegmentInfo`/
`FieldInfos` rather than also requiring Rust to write `.si`/`.fnm` -- this keeps
each write-path slice scoped to exactly the one format it's verifying, the same
way the read-path fixtures below call one codec-level `open`/`document` directly
rather than going through a full `IndexReader`. `VerifyFieldInfos.java` follows
the same pattern for `.fnm`: it opens the file directly through
`Lucene94FieldInfosFormat.read` with a hand-built `SegmentInfo` (no `.si` writer
needed), then checks every field's properties against `manifest.properties`.

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
