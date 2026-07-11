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
for cls in GenPrimitives GenCodecUtil GenSegmentInfo GenSegmentInfos GenLiveDocs GenFieldInfos GenNorms GenNumericDocValues; do
  java -cp "classes:$JAR" $cls data
done
```

`data/` is checked in (small, deterministic) so `cargo test` works without Java
installed; regenerate and re-commit whenever the pinned Lucene version changes.

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
- `GenNumericDocValues.java` — a real single-segment `IndexWriter` session
  (`numeric_dv_index/` subdirectory) with three NUMERIC-only doc-values
  fields: "varying" (arbitrary signed values, exercises plain delta
  compression), "gcd" (values sharing a large common divisor, exercises
  GCD compression), and "sparse" (present on only 3 of 5 docs, exercises
  the `IndexedDISI` path — same mechanism as `GenNorms.java`'s sparse
  field). Also dumps the segment's `.fnm` since parsing `.dvm` requires the
  field infos to check each field's doc-values-skip-index configuration.
  Expected values come from reading them back through Lucene's own
  `Lucene90DocValuesProducer.getNumeric`, not our own arithmetic.
