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
for cls in GenPrimitives GenCodecUtil GenSegmentInfo GenSegmentInfos; do
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
