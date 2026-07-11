# Parity matrix

Java file → ported/partial/not-needed/deferred. Populated per PR from Phase 1.
Pinned Lucene version: **10.5.0** (matches OpenSearch `gradle/libs.versions.toml`).

## lucene-util

| Java | Rust | Status |
|---|---|---|
| `util/BitUtil.zigZagEncode/Decode` | `lucene-util/src/zigzag.rs` | ported, fixture-verified |
| `Long.toString/parseLong(_, 36)` (generation ↔ base-36 filename suffix) | `lucene-util/src/base36.rs` | ported, round-trip tested |
| `util/FixedBitSet` (dense bitset: get/set/clear/cardinality) | `lucene-util/src/fixed_bit_set.rs` | ported (the subset `.liv` reading needs); no `SparseFixedBitSet` (in-memory-only optimization, not a format difference) |

## lucene-store

| Java | Rust | Status |
|---|---|---|
| `store/DataInput` (vint/vlong/zlong) | `lucene-store/src/data_input.rs` | ported, fixture-verified |
| `util/GroupVIntUtil` | `lucene-store/src/data_input.rs::read_group_vints` | ported, fixture-verified |
| `store/DataInput.readString` | `lucene-store/src/data_input.rs::read_string` | ported |
| `codecs/CodecUtil` (header/index-header/footer, CRC-32) | `lucene-store/src/codec_util.rs` | ported, fixture-verified (incl. corrupted-checksum case) |
| `codecs/CodecUtil.retrieveChecksum` (structural-only footer check, no full-file CRC) | `lucene-store/src/codec_util.rs::retrieve_checksum` | ported, unit-tested |
| `store/Directory`, `FSDirectory` (listing + whole-file read) | `lucene-store/src/directory.rs::{Directory, FsDirectory}` | ported (read-only), fixture-verified |
| `store/MMapDirectory` | `lucene-store/src/directory.rs::MmapDirectory` | ported (read-only), fixture-verified; this crate's only `unsafe` |
| `index/SegmentInfos.getLastCommitGeneration`, `generationFromSegmentsFileName`, `IndexFileNames.fileNameFromGeneration` | `lucene-store/src/directory.rs::{last_commit_generation, generation_from_segments_file_name, segments_file_name, read_latest_commit}` | ported, fixture-verified end-to-end (open dir → find latest commit → parse `segments_N`) |
| `store/IndexInput` slicing/cloning over real files | — | not started (`SliceInput` covers the in-memory case; `Directory::open` currently returns a whole-file buffer, not a lazily-sliced `IndexInput`) |
| `store/Directory` write side (`createOutput`, locking) | — | deferred to Phase 5 (write path) |
| `codecs/CodecUtil.writeHeader/writeFooter` (encode side) | — | deferred to Phase 5 (write path) |

## lucene-codecs

| Java | Rust | Status |
|---|---|---|
| `codecs/lucene90/Lucene90LiveDocsFormat` (`.liv` read) | `lucene-codecs/src/live_docs.rs` | ported (read-only), fixture-verified against a real IndexWriter deletion (2 of 5 docs deleted by term, `NoMergePolicy` to keep the segment intact) |
| — `SparseFixedBitSet`/`SparseLiveDocs` in-memory choice | — | not applicable to this port: the on-disk bytes are identical dense bits regardless of Java's in-memory representation choice |
| `codecs/lucene90/Lucene90LiveDocsFormat.writeLiveDocs` | — | deferred to Phase 5 (write path) |
| `codecs/lucene94/Lucene94FieldInfosFormat` (`.fnm` read, incl. `FieldInfo.checkConsistency`) | `lucene-codecs/src/field_infos.rs` | ported (read-only), fixture-verified against a real IndexWriter (7 field shapes + a soft-deletes field introduced by a later DV-update generation) |
| `codecs/lucene94/Lucene94FieldInfosFormat.write` | — | deferred to Phase 5 (write path) |
| `codecs/lucene90/Lucene90NormsFormat` (`.nvm`/`.nvd` read: empty/dense/sparse) | `lucene-codecs/src/norms.rs` | ported (read-only), fixture-verified against real per-doc norm values from Lucene's own `NormsProducer`, including a real sparse field (some docs missing it entirely) |
| `codecs/lucene90/IndexedDISI` (sparse doc-id-set: SPARSE/DENSE/ALL blocks) | `lucene-codecs/src/indexed_disi.rs` | ported as a **one-shot decode to `Vec<i32>`**, not Java's lazy seekable iterator — see the module doc for why (this port isn't in the hot-path-perf phase yet); jump table and DENSE rank bytes are parsed past but never used, since a full sequential decode doesn't need to skip ahead |
| `codecs/lucene90/Lucene90DocValuesFormat`/`Lucene90DocValuesProducer` (`.dvm`/`.dvd` read, **NUMERIC and BINARY fields only**) | `lucene-codecs/src/doc_values.rs` | ported (read-only), fixture-verified against a real IndexWriter: numeric plain-varying dense, numeric GCD-compressed dense, numeric sparse (`IndexedDISI`), binary fixed-length dense, binary variable-length dense (`DirectMonotonicReader`), binary variable-length sparse (`IndexedDISI` + `DirectMonotonicReader`); sorted/sorted-set/sorted-numeric doc values and per-field doc-values skip indexes are out of scope (return `Error::UnsupportedFieldType`/`UnsupportedSkipIndex` rather than misdecode), as is the varying-bits-per-value block split (`Error::UnsupportedVaryingBpvBlocks`) — see the module doc |
| `util/packed/DirectReader` (bit-packed integer array read) | `lucene-codecs/src/direct_reader.rs` | ported as **one generic bit-position formula**, not Java's thirteen width-specialized `DirectPackedReaderN` classes — those exist for JIT monomorphism, a concern this port doesn't have yet |
| `util/packed/DirectMonotonicReader` (monotonic sequence read, blocks of min/avg/bit-packed-delta) | `lucene-codecs/src/direct_monotonic.rs` | ported (read-only); used by BINARY doc values' variable-length address blocks |
| `codecs/lucene90/Lucene90NormsConsumer` (write side), `IndexedDISI.writeBitSet`, `Lucene90DocValuesConsumer` (write side) | — | deferred to Phase 5 (write path) |
| `codecs/lucene90/Lucene90DocValuesProducer` sorted/sorted-set/sorted-numeric doc values, terms-dict (LZ4) | — | not started — a separate future port (see `doc_values.rs` module doc for why it isn't just "skip past the bytes") |
| `codecs/lucene90/Lucene90CompoundFormat`/`Lucene90CompoundReader` (`.cfs`/`.cfe` read) | `lucene-codecs/src/compound_format.rs` | ported (read-only), fixture-verified against a real `useCompoundFile=true` IndexWriter segment (12 packed sub-files); preserves Java's exact-version cross-check between `.cfs`/`.cfe` and the total-length-vs-entries-table sanity check |
| `codecs/lucene90/Lucene90CompoundFormat.write` | — | deferred to Phase 5 (write path) |
| `codecs/lucene90/Lucene90StoredFieldsFormat`/`Lucene90CompressingStoredFieldsReader`/`FieldsIndexReader` (`.fdt`/`.fdx`/`.fdm` read, **`Mode.BEST_SPEED` only**) | `lucene-codecs/src/stored_fields.rs` | ported (read-only), fixture-verified against a real IndexWriter (6 docs, one field of each of the 6 stored-field types, varying lengths so the chunk uses the bulk multi-doc framing); `Mode.BEST_COMPRESSION` (DEFLATE, 48KB blocks) is out of scope — its distinct `.fdt` codec name makes the header check reject it cleanly with no extra plumbing needed |
| `util/compress/LZ4.decompress` (decompress-only; no compressor ported) | `lucene-codecs/src/lz4.rs` | ported (read-only) |
| `codecs/lucene90/LZ4WithPresetDictCompressionMode` (dictionary + fixed-size sub-blocks) | `lucene-codecs/src/stored_fields.rs::decompress_unit` | ported as an eager full decode (no partial/lazy read path — this port hands back a materialized `Document`, not a streaming reader); the per-block *compressed*-length vints exist in Java only to support seeking without decompressing everything before it, so they're read to keep the cursor aligned and discarded |
| `codecs/lucene90/compressing/StoredFieldsInts` (bulk per-doc field-count/length arrays) | `lucene-codecs/src/stored_fields.rs::read_bulk_ints` | ported with the exact on-disk transposed-block layout preserved (values decode identically to Java's SIMD-shaped bulk reader), but decoded via a plain per-value loop rather than replicating the bulk/SIMD machinery itself |
| `codecs/lucene90/compressing/Lucene90CompressingStoredFieldsWriter` (write side), `LZ4` compressor | — | deferred to Phase 5 (write path) |

## lucene-index

| Java | Rust | Status |
|---|---|---|
| `codecs/lucene99/Lucene99SegmentInfoFormat` (`.si` read) | `lucene-index/src/segment_info.rs` | ported (read-only), fixture-verified |
| — index-sorted segments (`numSortFields > 0`, `SortFieldProvider`) | — | unsupported by design, returns `Error::UnsupportedIndexSort` (needs a Java-plugin-class registry decision) |
| `index/SegmentInfos` (`segments_N` read: header/footer, per-segment commit metadata, user data) | `lucene-index/src/segment_infos.rs` | ported (read-only), fixture-verified against a real 2-commit `IndexWriter` output |
| `codecs/lucene99/Lucene99SegmentInfoFormat.write`, `SegmentInfos.write` | — | deferred to Phase 5 (write path) |

