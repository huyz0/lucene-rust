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

## lucene-index

| Java | Rust | Status |
|---|---|---|
| `codecs/lucene99/Lucene99SegmentInfoFormat` (`.si` read) | `lucene-index/src/segment_info.rs` | ported (read-only), fixture-verified |
| — index-sorted segments (`numSortFields > 0`, `SortFieldProvider`) | — | unsupported by design, returns `Error::UnsupportedIndexSort` (needs a Java-plugin-class registry decision) |
| `index/SegmentInfos` (`segments_N` read: header/footer, per-segment commit metadata, user data) | `lucene-index/src/segment_infos.rs` | ported (read-only), fixture-verified against a real 2-commit `IndexWriter` output |
| `codecs/lucene99/Lucene99SegmentInfoFormat.write`, `SegmentInfos.write` | — | deferred to Phase 5 (write path) |

