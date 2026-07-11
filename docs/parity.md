# Parity matrix

Java file → ported/partial/not-needed/deferred. Populated per PR from Phase 1.
Pinned Lucene version: **10.5.0** (matches OpenSearch `gradle/libs.versions.toml`).

## lucene-util

| Java | Rust | Status |
|---|---|---|
| `util/BitUtil.zigZagEncode/Decode` | `lucene-util/src/zigzag.rs` | ported, fixture-verified |

## lucene-store

| Java | Rust | Status |
|---|---|---|
| `store/DataInput` (vint/vlong/zlong) | `lucene-store/src/data_input.rs` | ported, fixture-verified |
| `util/GroupVIntUtil` | `lucene-store/src/data_input.rs::read_group_vints` | ported, fixture-verified |
| `store/DataInput.readString` | `lucene-store/src/data_input.rs::read_string` | ported |
| `codecs/CodecUtil` (header/index-header/footer, CRC-32) | `lucene-store/src/codec_util.rs` | ported, fixture-verified (incl. corrupted-checksum case) |
| `store/Directory`, `FSDirectory`, `MMapDirectory` | — | not started |
| `store/IndexInput` slicing/cloning over real files | — | not started (`SliceInput` covers the in-memory case) |
| `codecs/CodecUtil.writeHeader/writeFooter` (encode side) | — | deferred to Phase 5 (write path) |

## lucene-index

| Java | Rust | Status |
|---|---|---|
| `codecs/lucene99/Lucene99SegmentInfoFormat` (`.si` read) | `lucene-index/src/segment_info.rs` | ported (read-only), fixture-verified |
| — index-sorted segments (`numSortFields > 0`, `SortFieldProvider`) | — | unsupported by design, returns `Error::UnsupportedIndexSort` (needs a Java-plugin-class registry decision) |
| `index/SegmentInfos` (`segments_N` read: header/footer, per-segment commit metadata, user data) | `lucene-index/src/segment_infos.rs` | ported (read-only), fixture-verified against a real 2-commit `IndexWriter` output |
| `index/SegmentInfos` — locating the latest `segments_N` in a directory (`SEGMENTS_GEN`, listing) | — | not started (needs `Directory` from Phase 1) |
| `codecs/lucene99/Lucene99SegmentInfoFormat.write`, `SegmentInfos.write` | — | deferred to Phase 5 (write path) |

