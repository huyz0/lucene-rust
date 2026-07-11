# Parity matrix

Java file → ported/partial/not-needed/deferred. Populated per PR from Phase 1.
Pinned Lucene version: **10.5.0** (matches OpenSearch `gradle/libs.versions.toml`).

## lucene-util

| Java | Rust | Status |
|---|---|---|
| `util/BitUtil.zigZagEncode/Decode` | `lucene-util/src/zigzag.rs` | ported, fixture-verified |
| `Long.toString/parseLong(_, 36)` (generation ↔ base-36 filename suffix) | `lucene-util/src/base36.rs` | ported, round-trip tested |

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

## lucene-index

| Java | Rust | Status |
|---|---|---|
| `codecs/lucene99/Lucene99SegmentInfoFormat` (`.si` read) | `lucene-index/src/segment_info.rs` | ported (read-only), fixture-verified |
| — index-sorted segments (`numSortFields > 0`, `SortFieldProvider`) | — | unsupported by design, returns `Error::UnsupportedIndexSort` (needs a Java-plugin-class registry decision) |
| `index/SegmentInfos` (`segments_N` read: header/footer, per-segment commit metadata, user data) | `lucene-index/src/segment_infos.rs` | ported (read-only), fixture-verified against a real 2-commit `IndexWriter` output |
| `codecs/lucene99/Lucene99SegmentInfoFormat.write`, `SegmentInfos.write` | — | deferred to Phase 5 (write path) |

