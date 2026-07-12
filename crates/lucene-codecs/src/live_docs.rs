//! Port of `org.apache.lucene.codecs.lucene90.Lucene90LiveDocsFormat` (`.liv`
//! files) — both read and write.
//!
//! The `.liv` file is optional: it exists only when a segment has deletions
//! (`SegmentCommitInfo.hasDeletions()`, i.e. `del_gen != -1`). Its on-disk shape
//! is always the same dense bit array regardless of how Lucene chooses to
//! represent it in memory (`FixedBitSet` vs `SparseFixedBitSet` is a Java-side
//! in-memory optimization over identical bytes — this port only implements the
//! dense reader, since both would decode to the same bits).
//!
//! Wire format (little-endian throughout except the header/footer's BE fields):
//! ```text
//! IndexHeader(codec="Lucene90LiveDocs", version=0, id, suffix=delGen base-36)
//! Bits --> i64 * bits2words(maxDoc)   (bit=1 means live, bit=0 means deleted)
//! Footer
//! ```

use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::data_input::{DataInput, SliceInput};
use lucene_store::data_output::DataOutput;
use lucene_util::fixed_bit_set::{bits2words, FixedBitSet};

const CODEC_NAME: &str = "Lucene90LiveDocs";
const VERSION_START: i32 = 0;
const VERSION_CURRENT: i32 = 0;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    #[error("bits.deleted={actual} info.delcount={expected}")]
    DelCountMismatch { actual: usize, expected: usize },
}

pub type Result<T> = std::result::Result<T, Error>;

/// Parses a whole `.liv` file already read into memory.
///
/// `segment_id` must match the owning segment's id (from its `.si`/`segments_N`
/// entry); `del_gen` is the segment commit's deletion generation (used only to
/// validate the index header's suffix, matching `Long.toString(delGen, 36)`);
/// `max_doc` and `expected_del_count` come from the segment's `SegmentCommitInfo`
/// and are cross-checked against the bits actually read.
pub fn parse(
    buf: &[u8],
    segment_id: &[u8; ID_LENGTH],
    del_gen: i64,
    max_doc: usize,
    expected_del_count: usize,
) -> Result<FixedBitSet> {
    let mut input = SliceInput::new(buf);
    let suffix = lucene_util::base36::to_base36(del_gen);

    codec_util::check_index_header(
        &mut input,
        CODEC_NAME,
        VERSION_START,
        VERSION_CURRENT,
        segment_id,
        &suffix,
    )?;

    let mut words = vec![0i64; bits2words(max_doc)];
    input.read_i64s(&mut words)?;
    let live_docs = FixedBitSet::from_words(words.into_iter().map(|w| w as u64).collect(), max_doc);

    codec_util::check_footer(&mut input, buf.len())?;

    let actual_del_count = max_doc - live_docs.cardinality();
    if actual_del_count != expected_del_count {
        return Err(Error::DelCountMismatch {
            actual: actual_del_count,
            expected: expected_del_count,
        });
    }

    Ok(live_docs)
}

/// Writes a `.liv` file's bytes for `live_docs` (a dense bitset, bit=1 means
/// live, bit=0 means deleted), matching `Lucene90LiveDocsFormat.writeLiveDocs`.
///
/// Real Lucene's `writeBits` copies the input `Bits` in 1024-bit batches via
/// `Bits#applyMask` purely as a performance trick over the generic `Bits`
/// interface (which might not be word-addressable); since this port's input
/// is always an already-word-addressable [`FixedBitSet`], the batching has no
/// observable effect on the output bytes -- writing `live_docs.words()`
/// directly produces the exact same `.liv` bytes real Lucene would, just
/// without replicating an optimization this port doesn't need. `del_gen` must
/// be the segment commit's *next* deletion generation
/// (`SegmentCommitInfo.getNextDelGen()`), used both for the output file name
/// (by the caller) and the index header's suffix (`Long.toString(delGen,
/// 36)`), matching the read side's `parse`.
///
/// `expected_del_count` is `info.getDelCount() + newDelCount` in Java's
/// signature (the delete count the caller expects after this write); the
/// actual count derived from `live_docs` is cross-checked against it, exactly
/// like `parse` cross-checks the count it read.
pub fn write(
    live_docs: &FixedBitSet,
    segment_id: &[u8; ID_LENGTH],
    del_gen: i64,
    expected_del_count: usize,
) -> Result<Vec<u8>> {
    let suffix = lucene_util::base36::to_base36(del_gen);

    let mut buf: Vec<u8> = Vec::new();
    codec_util::write_index_header(&mut buf, CODEC_NAME, VERSION_CURRENT, segment_id, &suffix);

    for &word in live_docs.words() {
        buf.write_i64(word as i64);
    }

    codec_util::write_footer(&mut buf);

    let actual_del_count = live_docs.len() - live_docs.cardinality();
    if actual_del_count != expected_del_count {
        return Err(Error::DelCountMismatch {
            actual: actual_del_count,
            expected: expected_del_count,
        });
    }

    Ok(buf)
}

#[cfg(test)]
mod write_tests {
    use super::*;

    const SEGMENT_ID: [u8; ID_LENGTH] = *b"0123456789abcdef";

    fn round_trip(live_docs: &FixedBitSet, del_gen: i64, expected_del_count: usize) -> FixedBitSet {
        let bytes = write(live_docs, &SEGMENT_ID, del_gen, expected_del_count).unwrap();
        parse(
            &bytes,
            &SEGMENT_ID,
            del_gen,
            live_docs.len(),
            expected_del_count,
        )
        .unwrap()
    }

    #[test]
    fn all_live_no_deletions() {
        let max_doc = 5;
        let mut bs = FixedBitSet::new(max_doc);
        for i in 0..max_doc {
            bs.set(i);
        }
        let decoded = round_trip(&bs, 1, 0);
        assert_eq!(decoded.cardinality(), max_doc);
        for i in 0..max_doc {
            assert!(decoded.get(i));
        }
    }

    #[test]
    fn some_deleted() {
        let max_doc = 5;
        let mut bs = FixedBitSet::new(max_doc);
        for i in 0..max_doc {
            bs.set(i);
        }
        bs.clear(1);
        bs.clear(3);
        let decoded = round_trip(&bs, 7, 2);
        assert!(decoded.get(0));
        assert!(!decoded.get(1));
        assert!(decoded.get(2));
        assert!(!decoded.get(3));
        assert!(decoded.get(4));
        assert_eq!(decoded.cardinality(), 3);
    }

    #[test]
    fn doc_count_not_multiple_of_64_word_boundary() {
        // 130 bits -> 3 words, last word only holds 2 live bits. Exercises the
        // partial last-word case for both writer and reader.
        let max_doc = 130;
        let mut bs = FixedBitSet::new(max_doc);
        for i in 0..max_doc {
            bs.set(i);
        }
        // Delete a doc in the final partial word.
        bs.clear(129);
        let decoded = round_trip(&bs, 3, 1);
        assert!(decoded.get(128));
        assert!(!decoded.get(129));
        assert_eq!(decoded.cardinality(), max_doc - 1);
    }

    #[test]
    fn del_count_mismatch_is_rejected() {
        let max_doc = 4;
        let mut bs = FixedBitSet::new(max_doc);
        for i in 0..max_doc {
            bs.set(i);
        }
        bs.clear(0);
        let err = write(&bs, &SEGMENT_ID, 1, 0).unwrap_err();
        assert!(matches!(
            err,
            Error::DelCountMismatch {
                actual: 1,
                expected: 0
            }
        ));
    }

    #[test]
    fn write_then_parse_rejects_wrong_segment_id() {
        let max_doc = 4;
        let mut bs = FixedBitSet::new(max_doc);
        for i in 0..max_doc {
            bs.set(i);
        }
        let bytes = write(&bs, &SEGMENT_ID, 1, 0).unwrap();
        let other_id: [u8; ID_LENGTH] = *b"fedcba9876543210";
        let result = parse(&bytes, &other_id, 1, max_doc, 0);
        assert!(result.is_err());
    }
}
