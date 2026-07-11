//! Port of `org.apache.lucene.codecs.lucene90.Lucene90LiveDocsFormat` (`.liv`
//! files) — read-only.
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
