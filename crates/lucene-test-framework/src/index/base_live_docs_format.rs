//! `BaseLiveDocsFormatTestCase`: the `.liv` format's round trip, over
//! `lucene_codecs::live_docs::{write, parse}` (`Lucene90LiveDocsFormat`)
//! through a directory.
//!
//! Differs:
//!
//! - `writeLiveDocs` is two byte functions: `live_docs::write` for a
//!   [`FixedBitSet`] and `live_docs::write_bits` for any other [`Bits`],
//!   which `testSerialization`'s non-`FixedBitSet` pass goes through.
//! - `@Monster testOverflow` (a `MAX_DOCS` bitset) is [`serialization`]
//!   with its size as a parameter; it runs smaller by default.

use lucene_codecs::live_docs;
use lucene_store::codec_util::ID_LENGTH;
use lucene_store::Directory;
use lucene_util::bits::Bits;
use lucene_util::fixed_bit_set::FixedBitSet;

use super::base_index_file_format::{write_file, SuiteTest};
use crate::util::lucene_test_case::LuceneTestCase;
use crate::util::test_util;

/// The live documents `writeLiveDocs` is handed: a [`FixedBitSet`] (its
/// fast path) or any other [`Bits`].
#[derive(Clone, Copy)]
pub enum LiveBits<'a> {
    /// A `FixedBitSet`.
    Fixed(&'a FixedBitSet),
    /// Any other `Bits`.
    Other(&'a dyn Bits),
}

/// A `Bits` view of a bit set that is not a `FixedBitSet`: Java's
/// anonymous `Bits` over `liveDocs`.
struct BitsView<'a>(&'a FixedBitSet);

impl Bits for BitsView<'_> {
    fn get(&self, index: usize) -> bool {
        // FBS: `Bits.get` is only asked below `length()`, the set's length.
        self.0.get(index)
    }

    fn length(&self) -> usize {
        self.0.len()
    }
}

/// `format.writeLiveDocs(bits, dir, sci, delCount)`: `<segment>_<gen>.liv`.
///
/// # Errors
/// The writer's refusal (a delete count the bits disagree with) or the
/// directory's, as text.
pub fn write_live_docs(
    dir: &dyn Directory,
    segment: &str,
    id: &[u8; ID_LENGTH],
    del_gen: i64,
    live: LiveBits<'_>,
    del_count: usize,
) -> Result<(), String> {
    let bytes = match live {
        LiveBits::Fixed(bits) => live_docs::write(bits, id, del_gen, del_count),
        LiveBits::Other(bits) => live_docs::write_bits(bits, id, del_gen, del_count),
    }
    .map_err(|e| e.to_string())?;
    let name = format!("{segment}_{}.liv", lucene_util::base36::to_base36(del_gen));
    write_file(dir, &name, &bytes).map_err(|e| e.to_string())
}

/// `format.readLiveDocs(dir, sci)`.
///
/// # Errors
/// The directory's refusal or the parser's, as text.
pub fn read_live_docs(
    dir: &dyn Directory,
    segment: &str,
    id: &[u8; ID_LENGTH],
    del_gen: i64,
    max_doc: usize,
    del_count: usize,
) -> Result<FixedBitSet, String> {
    let name = format!("{segment}_{}.liv", lucene_util::base36::to_base36(del_gen));
    let bytes = dir.open(&name).map_err(|e| e.to_string())?;
    live_docs::parse(&bytes, id, del_gen, max_doc, del_count).map_err(|e| e.to_string())
}

/// `testSerialization(maxDoc, numLiveDocs, fixedBitSet)`.
pub fn serialization(t: &mut LuceneTestCase, max_doc: usize, num_live: usize, fixed_bit_set: bool) {
    let mut live = FixedBitSet::new(max_doc);
    let bound = live.len();
    let bound_i32 = i32::try_from(bound).expect("maxDoc fits an int");
    if num_live > max_doc / 2 {
        live.set_range(0, bound);
        for _ in 0..max_doc - num_live {
            loop {
                let bit = t.random().next_int_bounded(bound_i32) as usize;
                if live.get(bit) {
                    live.clear(bit);
                    break;
                }
            }
        }
    } else {
        for _ in 0..num_live {
            loop {
                let bit = t.random().next_int_bounded(bound_i32) as usize;
                if !live.get(bit) {
                    live.set(bit);
                    break;
                }
            }
        }
    }
    let dir = t.new_directory();
    let _is_compound_file = t.random().next_boolean();
    let id = lucene_util::string_helper::random_id();
    let del_count = max_doc - num_live;
    let view = BitsView(&live);
    let bits = if fixed_bit_set {
        LiveBits::Fixed(&live)
    } else {
        LiveBits::Other(&view)
    };
    write_live_docs(&*dir, "foo", &id, 1, bits, del_count).expect("writeLiveDocs");
    let read = read_live_docs(&*dir, "foo", &id, 1, max_doc, del_count).expect("readLiveDocs");
    assert_eq!(read.len(), bound);
    for i in 0..bound {
        assert_eq!(live.get(i), read.get(i), "doc {i}");
    }
    dir.close().expect("close");
}

fn random_max_doc(t: &mut LuceneTestCase) -> usize {
    test_util::next_int(t.random(), 3, 1000) as usize
}

/// `testDenseLiveDocs`.
pub fn test_dense_live_docs(t: &mut LuceneTestCase) {
    let max_doc = random_max_doc(t);
    for fixed_bit_set in [false, true] {
        serialization(t, max_doc, max_doc - 1, fixed_bit_set);
    }
}

/// `testEmptyLiveDocs`.
pub fn test_empty_live_docs(t: &mut LuceneTestCase) {
    let max_doc = random_max_doc(t);
    for fixed_bit_set in [false, true] {
        serialization(t, max_doc, 0, fixed_bit_set);
    }
}

/// `testSparseLiveDocs`.
pub fn test_sparse_live_docs(t: &mut LuceneTestCase) {
    let max_doc = random_max_doc(t);
    for fixed_bit_set in [false, true] {
        serialization(t, max_doc, 1, fixed_bit_set);
    }
}

/// `@Monster testOverflow`, at `max_doc` documents (Java's is
/// `IndexWriter.MAX_DOCS`).
pub fn overflow(t: &mut LuceneTestCase, max_doc: usize) {
    serialization(t, max_doc, max_doc - 7, false);
}

/// Every `BaseLiveDocsFormatTestCase` test this port runs by default.
pub const TESTS: &[SuiteTest] = &[
    ("testDenseLiveDocs", test_dense_live_docs),
    ("testEmptyLiveDocs", test_empty_live_docs),
    ("testSparseLiveDocs", test_sparse_live_docs),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refusals_come_back_as_text() {
        let mut t = LuceneTestCase::new("refusals_come_back_as_text");
        let dir = t.new_directory();
        let id = lucene_util::string_helper::random_id();
        let mut live = FixedBitSet::new(10);
        live.set_range(0, 10);
        live.clear(3);
        let err = write_live_docs(&*dir, "foo", &id, 1, LiveBits::Fixed(&live), 2)
            .expect_err("bad delCount");
        assert!(!err.is_empty());
        let err = read_live_docs(&*dir, "foo", &id, 1, 10, 1).expect_err("no file");
        assert!(!err.is_empty());
        write_live_docs(&*dir, "foo", &id, 1, LiveBits::Fixed(&live), 1).expect("write");
        let err = read_live_docs(&*dir, "foo", &id, 1, 10, 2).expect_err("bad delCount");
        assert!(!err.is_empty());
        let err =
            write_live_docs(&*dir, "foo", &id, 1, LiveBits::Fixed(&live), 1).expect_err("exists");
        assert!(!err.is_empty());
        dir.close().expect("close");
    }
}
