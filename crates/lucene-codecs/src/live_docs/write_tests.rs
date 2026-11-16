#![cfg(test)]
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
fn write_bits_matches_write_for_any_bits() {
    struct View<'a>(&'a FixedBitSet);
    impl Bits for View<'_> {
        fn get(&self, index: usize) -> bool {
            self.0.get(index)
        }
        fn length(&self) -> usize {
            self.0.len()
        }
    }
    for max_doc in [0usize, 1, 63, 64, 1023, 1024, 1025, 3000] {
        let mut live = FixedBitSet::new(max_doc);
        let mut deleted = 0;
        for i in 0..max_doc {
            if i % 3 == 0 {
                deleted += 1;
            } else {
                live.set(i);
            }
        }
        let expected = write(&live, &SEGMENT_ID, 2, deleted).unwrap();
        let bytes = write_bits(&View(&live), &SEGMENT_ID, 2, deleted).unwrap();
        assert_eq!(bytes, expected, "maxDoc={max_doc}");
        assert!(matches!(
            write_bits(&View(&live), &SEGMENT_ID, 2, deleted + 1),
            Err(Error::DelCountMismatch { .. })
        ));
    }
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

/// A `.liv` whose last word has bits set past `maxDoc` is corrupt --
/// Java's `FixedBitSet` constructor asserts they are clear. Report it as a
/// decode error rather than panicking (which is what the `debug_assert` in
/// `FixedBitSet::from_words` would do) or silently miscounting deletions.
#[test]
fn ghost_bits_past_max_doc_are_rejected() {
    let max_doc = 3;
    let mut buf: Vec<u8> = Vec::new();
    codec_util::write_index_header(
        &mut buf,
        CODEC_NAME,
        VERSION_CURRENT,
        &SEGMENT_ID,
        &lucene_util::base36::to_base36(1),
    );
    // Docs 0..2 live, plus a ghost bit at position 5 (>= maxDoc).
    buf.write_i64(0b100_111);
    codec_util::write_footer(&mut buf);
    assert!(matches!(
        parse(&buf, &SEGMENT_ID, 1, max_doc, 0),
        Err(Error::GhostBitsSet { max_doc: 3 })
    ));
}

/// A `maxDoc` that is an exact multiple of 64 has no ghost-bit positions
/// at all, so a full last word is legitimate.
#[test]
fn a_full_last_word_is_fine_when_max_doc_is_a_multiple_of_64() {
    let max_doc = 64;
    let mut bs = FixedBitSet::new(max_doc);
    for i in 0..max_doc {
        bs.set(i);
    }
    let decoded = round_trip(&bs, 1, 0);
    assert_eq!(decoded.cardinality(), max_doc);
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
