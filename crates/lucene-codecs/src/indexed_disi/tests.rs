#![cfg(test)]
// The arithmetic gate is about values read off disk; a test's `i + 1` is
// not one. See docs/arithmetic-gate.md.
#![allow(clippy::arithmetic_side_effects)]

/// The cursor and the decode-everything path must agree for every document
/// in range, across all three block shapes and across block boundaries.
///
/// Two implementations of one lookup is the shape that diverges silently,
/// and this one has three encodings, a rank table and a sentinel to get
/// wrong. Checked over the whole doc-id range rather than at sampled
/// points, because the interesting answers are the `None`s.
///
/// Run at every legal `denseRankPower` as well as with no table at all:
/// the rank jump changes `word_index`/`number_of_ones` behind the answer,
/// so an off-by-one there is only visible as a wrong *ordinal*, never as a
/// wrong present/absent.
fn assert_cursor_matches(doc_ids: &[i32], max_doc: i32) {
    for power in [NO_RANK, 7, 9, 12, 15] {
        let (bytes, jumps) = write_with_dense_rank_power(doc_ids, power);
        let decoded = decode_doc_ids(&bytes, power).unwrap();
        assert_eq!(
            decoded, doc_ids,
            "fixture does not round-trip at power {power}"
        );

        // Every arm is run twice: once with the block jump table the
        // writer just emitted, and once with the table declared absent, so
        // the two `advanceBlock` paths are proved to give the same answers
        // over the same bytes. A jump table that lands the cursor on the
        // wrong block is invisible to a walk that never consults it.
        for declared_jumps in [jumps, 0] {
            let mut cursor = DisiCursor::new(&bytes, power, declared_jumps);
            for doc in 0..max_doc {
                let want = rank_of(&decoded, doc);
                let got = cursor.advance_exact(doc).unwrap();
                assert_eq!(
                    got, want,
                    "disagreement at doc {doc}, power {power}, jumps {declared_jumps}"
                );
            }

            // The same answers again, but reached by jumping straight to
            // each present doc rather than walking every doc id. This is
            // the path `rank_skip` is on: a stride of 512+ doc ids inside
            // a DENSE block is exactly what triggers it.
            let mut cursor = DisiCursor::new(&bytes, power, declared_jumps);
            for (ordinal, &doc) in decoded.iter().enumerate() {
                assert_eq!(
                    cursor.advance_exact(doc).unwrap(),
                    Some(ordinal),
                    "jumped lookup disagreed at doc {doc}, power {power}, \
                         jumps {declared_jumps}"
                );
            }
        }
    }
}

#[test]
fn cursor_matches_decode_for_a_sparse_block() {
    let docs: Vec<i32> = (0..50).map(|i| i * 7).collect();
    assert_cursor_matches(&docs, 400);
}

#[test]
fn cursor_matches_decode_for_a_dense_block() {
    // Above MAX_ARRAY_LENGTH (4095) and below BLOCK_SIZE, so DENSE.
    let docs: Vec<i32> = (0..10_000).map(|i| i * 6).collect();
    assert_cursor_matches(&docs, 65_536);
}

#[test]
fn cursor_matches_decode_for_an_all_block() {
    let docs: Vec<i32> = (0..65_536).collect();
    assert_cursor_matches(&docs, 65_536);
}

#[test]
fn cursor_matches_decode_across_block_boundaries() {
    // Block 0 sparse, block 1 entirely absent, block 2 dense: the cursor has
    // to walk a block with no values at all, which is the case a
    // "step to the next block" loop gets wrong.
    let mut docs: Vec<i32> = (0..40).map(|i| i * 3).collect();
    docs.extend((0..6_000).map(|i| 2 * 65_536 + i * 9));
    docs.retain(|&d| d < 3 * 65_536);
    docs.sort_unstable();
    docs.dedup();
    assert_cursor_matches(&docs, 3 * 65_536);
}

#[test]
fn cursor_matches_decode_for_an_all_block_followed_by_a_dense_one() {
    // An ALL block contributes 65,536 to the next block's ordinal base
    // while writing zero payload bytes -- the combination that breaks a
    // cursor which derives the base from bytes consumed instead of from
    // the header's `numValues`.
    let mut docs: Vec<i32> = (0..65_536).collect();
    docs.extend((0..5_000).map(|i| 65_536 + i * 13));
    assert_cursor_matches(&docs, 2 * 65_536);
}

/// Java's `advanceExact` asserts a non-decreasing target and is undefined
/// otherwise. This port makes the assert unconditional, because the two
/// alternatives are both worse: `None` is a legitimate answer ("no value
/// here"), so a violated contract would come back as a plausible wrong
/// number, and a silent wrong ordinal is worse still.
#[test]
#[should_panic(expected = "forward-only")]
fn cursor_panics_on_a_backward_doc_rather_than_answering() {
    let docs: Vec<i32> = (0..50).map(|i| i * 7).collect();
    let (bytes, jumps) = write(&docs);
    let mut cursor = DisiCursor::new(&bytes, NO_RANK, jumps);
    assert_eq!(cursor.advance_exact(70).unwrap(), Some(10));
    cursor.advance_exact(7).unwrap();
}

#[test]
#[should_panic(expected = "non-negative")]
fn cursor_panics_on_a_negative_doc() {
    let (bytes, jumps) = write(&[1, 2, 3]);
    DisiCursor::new(&bytes, NO_RANK, jumps)
        .advance_exact(-1)
        .unwrap();
}

/// `reset` is what makes a random-access caller possible on a forward-only
/// cursor -- `NumericReader` uses exactly this. Checked against the decoded
/// list over a deliberately shuffled access order.
#[test]
fn reset_restores_random_access_and_agrees_with_decode() {
    let mut docs: Vec<i32> = (0..8_000).map(|i| i * 8).collect();
    docs.extend((0..60).map(|i| 65_536 + i * 11));
    let (bytes, jumps) = write(&docs);
    let decoded = decode_doc_ids(&bytes, NO_RANK).unwrap();

    let mut cursor = DisiCursor::new(&bytes, NO_RANK, jumps);
    // A pseudo-random order over the whole range, including absent docs.
    let mut probe = 12_345u32;
    for _ in 0..5_000 {
        probe = probe.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let doc = (probe % (2 * 65_536)) as i32;
        if doc < cursor.doc_id() {
            cursor.reset();
        }
        assert_eq!(
            cursor.advance_exact(doc).unwrap(),
            rank_of(&decoded, doc),
            "disagreement at doc {doc}"
        );
    }
}

/// Java short-circuits a repeated `advanceExact(target)` on `target ==
/// doc`, returning the cached `exists`. Both answers have to survive it.
#[test]
fn repeating_the_same_doc_returns_the_same_answer() {
    let docs: Vec<i32> = (0..50).map(|i| i * 7).collect();
    let (bytes, jumps) = write(&docs);
    let mut cursor = DisiCursor::new(&bytes, NO_RANK, jumps);
    assert_eq!(cursor.advance_exact(70).unwrap(), Some(10));
    assert_eq!(cursor.advance_exact(70).unwrap(), Some(10));
    assert_eq!(cursor.advance_exact(71).unwrap(), None);
    assert_eq!(cursor.advance_exact(71).unwrap(), None);
    assert_eq!(cursor.advance_exact(77).unwrap(), Some(11));
}

/// `NO_MORE_DOCS` is the terminating block's payload, not a document. A
/// cursor that just scanned the sentinel block would report it present at
/// the field's cardinality.
#[test]
fn the_no_more_docs_sentinel_is_never_reported_as_present() {
    let (bytes, jumps) = write(&[1, 5, 100]);
    let mut cursor = DisiCursor::new(&bytes, NO_RANK, jumps);
    assert_eq!(cursor.advance_exact(NO_MORE_DOCS).unwrap(), None);
}

#[test]
fn a_doc_past_every_block_has_no_value() {
    let (bytes, jumps) = write(&[1, 5, 100]);
    let mut cursor = DisiCursor::new(&bytes, NO_RANK, jumps);
    assert_eq!(cursor.advance_exact(1_000_000).unwrap(), None);
    assert_eq!(cursor.advance_exact(2_000_000).unwrap(), None);
}

/// A region that ends before its sentinel block is corrupt, and the cursor
/// must say so rather than answering "no value" -- the region always ends
/// with a block whose index outranks every legal target, so walking off the
/// end can only mean truncation.
#[test]
fn a_truncated_region_is_an_error_not_a_missing_value() {
    let (bytes, _) = write(&[1, 5, 100]);
    let truncated = &bytes[..bytes.len() - 4];
    let mut cursor = DisiCursor::new(truncated, NO_RANK, 0);
    // Inside the surviving first block the answer is still knowable.
    assert_eq!(cursor.advance_exact(5).unwrap(), Some(1));
    // Past it, the sentinel block's header is gone.
    assert!(cursor.advance_exact(70_000).is_err());
}

/// `createRank`'s exact byte layout: a big-endian 2-byte count of the set
/// bits *before* each sub-block. Pinned rather than round-tripped, because
/// a rank table that is self-consistent but not Java's would still read
/// back correctly here and be silently wrong for a real Lucene reader.
#[test]
fn create_rank_matches_javas_byte_layout() {
    let mut words = vec![0u64; DENSE_BLOCK_LONGS as usize];
    // 64 set bits in each of the first 8 words: at power 9 (8 words per
    // entry) that is exactly 512 bits before the second sub-block.
    for w in words.iter_mut().take(8) {
        *w = u64::MAX;
    }
    // One more bit far along, in word 100 (sub-block 12).
    words[100] = 1;

    let rank = create_rank(&words, 9);
    assert_eq!(rank.len(), (DENSE_BLOCK_LONGS >> 2) as usize, "256 bytes");
    // Entry 0: nothing before the block starts.
    assert_eq!((rank[0], rank[1]), (0, 0));
    // Entry 1 covers words 8..16, and 512 bits precede it: 0x0200.
    assert_eq!((rank[2], rank[3]), (0x02, 0x00));
    // Entries 2..=12 all still see the same 512 bits.
    assert_eq!((rank[24], rank[25]), (0x02, 0x00));
    // Entry 13 covers words 104..112, past word 100's single bit: 513.
    assert_eq!((rank[26], rank[27]), (0x02, 0x01));
}

/// The rank table is only useful if `rank_skip` actually fires, and it only
/// fires on a jump of at least `2^(power-6)` words. This walks a DENSE
/// block in strides far larger than that at every legal power, so every
/// answer comes through `rank_skip` rather than through the word loop.
#[test]
fn rank_skip_produces_the_same_ordinals_as_a_full_walk() {
    // Every 5th doc present: 13,108 in block 0, comfortably DENSE.
    let docs: Vec<i32> = (0..13_100).map(|i| i * 5).collect();
    let plain = decode_doc_ids(&write(&docs).0, NO_RANK).unwrap();
    for power in [7u8, 9, 12, 15] {
        let (bytes, jumps) = write_with_dense_rank_power(&docs, power);
        let mut cursor = DisiCursor::new(&bytes, power, jumps);
        // Stride of 32768 bits' worth of docs, so even power 15's
        // `2^9`-word threshold is crossed on every step.
        for doc in (0..65_536).step_by(32_771) {
            assert_eq!(
                cursor.advance_exact(doc).unwrap(),
                rank_of(&plain, doc),
                "power {power}, doc {doc}"
            );
        }
    }
}

#[test]
fn or_into_words_sets_exactly_the_decoded_documents() {
    // SPARSE, DENSE (with and without a rank table) and ALL blocks.
    let mut docs: Vec<i32> = (0..3_000).map(|i| i * 7).collect();
    docs.extend((65_536..70_000).map(|i| i * 2).filter(|&d| d < 131_072));
    docs.extend(131_072..196_608);
    docs.push(200_001);
    docs.sort_unstable();
    docs.dedup();
    for power in [NO_RANK, 9] {
        let (bytes, _) = write_with_dense_rank_power(&docs, power);
        let mut words = vec![0u64; 200_064 / 64 + 1];
        or_into_words(&bytes, power, &mut words).unwrap();
        let mut got = Vec::new();
        for (i, &w) in words.iter().enumerate() {
            for b in 0..64 {
                if w >> b & 1 != 0 {
                    got.push((i * 64 + b) as i32);
                }
            }
        }
        assert_eq!(got, decode_doc_ids(&bytes, power).unwrap(), "power {power}");
    }
    // A set too small for the documents is corruption, whatever the block.
    for docs in [
        vec![5, 700],
        (0..10_000).map(|i| i * 3).collect(),
        (0..65_536).collect::<Vec<_>>(),
    ] {
        let (bytes, _) = write(&docs);
        let mut words = vec![0u64; 2];
        assert!(or_into_words(&bytes, NO_RANK, &mut words).is_err());
    }
}

#[test]
fn write_with_dense_rank_power_round_trips_through_decode() {
    let docs: Vec<i32> = (0..5_000).map(|i| i * 13).collect();
    for power in [NO_RANK, 7, 9, 15] {
        let (bytes, _) = write_with_dense_rank_power(&docs, power);
        assert_eq!(
            decode_doc_ids(&bytes, power).unwrap(),
            docs,
            "power {power}"
        );
    }
    // A rank table costs `1024 >> (power - 7)` bytes per DENSE block and
    // nothing at all elsewhere.
    assert_eq!(
        write_with_dense_rank_power(&docs, 9).0.len() - write(&docs).0.len(),
        256
    );
}

/// A negative doc id puts `block_base` at 0xFFFF0000, which the DENSE
/// path turns into an astronomic `words[rel / 64]` index. Strict ascent
/// alone does not rule it out.
#[test]
#[should_panic(expected = "doc ids must be non-negative")]
fn write_rejects_a_negative_doc_id() {
    write_with_dense_rank_power(&[-5, 1, 2], DEFAULT_DENSE_RANK_POWER);
}

#[test]
#[should_panic(expected = "denseRankPower")]
fn write_rejects_an_out_of_range_dense_rank_power() {
    write_with_dense_rank_power(&[1, 2, 3], 6);
}

use super::*;

fn write_block_header(out: &mut Vec<u8>, block: u16, num_values: u32) {
    out.extend_from_slice(&block.to_le_bytes());
    out.extend_from_slice(&((num_values - 1) as u16).to_le_bytes());
}

fn sentinel_block() -> Vec<u8> {
    let mut out = Vec::new();
    write_block_header(&mut out, (i32::MAX >> 16) as u16, 1);
    out.extend_from_slice(&((i32::MAX & 0xFFFF) as u16).to_le_bytes());
    out
}

#[test]
fn sparse_block_then_sentinel() {
    let mut data = Vec::new();
    write_block_header(&mut data, 0, 3);
    for v in [1u16, 5, 100] {
        data.extend_from_slice(&v.to_le_bytes());
    }
    data.extend_from_slice(&sentinel_block());

    let docs = decode_doc_ids(&data, NO_RANK).unwrap();
    assert_eq!(docs, vec![1, 5, 100]);
}

#[test]
fn all_block_then_sentinel() {
    let mut data = Vec::new();
    write_block_header(&mut data, 0, BLOCK_SIZE);
    data.extend_from_slice(&sentinel_block());

    let docs = decode_doc_ids(&data, NO_RANK).unwrap();
    assert_eq!(docs.len(), BLOCK_SIZE as usize);
    assert_eq!(docs[0], 0);
    assert_eq!(docs[BLOCK_SIZE as usize - 1], BLOCK_SIZE as i32 - 1);
}

#[test]
fn dense_block_without_rank_then_sentinel() {
    let mut data = Vec::new();
    write_block_header(&mut data, 0, MAX_ARRAY_LENGTH + 1); // smallest DENSE size
    let mut words = vec![0i64; DENSE_BLOCK_LONGS as usize];
    words[0] = 0b1011; // bits 0,1,3 set
    words[1] = 1 << 5; // doc 64+5 = 69
    for w in &words {
        data.extend_from_slice(&w.to_le_bytes());
    }
    data.extend_from_slice(&sentinel_block());

    let docs = decode_doc_ids(&data, NO_RANK).unwrap();
    assert_eq!(docs, vec![0, 1, 3, 69]);
}

#[test]
fn dense_block_with_rank_table_is_skipped_correctly() {
    let dense_rank_power = 9u8; // default: rank every 512 docs (8 longs)
    let rank_bytes = (DENSE_BLOCK_LONGS >> (dense_rank_power - 7)) as usize;

    let mut data = Vec::new();
    write_block_header(&mut data, 0, MAX_ARRAY_LENGTH + 1);
    data.extend(vec![0xAAu8; rank_bytes]); // rank table: content irrelevant, just skipped
    let mut words = vec![0i64; DENSE_BLOCK_LONGS as usize];
    words[0] = 1; // doc 0
    for w in &words {
        data.extend_from_slice(&w.to_le_bytes());
    }
    data.extend_from_slice(&sentinel_block());

    let docs = decode_doc_ids(&data, dense_rank_power).unwrap();
    assert_eq!(docs, vec![0]);
}

#[test]
fn multiple_blocks_across_ranges() {
    let mut data = Vec::new();
    write_block_header(&mut data, 0, 1);
    data.extend_from_slice(&5u16.to_le_bytes()); // doc 5
    write_block_header(&mut data, 1, 1);
    data.extend_from_slice(&7u16.to_le_bytes()); // doc (1<<16)|7 = 65543
    data.extend_from_slice(&sentinel_block());

    let docs = decode_doc_ids(&data, NO_RANK).unwrap();
    assert_eq!(docs, vec![5, 65543]);
}

#[test]
fn empty_structure_is_just_the_sentinel() {
    let data = sentinel_block();
    let docs = decode_doc_ids(&data, NO_RANK).unwrap();
    assert!(docs.is_empty());
}

#[test]
fn dense_rank_power_outside_javas_legal_range_is_rejected() {
    // Java's IndexedDISI constructor throws IllegalArgumentException for
    // anything but -1 (0xFF) or 7..=15. The byte comes off disk, so an
    // out-of-range one must be an error, not a shift-underflow panic.
    let mut data = Vec::new();
    write_block_header(&mut data, 0, MAX_ARRAY_LENGTH + 1); // forces DENSE
    data.extend(vec![0u8; DENSE_BLOCK_LONGS as usize * 8]);
    data.extend_from_slice(&sentinel_block());

    for bad in [0u8, 6, 16, 200] {
        assert!(
            decode_doc_ids(&data, bad).is_err(),
            "denseRankPower {bad} must be rejected"
        );
        let mut cursor = DisiCursor::new(&data, bad, 0);
        assert!(cursor.advance_exact(0).is_err(), "denseRankPower {bad}");
    }
    // The legal extremes still parse.
    for good in [7u8, 9, 15, NO_RANK] {
        let mut data = Vec::new();
        write_block_header(&mut data, 0, MAX_ARRAY_LENGTH + 1);
        data.extend(vec![0u8; dense_rank_bytes(good).unwrap()]);
        let mut words = vec![0i64; DENSE_BLOCK_LONGS as usize];
        words[0] = 1; // doc 0
        for w in &words {
            data.extend_from_slice(&w.to_le_bytes());
        }
        data.extend_from_slice(&sentinel_block());
        assert_eq!(
            decode_doc_ids(&data, good).unwrap(),
            vec![0],
            "power {good}"
        );
        assert_eq!(
            DisiCursor::new(&data, good, 0).advance_exact(0).unwrap(),
            Some(0),
            "power {good}"
        );
    }
}

#[test]
#[should_panic(expected = "strictly ascending")]
fn write_rejects_a_descending_pair_inside_a_block() {
    // Both docs land in block 0, so only a whole-slice check catches this.
    write(&[10, 20, 15]);
}

#[test]
#[should_panic(expected = "strictly ascending")]
fn write_rejects_duplicate_doc_ids() {
    write(&[10, 10]);
}

#[test]
fn rank_of_found_and_not_found() {
    let docs = vec![1, 5, 100];
    assert_eq!(rank_of(&docs, 5), Some(1));
    assert_eq!(rank_of(&docs, 2), None);
}

#[test]
fn decode_below_refuses_documents_past_the_segment_or_out_of_order() {
    // Every block shape, written by the real writer, decodes the same
    // bounded and unbounded while the segment holds its last document.
    let sparse: Vec<i32> = vec![3, 70_000, 70_001];
    let dense: Vec<i32> = (0..10_000).map(|d| d * 3).collect();
    let all: Vec<i32> = (65_536..131_072).collect();
    for docs in [&sparse, &dense, &all] {
        let bytes = write(docs).0;
        let last = *docs.last().unwrap();
        assert_eq!(
            &decode_doc_ids_below(&bytes, NO_RANK, last + 1).unwrap(),
            docs
        );
        assert!(decode_doc_ids_below(&bytes, NO_RANK, last).is_err());
    }
    // A DENSE block whose first word restates the SPARSE block before it.
    let mut data = Vec::new();
    write_block_header(&mut data, 0, 1);
    data.extend_from_slice(&40u16.to_le_bytes());
    write_block_header(&mut data, 0, MAX_ARRAY_LENGTH + 1);
    let mut words = vec![0i64; DENSE_BLOCK_LONGS as usize];
    words[0] = 1 << 40;
    for w in &words {
        data.extend_from_slice(&w.to_le_bytes());
    }
    data.extend_from_slice(&sentinel_block());
    assert!(decode_doc_ids_below(&data, NO_RANK, 1 << 20).is_err());
    // A block numbered past `i32::MAX` is refused even unbounded, rather
    // than wrapped into a negative document.
    let mut data = Vec::new();
    write_block_header(&mut data, 0xFFFF, 1);
    data.extend_from_slice(&7u16.to_le_bytes());
    data.extend_from_slice(&sentinel_block());
    assert!(decode_doc_ids(&data, NO_RANK).is_err());
}

#[test]
fn truncated_input_is_eof_error() {
    let data = vec![0u8; 2]; // half a block header
    assert!(decode_doc_ids(&data, NO_RANK).is_err());
}

// --- `write` (the writer added alongside the pre-existing reader above) ---

#[test]
fn write_empty_doc_ids_round_trips_to_empty() {
    let (data, _) = write(&[]);
    assert_eq!(decode_doc_ids(&data, NO_RANK).unwrap(), Vec::<i32>::new());
}

#[test]
fn write_single_doc_id_round_trips() {
    let (data, _) = write(&[42]);
    assert_eq!(decode_doc_ids(&data, NO_RANK).unwrap(), vec![42]);
}

#[test]
fn write_exactly_max_array_length_stays_sparse_shape() {
    // 4095 (MAX_ARRAY_LENGTH) present docs in one block: the boundary
    // value itself must still take the SPARSE-as-shorts shape (`<=`,
    // not `<`), not spill into the DENSE bitset shape one doc early.
    let doc_ids: Vec<i32> = (0..MAX_ARRAY_LENGTH as i32).collect();
    let (data, _) = write(&doc_ids);
    assert_eq!(decode_doc_ids(&data, NO_RANK).unwrap(), doc_ids);
}

#[test]
fn write_one_more_than_max_array_length_switches_to_dense_shape() {
    // 4096 present docs: one past the SPARSE/DENSE boundary, must
    // decode identically via the DENSE bitset shape instead.
    let doc_ids: Vec<i32> = (0..(MAX_ARRAY_LENGTH as i32 + 1)).collect();
    let (data, _) = write(&doc_ids);
    assert_eq!(decode_doc_ids(&data, NO_RANK).unwrap(), doc_ids);
}

#[test]
fn write_exactly_block_size_minus_one_stays_dense_shape() {
    // 65535 present docs (one short of a full 65536-doc block): must
    // NOT be mistaken for the ALL shape (which requires every doc in
    // the block, i.e. exactly BLOCK_SIZE).
    let doc_ids: Vec<i32> = (0..(BLOCK_SIZE as i32 - 1)).collect();
    let (data, _) = write(&doc_ids);
    assert_eq!(decode_doc_ids(&data, NO_RANK).unwrap(), doc_ids);
}

#[test]
fn write_exactly_block_size_uses_all_shape() {
    // Every doc in the block present: the ALL shape (zero body bytes
    // for the block, per real Lucene's IndexedDISIBuilder).
    let doc_ids: Vec<i32> = (0..BLOCK_SIZE as i32).collect();
    let (data, _) = write(&doc_ids);
    assert_eq!(decode_doc_ids(&data, NO_RANK).unwrap(), doc_ids);
}

/// `flushBlockJumps`' exact byte layout, hand-derived from Java rather
/// than round-tripped: a self-consistent-but-not-Java table round-trips
/// through this port's own cursor and is silently wrong for real Lucene,
/// which is precisely the trap `create_rank_matches_javas_byte_layout`
/// exists for on the rank table.
///
/// Two present docs in block 0, one in block **2** -- block 1 is empty, so
/// this also pins `addJumps`' fill of the skipped blocks with the *next*
/// real block's pair.
#[test]
fn jump_table_matches_javas_byte_layout() {
    let docs = vec![1, 2, 2 * BLOCK_SIZE as i32 + 5];
    let (bytes, count) = write(&docs);

    // Block 0 is SPARSE with 2 docs: 4 header bytes + 2 shorts = 8.
    // Block 2 is SPARSE with 1 doc: 4 + 2 = 6, at offset 8.
    // The sentinel block is another 6, at offset 14.
    // `blockCount = lastBlock + 1 = 4`, so four pairs trail the blocks.
    assert_eq!(count, 4);
    assert_eq!(bytes.len(), 20 + 4 * 8);

    let table = &bytes[20..];
    let entry = |i: usize| {
        (
            i32::from_le_bytes(table[i * 8..i * 8 + 4].try_into().unwrap()),
            i32::from_le_bytes(table[i * 8 + 4..i * 8 + 8].try_into().unwrap()),
        )
    };
    // (index, offset), where `index` is the cardinality *before* the block
    // the entry points at and `offset` is that block header's byte offset.
    assert_eq!(entry(0), (0, 0), "block 0");
    assert_eq!(entry(1), (2, 8), "empty block 1 points at block 2");
    assert_eq!(entry(2), (2, 8), "block 2");
    assert_eq!(
        entry(3),
        (3, 14),
        "the logical block after the last real one"
    );
}

/// Java: "Jumps with a single real entry + NO_MORE_DOCS is just wasted
/// space so we ignore that." Every doc in one block therefore produces no
/// table at all, and `jumpTableEntryCount` is `0`.
#[test]
fn a_single_block_region_writes_no_jump_table() {
    let (bytes, count) = write(&[1, 5, 100]);
    assert_eq!(count, 0);
    // 4 + 3 * 2 for the block, 4 + 2 for the sentinel, and nothing more.
    assert_eq!(bytes.len(), 10 + 6);
    // An *empty* region is the one case Java does write a table for:
    // `blockCount` is 1, not 2, so `flushBlockJumps`' single-real-block
    // exemption does not apply and one entry (pointing at the sentinel)
    // trails the sentinel block.
    let (empty, empty_count) = write(&[]);
    assert_eq!(empty_count, 1);
    assert_eq!(empty.len(), 6 + 8, "the sentinel block plus one entry");
}

/// The jump table is genuinely *read*: a table whose offsets point at the
/// wrong blocks gives different answers from the same bytes walked
/// block-by-block. Without this, a table that is written but never
/// consulted would pass every other test in this file.
#[test]
fn a_corrupted_jump_table_changes_the_answer_which_proves_it_is_read() {
    // Three blocks, so a jump from block 0 to block 2 is two ahead and
    // takes the table rather than the walk.
    let mut docs: Vec<i32> = vec![1, 2, 3];
    docs.push(BLOCK_SIZE as i32 + 7);
    docs.push(2 * BLOCK_SIZE as i32 + 9);
    let (bytes, count) = write(&docs);
    assert!(count >= 3);

    let target = 2 * BLOCK_SIZE as i32 + 9;
    let mut honest = DisiCursor::new(&bytes, NO_RANK, count);
    assert_eq!(honest.advance_exact(target).unwrap(), Some(4));

    // Point block 2's entry at block 1's header instead. The walk would
    // still be right; a cursor that trusts the table is not.
    let table_start = bytes.len() - count as usize * 8;
    let mut corrupt = bytes.clone();
    let block1_offset = i32::from_le_bytes(
        bytes[table_start + 8 + 4..table_start + 8 + 8]
            .try_into()
            .unwrap(),
    );
    corrupt[table_start + 16 + 4..table_start + 16 + 8]
        .copy_from_slice(&block1_offset.to_le_bytes());
    let mut lied_to = DisiCursor::new(&corrupt, NO_RANK, count);
    assert_ne!(
        lied_to.advance_exact(target).unwrap(),
        Some(4),
        "the cursor cannot be reading the jump table"
    );

    // Declaring the same bytes table-less walks the blocks and is right
    // again -- so the difference above is the table, not the corruption.
    let mut walked = DisiCursor::new(&corrupt[..table_start], NO_RANK, 0);
    assert_eq!(walked.advance_exact(target).unwrap(), Some(4));
}

/// A jump past the last covered block clamps to the table's final entry,
/// which always points at the `NO_MORE_DOCS` sentinel: Java's "If the
/// jumpTableEntryCount is exceeded, there are no further bits."
#[test]
fn a_jump_beyond_the_tables_last_block_lands_on_the_sentinel() {
    let docs = vec![1, BLOCK_SIZE as i32 + 7, 2 * BLOCK_SIZE as i32 + 9];
    let (bytes, count) = write(&docs);
    let mut cursor = DisiCursor::new(&bytes, NO_RANK, count);
    // Block 40 is far past every entry.
    assert_eq!(cursor.advance_exact(40 * BLOCK_SIZE as i32).unwrap(), None);
}

/// A `jumpTableEntryCount` larger than the region degrades to the
/// block-by-block walk instead of slicing past the end: the walk is always
/// correct, so a wrong count costs time, never an answer.
#[test]
fn an_over_large_jump_table_count_falls_back_to_walking_the_blocks() {
    let docs = vec![1, BLOCK_SIZE as i32 + 7, 2 * BLOCK_SIZE as i32 + 9];
    let (bytes, _) = write(&docs);
    let mut cursor = DisiCursor::new(&bytes, NO_RANK, i16::MAX);
    assert_eq!(
        cursor.advance_exact(2 * BLOCK_SIZE as i32 + 9).unwrap(),
        Some(2)
    );
}

#[test]
fn write_spans_a_block_boundary_correctly() {
    // One doc in the last slot of block 0 and one in the first slot of
    // block 1 -- proves block partitioning doesn't off-by-one at the
    // 65536 boundary itself.
    let doc_ids = vec![BLOCK_SIZE as i32 - 1, BLOCK_SIZE as i32];
    let (data, _) = write(&doc_ids);
    assert_eq!(decode_doc_ids(&data, NO_RANK).unwrap(), doc_ids);
}
