#![cfg(test)]
use super::*;

fn vint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

/// Block `i`'s wire values: its entry's `.doc`/`.pos`/`.pay` pointers
/// (the *next* block's start) and `posBufferUpto`.
fn doc_ptr(i: usize) -> i64 {
    1000 * (i as i64 + 1)
}
fn pos_ptr(i: usize) -> i64 {
    10 * (i as i64 + 1)
}
fn pay_ptr(i: usize) -> i64 {
    7 * (i as i64 + 1)
}
fn pos_upto(i: usize) -> i32 {
    (i % 5) as i32
}

/// `Lucene90SkipWriter` + `MultiLevelSkipListWriter` for a term whose
/// entry-carrying blocks end on `block_last` (the writer buffers a
/// block's entry when the next block starts). Base pointers are 0, each
/// entry's impacts are the one byte `i`. Returns the serialized skip
/// data after `pad` filler bytes.
fn write(block_last: &[i32], fields: SkipFields, levels: usize, pad: usize) -> Vec<u8> {
    let mut bufs: Vec<Vec<u8>> = vec![Vec::new(); levels];
    let mut last_doc = vec![0i32; levels];
    let mut last_doc_ptr = vec![0i64; levels];
    let mut last_pos_ptr = vec![0i64; levels];
    let mut last_pay_ptr = vec![0i64; levels];
    for (i, &doc) in block_last.iter().enumerate() {
        // `bufferSkip(df)`: how many levels this entry reaches.
        let mut blocks = i as i64 + 1;
        let mut n = 1;
        while blocks % SKIP_MULTIPLIER == 0 && n < levels {
            n += 1;
            blocks /= SKIP_MULTIPLIER;
        }
        let mut child = 0u64;
        for l in 0..n {
            let b = &mut bufs[l];
            vint(b, (doc - last_doc[l]) as u64);
            last_doc[l] = doc;
            vint(b, (doc_ptr(i) - last_doc_ptr[l]) as u64);
            last_doc_ptr[l] = doc_ptr(i);
            if fields.has_pos {
                vint(b, (pos_ptr(i) - last_pos_ptr[l]) as u64);
                last_pos_ptr[l] = pos_ptr(i);
                vint(b, pos_upto(i) as u64);
                if fields.has_payloads {
                    vint(b, 3);
                }
                if fields.has_offsets || fields.has_payloads {
                    vint(b, (pay_ptr(i) - last_pay_ptr[l]) as u64);
                    last_pay_ptr[l] = pay_ptr(i);
                }
            }
            b.push(1);
            b.push(i as u8);
            let new_child = b.len() as u64;
            if l != 0 {
                vint(b, child);
            }
            child = new_child;
        }
    }
    // `writeSkip`: levels top-down, each but level 0 length-prefixed.
    let mut out = vec![0xAAu8; pad];
    for l in (1..levels).rev() {
        vint(&mut out, bufs[l].len() as u64);
        out.extend_from_slice(&bufs[l]);
    }
    out.extend_from_slice(&bufs[0]);
    out
}

const DOCS_ONLY: SkipFields = SkipFields {
    has_pos: false,
    has_offsets: false,
    has_payloads: false,
};
const EVERYTHING: SkipFields = SkipFields {
    has_pos: true,
    has_offsets: true,
    has_payloads: true,
};

/// A term of `df` documents `0, 3, 6, ...`: the last documents of its
/// entry-carrying blocks, the number of levels Lucene writes, and the
/// documents.
fn term(df: i32) -> (Vec<i32>, usize, Vec<i32>) {
    let docs: Vec<i32> = (0..df).map(|i| i * 3).collect();
    let full = df as usize / 128;
    // The last full block carries no entry when nothing follows it.
    let entries = if (df as usize).is_multiple_of(128) {
        full - 1
    } else {
        full
    };
    let block_last: Vec<i32> = (0..entries).map(|b| docs[128 * b + 127]).collect();
    let dc = i64::from(if df % 128 == 0 { df - 1 } else { df });
    let mut levels = 1;
    let mut x = dc / 128;
    while dc > 128 && x >= 8 {
        x /= 8;
        levels += 1;
    }
    (block_last, levels, docs)
}

fn check(df: i32, fields: SkipFields, targets: impl Iterator<Item = i32>, fresh: bool) {
    let (block_last, levels, docs) = term(df);
    let buf = write(&block_last, fields, levels, 5);
    let open = || SkipList::new(&buf, fields, 5, 0, 0, 0, df).unwrap();
    let mut s = open();
    assert_eq!(s.number_of_skip_levels, levels, "df {df}");
    assert_eq!(s.next_skip_doc(), -1);
    assert!(s.level1().is_none());
    let mut checked = 0;
    for t in targets {
        // `skipDoc[0]` starts at 0, so `skipTo(0)` reads no entry; the
        // cursor asks for at least 1, as `advanceShallow` does.
        let t = t.max(1);
        if fresh {
            s = open();
        }
        let at = docs.partition_point(|&d| d < t);
        if at == docs.len() {
            break;
        }
        let b = at / 128;
        s.skip_to(t).unwrap();
        checked += 1;
        let ctx = format!("df {df} target {t} block {b}");
        if b == 0 {
            assert_eq!((s.last_doc(), s.doc_pointer()), (0, 0), "{ctx}");
            assert_eq!(s.pos_state(), (0, 0, 0), "{ctx}");
        } else {
            assert_eq!(s.last_doc(), block_last[b - 1], "{ctx}");
            assert_eq!(s.doc_pointer(), doc_ptr(b - 1), "{ctx}");
            if fields.has_pos {
                assert_eq!(
                    s.pos_state(),
                    (pos_ptr(b - 1), pos_upto(b - 1), pay_ptr(b - 1)),
                    "{ctx}"
                );
            }
        }
        if b < block_last.len() {
            assert_eq!(s.next_skip_doc(), block_last[b], "{ctx}");
            assert_eq!(s.next_doc_pointer(), doc_ptr(b), "{ctx}");
            assert_eq!(s.level0_impacts(), &[b as u8], "{ctx}");
        } else {
            assert_eq!(s.next_skip_doc(), i32::MAX, "{ctx}");
            assert!(s.level0_impacts().is_empty(), "{ctx}");
        }
        let span_end = b / 8 * 8 + 7;
        if levels >= 2 && span_end < block_last.len() {
            assert_eq!(
                s.level1(),
                Some((block_last[span_end], &[span_end as u8][..])),
                "{ctx}"
            );
        } else {
            assert!(s.level1().is_none(), "{ctx}");
        }
    }
    assert!(checked > 3, "df {df}: only {checked} targets");
}

#[test]
fn skip_to_lands_on_the_block_holding_the_target_at_every_level() {
    // 3 levels (entries every 128, 1,024 and 8,192 docs) and a tail.
    let df = 128 * 70 + 5;
    check(df, DOCS_ONLY, (0..df * 3 + 2).step_by(7), false);
    check(df, EVERYTHING, (0..df * 3 + 2).step_by(389), false);
    // A cold list per target: the descent from the top level alone.
    check(df, EVERYTHING, (0..df * 3).step_by(1531), true);
    // Ends on a block boundary: `trim` drops the last block's entry.
    check(128 * 16, DOCS_ONLY, (0..128 * 48).step_by(5), false);
    // One entry, one level.
    check(129, EVERYTHING, 0..400, false);
}

#[test]
fn skip_to_jumps_across_whole_upper_level_spans() {
    let df = 128 * 600 + 1;
    let (_, levels, _) = term(df);
    assert_eq!(levels, 4);
    check(
        df,
        DOCS_ONLY,
        [1, 50_000, 50_001, 190_000, 230_390].into_iter(),
        false,
    );
}

#[test]
fn corrupt_skip_data_is_an_error_not_a_panic() {
    let (block_last, levels, _) = term(128 * 70 + 5);
    let buf = write(&block_last, DOCS_ONLY, levels, 0);
    for bad in [-1, buf.len() as i64 + 1] {
        assert!(SkipList::new(&buf, DOCS_ONLY, bad, 0, 0, 0, 8965).is_err());
    }
    // A level length past the end of the data.
    let mut long = buf.clone();
    long[0] = 0x7f;
    long.truncate(40);
    assert!(SkipList::new(&long, DOCS_ONLY, 0, 0, 0, 0, 8965).is_err());
    // A negative level length.
    let negative = [0xffu8, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01];
    assert!(SkipList::new(&negative, DOCS_ONLY, 0, 0, 0, 0, 8965).is_err());
    // Every truncation, and every byte flipped: an error or an answer,
    // never a panic.
    for cut in 0..buf.len() {
        if let Ok(mut s) = SkipList::new(&buf[..cut], DOCS_ONLY, 0, 0, 0, 0, 8965) {
            let _ = s.skip_to(26_000);
        }
    }
    for i in 0..buf.len() {
        let mut b = buf.clone();
        b[i] ^= 0xff;
        if let Ok(mut s) = SkipList::new(&b, DOCS_ONLY, 0, 0, 0, 0, 8965) {
            for t in [100, 5_000, 26_000] {
                if s.skip_to(t).is_err() {
                    break;
                }
                let _ = (s.level0_impacts(), s.level1(), s.next_doc_pointer());
            }
        }
    }
}

#[test]
fn a_child_pointer_that_does_not_fit_is_corrupt() {
    // Two levels by hand; the level-1 entry's child pointer is -1.
    let mut level1 = Vec::new();
    vint(&mut level1, 1023);
    vint(&mut level1, 8000);
    level1.push(0); // no impacts
    level1.extend_from_slice(&[0xff; 9]);
    level1.push(0x01);
    let mut buf = Vec::new();
    vint(&mut buf, level1.len() as u64);
    buf.extend_from_slice(&level1);
    buf.extend_from_slice(&[1, 1, 0]); // one level-0 entry
    let mut s = SkipList::new(&buf, DOCS_ONLY, 0, 0, 0, 0, 1100).unwrap();
    assert_eq!(s.number_of_skip_levels, 2);
    assert!(s.skip_to(1100).is_err());
}
