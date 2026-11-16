#![cfg(test)]
#![allow(clippy::arithmetic_side_effects)]
use super::*;
use crate::morph::resource::test_util::{header, vlong};

fn units(s: &str) -> Vec<u16> {
    s.encode_utf16().collect()
}

/// Walks `key` from the first arc; the output sum and final output at
/// the last unit, `None` when an arc is missing.
fn walk(f: &TokenInfoFst, key: &[u16]) -> Option<(i64, bool, i64)> {
    let mut arc = f.first_arc();
    let mut out = 0;
    for (i, &u) in key.iter().enumerate() {
        arc = f.find_target_arc(i32::from(u), &arc, i == 0).unwrap()?;
        out += arc.output();
    }
    Some((out, arc.is_final(), arc.next_final_output()))
}

#[test]
fn trie_maps_keys_to_ordinals() {
    let keys: Vec<Vec<u16>> = ["ab", "abc", "b", "日本"]
        .iter()
        .map(|s| units(s))
        .collect();
    let f = TokenInfoFst::from_sorted(&keys, 0x30FF, 0x3040).unwrap();
    assert_eq!(walk(&f, &units("ab")), Some((0, true, 0)));
    assert_eq!(walk(&f, &units("abc")), Some((0, true, 1)));
    assert_eq!(walk(&f, &units("a")), Some((0, false, 0)));
    assert_eq!(walk(&f, &units("日本")), Some((0, true, 3)));
    assert_eq!(walk(&f, &units("x")), None);
    assert_eq!(walk(&f, &units("abcd")), None);
    assert!(!f.first_arc().is_final());
    assert_eq!(f.empty_output(), None);
    let first = f.first_arc();
    assert_eq!(f.find_target_arc(-1, &first, false).unwrap(), None);
    // Cached root arcs: a cached label answers whatever the follow arc.
    let k: Vec<Vec<u16>> = vec![units("あい")];
    let f = TokenInfoFst::from_sorted(&k, 0x30FF, 0x3040).unwrap();
    let a = f.find_target_arc(0x3042, &first, true).unwrap().unwrap();
    assert_eq!(f.find_target_arc(0x3042, &a, true).unwrap(), Some(a));
    assert_eq!(f.find_target_arc(0x3042, &a, false).unwrap(), None);
    assert_eq!(a.label(), 0x3042);
    assert!(!a.is_last());
    assert!(a.target() > 0);
    // An empty key is the empty output; a repeated key fails.
    let e = TokenInfoFst::from_sorted(&[vec![], units("a")], 1, 0).unwrap();
    assert!(e.first_arc().is_final());
    assert_eq!(e.empty_output(), Some(0));
    assert!(TokenInfoFst::from_sorted(&[units("a"), units("a")], 1, 0).is_err());
    assert!(TokenInfoFst::from_sorted(&[], 0, 1).is_err());
}

/// An FST file whose body is `body` (bytes in file order; the reader
/// starts at `start`).
fn fst_file(input_type: u8, empty: Option<u64>, start: u64, body: &[u8]) -> Vec<u8> {
    let mut b = header("FST", 9);
    match empty {
        Some(v) => {
            let mut e = Vec::new();
            vlong(&mut e, v);
            e.reverse();
            b.push(1);
            vlong(&mut b, e.len() as u64);
            b.extend_from_slice(&e);
        }
        None => b.push(0),
    }
    b.push(input_type);
    vlong(&mut b, start);
    vlong(&mut b, body.len() as u64);
    b.extend_from_slice(body);
    b
}

#[test]
fn hand_built_list_node() {
    // One list node at 4..1 (read downward): arc 'a' (final, last, stop
    // node, output 5). Bytes in read order: flags, label, vlong output.
    let flags = BIT_FINAL_ARC | BIT_LAST_ARC | BIT_STOP_NODE | BIT_ARC_HAS_OUTPUT;
    let body = [0u8, 5, b'a', flags];
    let f = TokenInfoFst::read(&fst_file(0, Some(7), 3, &body), 0, 0).unwrap();
    assert_eq!(f.empty_output(), Some(7));
    let first = f.first_arc();
    assert!(first.is_final());
    assert_eq!(first.next_final_output(), 7);
    let a = f
        .find_target_arc(i32::from(b'a'), &first, false)
        .unwrap()
        .unwrap();
    assert_eq!(
        (a.output(), a.is_final(), a.target()),
        (5, true, FINAL_END_NODE)
    );
    assert_eq!(
        f.find_target_arc(i32::from(b'b'), &first, false).unwrap(),
        None
    );
    assert_eq!(f.find_target_arc(0, &first, false).unwrap(), None);
    assert_eq!(f.find_target_arc(i32::from(b'a'), &a, false).unwrap(), None);
    // END_LABEL off a final arc, with and without a target.
    let end = f
        .find_target_arc(END_LABEL, &first, false)
        .unwrap()
        .unwrap();
    assert_eq!(
        (end.label(), end.output(), end.is_last()),
        (END_LABEL, 7, false)
    );
    let end = f.find_target_arc(END_LABEL, &a, false).unwrap().unwrap();
    assert!(end.is_last());
    let mut not_final = first;
    not_final.flags = 0;
    assert_eq!(
        f.find_target_arc(END_LABEL, &not_final, false).unwrap(),
        None
    );
    // A zero empty output carries no final-output flag.
    let z = TokenInfoFst::read(&fst_file(0, Some(0), 3, &body), 0, 0).unwrap();
    assert_eq!(z.first_arc().flags & BIT_ARC_HAS_FINAL_OUTPUT, 0);
}

#[test]
fn hand_built_target_next_arcs() {
    // Read order from the start node: arc 'a' (target-next, output 3,
    // final output 5), arc 'b' (last, stop, final), then the next node:
    // arc 'c' (last, stop, final, output 7).
    let read_order = [
        BIT_TARGET_NEXT | BIT_ARC_HAS_OUTPUT | BIT_ARC_HAS_FINAL_OUTPUT,
        b'a',
        3,
        5,
        BIT_LAST_ARC | BIT_STOP_NODE | BIT_FINAL_ARC,
        b'b',
        BIT_LAST_ARC | BIT_STOP_NODE | BIT_FINAL_ARC | BIT_ARC_HAS_OUTPUT,
        b'c',
        7,
    ];
    let mut body = read_order.to_vec();
    body.reverse();
    let f = TokenInfoFst::read(&fst_file(0, None, 8, &body), 0, 0).unwrap();
    let first = f.first_arc();
    let a = f
        .find_target_arc(i32::from(b'a'), &first, false)
        .unwrap()
        .unwrap();
    assert_eq!(
        (a.output(), a.next_final_output(), a.is_final()),
        (3, 5, false)
    );
    let c = f
        .find_target_arc(i32::from(b'c'), &a, false)
        .unwrap()
        .unwrap();
    assert_eq!((c.output(), c.is_final()), (7, true));
    let b = f
        .find_target_arc(i32::from(b'b'), &first, false)
        .unwrap()
        .unwrap();
    assert!(b.is_final() && b.is_last());
    assert_eq!(
        f.find_target_arc(i32::from(b'A'), &first, false).unwrap(),
        None
    );
    assert_eq!(
        f.find_target_arc(i32::from(b'z'), &first, false).unwrap(),
        None
    );
    // A non-final stop arc ends at no node.
    let mut odd = read_order.to_vec();
    odd[4] = BIT_LAST_ARC | BIT_STOP_NODE;
    odd.reverse();
    let g = TokenInfoFst::read(&fst_file(0, None, 8, &odd), 0, 0).unwrap();
    let b = g
        .find_target_arc(i32::from(b'b'), &g.first_arc(), false)
        .unwrap()
        .unwrap();
    assert_eq!((b.is_final(), b.target()), (false, NON_FINAL_END_NODE));
}

#[test]
fn corrupt_files_fail_without_panicking() {
    let flags = BIT_FINAL_ARC | BIT_LAST_ARC | BIT_STOP_NODE | BIT_ARC_HAS_OUTPUT;
    let file = fst_file(1, None, 4, &[0u8, 5, b'a', 0, flags]);
    assert!(TokenInfoFst::read(&file, 0x100, 0).is_ok());
    for cut in 0..file.len() {
        let _ = TokenInfoFst::read(&file[..cut], 0x100, 0);
    }
    for i in 0..file.len() {
        let mut f = file.clone();
        f[i] ^= 0xFF;
        if let Ok(t) = TokenInfoFst::read(&f, 0x100, 0) {
            let _ = walk(&t, &units("a"));
        }
    }
    let bad_type = fst_file(7, None, 1, &[0]);
    let e = TokenInfoFst::read(&bad_type, 0, 0).unwrap_err();
    assert!(e.to_string().contains("invalid input type 7"), "{e}");
    let mut neg = header("FST", 9);
    neg.extend_from_slice(&[1, 0xFF, 0xFF, 0xFF, 0xFF, 0x0F]);
    assert!(TokenInfoFst::read(&neg, 0, 0).is_err());
    let start_outside = fst_file(0, None, 99, &[0, 0]);
    assert!(TokenInfoFst::read(&start_outside, 0x100, 0).is_err());
}

#[test]
fn bit_table_helpers() {
    // A table of two bytes 0b0000_0101, 0b1000_0000 at positions 1, 0
    // (read downward from 1).
    let bytes = [0b1000_0000u8, 0b0000_0101];
    let arc = FstArc {
        bit_table_start: 1,
        num_arcs: 16,
        ..FstArc::default()
    };
    let mut r = Reader {
        bytes: &bytes,
        pos: 0,
    };
    assert!(is_bit_set(0, &arc, &mut r).unwrap());
    assert!(!is_bit_set(1, &arc, &mut r).unwrap());
    assert!(is_bit_set(15, &arc, &mut r).unwrap());
    assert_eq!(count_bits(&arc, &mut r).unwrap(), 3);
    assert_eq!(count_bits_up_to(2, &arc, &mut r).unwrap(), 1);
    assert_eq!(count_bits_up_to(8, &arc, &mut r).unwrap(), 2);
    assert_eq!(count_bits_up_to(15, &arc, &mut r).unwrap(), 2);
    assert_eq!(num_presence_bytes(9), 2);
}

#[test]
fn readers_decode_labels_and_varints() {
    let bytes = [0x01u8, 0x02, 0x81, 0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0x7F];
    let mut r = Reader {
        bytes: &bytes,
        pos: 1,
    };
    assert_eq!(r.read_short().unwrap(), 0x0102);
    let mut r = Reader {
        bytes: &bytes,
        pos: 1,
    };
    let fst = |input_type, version| Fst {
        bytes: Vec::new(),
        input_type,
        version,
        start_node: 0,
        empty_output: None,
    };
    assert_eq!(fst(InputType::Byte2, 7).read_label(&mut r).unwrap(), 0x0201);
    let mut r = Reader {
        bytes: &[0x01, 0x81],
        pos: 1,
    };
    assert_eq!(fst(InputType::Byte4, 9).read_label(&mut r).unwrap(), 0x81);
    let five = [0x1F, 0xFF, 0xFF, 0xFF, 0xFF];
    let mut r = Reader {
        bytes: &five,
        pos: 4,
    };
    assert!(r.read_vint().is_err());
    let ok = [0x0F, 0xFF, 0xFF, 0xFF, 0xFF];
    let mut r = Reader { bytes: &ok, pos: 4 };
    assert_eq!(r.read_vint().unwrap(), -1);
    let ten = [0xFF; 10];
    let mut r = Reader {
        bytes: &ten,
        pos: 9,
    };
    assert!(r.read_vlong().is_err());
    let mut r = Reader {
        bytes: &ten,
        pos: -1,
    };
    assert!(r.read_byte().is_err());
}
