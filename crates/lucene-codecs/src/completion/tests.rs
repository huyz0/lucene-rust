#![cfg(test)]
use super::*;
use crate::fst_util::{get_bytes, TypedFst};

fn payload(surface: &str, weight: i32, type_byte: u8) -> Vec<u8> {
    let mut out = Vec::new();
    out.write_vint(surface.len() as i32);
    out.extend_from_slice(surface.as_bytes());
    out.write_vint(weight.wrapping_add(1));
    out.push(type_byte);
    out
}

fn doc(id: i32, surface: &str, weight: i32) -> TermDoc {
    TermDoc {
        doc_id: id,
        payloads: vec![payload(surface, weight, 0)],
    }
}

#[test]
fn weights_encode_to_the_complement_and_back() {
    assert_eq!(encode_weight(0).unwrap(), i64::from(i32::MAX));
    assert_eq!(encode_weight(i64::from(i32::MAX)).unwrap(), 0);
    assert_eq!(decode_weight(encode_weight(1234).unwrap()), 1234);
    assert!(encode_weight(-1).is_err());
    assert!(encode_weight(i64::from(i32::MAX) + 1).is_err());
}

#[test]
fn payload_is_surface_sep_and_vint_doc() {
    let p = make_payload(b"caf\xc3\xa9", 300);
    assert_eq!(&p[..5], b"caf\xc3\xa9");
    assert_eq!(p[5], PAYLOAD_SEP);
    assert_eq!(&p[6..], &[0xAC, 0x02]);
    assert_eq!(payload_sep_index(&p, PAYLOAD_SEP), Some(5));
    assert_eq!(payload_sep_index(b"none", PAYLOAD_SEP), None);
}

#[test]
fn dedup_bytes_grow_with_the_number_of_dedup_bytes() {
    assert_eq!(max_num_arcs_for_dedup_byte(1), 3);
    assert_eq!(max_num_arcs_for_dedup_byte(5), 11);
    assert_eq!(max_num_arcs_for_dedup_byte(6), 13 * 6);
    assert_eq!(max_num_arcs_for_dedup_byte(7), 15 * 7);
    assert_eq!(max_num_arcs_for_dedup_byte(200), 255);
    assert_eq!(max_num_arcs_for_dedup_byte(20), 255);
}

#[test]
fn hash_map_order_is_by_bucket_then_insertion() {
    // "a" = 97 -> bucket 1, "b" -> 2, "p" = 112 -> 0
    assert_eq!(java_hash_map_order(&["b", "a", "p"]), vec![2, 1, 0]);
    assert_eq!(java_hash_map_order(&[]), Vec::<usize>::new());
    // "q" = 113 shares bucket 1 with "a" in 16 slots but not in 32: the
    // thirteenth key resizes the table, so "q" moves behind "l"
    let mut keys = vec!["q".to_string()];
    keys.extend((b'a'..=b'l').map(|c| (c as char).to_string()));
    let refs: Vec<&str> = keys.iter().map(String::as_str).collect();
    let want: Vec<usize> = (1..13).chain([0]).collect();
    assert_eq!(java_hash_map_order(&refs), want);
    assert_eq!(java_hash_map_order(&refs[..12])[..2], [0, 1]);
}

#[test]
fn a_field_round_trips_through_both_files() {
    let id = [7u8; ID_LENGTH];
    let mut w = CompletionFieldWriter::new();
    w.write_term(b"cat", &[doc(0, "Cat", 5), doc(3, "cat", 9)])
        .unwrap();
    w.write_term(b"dog", &[doc(1, "Dog", 2)]).unwrap();
    let mut files = CompletionFilesWriter::new("Completion104", "Lucene104", id, "Completion104_0");
    files.add_field("suggest", 4, w).unwrap();
    // a field that held nothing leaves no entry
    files
        .add_field("empty", 5, CompletionFieldWriter::new())
        .unwrap();
    let (lkp, cmp) = files.finish();

    let opened =
        CompletionFields::open("Completion104", &id, "Completion104_0", &cmp, lkp).unwrap();
    assert_eq!(opened.delegate_format, "Lucene104");
    assert_eq!(opened.fields.len(), 1);
    let (number, meta) = opened.fields[0];
    assert_eq!(number, 4);
    assert_eq!(
        (meta.min_weight, meta.max_weight, meta.type_byte),
        (2, 9, 0)
    );
    assert!(opened.suggester(5).unwrap().is_none());
    let s = opened.suggester(4).unwrap().unwrap();
    assert_eq!(s.payload_sep, i32::from(PAYLOAD_SEP));
    assert_eq!(s.max_analyzed_paths_per_output, 2);
    // "dog" + END_BYTE + dedup byte 0
    let out = get_bytes::<NrtOutputs>(&s.fst, b"dog\x00\x00")
        .unwrap()
        .unwrap();
    assert_eq!(decode_weight(out.first), 2);
    assert_eq!(out.second, make_payload(b"Dog", 1));
    let _: &TypedFst<NrtOutputs> = &s.fst;
}

#[test]
fn open_refuses_a_wrong_segment_or_a_damaged_table() {
    let id = [1u8; ID_LENGTH];
    let mut w = CompletionFieldWriter::new();
    w.write_term(b"a", &[doc(0, "a", 1)]).unwrap();
    let mut files = CompletionFilesWriter::new("Completion104", "Lucene104", id, "S");
    files.add_field("f", 0, w).unwrap();
    let (lkp, cmp) = files.finish();
    let other = [2u8; ID_LENGTH];
    assert!(CompletionFields::open("Completion104", &other, "S", &cmp, lkp.clone()).is_err());
    assert!(CompletionFields::open("Completion104", &id, "T", &cmp, lkp.clone()).is_err());
    assert!(CompletionFields::open("Other", &id, "S", &cmp, lkp.clone()).is_err());
    let mut bad = cmp.clone();
    let n = bad.len();
    bad[n - 10] ^= 0xFF;
    assert!(CompletionFields::open("Completion104", &id, "S", &bad, lkp.clone()).is_err());
    assert!(CompletionFields::open("Completion104", &id, "S", &cmp, lkp[..8].to_vec()).is_err());
}

#[test]
fn mixed_types_and_bad_weights_are_refused() {
    let mut w = CompletionFieldWriter::new();
    w.write_term(
        b"a",
        &[TermDoc {
            doc_id: 0,
            payloads: vec![payload("a", 1, 0), payload("a", 2, 1)],
        }],
    )
    .unwrap_err();
    // a surface form whose weight does not fit
    let mut w = CompletionFieldWriter::new();
    let bad = TermDoc {
        doc_id: 0,
        payloads: vec![{
            let mut p = Vec::new();
            p.write_vint(1);
            p.push(b'a');
            p.write_vint(0); // weight -1
            p.push(0);
            p
        }],
    };
    assert!(w.write_term(b"a", &[bad]).is_err());
}

#[test]
fn load_refuses_a_non_positive_path_count() {
    let mut w = CompletionFieldWriter::new();
    w.write_term(b"a", &[doc(0, "a", 1)]).unwrap();
    let mut out = Vec::new();
    assert!(w.finish(&mut out).unwrap());
    // the three trailing vints: paths per output, END_BYTE, PAYLOAD_SEP
    let n = out.len();
    out[n - 3] = 0;
    assert!(NrtSuggester::load(&out).is_err());
}

#[test]
fn many_surface_forms_of_one_term_get_dedup_bytes() {
    let mut w = CompletionFieldWriter::new();
    let docs: Vec<TermDoc> = (0..40).map(|i| doc(i, "x", i)).collect();
    w.write_term(b"x", &docs).unwrap();
    let mut out = Vec::new();
    assert!(w.finish(&mut out).unwrap());
    let s = NrtSuggester::load(&out).unwrap();
    assert_eq!(s.max_analyzed_paths_per_output, 40);
}
