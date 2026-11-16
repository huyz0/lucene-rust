#![cfg(test)]
use super::super::term_vectors_writer::{
    term_vectors_file_name, write, SimpleTextTermVectorsWriter,
};
use super::*;
use crate::field_infos::{FieldInfo, FieldInfos, IndexOptions};
use crate::term_vectors::TermVectorsDocument;

fn infos() -> FieldInfos {
    let f = |name: &str, n: i32| {
        FieldInfo::new(name, n)
            .with_index_options(IndexOptions::DocsAndFreqsAndPositionsAndOffsets)
            .with_store_term_vectors(true)
    };
    FieldInfos::new(vec![f("b", 0), f("a", 1)]).unwrap()
}

fn term(t: &[u8], pos: bool, off: bool, pay: bool) -> TermVectorTerm {
    TermVectorTerm {
        term: t.to_vec(),
        freq: 2,
        positions: pos.then(|| vec![1, 4]),
        start_offsets: off.then(|| vec![0, 9]),
        end_offsets: off.then(|| vec![3, 12]),
        payloads: pay.then(|| vec![vec![], b"\n\\".to_vec()]),
    }
}

fn field(n: i32, pos: bool, off: bool, pay: bool, terms: Vec<TermVectorTerm>) -> TermVectorField {
    TermVectorField {
        field_number: n,
        has_positions: pos,
        has_offsets: off,
        has_payloads: pay,
        terms,
    }
}

fn docs() -> Vec<TermVectorsDocument> {
    vec![
        TermVectorsDocument {
            fields: vec![
                field(0, true, false, true, vec![term(b"x", true, false, true)]),
                field(1, false, true, false, vec![term(b"m", false, true, false)]),
            ],
        },
        TermVectorsDocument::default(),
        TermVectorsDocument {
            fields: vec![field(
                1,
                false,
                false,
                false,
                vec![
                    term(b"q", false, false, false),
                    term(b"z", false, false, false),
                ],
            )],
        },
    ]
}

#[test]
fn documents_read_back_with_fields_by_name() {
    let data = write(&docs(), &infos()).unwrap();
    let r = SimpleTextTermVectorsReader::open(&data, 3).unwrap();
    let metadata = r.rebind(&[]);
    let r = metadata.rebind(&data);
    assert!(Arc::ptr_eq(&metadata.offsets, &r.offsets));
    assert!(metadata.get(0).is_err());
    assert_eq!(r.max_doc(), 3);
    let d0 = r.get(0).unwrap().unwrap();
    assert_eq!(
        d0.iter().map(|f| f.0.as_str()).collect::<Vec<_>>(),
        ["a", "b"]
    );
    assert_eq!(d0[1].1, docs()[0].fields[0]);
    assert_eq!(d0[0].1, docs()[0].fields[1]);
    assert!(r.get(1).unwrap().is_none());
    assert!(r.get(3).is_err() && r.get(-1).is_err());
    assert_eq!(term_vectors_file_name("_0"), "_0.tvc");
}

#[test]
fn terms_enum_and_postings() {
    let data = write(&docs(), &infos()).unwrap();
    let r = SimpleTextTermVectorsReader::open(&data, 3).unwrap();
    let d0 = r.get(0).unwrap().unwrap();
    // Positions and payloads, no offsets.
    let mut te = SimpleTextTvTermsEnum::new(&d0[1].1);
    assert!(
        te.term().is_none() && te.total_term_freq().is_err() && te.postings(POSITIONS).is_err()
    );
    assert_eq!(te.next(), Some(&b"x"[..]));
    assert_eq!((te.doc_freq(), te.total_term_freq().unwrap()), (1, 2));
    let mut p = te.postings(POSITIONS).unwrap();
    assert_eq!((p.doc_id(), p.cost()), (-1, 1));
    assert_eq!(p.payload(), None, "before the first position");
    assert_eq!(p.next_doc(), 0);
    assert_eq!(p.freq(), 2);
    assert_eq!(p.next_position().unwrap(), 1);
    assert_eq!(
        (p.start_offset(), p.end_offset(), p.payload()),
        (-1, -1, None)
    );
    assert_eq!(p.next_position().unwrap(), 4);
    assert_eq!(p.payload(), Some(&b"\n\\"[..]));
    assert!(p.next_position().is_err());
    assert_eq!(p.next_doc(), NO_MORE_DOCS);
    assert_eq!(te.next(), None);
    // Offsets only: positions read -1.
    let mut te = SimpleTextTvTermsEnum::new(&d0[0].1);
    assert_eq!(te.seek_ceil(b"m"), SeekStatus::Found);
    let mut p = te.postings(POSITIONS).unwrap();
    assert_eq!(p.advance(0), 0);
    assert_eq!(p.next_position().unwrap(), -1);
    assert_eq!((p.start_offset(), p.end_offset()), (0, 3));
    assert_eq!(te.seek_ceil(b"a"), SeekStatus::NotFound);
    assert_eq!(te.seek_ceil(b"n"), SeekStatus::End);
    // Freqs only; and freq 1 without FREQS.
    let d2 = r.get(2).unwrap().unwrap();
    let mut te = SimpleTextTvTermsEnum::new(&d2[0].1);
    assert_eq!(te.seek_ceil(b"r"), SeekStatus::NotFound);
    assert_eq!(te.term(), Some(&b"z"[..]));
    let mut p = te.postings(POSITIONS).unwrap();
    assert_eq!(p.freq(), 2);
    assert_eq!(p.next_position().unwrap(), -1);
    assert_eq!(
        (p.start_offset(), p.end_offset(), p.payload()),
        (-1, -1, None)
    );
    assert_eq!(p.advance(5), NO_MORE_DOCS);
    assert_eq!(te.postings(0).unwrap().freq(), 1);
    assert_eq!(te.postings(FREQS).unwrap().freq(), 2);
}

#[test]
fn bad_files_and_counts_are_refused() {
    let data = write(&docs(), &infos()).unwrap();
    assert!(
        SimpleTextTermVectorsReader::open(&data, 2).is_err(),
        "more than maxDoc"
    );
    assert!(
        SimpleTextTermVectorsReader::open(&data, 4).is_err(),
        "fewer"
    );
    assert!(SimpleTextTermVectorsReader::open(&data, -1).is_err());
    assert!(
        SimpleTextTermVectorsReader::open(b"doc 0\n", 1).is_err(),
        "no END"
    );
    let mut damaged = data.clone();
    let at = damaged.iter().position(|&b| b == b'x').unwrap();
    damaged[at] = b'y';
    assert!(
        SimpleTextTermVectorsReader::open(&damaged, 3).is_err(),
        "checksum"
    );
    // A negative freq with positions.
    let mut w = SimpleTextTermVectorsWriter::new();
    let info = infos();
    w.start_document(1);
    w.start_field(info.field_by_number(0).unwrap(), 1, true, false, false);
    w.start_term(b"t", -1);
    let data = w.finish(1).unwrap();
    let r = SimpleTextTermVectorsReader::open(&data, 1).unwrap();
    assert!(r.get(0).is_err());
    // A malformed numfields line.
    let text = String::from_utf8(data)
        .unwrap()
        .replacen("numfields 1", "numfields x", 1);
    let mut resealed = text.as_bytes()[..text.rfind("checksum ").unwrap()].to_vec();
    util::write_checksum(&mut resealed);
    let r = SimpleTextTermVectorsReader::open(&resealed, 1).unwrap();
    assert!(r.get(0).is_err());
}

#[test]
fn repeated_terms_and_fields_keep_the_last_value() {
    let info = infos();
    let mut w = SimpleTextTermVectorsWriter::new();
    w.start_document(2);
    w.start_field(info.field_by_number(0).unwrap(), 3, false, false, false);
    for (t, freq) in [(&b"b"[..], 1), (b"a", 2), (b"b", 3)] {
        w.start_term(t, freq);
    }
    w.start_field(info.field_by_number(0).unwrap(), 1, false, false, false);
    w.start_term(b"z", 4);
    let data = w.finish(1).unwrap();
    let r = SimpleTextTermVectorsReader::open(&data, 1).unwrap();
    let doc = r.get(0).unwrap().unwrap();
    assert_eq!(doc.len(), 1, "one field named b");
    let terms: Vec<_> = doc[0]
        .1
        .terms
        .iter()
        .map(|t| (t.term.clone(), t.freq))
        .collect();
    assert_eq!(terms, [(b"z".to_vec(), 4)], "the second `b` field wins");
    let mut w = SimpleTextTermVectorsWriter::new();
    w.start_document(1);
    w.start_field(info.field_by_number(0).unwrap(), 3, false, false, false);
    for (t, freq) in [(&b"b"[..], 1), (b"a", 2), (b"b", 3)] {
        w.start_term(t, freq);
    }
    let data = w.finish(1).unwrap();
    let r = SimpleTextTermVectorsReader::open(&data, 1).unwrap();
    let doc = r.get(0).unwrap().unwrap();
    let terms: Vec<_> = doc[0]
        .1
        .terms
        .iter()
        .map(|t| (t.term.clone(), t.freq))
        .collect();
    assert_eq!(terms, [(b"a".to_vec(), 2), (b"b".to_vec(), 3)]);
    let mut v = vec![(2, 'x'), (1, 'y'), (2, 'z'), (1, 'w')];
    sort_keep_last(&mut v, |a, b| a.0.cmp(&b.0));
    assert_eq!(v, [(1, 'w'), (2, 'z')]);
}

#[test]
fn the_writer_refuses_bad_input() {
    let mut w = SimpleTextTermVectorsWriter::new();
    w.start_document(0);
    assert!(matches!(w.finish(2), Err(Error::Runtime(_))));
    let mut bad = docs();
    bad[0].fields[0].field_number = 9;
    assert!(write(&bad, &infos()).is_err(), "unknown field");
    let mut bad = docs();
    bad[0].fields[0].terms[0].positions = Some(vec![1]);
    assert!(write(&bad, &infos()).is_err(), "short occurrences");
}
