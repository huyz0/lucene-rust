#![cfg(test)]
use super::*;
use lucene_codecs::field_infos;
use lucene_codecs::term_vectors;

// Reuses the same checked-in real-Lucene fixture the differential test
// in `crates/lucene-codecs/tests/term_vectors_fixtures.rs` opens (see
// that test / `fixtures/src/GenTermVectors.java` for its shape: doc 0 --
// field "text" with repeated terms, positions+offsets+payloads on some
// occurrences; doc 1 -- two fields "text"/"title"; doc 2 -- no term
// vectors at all). This module's own logic (field-name resolution,
// out-of-range propagation, span computation) doesn't touch any new
// byte-format decoding, so these tests reuse the fixture already verified
// against real Lucene, the same real bytes
// `term_vectors_fixtures.rs` differentially checks.
fn dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/term_vectors_index/"
    )
    .to_string()
}

struct Manifest {
    kv: Vec<(String, String)>,
}

impl Manifest {
    fn load() -> Self {
        let text = std::fs::read_to_string(format!("{}manifest.properties", dir()))
            .expect("run fixtures generator first (GenTermVectors)");
        let kv = text
            .lines()
            .filter_map(|l| l.split_once('='))
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        Manifest { kv }
    }

    fn get(&self, key: &str) -> &str {
        self.kv
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
            .unwrap_or_else(|| panic!("manifest key {key} missing"))
    }
}

fn id_from_hex(hex: &str) -> [u8; 16] {
    let mut id = [0u8; 16];
    for (i, byte) in id.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
    }
    id
}

fn open_fixture() -> (
    term_vectors::TermVectorsReader<'static>,
    field_infos::FieldInfos,
) {
    let manifest = Manifest::load();
    let id = id_from_hex(manifest.get("id_hex"));

    let tvd = std::fs::read(format!("{}{}.raw", dir(), manifest.get("tvd_file_name"))).unwrap();
    let tvx = std::fs::read(format!("{}{}.raw", dir(), manifest.get("tvx_file_name"))).unwrap();
    let tvm = std::fs::read(format!("{}{}.raw", dir(), manifest.get("tvm_file_name"))).unwrap();
    let fnm = std::fs::read(format!("{}_0.fnm", dir())).unwrap();

    // Leaked to get a `'static` borrow -- fine for a test-only helper
    // that runs once per test process.
    let tvd: &'static [u8] = Box::leak(tvd.into_boxed_slice());
    let tvx: &'static [u8] = Box::leak(tvx.into_boxed_slice());

    let reader = term_vectors::open(tvd, tvx, &tvm, &id, "").unwrap();
    let fis = field_infos::parse(&fnm, &id, "").unwrap();
    (reader, fis)
}

#[test]
fn resolves_field_by_name_and_returns_terms_with_offsets() {
    let (reader, fis) = open_fixture();

    let field = term_vector_for_doc(&reader, &fis, 0, "text")
        .unwrap()
        .expect("doc 0's \"text\" field has a term vector");
    assert!(field.has_offsets);
    // "car:1:1,4,7,bbcc;cat:2:0,0,3,aa:2,8,11,NONE" (manifest).
    assert_eq!(field.terms.len(), 2);
    assert_eq!(field.terms[0].term, b"car");
    assert_eq!(field.terms[0].start_offsets, Some(vec![4]));
    assert_eq!(field.terms[0].end_offsets, Some(vec![7]));
    assert_eq!(field.terms[1].term, b"cat");
    assert_eq!(field.terms[1].start_offsets, Some(vec![0, 8]));
    assert_eq!(field.terms[1].end_offsets, Some(vec![3, 11]));
}

#[test]
fn matched_term_offsets_finds_and_sorts_spans_from_real_fixture() {
    let (reader, fis) = open_fixture();
    let field = term_vector_for_doc(&reader, &fis, 0, "text")
        .unwrap()
        .unwrap();

    let spans = matched_term_offsets(&field, &["cat".to_string(), "car".to_string()]).unwrap();
    assert_eq!(
        spans,
        vec![
            TermOffsetSpan {
                term: "cat".to_string(),
                start_offset: 0,
                end_offset: 3
            },
            TermOffsetSpan {
                term: "car".to_string(),
                start_offset: 4,
                end_offset: 7
            },
            TermOffsetSpan {
                term: "cat".to_string(),
                start_offset: 8,
                end_offset: 11
            },
        ]
    );
}

#[test]
fn unmatched_term_yields_empty_spans() {
    let (reader, fis) = open_fixture();
    let field = term_vector_for_doc(&reader, &fis, 0, "text")
        .unwrap()
        .unwrap();

    let spans = matched_term_offsets(&field, &["fox".to_string()]).unwrap();
    assert!(spans.is_empty());
}

#[test]
fn missing_field_name_returns_none() {
    let (reader, fis) = open_fixture();
    assert!(term_vector_for_doc(&reader, &fis, 0, "no_such_field")
        .unwrap()
        .is_none());
}

#[test]
fn field_known_but_not_stored_for_this_doc_returns_none() {
    let (reader, fis) = open_fixture();
    // doc 0 only has a "text" term vector; "title" exists as a field
    // (doc 1 has it) but wasn't stored for doc 0.
    assert!(term_vector_for_doc(&reader, &fis, 0, "title")
        .unwrap()
        .is_none());
}

#[test]
fn doc_with_no_term_vectors_at_all_returns_none() {
    let (reader, fis) = open_fixture();
    // doc 2 has no term vectors stored for any field (manifest:
    // "doc.2.fields=NONE").
    assert!(term_vector_for_doc(&reader, &fis, 2, "text")
        .unwrap()
        .is_none());
}

#[test]
fn out_of_range_doc_id_is_an_error() {
    let (reader, fis) = open_fixture();
    assert!(matches!(
        term_vector_for_doc(&reader, &fis, 999, "text"),
        Err(crate::Error::TermVectors(_))
    ));
}

#[test]
fn spans_starting_at_the_same_offset_are_ordered_by_end_then_term() {
    // Real Lucene's `OffsetsEnum.compareTo` breaks a start-offset tie on
    // end offset, then on the term itself -- the ordering an
    // overlapping-token (synonym) field produces. Built directly, since
    // the checked-in fixture has no overlapping tokens.
    let field = TermVectorField {
        field_number: 0,
        has_positions: true,
        has_offsets: true,
        has_payloads: false,
        terms: vec![
            lucene_codecs::term_vectors::TermVectorTerm {
                term: b"zebra".to_vec(),
                freq: 1,
                positions: None,
                start_offsets: Some(vec![0]),
                end_offsets: Some(vec![5]),
                payloads: None,
            },
            lucene_codecs::term_vectors::TermVectorTerm {
                term: b"alpha".to_vec(),
                freq: 1,
                positions: None,
                start_offsets: Some(vec![0]),
                end_offsets: Some(vec![5]),
                payloads: None,
            },
            lucene_codecs::term_vectors::TermVectorTerm {
                term: b"mid".to_vec(),
                freq: 1,
                positions: None,
                start_offsets: Some(vec![0]),
                end_offsets: Some(vec![3]),
                payloads: None,
            },
        ],
    };
    let spans = matched_term_offsets(
        &field,
        &["zebra".to_string(), "alpha".to_string(), "mid".to_string()],
    )
    .unwrap();
    assert_eq!(
        spans
            .iter()
            .map(|s| (s.term.as_str(), s.start_offset, s.end_offset))
            .collect::<Vec<_>>(),
        vec![("mid", 0, 3), ("alpha", 0, 5), ("zebra", 0, 5)]
    );
}

#[test]
fn no_offsets_field_yields_none_from_matched_term_offsets() {
    let field = TermVectorField {
        field_number: 0,
        has_positions: true,
        has_offsets: false,
        has_payloads: false,
        terms: vec![],
    };
    assert!(matched_term_offsets(&field, &["cat".to_string()]).is_none());
}

#[test]
fn empty_matched_terms_yields_empty_spans_not_none() {
    let (reader, fis) = open_fixture();
    let field = term_vector_for_doc(&reader, &fis, 0, "text")
        .unwrap()
        .unwrap();
    let spans = matched_term_offsets(&field, &[]).unwrap();
    assert!(spans.is_empty());
}
