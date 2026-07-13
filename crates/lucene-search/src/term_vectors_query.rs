//! Query-facing read API over already-decoded term vectors (task #39).
//!
//! `crates/lucene-codecs/src/term_vectors.rs` (task #3's write side, task
//! #26's merge support) already fully decodes one document's term vectors
//! into [`lucene_codecs::term_vectors::TermVectorsDocument`] --
//! [`lucene_codecs::term_vectors::TermVectorField`] already carries, per
//! field, every term with its frequency and (whichever of these the field
//! had turned on at index time) positions, start/end character offsets, and
//! payloads. This module does not add any new byte-format decoding; it's a
//! thin adapter that (1) resolves a caller-friendly `(doc, field name)` pair
//! to the codec's `(doc, field number)` shape via [`FieldInfos`], and (2)
//! adds one small highlighting-adjacent primitive on top of data the codec
//! already produces.
//!
//! **Positions and payloads are also already decodable** (see
//! `term_vectors.rs`'s own tests), but this module's [`matched_term_offsets`]
//! only needs start/end offsets to compute highlight spans, so it doesn't
//! surface those here -- callers who need positions/payloads can read them
//! straight off [`lucene_codecs::term_vectors::TermVectorTerm`] via
//! [`term_vector_for_doc`]'s result.
//!
//! ## What's real "highlighting" support here, and what isn't
//!
//! [`matched_term_offsets`] computes character-offset spans for a set of
//! matched terms in one document's one field -- exactly the primitive a
//! `Highlighter`/`UnifiedHighlighter` needs to slice the original text at
//! match boundaries. It only works when that field was indexed with
//! `IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS` (the codec's
//! `has_offsets` flag) -- offsets are per-field, not always present, exactly
//! like real Lucene. There is no fragmenting, scoring of fragments, or
//! `PassageFormatter`-equivalent here: this is the offset-lookup primitive a
//! real highlighter is built on top of, not a highlighter itself. Building
//! an actual `UnifiedHighlighter`-equivalent (fragment selection, snippet
//! assembly, `BreakIterator`-style boundary snapping) is out of scope for
//! this task and left for a follow-up slice.

use lucene_codecs::field_infos::FieldInfos;
use lucene_codecs::term_vectors::{TermVectorField, TermVectorsReader};

use crate::Result;

/// One matched term's character-offset span within a document's field, as
/// computed by [`matched_term_offsets`] -- a named struct rather than a
/// `(String, i32, i32)` tuple, matching this crate's convention for
/// multi-field result items (e.g. [`crate::collector::ScoreDoc`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TermOffsetSpan {
    pub term: String,
    pub start_offset: i32,
    pub end_offset: i32,
}

/// Looks up one document's term vector for one field, by name.
///
/// - Returns `Ok(None)` when the field name isn't in `field_infos` at all,
///   when the document has no term vectors stored, or when it has term
///   vectors for other fields but not this one -- all "no term vector for
///   this (doc, field)" outcomes, not errors, mirroring real
///   `IndexReader.getTermVector(doc, field)`'s `null` return for the same
///   cases.
/// - Returns `Err` when `doc_id` is out of range for `reader`, or the
///   underlying `.tvd`/`.tvx` decode fails -- propagated straight from
///   [`TermVectorsReader::document`] via `?`, the same "let the codec's own
///   error surface, don't invent a second convention" approach every other
///   `lucene-search` query function already takes for its underlying codec
///   reader (see [`crate::search_term_query`]'s `blocktree::Error`
///   passthrough, or [`crate::doc_value_query`]'s `doc_values::Error`/
///   `norms::Error` passthrough).
pub fn term_vector_for_doc(
    reader: &TermVectorsReader<'_>,
    field_infos: &FieldInfos,
    doc_id: i32,
    field: &str,
) -> Result<Option<TermVectorField>> {
    let Some(field_info) = field_infos.fields.iter().find(|fi| fi.name == field) else {
        return Ok(None);
    };
    let Some(doc) = reader.document(doc_id)? else {
        return Ok(None);
    };
    Ok(doc
        .fields
        .into_iter()
        .find(|f| f.field_number == field_info.number))
}

/// Computes character-offset spans for `matched_terms`' occurrences in
/// `field`'s term vector -- the primitive a highlighter walks to slice the
/// original field text at match boundaries.
///
/// Returns `None` when `field` doesn't have offsets stored
/// (`field.has_offsets == false`) -- there is nothing to compute a span
/// from, and this deliberately does not fall back to guessing offsets from
/// positions or term order (that would be fabricating data this port never
/// decoded, exactly what the task's scoping guidance warns against).
///
/// When offsets are available, returns one [`TermOffsetSpan`] per occurrence
/// of any term in `matched_terms` (case-sensitive, matched against the
/// term's raw UTF-8 bytes), across all of that term's occurrences (a
/// repeated matched term contributes multiple spans), sorted ascending by
/// `start_offset` -- the order a highlighter walking a document
/// left-to-right wants.
pub fn matched_term_offsets(
    field: &TermVectorField,
    matched_terms: &[String],
) -> Option<Vec<TermOffsetSpan>> {
    if !field.has_offsets {
        return None;
    }
    let mut spans = Vec::new();
    for term in &field.terms {
        if !matched_terms.iter().any(|m| m.as_bytes() == term.term) {
            continue;
        }
        let term_str = String::from_utf8_lossy(&term.term).into_owned();
        let starts = term.start_offsets.as_deref().unwrap_or(&[]);
        let ends = term.end_offsets.as_deref().unwrap_or(&[]);
        for (&start_offset, &end_offset) in starts.iter().zip(ends.iter()) {
            spans.push(TermOffsetSpan {
                term: term_str.clone(),
                start_offset,
                end_offset,
            });
        }
    }
    spans.sort_by_key(|s| s.start_offset);
    Some(spans)
}

#[cfg(test)]
mod tests {
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
    // byte-format decoding, so per the `differential-testing` skill's
    // precedent for composition/wiring tasks (#36-38), no new Java fixture
    // was generated for this task -- these tests reuse the fixture task #3
    // already verified against real Lucene, the same real bytes
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
}
