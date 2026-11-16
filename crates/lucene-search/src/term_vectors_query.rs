//! Query-facing read API over already-decoded term vectors.
//!
//! `crates/lucene-codecs/src/term_vectors.rs` fully decodes one document's
//! term vectors into [`lucene_codecs::term_vectors::TermVectorsDocument`] --
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
//! real highlighter is built on top of, not a highlighter itself.
//! `lucene_highlighter::highlighter` is the layer above -- it takes these spans plus the
//! original field text and does the passage assembly, `PassageScorer`-based
//! selection and `DefaultPassageFormatter` rendering (only the
//! `BreakIterator`-based boundary chooser stays a documented simplification
//! there).

use lucene_codecs::field_infos::FieldInfos;
use lucene_codecs::term_vectors::{TermVectorField, TermVectorsReader};

use crate::Result;

/// One matched term's offset span within a document's field, as computed by
/// [`matched_term_offsets`] -- a named struct rather than a
/// `(String, i32, i32)` tuple, matching this crate's convention for
/// multi-field result items (e.g. [`crate::collector::ScoreDoc`]).
///
/// **Unit: UTF-16 code units (Java `char`s).** These are `OffsetAttribute`'s
/// own numbers, decoded verbatim off the `.tvd`/`.pos` by `lucene-codecs`
/// and never reinterpreted here, and they are indices into the original
/// `String` the indexing-time analyzer saw -- which is what makes them UTF-16
/// rather than bytes or Unicode scalars. `lucene_highlighter::highlighter` slices the
/// stored text with them in that unit; see its module doc.
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
/// repeated matched term contributes multiple spans), in real Lucene's
/// `OffsetsEnum.compareTo` order: ascending `start_offset`, then ascending
/// `end_offset`, then ascending term. That full ordering (not just a
/// `start_offset` key) is what makes the result
/// deterministic when two different terms start at the same offset -- the
/// synonym/overlapping-token case a highlighter has to render consistently.
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
    spans.sort_by(|a, b| {
        a.start_offset
            .cmp(&b.start_offset)
            .then_with(|| a.end_offset.cmp(&b.end_offset))
            .then_with(|| a.term.cmp(&b.term))
    });
    Some(spans)
}

#[cfg(test)]
mod tests;
