//! An in-memory, indexing-side tokenize-and-invert builder: real Lucene's
//! `DocumentsWriterPerThread`/`IndexingChain`'s job of running each
//! document's indexed field text through an [`Analyzer`] and building an
//! in-memory inverted index (term -> per-doc positions/offsets), ready in
//! *shape* to be handed to a postings encoder -- but with no such encoder to
//! hand it to yet.
//!
//! # Scope reality (read this before assuming more than it says)
//!
//! This port's segment writer (`crate::segment_writer`) has **no write-side
//! postings encoder at all** -- confirmed by that module's own "What this
//! deliberately is not" doc comment, which states every flushed field is
//! `IndexOptions::None` (stored-only), because there is no reusable
//! write-side postings/doc-values/points/vectors format built yet. That
//! means there is nowhere in this port to *persist* a real inverted index
//! today.
//!
//! Given that, this module is scoped down honestly to exactly what's
//! buildable and valuable right now: an in-memory inverted-index **builder**
//! that takes a batch of documents' indexed field text, runs each through an
//! [`Analyzer`] (`lucene-analysis`, already wired into query-side analysis by
//! a prior task), and produces the in-memory data structure real Lucene's
//! `TermsHashPerField`/`FreqProxTermsWriterPerField` build while indexing a
//! document -- a term dictionary grouping postings by `(field, term)`, each
//! entry holding a doc-ID-sorted list of per-doc term frequency, positions,
//! and offsets.
//!
//! **This is real, testable work** (it is the exact tokenize-and-invert
//! logic a future postings writer will need as its input), but:
//!
//! - **Nothing downstream can consume it yet.** There is no code path from
//!   [`InMemoryInvertedIndex`] to any file on disk. A future postings writer
//!   (not yet built) is required before this becomes part of a real,
//!   persisted, searchable index.
//! - **This does NOT make documents indexed/searchable via analyzed text.**
//!   Nothing in `crate::segment_writer` reads from or writes this structure;
//!   flushing a segment today still only writes stored fields. Do not treat
//!   this module's existence as closing that gap -- it narrows the *design*
//!   gap (the shape of the eventual postings-writer input now exists and is
//!   tested) but not the *persistence* gap.
//!
//! # Why this output shape anticipates a future postings writer
//!
//! A real postings writer (`Lucene104PostingsWriter`, read-side ported in
//! `lucene_codecs::postings`) needs, per term, in ascending doc-ID order:
//! doc ID, term frequency, and (for `DOCS_AND_FREQS_AND_POSITIONS[_AND_OFFSETS]`
//! fields) each occurrence's position and character offset span. This module's
//! `Vec<PostingEntry>` (sorted by `doc_id`, each entry carrying `term_freq`
//! (`positions.len()`), `positions: Vec<i32>`, and `offsets: Vec<(i32, i32)>`
//! parallel to `positions`) carries exactly that information, grouped per
//! doc -- a future encoder can iterate `postings` in order without needing
//! to re-derive doc-ID ordering or re-group occurrences into a frequency
//! count. This is a row-oriented (per-doc) accumulator, not a structurally
//! identical match to `lucene_codecs::postings`' own read-side columnar
//! shape (`docs: Vec<i32>`, `freqs: Vec<i32>`, separately-decoded positions)
//! -- a future writer will still transform between the two, same as real
//! Lucene's own `TermsHashPerField` (also row/doc-oriented before final
//! encoding) does relative to its own on-disk columnar format.

use lucene_analysis::Analyzer;
use std::collections::BTreeMap;

/// One document's occurrence of a term within a single field: its position
/// (already position-increment-resolved by the analyzer, i.e. an absolute
/// position, not just an increment) and its offset span, passed through
/// opaquely from [`lucene_analysis::Token`] -- see that type's own doc
/// comment for a real, currently-latent unit caveat (these are UTF-8 byte
/// offsets, not character offsets, which matters once a future writer
/// persists this structure and something downstream assumes char offsets).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Occurrence {
    pub position: i32,
    pub start_offset: i32,
    pub end_offset: i32,
}

/// One document's postings for one `(field, term)`: term frequency
/// (`occurrences.len()`) plus every occurrence's position and offsets, in
/// the order they occurred in the document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PostingEntry {
    pub doc_id: i32,
    pub occurrences: Vec<Occurrence>,
}

impl PostingEntry {
    /// Real Lucene's per-doc term frequency for this term: the number of
    /// occurrences recorded, i.e. `occurrences.len()`.
    pub fn term_freq(&self) -> i32 {
        self.occurrences.len() as i32
    }

    /// This entry's positions, in occurrence order -- convenience view over
    /// `occurrences` matching the shape a positions-stream encoder wants.
    pub fn positions(&self) -> Vec<i32> {
        self.occurrences.iter().map(|o| o.position).collect()
    }

    /// This entry's `(start_offset, end_offset)` spans, in occurrence order,
    /// parallel to [`Self::positions`].
    pub fn offsets(&self) -> Vec<(i32, i32)> {
        self.occurrences
            .iter()
            .map(|o| (o.start_offset, o.end_offset))
            .collect()
    }
}

/// A `(field_name, term_bytes)` key, matching real Lucene's per-field term
/// dictionary: the same term text in two different fields is two distinct
/// entries, never merged.
pub type TermKey = (String, String);

/// The in-memory inverted index built by [`invert_documents`]: a term
/// dictionary keyed by `(field, term)`, each mapping to a doc-ID-sorted
/// posting list. Uses a [`BTreeMap`] so both the field/term ordering and
/// (via the doc-append order below) doc ordering are deterministic and
/// match real Lucene's sorted-term-dictionary iteration order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InMemoryInvertedIndex {
    pub terms: BTreeMap<TermKey, Vec<PostingEntry>>,
}

impl InMemoryInvertedIndex {
    /// Looks up the posting list for a `(field, term)` pair, if present.
    pub fn postings(&self, field: &str, term: &str) -> Option<&[PostingEntry]> {
        self.terms
            .get(&(field.to_string(), term.to_string()))
            .map(Vec::as_slice)
    }
}

/// Tokenizes and inverts a batch of documents' indexed field text via
/// `analyzer`, producing an [`InMemoryInvertedIndex`].
///
/// `docs` is `(doc_id, field_name, text)` triples: a document with multiple
/// indexed fields is represented as multiple entries sharing the same
/// `doc_id`; a batch with multiple documents is multiple `doc_id` values.
/// `docs` need not be sorted by `doc_id` or grouped by field, and need not
/// even be internally consistent about doc-ID order across fields --
/// this function sorts each `(field, term)` key's posting list by `doc_id`
/// itself before returning, so the doc-ID-sorted invariant genuinely holds
/// regardless of input order, rather than being a caller obligation to
/// uphold.
pub fn invert_documents(docs: &[(i32, &str, &str)], analyzer: &Analyzer) -> InMemoryInvertedIndex {
    let mut index = InMemoryInvertedIndex::default();

    for &(doc_id, field, text) in docs {
        let tokens = analyzer.analyze(text);

        // Resolve position increments to absolute positions and group by
        // term within this single (doc, field), matching real Lucene's
        // TermsHashPerField accumulating one PostingEntry per (doc, field,
        // term) even when a term occurs multiple times.
        let mut position = -1i32;
        let mut per_term: BTreeMap<String, Vec<Occurrence>> = BTreeMap::new();
        for token in tokens {
            position += token.position_increment;
            per_term.entry(token.term).or_default().push(Occurrence {
                position,
                start_offset: token.start_offset,
                end_offset: token.end_offset,
            });
        }

        for (term, occurrences) in per_term {
            let key = (field.to_string(), term);
            index.terms.entry(key).or_default().push(PostingEntry {
                doc_id,
                occurrences,
            });
        }
    }

    // Enforce the doc-ID-sorted invariant directly, rather than trusting
    // callers to supply `docs` in ascending doc-ID order -- a stable sort
    // preserves each doc's own occurrence order when doc_ids happen to tie
    // (which can't happen across distinct documents, but keeps this
    // correct if a caller ever passes the same doc_id twice for one field).
    for postings in index.terms.values_mut() {
        postings.sort_by_key(|entry| entry.doc_id);
    }

    index
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn occ(position: i32, start: i32, end: i32) -> Occurrence {
        Occurrence {
            position,
            start_offset: start,
            end_offset: end,
        }
    }

    #[test]
    fn single_doc_single_field_inverts_correctly() {
        let analyzer = Analyzer::standard(None);
        let docs = vec![(0, "body", "the quick fox")];
        let index = invert_documents(&docs, &analyzer);

        assert_eq!(
            index.postings("body", "the"),
            Some(
                &[PostingEntry {
                    doc_id: 0,
                    occurrences: vec![occ(0, 0, 3)],
                }][..]
            )
        );
        assert_eq!(
            index.postings("body", "quick"),
            Some(
                &[PostingEntry {
                    doc_id: 0,
                    occurrences: vec![occ(1, 4, 9)],
                }][..]
            )
        );
        assert_eq!(
            index.postings("body", "fox"),
            Some(
                &[PostingEntry {
                    doc_id: 0,
                    occurrences: vec![occ(2, 10, 13)],
                }][..]
            )
        );
        assert_eq!(index.terms.len(), 3);
    }

    #[test]
    fn multiple_docs_sharing_a_term_are_doc_id_sorted() {
        let analyzer = Analyzer::standard(None);
        let docs = vec![
            (0, "body", "fox"),
            (1, "body", "fox jumps"),
            (2, "body", "the fox runs"),
        ];
        let index = invert_documents(&docs, &analyzer);

        let postings = index.postings("body", "fox").unwrap();
        assert_eq!(postings.len(), 3);
        assert_eq!(postings[0].doc_id, 0);
        assert_eq!(postings[1].doc_id, 1);
        assert_eq!(postings[2].doc_id, 2);
    }

    /// The doc-ID-sorted invariant must hold even when `docs` is supplied
    /// out of order -- `invert_documents` sorts each term's posting list
    /// itself rather than trusting the caller to pre-sort. Passing input in
    /// REVERSE doc-ID order is the strongest possible proof this isn't just
    /// an artifact of already-ascending test fixtures.
    #[test]
    fn out_of_order_input_docs_still_produce_doc_id_sorted_postings() {
        let analyzer = Analyzer::standard(None);
        let docs = vec![
            (2, "body", "the fox runs"),
            (0, "body", "fox"),
            (1, "body", "fox jumps"),
        ];
        let index = invert_documents(&docs, &analyzer);

        let postings = index.postings("body", "fox").unwrap();
        assert_eq!(postings.len(), 3);
        assert_eq!(
            postings.iter().map(|e| e.doc_id).collect::<Vec<_>>(),
            vec![0, 1, 2],
            "postings must be doc-ID-sorted regardless of input order: {postings:?}"
        );
    }

    #[test]
    fn repeated_term_in_one_doc_has_correct_freq_and_all_positions() {
        let analyzer = Analyzer::standard(None);
        // "fox" occurs at positions 0 and 3 (0-indexed: fox=0, saw=1,
        // another=2, fox=3).
        let docs = vec![(0, "body", "fox saw another fox")];
        let index = invert_documents(&docs, &analyzer);

        let postings = index.postings("body", "fox").unwrap();
        assert_eq!(postings.len(), 1);
        let entry = &postings[0];
        assert_eq!(entry.doc_id, 0);
        assert_eq!(entry.term_freq(), 2);
        assert_eq!(entry.positions(), vec![0, 3]);
        assert_eq!(entry.offsets(), vec![(0, 3), (16, 19)]);
    }

    #[test]
    fn multiple_fields_on_same_doc_are_independent() {
        let analyzer = Analyzer::standard(None);
        let docs = vec![(0, "title", "fox"), (0, "body", "fox and hound")];
        let index = invert_documents(&docs, &analyzer);

        // Same term "fox" in two different fields must be two distinct
        // entries, not merged into one.
        assert_eq!(index.terms.len(), 4); // title/fox, body/fox, body/and, body/hound
        let title_fox = index.postings("title", "fox").unwrap();
        let body_fox = index.postings("body", "fox").unwrap();
        assert_eq!(title_fox.len(), 1);
        assert_eq!(body_fox.len(), 1);
        assert_eq!(title_fox[0].occurrences, vec![occ(0, 0, 3)]);
        assert_eq!(body_fox[0].occurrences, vec![occ(0, 0, 3)]);
        assert!(index.postings("title", "and").is_none());
    }

    #[test]
    fn stopword_filtered_text_excludes_stopword_preserves_positions() {
        let stopwords: HashSet<String> = ["the".to_string()].into_iter().collect();
        let analyzer = Analyzer::standard(Some(&stopwords));
        // "the quick fox": "the" removed, "quick" absorbs the skipped
        // position (position_increment 2), so "quick" lands at absolute
        // position 1 and "fox" at position 2 -- not 0/1, which would happen
        // if the stopword's position gap were silently dropped instead of
        // preserved.
        let docs = vec![(0, "body", "the quick fox")];
        let index = invert_documents(&docs, &analyzer);

        assert!(index.postings("body", "the").is_none());
        let quick = index.postings("body", "quick").unwrap();
        assert_eq!(quick[0].occurrences, vec![occ(1, 4, 9)]);
        let fox = index.postings("body", "fox").unwrap();
        assert_eq!(fox[0].occurrences, vec![occ(2, 10, 13)]);
    }

    #[test]
    fn empty_docs_batch_yields_empty_index() {
        let analyzer = Analyzer::standard(None);
        let index = invert_documents(&[], &analyzer);
        assert!(index.terms.is_empty());
    }

    #[test]
    fn postings_lookup_returns_none_for_unknown_term() {
        let analyzer = Analyzer::standard(None);
        let docs = vec![(0, "body", "fox")];
        let index = invert_documents(&docs, &analyzer);
        assert!(index.postings("body", "nonexistent").is_none());
        assert!(index.postings("nonexistent-field", "fox").is_none());
    }
}
