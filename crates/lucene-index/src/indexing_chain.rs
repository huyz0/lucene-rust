//! An in-memory, indexing-side tokenize-and-invert builder: real Lucene's
//! `DocumentsWriterPerThread`/`IndexingChain`'s job of running each
//! document's indexed field text through an [`Analyzer`] and building an
//! in-memory inverted index (term -> per-doc positions/offsets), ready in
//! *shape* to be handed to a postings encoder.
//!
//! # Scope reality (read this before assuming more than it says)
//!
//! **This module is wired all the way through to a persisted, searchable
//! index.** `crate::index_writer::IndexWriter`'s private
//! `build_postings_output` helper calls [`invert_documents`] directly for
//! every field opted into postings via
//! `IndexWriter::set_postings_field`/`IndexWriter::add_postings_field`, then
//! feeds its output into
//! [`lucene_codecs::postings_writer::write_fields`], which produces real
//! `.doc`/`.tim`/`.tip`/`.tmd` bytes (and, for a field whose `index_options`
//! indexes positions/offsets, real `.pos`/`.pay` bytes too) that
//! `IndexWriter::commit` writes to `dir` like any other segment file. A
//! caller that adds real text via `IndexWriter::add_document`, opts a field
//! into `IndexOptions::DocsAndFreqsAndPositions` (or
//! `...AndPositionsAndOffsets`) postings, and commits ends up with a segment
//! `lucene_search`'s `PhraseQuery` can search correctly -- this is a genuine
//! add_document -> tokenize -> invert -> write -> commit -> search round
//! trip, not just an in-memory data structure.
//!
//! Payload bytes are wired through too:
//! [`invert_documents_with_payloads`] takes the stand-in for Lucene's
//! `PayloadAttribute` (see [`PayloadSource`]) and fills
//! [`PostingEntry::payloads`] for the fields whose `FieldInfo.store_payloads`
//! is set, which `IndexWriter::build_postings_output` forwards to
//! `postings_writer`'s `has_payloads`/`TermPostings::payloads`. What is still
//! not wired up from this module's output is term-vector-style per-document
//! random access to positions/offsets outside of the postings path (see
//! `crate::term_vectors` for that instead, which has its own, separate
//! indexing pass) -- see `docs/parity.md` for the exact current line.
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
use std::collections::{BTreeMap, HashMap};

/// One document's occurrence of a term within a single field: its position
/// (already position-increment-resolved by the analyzer, i.e. an absolute
/// position, not just an increment) and its offset span, passed through
/// opaquely from [`lucene_analysis::Token`].
///
/// The offsets are **UTF-16 code units** -- Java `char` indices into the field
/// text, exactly what `OffsetAttribute` reports, which is what real Lucene
/// reads back out of the `.pos`/`.pay`/`.tvd` this structure feeds. They were
/// UTF-8 byte offsets until c33; nothing here converts, so the unit is
/// whatever the analyzer emits, and it is pinned there
/// (`crates/lucene-analysis/tests/analysis_fixtures.rs`, the `utf16_*` cases).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Occurrence {
    pub position: i32,
    pub start_offset: i32,
    pub end_offset: i32,
}

/// One document's postings for one `(field, term)`: term frequency
/// (`occurrences.len()`) plus every occurrence's position and offsets, in
/// the order they occurred in the document.
///
/// `payloads` is the per-occurrence payload byte string real Lucene reads off
/// `PayloadAttribute` in `IndexingChain`'s `PerField.invert` loop. It is
/// either **empty** (this field does not store payloads -- the overwhelmingly
/// common case, and the one where paying 24 bytes per posting entry for an
/// always-empty `Vec` would be the wrong trade) or exactly parallel to
/// `occurrences`, with an empty `Vec<u8>` where an occurrence carried no
/// payload. That is the same "empty means no payload for this occurrence,
/// presence is a per-field property" convention
/// [`lucene_codecs::postings_writer::TermPostings::payloads`] documents, and
/// the same one Java uses (`Lucene104PostingsWriter.addPosition` treats a
/// `null` payload and a zero-length one identically).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PostingEntry {
    pub doc_id: i32,
    pub occurrences: Vec<Occurrence>,
    pub payloads: Vec<Vec<u8>>,
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

    /// Whether this entry carries a payload slot per occurrence -- i.e.
    /// whether the field it belongs to stores payloads at all. When true,
    /// `payloads.len() == occurrences.len()`; when false, `payloads` is empty.
    /// Those are the only two states [`invert_documents_with_payloads`] ever
    /// produces, and `postings_writer` rejects anything in between.
    pub fn has_payloads(&self) -> bool {
        !self.payloads.is_empty()
    }
}

/// One token's worth of context handed to a [`PayloadSource`], mirroring what
/// a real Lucene `TokenFilter` can see when it sets `PayloadAttribute` on the
/// token stream: the field being analysed, the document it is being analysed
/// for, and the token itself after position-increment resolution.
///
/// Passed by reference so a source that ignores most of it costs nothing, and
/// so adding a field here does not break existing sources.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PayloadContext<'a> {
    pub field: &'a str,
    pub term: &'a str,
    pub doc_id: i32,
    pub position: i32,
    pub start_offset: i32,
    pub end_offset: i32,
}

/// This port's stand-in for real Lucene's `PayloadAttribute`.
///
/// In Java a payload is not produced by the indexing chain at all: any
/// `TokenFilter` in the analyzer may call `PayloadAttribute.setPayload`, and
/// `IndexingChain`'s invert loop simply reads whatever the attribute holds for
/// the current token (`TermsHashPerField.writeProx`'s `payload` argument).
/// [`lucene_analysis::Token`] carries no payload attribute, so the supplier is
/// passed in here instead -- same layering, one indirection instead of an
/// attribute lookup: the analysis side decides the bytes, the indexing chain
/// only records them.
///
/// Returning `None` means "this token has no payload", which is what a `null`
/// `PayloadAttribute` means in Java, and is recorded as a zero-length entry
/// (see [`PostingEntry::payloads`]).
pub type PayloadSource<'a> = &'a dyn Fn(&PayloadContext<'_>) -> Option<Vec<u8>>;

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
    /// The exact heap this structure occupies, in bytes -- real Lucene's
    /// `Accountable.ramBytesUsed()` over what `IndexingChain`'s
    /// `TermsHash`/`BytesRefHash`/`ByteBlockPool` triple would hold for the same
    /// content.
    ///
    /// Counted, not estimated: every `BTreeMap` node slot, both key `String`s
    /// per term, the `Vec<PostingEntry>` per term, and the `Vec<Occurrence>`
    /// (plus, for a `store_payloads` field, the `Vec<Vec<u8>>` of payload slots
    /// and each slot's own bytes) per posting entry, each at its **capacity**
    /// rather than its length (an over-allocated `Vec` occupies its
    /// capacity).
    ///
    /// This is the number that makes the memory shape of a flush legible.
    /// Measured on `benchmarks/rust-runner`'s `index-bench` corpus (20k docs x
    /// 40 tokens drawn from a 20k-word vocabulary): **8.3 MB of document text
    /// becomes 78.5 MB here, 9.4x**. Almost all of it is per-occurrence: with
    /// that much term diversity nearly every `(doc, term)` pair is unique, so a
    /// ~6-byte token becomes a `(String, String)` key slot, a [`PostingEntry`],
    /// and a `Vec<Occurrence>` whose first `push` reserves capacity 4 -- 48
    /// bytes of allocation for 12 bytes of payload. Real Lucene pays *zero*
    /// heap objects per occurrence (a token becomes a few bytes in a
    /// `ByteBlockPool` slice), which is the divergence
    /// `docs/sweep/m2/LEDGER.md` records as the block-pool redesign. Handing
    /// the `Vec` surplus back with `shrink_to_fit` was tried and rejected: it
    /// cuts this figure to 5.98x but costs 25-60% indexing throughput and moves
    /// peak RSS not at all, because glibc keeps the freed 48-byte chunks in its
    /// arena.
    ///
    /// `BTreeMap`'s node overhead is charged as one `(K, V)` slot per entry;
    /// B-tree nodes are allocated in blocks of up to 11 entries, so the true
    /// figure is this plus a small, bounded per-node constant.
    // ARITH: every term added is the size of a live allocation (the per-entry
    // `(K, V)` slot charge is the documented approximation of `BTreeMap`'s node
    // overhead), so the running total is within a small constant factor of the
    // bytes this process actually holds -- a `usize` cannot overflow summing
    // sizes that fit in the address space. Written as
    // plain `+=` rather than `checked_add` because this walks every term in
    // the segment and is called per flush.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn ram_bytes_used(&self) -> usize {
        let mut bytes = std::mem::size_of::<Self>();
        for ((field, term), postings) in &self.terms {
            bytes += std::mem::size_of::<(TermKey, Vec<PostingEntry>)>();
            bytes += field.capacity() + term.capacity();
            bytes += postings.capacity() * std::mem::size_of::<PostingEntry>();
            for entry in postings {
                bytes += entry.occurrences.capacity() * std::mem::size_of::<Occurrence>();
                // Payload slots exist only for a `store_payloads` field, so
                // this is zero for the common case rather than a per-entry
                // constant. Each slot's own heap is counted too: a payload is
                // a byte string, and its bytes are as real as its header.
                bytes += entry.payloads.capacity() * std::mem::size_of::<Vec<u8>>();
                for payload in &entry.payloads {
                    bytes += payload.capacity();
                }
            }
        }
        bytes
    }

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
    invert_documents_with_payloads(docs, analyzer, &[], &|_| None)
}

/// `IndexWriter.MAX_POSITION`: the largest position a term occurrence may
/// carry. Java's value is `Integer.MAX_VALUE - 128`, leaving the postings
/// codecs the headroom they need for their own sentinels.
///
/// `check_index` carries a private copy of the same constant, for the read
/// side of the identical rule (`postings.positions_valid`). The two should be
/// one definition -- this is the writer's half, so this is the one to keep --
/// but that module is another batch's, so the consolidation is left as a
/// follow-up rather than done across the boundary.
pub const MAX_POSITION: i32 = i32::MAX - 128;

/// `IndexingChain.PerField.invert`'s `invertState.position += posIncr`, with
/// the two guards Java wraps around it.
///
/// Java detects the overflow *after the fact* (`if (invertState.position <
/// invertState.lastPosition)` — a wrapped `int` went backwards) and throws
/// `IllegalArgumentException("position overflowed Integer.MAX_VALUE")`, and
/// separately rejects anything past `IndexWriter.MAX_POSITION`. Neither guard
/// existed here, and in Rust the bare `+=` is not a wrap but a **panic** in a
/// debug build: a single field value long enough to carry 2^31 token
/// positions, or an analyzer chain handing back a large position increment,
/// takes the process down rather than the document.
///
/// [`invert_documents_with_payloads`] is infallible by signature (it returns
/// an [`InMemoryInvertedIndex`], not a `Result`), so the guard clamps at
/// [`MAX_POSITION`] rather than raising Java's exception. Clamping is the
/// conservative direction here and not a silent wrong answer: positions only
/// ever collapse *together* at the ceiling, so a phrase or span query can
/// return a false positive on a document past 2^31 positions, where a wrapped
/// negative position would instead be encoded as a garbage vint delta and
/// corrupt the `.pos` file for every document after it.
#[inline]
fn advance_position(position: i32, increment: i32) -> i32 {
    match position.checked_add(increment) {
        Some(next) if next <= MAX_POSITION => next,
        _ => MAX_POSITION,
    }
}

/// [`invert_documents`] plus payloads: `payload_fields` names the fields whose
/// `FieldInfo.store_payloads` is set (Lucene's per-field `hasPayloads`, which
/// is a field property, never a per-token one), and `source` supplies each
/// token's payload bytes the way a `PayloadAttribute`-setting `TokenFilter`
/// would.
///
/// Every occurrence of a **listed** field gets a payload slot -- an empty
/// `Vec<u8>` where `source` returned `None` -- so [`PostingEntry::payloads`]
/// is exactly parallel to [`PostingEntry::occurrences`] and satisfies
/// `postings_writer`'s `payloads[i].len() == freq` obligation without the
/// caller re-deriving it. A field not in `payload_fields` gets no slots and
/// pays nothing, which is why the gate is a parameter rather than something
/// `source` signals by returning `None`: "no payload on this token" and "this
/// field has no payloads" are different states on the wire, and only the
/// second omits `.pay`'s payload-length stream entirely.
///
/// `payload_fields` is scanned linearly once per (document, field), not once
/// per token -- a segment has a handful of payload fields at most, so a set
/// would cost more to build than the scan saves.
pub fn invert_documents_with_payloads(
    docs: &[(i32, &str, &str)],
    analyzer: &Analyzer,
    payload_fields: &[&str],
    source: PayloadSource<'_>,
) -> InMemoryInvertedIndex {
    // Accumulate in a hash map, not the result's `BTreeMap`, then build the
    // ordered map once at the end. The dictionary is touched once per
    // (document, term) occurrence group -- the dominant term in this
    // function's cost -- while it is *ordered* exactly once, at the end. A
    // `BTreeMap` here instead pays an O(log n) chain of `String` comparisons
    // on every one of those touches, which is the shape real Lucene
    // deliberately avoids too: `TermsHashPerField` accumulates through a
    // `BytesRefHash` (open-addressed, hash-keyed) and only sorts the term
    // dictionary when the segment is flushed.
    let mut acc: HashMap<TermKey, Vec<PostingEntry>> = HashMap::new();
    // Reused across documents so the per-document grouping map is allocated
    // once for the whole batch rather than once per (document, field).
    let mut per_term: HashMap<String, (Vec<Occurrence>, Vec<Vec<u8>>)> = HashMap::new();

    for &(doc_id, field, text) in docs {
        let tokens = analyzer.analyze(text);
        let field_has_payloads = payload_fields.contains(&field);

        // Resolve position increments to absolute positions and group by
        // term within this single (doc, field), matching real Lucene's
        // TermsHashPerField accumulating one PostingEntry per (doc, field,
        // term) even when a term occurs multiple times.
        per_term.clear();
        let mut position = -1i32;
        for token in tokens {
            position = advance_position(position, token.position_increment);
            let occurrence = Occurrence {
                position,
                start_offset: token.start_offset,
                end_offset: token.end_offset,
            };
            // Ask the source before the term `String` is moved into the map.
            // A field without payloads never calls it at all, so a source is
            // free to be expensive without taxing every other field.
            let payload = if field_has_payloads {
                source(&PayloadContext {
                    field,
                    term: &token.term,
                    doc_id,
                    position,
                    start_offset: occurrence.start_offset,
                    end_offset: occurrence.end_offset,
                })
                .unwrap_or_default()
            } else {
                Vec::new()
            };
            let slot = per_term.entry(token.term).or_default();
            slot.0.push(occurrence);
            if field_has_payloads {
                slot.1.push(payload);
            }
        }

        for (term, (occurrences, payloads)) in per_term.drain() {
            let key = (field.to_string(), term);
            acc.entry(key).or_default().push(PostingEntry {
                doc_id,
                occurrences,
                payloads,
            });
        }
    }

    // Enforce the doc-ID-sorted invariant directly, rather than trusting
    // callers to supply `docs` in ascending doc-ID order -- a stable sort
    // preserves each doc's own occurrence order when doc_ids happen to tie
    // (which can't happen across distinct documents, but keeps this
    // correct if a caller ever passes the same doc_id twice for one field).
    // Note this also un-does the hash map's arbitrary per-document iteration
    // order above: each (field, term) receives at most one entry per document,
    // so sorting by `doc_id` fully determines the list.
    let mut entries: Vec<(TermKey, Vec<PostingEntry>)> = acc.into_iter().collect();
    for (_, postings) in entries.iter_mut() {
        postings.sort_by_key(|entry| entry.doc_id);
    }
    entries.sort_unstable_by(|a, b| a.0.cmp(&b.0));

    InMemoryInvertedIndex {
        // `BTreeMap::from_iter` over already-sorted, deduplicated pairs builds
        // the tree bottom-up in O(n) rather than n O(log n) insertions.
        terms: entries.into_iter().collect(),
    }
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

    /// c33: the offsets this module forwards into `.pos`/`.pay`/`.tvd` are
    /// Java `char` indices, so a JVM reader that does
    /// `text.substring(startOffset, endOffset)` gets the term back.
    ///
    /// `invert_documents` does not convert anything -- it passes
    /// [`lucene_analysis::Token`]'s offsets straight through -- so this test
    /// is what makes that pass-through a checked claim rather than an assumed
    /// one, over text where UTF-8 bytes, Unicode scalars and UTF-16 code units
    /// all disagree.
    #[test]
    fn offsets_forwarded_to_the_codec_slice_the_source_text_as_java_chars() {
        let analyzer = Analyzer::standard(None);
        // "caf\u{E9}" is 4 chars / 5 bytes; "\u{4E16}" is 1 char / 3 bytes;
        // "\u{1D306}" is 2 chars / 1 scalar / 4 bytes.
        let text = "alpha caf\u{E9} \u{4E16} \u{1D306} omega";
        let docs = vec![(0, "body", text)];
        let index = invert_documents(&docs, &analyzer);

        // What a JVM caller does with an OffsetAttribute: index the text by
        // UTF-16 code unit.
        let units: Vec<u16> = text.encode_utf16().collect();
        let mut seen = 0;
        for ((_, term), entries) in &index.terms {
            for entry in entries {
                for occurrence in &entry.occurrences {
                    let (start, end) = (occurrence.start_offset, occurrence.end_offset);
                    assert!(start >= 0 && end >= start, "offsets out of order");
                    let slice = String::from_utf16(&units[start as usize..end as usize])
                        .expect("a Java `char` span of the source text");
                    assert_eq!(&slice, term, "offsets {start}..{end} do not name the term");
                    seen += 1;
                }
            }
        }
        assert_eq!(seen, 4, "expected alpha/caf\u{E9}/\u{4E16}/omega");
        // A byte-offset producer would have put "omega" at 21, not 16.
        assert_eq!(text.len(), 26);
        // Position 3, not 4: `U+1D306` is a symbol, so neither real Lucene's
        // `StandardTokenizer` (see the `utf16_astral_symbol` fixture case,
        // whose two tokens are one position apart across it) nor this port
        // emits a token for it, and a skipped run of text consumes no
        // position. The unit fix does not touch positions -- which is exactly
        // why one is asserted here alongside the offsets.
        assert_eq!(
            index.postings("body", "omega").expect("omega")[0].occurrences[0],
            occ(3, 16, 21)
        );
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
                    payloads: vec![],
                }][..]
            )
        );
        assert_eq!(
            index.postings("body", "quick"),
            Some(
                &[PostingEntry {
                    doc_id: 0,
                    occurrences: vec![occ(1, 4, 9)],
                    payloads: vec![],
                }][..]
            )
        );
        assert_eq!(
            index.postings("body", "fox"),
            Some(
                &[PostingEntry {
                    doc_id: 0,
                    occurrences: vec![occ(2, 10, 13)],
                    payloads: vec![],
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

    /// The byte count has to be a *count*, not a constant: adding a document
    /// with new terms must move it, and every component (keys, posting vectors,
    /// occurrence vectors) must be represented. Asserted as a lower bound
    /// derived from the content rather than an exact figure, since `Vec`
    /// capacity growth is an allocator/`RawVec` detail.
    #[test]
    fn ram_bytes_used_counts_keys_postings_and_occurrences() {
        let analyzer = Analyzer::standard(None);
        let empty = invert_documents(&[], &analyzer);
        assert_eq!(
            empty.ram_bytes_used(),
            std::mem::size_of::<InMemoryInvertedIndex>()
        );

        let one = invert_documents(&[(0, "body", "alpha beta")], &analyzer);
        let two = invert_documents(
            &[(0, "body", "alpha beta"), (1, "body", "gamma delta")],
            &analyzer,
        );
        assert!(one.ram_bytes_used() > empty.ram_bytes_used());
        assert!(
            two.ram_bytes_used() > one.ram_bytes_used(),
            "four terms must cost more than two: {} vs {}",
            two.ram_bytes_used(),
            one.ram_bytes_used()
        );

        // A lower bound built purely from the content: two terms, each with a
        // "body" key (4 bytes) plus the term bytes, one posting entry and one
        // occurrence.
        let floor = 2
            * (std::mem::size_of::<(TermKey, Vec<PostingEntry>)>()
                + 4
                + 5
                + std::mem::size_of::<PostingEntry>()
                + std::mem::size_of::<Occurrence>());
        assert!(
            one.ram_bytes_used() >= floor,
            "{} < content floor {floor}",
            one.ram_bytes_used()
        );

        // Repeating a term costs another `Occurrence` but no new key, so the
        // figure never *drops* -- and does not necessarily rise, because a
        // `Vec`'s second element usually fits in the capacity its first
        // allocation already reserved. Counting capacity rather than length is
        // the point: the memory is occupied either way.
        let repeated = invert_documents(&[(0, "body", "alpha beta alpha")], &analyzer);
        assert!(repeated.ram_bytes_used() >= one.ram_bytes_used());
    }

    /// A `store_payloads` field gets one payload slot per occurrence, in
    /// occurrence order, with a `None` from the source recorded as a
    /// zero-length entry -- Lucene's `null`-`PayloadAttribute` equivalent.
    /// The control is the second field in the same batch: a field not named
    /// in `payload_fields` gets no slots at all, so "payloads are present"
    /// cannot be something this function does unconditionally.
    #[test]
    fn payload_slots_are_filled_only_for_the_fields_that_declare_payloads() {
        let analyzer = Analyzer::standard(None);
        let docs = vec![(0, "body", "fox saw a fox"), (0, "title", "fox")];
        let index = invert_documents_with_payloads(&docs, &analyzer, &["body"], &|ctx| {
            // Every second position carries no payload, so the zero-length
            // case is exercised alongside the non-empty one.
            if ctx.position % 2 == 0 {
                Some(format!("{}:{}", ctx.term, ctx.position).into_bytes())
            } else {
                None
            }
        });

        let body = index.postings("body", "fox").unwrap();
        assert_eq!(body.len(), 1);
        assert!(body[0].has_payloads());
        assert_eq!(body[0].term_freq(), 2);
        assert_eq!(body[0].positions(), vec![0, 3]);
        // Position 0 is even (payload), position 3 is odd (none).
        assert_eq!(body[0].payloads, vec![b"fox:0".to_vec(), Vec::new()]);

        let title = index.postings("title", "fox").unwrap();
        assert!(!title[0].has_payloads());
        assert!(title[0].payloads.is_empty());
    }

    /// The context handed to a source is the whole token, not just its text:
    /// a source keyed on the document or the offsets has to be able to see
    /// them, and `field` is what lets one source serve several fields.
    #[test]
    fn the_payload_source_sees_the_field_document_position_and_offsets() {
        let analyzer = Analyzer::standard(None);
        let docs = vec![(7, "body", "the quick fox")];
        let seen = std::cell::RefCell::new(Vec::new());
        let index = invert_documents_with_payloads(&docs, &analyzer, &["body"], &|ctx| {
            seen.borrow_mut().push((
                ctx.field.to_string(),
                ctx.term.to_string(),
                ctx.doc_id,
                ctx.position,
                ctx.start_offset,
                ctx.end_offset,
            ));
            None
        });
        assert_eq!(
            seen.into_inner(),
            vec![
                ("body".into(), "the".into(), 7, 0, 0, 3),
                ("body".into(), "quick".into(), 7, 1, 4, 9),
                ("body".into(), "fox".into(), 7, 2, 10, 13),
            ]
        );
        // And the slots are still there, all empty.
        assert_eq!(
            index.postings("body", "fox").unwrap()[0].payloads,
            vec![Vec::<u8>::new()]
        );
    }

    /// `invert_documents` is the no-payloads shorthand, and must stay exactly
    /// that: no slots, and a source that would panic if it were ever called.
    #[test]
    fn invert_documents_records_no_payload_slots_at_all() {
        let analyzer = Analyzer::standard(None);
        let index = invert_documents(&[(0, "body", "fox")], &analyzer);
        assert!(index.postings("body", "fox").unwrap()[0]
            .payloads
            .is_empty());
        // Same batch through the payload entry point with an empty field
        // list: identical output, which is what makes the delegation safe.
        let same = invert_documents_with_payloads(&[(0, "body", "fox")], &analyzer, &[], &|_| {
            panic!("a field not in payload_fields must never reach the source")
        });
        assert_eq!(index, same);
    }

    /// Payload bytes are heap this structure occupies, so `ram_bytes_used`
    /// has to count them -- both the slot vector and each slot's own bytes.
    #[test]
    fn ram_bytes_used_counts_payload_slots_and_their_bytes() {
        let analyzer = Analyzer::standard(None);
        let docs = [(0, "body", "alpha beta")];
        let without = invert_documents(&docs, &analyzer);
        let with_empty = invert_documents_with_payloads(&docs, &analyzer, &["body"], &|_| None);
        let with_bytes =
            invert_documents_with_payloads(&docs, &analyzer, &["body"], &|_| Some(vec![0u8; 64]));

        assert!(
            with_empty.ram_bytes_used() > without.ram_bytes_used(),
            "the slot vector itself is real memory: {} vs {}",
            with_empty.ram_bytes_used(),
            without.ram_bytes_used()
        );
        assert!(
            with_bytes.ram_bytes_used() >= with_empty.ram_bytes_used() + 2 * 64,
            "64 payload bytes on each of two terms must show up: {} vs {}",
            with_bytes.ram_bytes_used(),
            with_empty.ram_bytes_used()
        );
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

    /// The position accumulator was a bare `position += posIncr` on an `i32`.
    /// Java guards it twice — it detects the wrap after the fact (`position <
    /// lastPosition`) and rejects anything past `IndexWriter.MAX_POSITION` —
    /// and neither guard was ported. In Rust the bare `+=` is not a wrap but a
    /// **panic** in a debug build, so one field value carrying 2^31 token
    /// positions takes the process down instead of the document.
    #[test]
    fn position_accumulator_clamps_instead_of_overflowing() {
        // The ordinary case is untouched: -1 is the pre-first-token seed, and
        // a standard increment of 1 lands the first token at position 0.
        assert_eq!(advance_position(-1, 1), 0);
        assert_eq!(advance_position(0, 1), 1);
        // A synonym's zero increment stays put.
        assert_eq!(advance_position(7, 0), 7);

        // Past the ceiling: clamped, never wrapped, and never negative.
        assert_eq!(advance_position(MAX_POSITION, 1), MAX_POSITION);
        assert_eq!(advance_position(MAX_POSITION - 1, 5), MAX_POSITION);
        assert_eq!(advance_position(i32::MAX - 1, 1), MAX_POSITION);
        assert_eq!(advance_position(i32::MAX, i32::MAX), MAX_POSITION);
        assert!(advance_position(MAX_POSITION, i32::MAX) >= 0);
    }
}
