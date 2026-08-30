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
//! [`TermPostingList`]'s flat payload run for the fields whose
//! `FieldInfo.store_payloads` is set, which
//! `IndexWriter::build_postings_output` moves straight into
//! `postings_writer`'s `has_payloads`/`TermPostings::payload_bytes`. What is still
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
use lucene_codecs::postings_writer;
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
/// **Payloads are not here.** They live once per `(field, term)`, on
/// [`TermPostingList`], because a payload slot per posting entry is a heap
/// object per posting entry: c23 measured the nested
/// `Vec<Vec<u8>>`-per-entry shape this used to have at 26 us/doc and ~190 MB
/// per 50 000 documents, with an all-empty-payload control costing the same,
/// which is what identifies the *slot* rather than the bytes as the cost.
/// Java pays neither -- `FreqProxTermsWriterPerField.writeProx` appends the payload
/// length and its bytes into the term's existing `ByteSlicePool` stream, so
/// a payload costs bytes in a pool and no object at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PostingEntry {
    pub doc_id: i32,
    pub occurrences: Vec<Occurrence>,
}

/// One `(field, term)` key's whole posting list: the doc-ID-sorted
/// [`PostingEntry`]s, plus -- for a field whose `FieldInfo.store_payloads` is
/// set -- every occurrence's payload bytes in one flat run.
///
/// `payload_bytes` is the concatenation of every occurrence's payload, in
/// `entries` order and then occurrence order within each entry;
/// `payload_lengths` is one length per occurrence in the same order, so
/// occurrence `k` of entry `i` has length `payload_lengths[o + k]` where `o`
/// is the number of occurrences in entries `0..i`. A zero length is a real
/// state, not an absent one: Java treats a `null` `PayloadAttribute` and a
/// zero-length one identically (`Lucene104PostingsWriter.addPosition`), and
/// payload *presence* is a per-field property (`FieldInfo.hasPayloads()`),
/// never a per-occurrence one.
///
/// Both vectors are **empty** for a field that does not store payloads --
/// the overwhelmingly common case, which then pays nothing per occurrence and
/// two empty `Vec` headers per term rather than per posting entry. This is
/// the same flat shape
/// [`lucene_codecs::postings_writer::TermPostings::payload_bytes`] takes, so
/// `IndexWriter::build_postings_output` moves the two vectors across without
/// copying or re-materializing anything.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TermPostingList {
    pub entries: Vec<PostingEntry>,
    pub payload_bytes: Vec<u8>,
    pub payload_lengths: Vec<u32>,
}

impl TermPostingList {
    /// Whether this posting list carries a payload slot per occurrence --
    /// i.e. whether the field it belongs to stores payloads at all. When
    /// true, `payload_lengths.len()` is the sum of every entry's
    /// `occurrences.len()`; when false, both payload vectors are empty.
    /// Those are the only two states [`invert_documents_with_payloads`] ever
    /// produces, and `postings_writer` rejects anything in between.
    pub fn has_payloads(&self) -> bool {
        !self.payload_lengths.is_empty()
    }

    /// Occurrence `k` of entry `i`'s payload bytes, or `None` when this list
    /// carries no payloads. Reconstructs the nested view the flat run
    /// replaces, for a caller that wants one occurrence rather than the run.
    /// Nothing in the write path does -- `build_postings_output` moves the run
    /// whole -- so today this exists for the tests that assert the run means
    /// what it says.
    ///
    /// O(number of occurrences before it), because the flat run stores
    /// lengths rather than offsets -- exactly as
    /// `Lucene104PostingsWriter`'s own payload byte run does, and for the
    /// same reason: the writer consumes it strictly in order.
    pub fn payload(&self, entry_index: usize, occurrence: usize) -> Option<&[u8]> {
        if !self.has_payloads() {
            return None;
        }
        let before: usize = self
            .entries
            .get(..entry_index)?
            .iter()
            .map(|e| e.occurrences.len())
            .sum();
        let index = before.checked_add(occurrence)?;
        if occurrence >= self.entries.get(entry_index)?.occurrences.len() {
            return None;
        }
        let start: usize = self
            .payload_lengths
            .get(..index)?
            .iter()
            .map(|&l| l as usize)
            .sum();
        let len = *self.payload_lengths.get(index)? as usize;
        self.payload_bytes.get(start..start.checked_add(len)?)
    }

    /// Sorts `entries` by `doc_id`, carrying the flat payload run with them.
    ///
    /// The check comes first because the sorted case is the only one
    /// `IndexWriter` produces (it inverts in ascending doc-ID order), and
    /// permuting the run means a prefix sum over every occurrence. The
    /// unsorted case is real -- `invert_documents*` takes `(doc_id, field,
    /// text)` triples in any order and says so -- and is what
    /// `sorting_entries_by_doc_id_carries_the_payload_run_with_them` drives.
    /// A stable sort preserves each doc's own occurrence order when `doc_id`s
    /// tie.
    fn sort_by_doc_id(&mut self) {
        if self.entries.windows(2).all(|w| w[0].doc_id <= w[1].doc_id) {
            return;
        }
        if !self.has_payloads() {
            self.entries.sort_by_key(|entry| entry.doc_id);
            return;
        }
        let counts: Vec<u32> = self
            .entries
            .iter()
            .map(|e| e.occurrences.len() as u32)
            .collect();
        let mut order: Vec<usize> = (0..self.entries.len()).collect();
        order.sort_by_key(|&i| self.entries[i].doc_id);
        let (bytes, lengths) = postings_writer::permute_payload_run(
            &self.payload_bytes,
            &self.payload_lengths,
            &counts,
            &order,
        );
        self.payload_bytes = bytes;
        self.payload_lengths = lengths;
        // The same permutation applied to `entries`, moving rather than
        // re-sorting, so the two cannot disagree about which order was used.
        let mut entries: Vec<PostingEntry> = Vec::with_capacity(self.entries.len());
        let mut taken: Vec<Option<PostingEntry>> = self.entries.drain(..).map(Some).collect();
        for &i in &order {
            entries.push(taken[i].take().expect("each index appears once in `order`"));
        }
        self.entries = entries;
    }
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
/// the current token (`FreqProxTermsWriterPerField.writeProx`'s `payload` argument).
/// [`lucene_analysis::Token`] carries no payload attribute, so the supplier is
/// passed in here instead -- same layering, one indirection instead of an
/// attribute lookup: the analysis side decides the bytes, the indexing chain
/// only records them.
///
/// Returning `None` means "this token has no payload", which is what a `null`
/// `PayloadAttribute` means in Java, and is recorded as a zero-length run
/// (see [`TermPostingList::payload_lengths`]).
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
    pub terms: BTreeMap<TermKey, TermPostingList>,
}

impl InMemoryInvertedIndex {
    /// The exact heap this structure occupies, in bytes -- real Lucene's
    /// `Accountable.ramBytesUsed()` over what `IndexingChain`'s
    /// `TermsHash`/`BytesRefHash`/`ByteBlockPool` triple would hold for the same
    /// content.
    ///
    /// Counted, not estimated: every `BTreeMap` node slot, both key `String`s
    /// per term, the [`TermPostingList`] per term (its `Vec<PostingEntry>`
    /// and, for a `store_payloads` field, its two flat payload vectors), and
    /// the `Vec<Occurrence>` per posting entry, each at its **capacity**
    /// rather than its length (an over-allocated `Vec` occupies its
    /// capacity).
    ///
    /// This is the number that makes the memory shape of a flush legible.
    /// Measured on `benchmarks/rust-runner`'s `index-bench` corpus (20k docs x
    /// 40 tokens drawn from a 20k-word vocabulary, 4.90 MB of body text):
    /// **102.5 MB here before c38, 75.8 MB after**, against Java's *zero*
    /// heap objects per occurrence (a token becomes a few bytes in a
    /// `ByteBlockPool` slice). What is left is per-occurrence and structural:
    /// nearly every `(doc, term)` pair is unique on that corpus, so a ~6-byte
    /// token becomes a `(String, String)` key slot, a [`PostingEntry`], and a
    /// `Vec<Occurrence>` whose first `push` reserves capacity 4 -- 48 bytes of
    /// allocation for 12 bytes of payload. That is the divergence
    /// `docs/sweep/m2/LEDGER.md` records as the block-pool redesign, and it is
    /// a milestone rather than a batch (see
    /// `docs/sweep/m2/c38-allocation-shape.md`). Handing the `Vec` surplus
    /// back with `shrink_to_fit` was tried and rejected: it cuts this figure
    /// sharply but costs 25-60% indexing throughput and moves peak RSS not at
    /// all, because glibc keeps the freed 48-byte chunks in its arena.
    ///
    /// `crates/lucene-index/examples/invert_memory.rs` is the instrument that
    /// produces the numbers above.
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
            bytes += std::mem::size_of::<(TermKey, TermPostingList)>();
            bytes += field.capacity() + term.capacity();
            bytes += postings.entries.capacity() * std::mem::size_of::<PostingEntry>();
            // The flat payload run exists only for a `store_payloads` field,
            // so this is zero for the common case rather than a per-entry
            // constant -- and it is charged once per term rather than once
            // per posting entry, which is the whole point of the shape.
            bytes += postings.payload_bytes.capacity();
            bytes += postings.payload_lengths.capacity() * std::mem::size_of::<u32>();
            for entry in &postings.entries {
                bytes += entry.occurrences.capacity() * std::mem::size_of::<Occurrence>();
            }
        }
        bytes
    }

    /// Looks up the posting list for a `(field, term)` pair, if present.
    pub fn postings(&self, field: &str, term: &str) -> Option<&[PostingEntry]> {
        self.posting_list(field, term)
            .map(|list| list.entries.as_slice())
    }

    /// Looks up a `(field, term)` pair's whole [`TermPostingList`] -- the
    /// entries plus the flat payload run -- if present.
    pub fn posting_list(&self, field: &str, term: &str) -> Option<&TermPostingList> {
        self.terms.get(&(field.to_string(), term.to_string()))
    }
}

/// Tokenizes and inverts a batch of documents' indexed field text via
/// `analyzer`, producing an [`InMemoryInvertedIndex`].
///
/// `docs` is `(doc_id, field_name, text)` triples: a document with multiple
/// indexed fields is represented as multiple entries sharing the same
/// `doc_id`; a batch with multiple documents is multiple `doc_id` values; and
/// **several entries sharing one `(doc_id, field_name)` are the values of one
/// multi-valued field**, inverted through a single `FieldInvertState` the way
/// Java's `PerField` does (see [`invert_documents_with_payloads`]).
///
/// `docs` need not be sorted by `doc_id`, grouped by field, or internally
/// consistent about doc-ID order across fields. Multi-valued entries are
/// gathered by key, not by adjacency, so interleaving two fields' values does
/// not restart either one's position counter; **the order of one field's own
/// values is the order they appear in `docs`**, which is the only thing about
/// the input order that can change the answer (it is `Document.getFields`'
/// order in Java). Each `(field, term)` key's posting list is sorted by
/// `doc_id` before returning, so the doc-ID-sorted invariant holds regardless
/// of input order rather than being a caller obligation.
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

/// One `(document, field, term)` group as [`invert_documents_with_payloads`]
/// accumulates it, before it becomes a [`PostingEntry`] and a slice of its
/// term's payload run: the occurrences, and the group's own flat
/// `(bytes, lengths)` payload pair.
///
/// The payload halves are the same flat shape [`TermPostingList`] holds, not
/// a `Vec<Vec<u8>>` of per-occurrence slots -- two allocations per group
/// rather than one plus one per non-empty payload, and, because the group's
/// run is appended to the term's run and then dropped, nothing per occurrence
/// stays live at all.
type TermGroup = (Vec<Occurrence>, Vec<u8>, Vec<u32>);

/// [`invert_documents`] plus payloads: `payload_fields` names the fields whose
/// `FieldInfo.store_payloads` is set (Lucene's per-field `hasPayloads`, which
/// is a field property, never a per-token one), and `source` supplies each
/// token's payload bytes the way a `PayloadAttribute`-setting `TokenFilter`
/// would.
///
/// Every occurrence of a **listed** field gets a payload length -- zero where
/// `source` returned `None` -- so [`TermPostingList::payload_lengths`] is
/// exactly parallel to the field's occurrences and satisfies
/// `postings_writer`'s "one length per occurrence" obligation without the
/// caller re-deriving it. A field not in `payload_fields` gets no run and
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
    let mut acc: HashMap<TermKey, TermPostingList> = HashMap::new();
    // Reused across documents so the per-document grouping map is allocated
    // once for the whole batch rather than once per (document, field).
    let mut per_term: HashMap<String, TermGroup> = HashMap::new();

    // One iteration per **(document, field)**, not per input tuple: entries
    // sharing a doc ID and field name are the *values* of one multi-valued
    // field, and Java runs them through one `FieldInvertState`
    // (`PerField.invert(docID, field, first)` resets it only when `first`,
    // which `IndexingChain.processField` sets from `pf.fieldGen != fieldGen`).
    // Splitting them, as this loop used to, restarted both counters at each
    // value -- so two values both began at position 0 and offset 0, which is a
    // phrase match across a value boundary that Lucene does not have, and two
    // occurrences claiming the same offsets.
    //
    // Gathered **by key, not by adjacency**: `PerField` owns its state for the
    // whole document, so Java does not care whether another field's value sits
    // between two of this one's, and a "consecutive runs only" grouping would
    // silently re-create the defect above for `[(0,"f",..), (0,"g",..),
    // (0,"f",..)]`. `groups` holds one entry per `(doc_id, field)` in
    // first-appearance order, each listing that key's value indices in input
    // order -- `Document.getFields(name)`' order, the one property of the
    // input that legitimately decides the answer.
    //
    // Keyed through a `HashMap`, not a scan over the groups: a flush is tens
    // of thousands of documents times a handful of fields, so a linear
    // first-appearance search would be quadratic in the batch. The `Vec` is
    // what keeps first-appearance order, which a `HashMap` alone would lose.
    let mut group_keys: Vec<(i32, &str)> = Vec::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut group_of: HashMap<(i32, &str), usize> = HashMap::new();
    for (i, &(doc_id, field, _)) in docs.iter().enumerate() {
        match group_of.get(&(doc_id, field)) {
            Some(&at) => groups[at].push(i),
            None => {
                group_of.insert((doc_id, field), groups.len());
                group_keys.push((doc_id, field));
                groups.push(vec![i]);
            }
        }
    }

    for (group, &(doc_id, field)) in groups.iter().zip(group_keys.iter()) {
        let field_has_payloads = payload_fields.contains(&field);

        // Resolve position increments to absolute positions and group by
        // term within this single (doc, field), matching real Lucene's
        // TermsHashPerField accumulating one PostingEntry per (doc, field,
        // term) even when a term occurs multiple times.
        per_term.clear();
        // `FieldInvertState.reset()`: `position = -1`, `offset = 0`.
        let mut position = -1i32;
        let mut offset = 0i32;
        for &index in group {
            let text = docs[index].2;
            let stream = analyzer.analyze_stream(text);
            for token in stream.tokens {
                position = advance_position(position, token.position_increment);
                let occurrence = Occurrence {
                    position,
                    // `IndexingChain`: `startOffset = invertState.offset +
                    // offsetAttribute.startOffset()`. `offset` is 0 for the
                    // first (and only, for a single-valued field) value, so
                    // this is the identity in the common case.
                    start_offset: offset.saturating_add(token.start_offset),
                    end_offset: offset.saturating_add(token.end_offset),
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
                    slot.1.extend_from_slice(&payload);
                    slot.2.push(payload.len() as u32);
                }
            }
            // `stream.end()`, then the two attribute reads
            // `invertTokenStream` makes right after it, then the analyzer's
            // per-field gaps. Both gaps are applied after *every* value, as
            // Java's are (the last value's are simply never observed).
            position = advance_position(position, stream.final_position_increment);
            offset = offset.saturating_add(stream.final_offset);
            position = advance_position(position, analyzer.position_increment_gap());
            offset = offset.saturating_add(analyzer.offset_gap());
        }

        for (term, (occurrences, payload_bytes, payload_lengths)) in per_term.drain() {
            let key = (field.to_string(), term);
            let list = acc.entry(key).or_default();
            // Appended to the term's own run, so the group's two vectors are
            // freed here rather than staying live until the flush -- the
            // difference between a payload costing bytes in a shared run and
            // costing a heap object per occurrence.
            list.payload_bytes.extend_from_slice(&payload_bytes);
            list.payload_lengths.extend_from_slice(&payload_lengths);
            list.entries.push(PostingEntry {
                doc_id,
                occurrences,
            });
        }
    }

    // Enforce the doc-ID-sorted invariant directly, rather than trusting
    // callers to supply `docs` in ascending doc-ID order. The grouped loop
    // above already gives each `(field, term)` at most one entry per document,
    // so this only reorders; the sort is stable anyway, which keeps it
    // insensitive to the hash map's arbitrary per-document iteration order.
    let mut entries: Vec<(TermKey, TermPostingList)> = acc.into_iter().collect();
    for (_, postings) in entries.iter_mut() {
        postings.sort_by_doc_id();
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
    /// Java's `PerField` owns its `FieldInvertState` for the whole document,
    /// so two values of one field are one field however many *other* fields'
    /// values sit between them (`processField` resets only when
    /// `pf.fieldGen != fieldGen`). A grouping that only joined *consecutive*
    /// tuples would restart the second value at position 0 and offset 0 --
    /// the very defect the grouping exists to remove -- for an input this
    /// function's own contract accepts.
    #[test]
    fn a_fields_values_are_one_field_even_with_another_fields_value_between_them() {
        let analyzer = Analyzer::standard(None).with_position_increment_gap(7);
        let interleaved = invert_documents(
            &[(0, "f", "alpha"), (0, "g", "zulu"), (0, "f", "beta")],
            &analyzer,
        );
        let adjacent = invert_documents(
            &[(0, "f", "alpha"), (0, "f", "beta"), (0, "g", "zulu")],
            &analyzer,
        );
        assert_eq!(interleaved.terms, adjacent.terms);

        // ... and the second value really is offset, not restarted: "alpha" is
        // 5 characters, the offset gap is 1, and the position gap is 7.
        let beta = interleaved.posting_list("f", "beta").expect("beta indexed");
        assert_eq!(beta.entries.len(), 1);
        assert_eq!(
            beta.entries[0].occurrences,
            vec![Occurrence {
                position: 8,
                start_offset: 6,
                end_offset: 10,
            }]
        );
        // The interposed field is unaffected by either gap.
        let zulu = interleaved.posting_list("g", "zulu").expect("zulu indexed");
        assert_eq!(
            zulu.entries[0].occurrences,
            vec![Occurrence {
                position: 0,
                start_offset: 0,
                end_offset: 4,
            }]
        );
    }

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
        for ((_, term), list) in &index.terms {
            for entry in &list.entries {
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

        let body = index.posting_list("body", "fox").unwrap();
        assert_eq!(body.entries.len(), 1);
        assert!(body.has_payloads());
        assert_eq!(body.entries[0].term_freq(), 2);
        assert_eq!(body.entries[0].positions(), vec![0, 3]);
        // Position 0 is even (payload), position 3 is odd (none).
        assert_eq!(body.payload_lengths, vec![5, 0]);
        assert_eq!(body.payload_bytes, b"fox:0".to_vec());
        assert_eq!(body.payload(0, 0), Some(&b"fox:0"[..]));
        assert_eq!(body.payload(0, 1), Some(&b""[..]));

        let title = index.posting_list("title", "fox").unwrap();
        assert!(!title.has_payloads());
        assert!(title.payload_lengths.is_empty());
        assert!(title.payload_bytes.is_empty());
        assert_eq!(title.payload(0, 0), None);
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
        // And the run is still there, one zero length per occurrence.
        let list = index.posting_list("body", "fox").unwrap();
        assert_eq!(list.payload_lengths, vec![0]);
        assert!(list.payload_bytes.is_empty());
    }

    /// `invert_documents` is the no-payloads shorthand, and must stay exactly
    /// that: no slots, and a source that would panic if it were ever called.
    #[test]
    fn invert_documents_records_no_payload_slots_at_all() {
        let analyzer = Analyzer::standard(None);
        let index = invert_documents(&[(0, "body", "fox")], &analyzer);
        assert!(!index.posting_list("body", "fox").unwrap().has_payloads());
        // Same batch through the payload entry point with an empty field
        // list: identical output, which is what makes the delegation safe.
        let same = invert_documents_with_payloads(&[(0, "body", "fox")], &analyzer, &[], &|_| {
            panic!("a field not in payload_fields must never reach the source")
        });
        assert_eq!(index, same);
    }

    /// The allocation-shape assertion behind c38's item 2: a
    /// [`PostingEntry`] carries **no payload slot at all**, so a payload
    /// costs nothing per posting entry for the overwhelmingly common field
    /// that has none. Against the shape this replaced -- a `Vec<Vec<u8>>` on
    /// every entry -- this is 24 bytes smaller per entry on a 64-bit target,
    /// which on `index-bench`'s corpus is 19 MB of the 27 MB the change
    /// removed from `InMemoryInvertedIndex`.
    ///
    /// Written against `size_of` rather than a byte count so it is a
    /// statement about the type, and so it holds on a 32-bit target too.
    #[test]
    fn a_posting_entry_carries_no_payload_slot() {
        assert_eq!(
            std::mem::size_of::<PostingEntry>(),
            std::mem::size_of::<Vec<Occurrence>>() + std::mem::size_of::<usize>(),
            "a PostingEntry is a doc id and its occurrences, nothing else"
        );
    }

    /// c23's finding, asserted from the other side: the cost of payloads must
    /// be the **bytes**, not a per-occurrence slot. An all-empty-payload field
    /// therefore costs one `u32` length per occurrence and nothing else --
    /// where the nested shape cost a `Vec` header per posting entry (24 bytes)
    /// plus one allocation per non-empty payload.
    ///
    /// The bound is expressed as "cheaper than one empty `Vec` header per
    /// occurrence", which is exactly what the nested shape charged before a
    /// single payload byte was stored, so `Vec`'s growth slack (the run runs
    /// about 8.7 bytes per occurrence here against the 4 it holds) cannot
    /// make it flaky while it still fails outright against the old shape.
    #[test]
    fn an_all_empty_payload_field_costs_only_a_length_per_occurrence() {
        let analyzer = Analyzer::standard(None);
        let texts: Vec<String> = (0..200).map(|d| format!("alpha beta doc{d}")).collect();
        let docs: Vec<(i32, &str, &str)> = texts
            .iter()
            .enumerate()
            .map(|(d, t)| (d as i32, "body", t.as_str()))
            .collect();
        let without = invert_documents(&docs, &analyzer);
        let with_empty = invert_documents_with_payloads(&docs, &analyzer, &["body"], &|_| None);

        let occurrences: usize = with_empty
            .terms
            .values()
            .flat_map(|l| l.entries.iter())
            .map(|e| e.occurrences.len())
            .sum();
        assert_eq!(occurrences, 600, "200 docs x 3 tokens");
        let extra = with_empty.ram_bytes_used() - without.ram_bytes_used();
        assert!(
            extra < occurrences * std::mem::size_of::<Vec<u8>>(),
            "an all-empty-payload field cost {extra} bytes for {occurrences} occurrences, \
             which is not cheaper than a vector header each"
        );
        assert!(
            extra >= occurrences * 4,
            "but it must still cost the lengths it stores: {extra}"
        );
    }

    /// Payload bytes are heap this structure occupies, so `ram_bytes_used`
    /// has to count the flat run -- both its lengths and its bytes.
    #[test]
    fn ram_bytes_used_counts_the_payload_run_and_its_bytes() {
        let analyzer = Analyzer::standard(None);
        let docs = [(0, "body", "alpha beta")];
        let without = invert_documents(&docs, &analyzer);
        let with_empty = invert_documents_with_payloads(&docs, &analyzer, &["body"], &|_| None);
        let with_bytes =
            invert_documents_with_payloads(&docs, &analyzer, &["body"], &|_| Some(vec![0u8; 64]));

        assert!(
            with_empty.ram_bytes_used() > without.ram_bytes_used(),
            "the length run itself is real memory: {} vs {}",
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

    /// The flat run is laid out in `entries` order, so the doc-ID sort has to
    /// carry it. Documents are handed in **descending** doc-ID order with a
    /// distinct, distinctly-*sized* payload per document, so a sort that
    /// permuted the entries and left the run alone would hand every document
    /// somebody else's payload -- and a sort that permuted fixed-width slices
    /// would too.
    #[test]
    fn sorting_entries_by_doc_id_carries_the_payload_run_with_them() {
        let analyzer = Analyzer::standard(None);
        // doc 2 has one occurrence, doc 1 has two, doc 0 has three, so the
        // per-document runs have three different lengths as well as three
        // different byte counts.
        let docs = vec![
            (2, "body", "fox"),
            (1, "body", "fox fox"),
            (0, "body", "fox fox fox"),
        ];
        let index = invert_documents_with_payloads(&docs, &analyzer, &["body"], &|ctx| {
            Some(vec![ctx.doc_id as u8; ctx.doc_id as usize + 1])
        });
        let list = index.posting_list("body", "fox").expect("fox");
        assert_eq!(
            list.entries.iter().map(|e| e.doc_id).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(list.payload_lengths, vec![1, 1, 1, 2, 2, 3]);
        assert_eq!(list.payload_bytes, vec![0, 0, 0, 1, 1, 1, 1, 2, 2, 2]);
        for (entry_index, doc_id) in [0u8, 1, 2].into_iter().enumerate() {
            let expected = vec![doc_id; doc_id as usize + 1];
            for occurrence in 0..list.entries[entry_index].occurrences.len() {
                assert_eq!(
                    list.payload(entry_index, occurrence),
                    Some(&expected[..]),
                    "entry {entry_index}, occurrence {occurrence}"
                );
            }
        }
        // Out of range in either dimension is `None`, not a panic.
        assert_eq!(list.payload(3, 0), None);
        assert_eq!(list.payload(0, 3), None);
    }

    /// The same sort with no payloads at all must still order the entries --
    /// the early return for a payload-free list is a separate branch.
    #[test]
    fn sorting_entries_by_doc_id_works_without_payloads() {
        let analyzer = Analyzer::standard(None);
        let docs = vec![(2, "body", "fox"), (0, "body", "fox"), (1, "body", "fox")];
        let index = invert_documents(&docs, &analyzer);
        let list = index.posting_list("body", "fox").expect("fox");
        assert_eq!(
            list.entries.iter().map(|e| e.doc_id).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert!(!list.has_payloads());
        assert_eq!(list.payload(0, 0), None);
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
