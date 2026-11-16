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
/// reads back out of the `.pos`/`.pay`/`.tvd` this structure feeds. Nothing here
/// converts, so the unit is
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
/// object per posting entry: a nested `Vec<Vec<u8>>`-per-entry shape
/// measured 26 us/doc and ~190 MB
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
    /// Per field, every inverted document's [`FieldEnd`], ascending by
    /// document: the part of `FieldInvertState` the postings cannot
    /// reconstruct.
    pub field_ends: BTreeMap<String, Vec<(i32, FieldEnd)>>,
}

/// Where `IndexingChain.PerField.invert` left one document's field: the
/// `FieldInvertState` values a posting list does not carry.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FieldEnd {
    /// `getPosition()`: after the last value's final increment and gap.
    pub position: i32,
    /// `getOffset()`: after the last value's final offset and gap.
    pub offset: i32,
    /// `getNumOverlap()`: tokens at a position increment of `0`.
    pub num_overlap: i32,
    /// The last value's `end()` increment and offset: its attribute source,
    /// every other attribute cleared by the standard chain's `end()`.
    pub final_increment: i32,
    pub final_offset: i32,
}

impl InMemoryInvertedIndex {
    /// `field`'s [`FieldEnd`] for `doc`, if it inverted that document.
    pub fn field_end(&self, field: &str, doc: i32) -> Option<FieldEnd> {
        let ends = self.field_ends.get(field)?;
        ends.binary_search_by_key(&doc, |(d, _)| *d)
            .ok()
            .map(|i| ends[i].1)
    }
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
    /// **75.8 MB here**, against Java's *zero*
    /// heap objects per occurrence (a token becomes a few bytes in a
    /// `ByteBlockPool` slice). What is left is per-occurrence and structural:
    /// nearly every `(doc, term)` pair is unique on that corpus, so a ~6-byte
    /// token becomes a `(String, String)` key slot, a [`PostingEntry`], and a
    /// `Vec<Occurrence>` whose first `push` reserves capacity 4 -- 48 bytes of
    /// allocation for 12 bytes of payload. Closing it needs a block-pool
    /// redesign. Handing the `Vec` surplus back with `shrink_to_fit` was tried
    /// and rejected: it cuts this figure sharply but costs 25-60% indexing
    /// throughput and moves peak RSS not at all, because glibc keeps the freed
    /// 48-byte chunks in its arena.
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
        for (field, ends) in &self.field_ends {
            bytes += field.capacity() + ends.capacity() * std::mem::size_of::<(i32, FieldEnd)>();
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
/// one definition -- this is the writer's half, so this is the one to keep.
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
    // Per field, then per term: a flat `(field, term)` key allocated the
    // field's name again for every (document, term) pair.
    let mut acc: HashMap<String, HashMap<String, TermPostingList>> = HashMap::new();
    // Reused across documents so the per-document grouping map is allocated
    // once for the whole batch rather than once per (document, field).
    let mut per_term: HashMap<String, TermGroup> = HashMap::new();

    // One iteration per **(document, field)**, not per input tuple: entries
    // sharing a doc ID and field name are the *values* of one multi-valued
    // field, and Java runs them through one `FieldInvertState`
    // (`PerField.invert(docID, field, first)` resets it only when `first`,
    // which `IndexingChain.processField` sets from `pf.fieldGen != fieldGen`).
    // Splitting them would restart both counters at each value -- so two
    // values would both begin at position 0 and offset 0, giving a phrase
    // match across a value boundary that Lucene does not have, and two
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
    let mut field_ends: BTreeMap<String, Vec<(i32, FieldEnd)>> = BTreeMap::new();
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
        let mut end = FieldEnd::default();
        for &index in group {
            let text = docs[index].2;
            let stream = analyzer.analyze_stream(text);
            for token in stream.tokens {
                if token.position_increment == 0 {
                    end.num_overlap = end.num_overlap.saturating_add(1);
                }
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
            end.final_increment = stream.final_position_increment;
            end.final_offset = stream.final_offset;
        }
        end.position = position;
        end.offset = offset;
        // Looked up before inserting, so a field's name is allocated once.
        match field_ends.get_mut(field) {
            Some(ends) => ends.push((doc_id, end)),
            None => {
                field_ends.insert(field.to_string(), vec![(doc_id, end)]);
            }
        }

        let field_acc = match acc.get_mut(field) {
            Some(terms) => terms,
            None => acc.entry(field.to_string()).or_default(),
        };
        for (term, (occurrences, payload_bytes, payload_lengths)) in per_term.drain() {
            let list = field_acc.entry(term).or_default();
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
    let mut entries: Vec<(TermKey, TermPostingList)> = acc
        .into_iter()
        .flat_map(|(field, terms)| {
            terms
                .into_iter()
                .map(move |(term, list)| ((field.clone(), term), list))
        })
        .collect();
    for (_, postings) in entries.iter_mut() {
        postings.sort_by_doc_id();
    }
    entries.sort_unstable_by(|a, b| a.0.cmp(&b.0));

    for ends in field_ends.values_mut() {
        ends.sort_by_key(|(doc, _)| *doc);
    }
    InMemoryInvertedIndex {
        // `BTreeMap::from_iter` over already-sorted, deduplicated pairs builds
        // the tree bottom-up in O(n) rather than n O(log n) insertions.
        terms: entries.into_iter().collect(),
        field_ends,
    }
}

/// Java's `String.hashCode()`: `s[0]*31^(n-1) + ... + s[n-1]` over the
/// UTF-16 code units, wrapping.
pub fn java_string_hash(s: &str) -> i32 {
    s.encode_utf16().fold(0i32, |h, unit| {
        h.wrapping_mul(31).wrapping_add(i32::from(unit))
    })
}

/// The order `IndexingChain`'s flush loops (`writeDocValues`, `writePoints`)
/// visit fields in: its `fieldHash`, walked bucket by bucket, each bucket's
/// chain from its head. `names` are the fields in the order the chain first
/// saw them (`getOrAddPerField`); the result lists their indexes in walk
/// order.
///
/// `fieldHash` starts with 2 buckets; a new field is put at the head of
/// bucket `hashCode() & hashMask`, and once there are at least half as many
/// fields as buckets the table doubles (`rehash`, which moves each old
/// bucket's chain, head first, onto the heads of the new buckets).
pub fn field_hash_order(names: &[&str]) -> Vec<usize> {
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); 2];
    for (i, name) in names.iter().enumerate() {
        let mask = buckets.len().saturating_sub(1);
        // `hashCode() & hashMask`: the mask is below the table size, so the
        // `as` keeps exactly the bits Java keeps.
        #[allow(clippy::cast_sign_loss)]
        let pos = (java_string_hash(name) as u32 as usize) & mask;
        buckets[pos].insert(0, i);
        if i.saturating_add(1) >= buckets.len() / 2 {
            let doubled = buckets.len().saturating_mul(2);
            let mut rehashed: Vec<Vec<usize>> = vec![Vec::new(); doubled];
            for chain in &buckets {
                for &field in chain {
                    #[allow(clippy::cast_sign_loss)]
                    let pos = (java_string_hash(names[field]) as u32 as usize)
                        & doubled.saturating_sub(1);
                    rehashed[pos].insert(0, field);
                }
            }
            buckets = rehashed;
        }
    }
    buckets.into_iter().flatten().collect()
}

/// The order a `java.util.HashMap<String, _>` iterates `names`, inserted in
/// the order given: bucket by bucket at the table's final capacity (16,
/// doubling once more than three quarters full), each bucket in insertion
/// order (a resize keeps it). A key's bucket is `h ^ (h >>> 16)` of its
/// `hashCode()`, masked. Returns the names' indexes in that order; a bucket
/// long enough to be a tree (eight keys) is not modelled.
pub fn java_hash_map_order(names: &[&str]) -> Vec<usize> {
    let mut capacity = 16usize;
    while names.len().saturating_mul(4) > capacity.saturating_mul(3) {
        capacity = capacity.saturating_mul(2);
    }
    let bucket = |name: &str| {
        #[allow(clippy::cast_sign_loss)]
        let h = java_string_hash(name) as u32;
        ((h ^ (h >> 16)) as usize) & capacity.saturating_sub(1)
    };
    let mut order: Vec<usize> = (0..names.len()).collect();
    order.sort_by_key(|&i| (bucket(names[i]), i));
    order
}

#[cfg(test)]
mod tests;
