//! The indexing-time inverted index for the common case, built the way
//! Lucene's `IndexingChain` + `TermsHashPerField` + `FreqProxTermsWriter`
//! build theirs: per field, a term hash over one byte arena handing out dense
//! `u32` term ids, and per term only what the postings writer will need --
//! `(doc, freq)` pairs, and for a positions field one flat run of positions
//! (and offsets) -- appended as each document is inverted.
//!
//! # Why this exists next to `indexing_chain`
//!
//! [`crate::indexing_chain::invert_documents_with_payloads`] is the general
//! path: payloads, term vectors, and the `InMemoryInvertedIndex` shape those
//! consumers read. It keys a `HashMap` by `String` per document and again by
//! `(String, String)` across the segment, and gives every posting its own
//! `Vec<Occurrence>` -- about four allocations and two SipHash lookups per
//! token, roughly 100 bytes per buffered posting against the 1-3 bytes a
//! Lucene byte slice spends. Measured end to end that made indexing 5x slower
//! than Lucene.
//!
//! Here a token costs a table probe over its bytes and, for a positions
//! field, a push onto its term's run. Allocations happen per *term* (when its
//! lists grow), not per token. Field lengths for norms are counted as the
//! tokens go by (`FieldInvertState.length`), so norms need no second pass over
//! the dictionary, and each field counts its documents directly instead of
//! inserting every posting into a `BTreeSet`.
//!
//! The output is exactly what the general path's consumers produced from its
//! index: [`lucene_codecs::postings_writer::TermPostings`] per field, sorted by term bytes,
//! and per-document field lengths -- asserted equal in this module's tests.

use lucene_analysis::Analyzer;
use lucene_codecs::field_infos::IndexOptions;
use lucene_codecs::postings_writer::TermPostings;

use crate::indexing_chain::MAX_POSITION;

/// FxHash's multiplier: a fast, well-mixing hash for short byte strings. No
/// defence against adversarial collisions is needed -- the table is per field
/// per segment and bounded by what one flush buffers.
const FX_K: u64 = 0x517c_c1b7_2722_0a95;

#[inline]
pub(crate) fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut h: u64 = bytes.len() as u64;
    let mut chunks = bytes.chunks_exact(8);
    for c in &mut chunks {
        let w = u64::from_le_bytes(c.try_into().expect("chunks_exact(8)"));
        h = (h.rotate_left(5) ^ w).wrapping_mul(FX_K);
    }
    let rest = chunks.remainder();
    if !rest.is_empty() {
        let mut buf = [0u8; 8];
        buf[..rest.len()].copy_from_slice(rest);
        h = (h.rotate_left(5) ^ u64::from_le_bytes(buf)).wrapping_mul(FX_K);
    }
    h
}

/// The first slot to probe for hash `h` in a table of `size` slots (a power
/// of two): the hash's **top** bits. `hash_bytes` ends in a multiply, and a
/// product's low bits depend only on the multiplicand's low bits -- for short
/// terms that is the first byte or two (`t` plus a digit for `t123`), so
/// masking the low bits put thousands of distinct terms on a handful of
/// slots and linear probing ran through all of them: 92% of indexing time,
/// 0.31x of Lucene. The high bits take every input bit into account.
#[inline]
fn home_slot(h: u64, size: usize) -> usize {
    // `size` is a power of two >= 1024 (`TermHash::new`, doubled by
    // `grow`), so the shift is in `1..=54`; `checked_shr` would only answer
    // `None` for a shift of 64, i.e. a one-slot table.
    let shift = u64::BITS.saturating_sub(size.trailing_zeros());
    h.checked_shr(shift).unwrap_or(0) as usize
}

/// `BytesRefHash`: term bytes in one arena, dense ids, open addressing.
#[derive(Debug, Default)]
pub(crate) struct TermHash {
    arena: Vec<u8>,
    /// Term `i` is `arena[starts[i]..starts[i + 1]]`.
    starts: Vec<u32>,
    /// `id + 1`, or `0` for an empty slot. Power-of-two sized.
    table: Vec<u32>,
    hashes: Vec<u64>,
}

impl TermHash {
    pub(crate) fn new() -> Self {
        TermHash {
            arena: Vec::new(),
            starts: vec![0],
            table: vec![0; 1024],
            hashes: Vec::new(),
        }
    }

    fn len(&self) -> usize {
        self.hashes.len()
    }

    pub(crate) fn term(&self, id: u32) -> &[u8] {
        let (a, b) = (
            self.starts[id as usize],
            self.starts[(id as usize).wrapping_add(1)],
        );
        &self.arena[a as usize..b as usize]
    }

    /// Heap bytes held.
    pub(crate) fn ram_bytes(&self) -> usize {
        self.arena
            .capacity()
            .saturating_add(self.starts.capacity().saturating_mul(4))
            .saturating_add(self.table.capacity().saturating_mul(4))
            .saturating_add(self.hashes.capacity().saturating_mul(8))
    }

    /// Forgets every term from id `len` on, as if never added.
    pub(crate) fn truncate(&mut self, len: usize) {
        if len >= self.len() {
            return;
        }
        self.arena.truncate(self.starts[len] as usize);
        self.starts.truncate(len.saturating_add(1));
        self.hashes.truncate(len);
        self.rehash(self.table.len());
    }

    /// The id of `bytes`, adding it if new; `true` when it was added.
    #[inline]
    pub(crate) fn add(&mut self, bytes: &[u8]) -> (u32, bool) {
        self.add_hashed(bytes, hash_bytes(bytes))
    }

    /// [`Self::add`] for bytes whose [`hash_bytes`] is already known -- a
    /// term moving from one table to another keeps its hash.
    #[inline]
    fn add_hashed(&mut self, bytes: &[u8], h: u64) -> (u32, bool) {
        // `table.len()` is a power of two, at least 1024.
        let mask = self.table.len().wrapping_sub(1);
        let mut slot = home_slot(h, self.table.len());
        loop {
            let entry = self.table[slot];
            if entry == 0 {
                break;
            }
            let id = entry.wrapping_sub(1);
            if self.hashes[id as usize] == h && self.term(id) == bytes {
                return (id, false);
            }
            slot = slot.wrapping_add(1) & mask;
        }
        let id = self.hashes.len() as u32;
        self.arena.extend_from_slice(bytes);
        self.starts.push(self.arena.len() as u32);
        self.hashes.push(h);
        self.table[slot] = id.wrapping_add(1);
        // Keep the load under 1/2.
        if self.hashes.len().saturating_mul(2) > self.table.len() {
            self.grow();
        }
        (id, true)
    }

    fn grow(&mut self) {
        self.rehash(self.table.len().saturating_mul(2));
    }

    /// Rebuilds the table at `size` slots over every term.
    fn rehash(&mut self, size: usize) {
        let mut table = vec![0u32; size];
        let mask = size.wrapping_sub(1);
        for (id, &h) in self.hashes.iter().enumerate() {
            let mut slot = home_slot(h, size);
            while table[slot] != 0 {
                slot = slot.wrapping_add(1) & mask;
            }
            table[slot] = (id as u32).wrapping_add(1);
        }
        self.table = table;
    }

    /// Every term id, sorted by term bytes -- `BytesRefHash.sort`.
    ///
    /// Java's sort is an MSB radix sort; this sorts each id under its first
    /// eight bytes as one big-endian integer, which orders every pair of
    /// terms differing there without touching the arena, and compares full
    /// bytes only within a run of equal prefixes.
    pub(crate) fn sorted_ids(&self) -> Vec<u32> {
        let mut keyed: Vec<(u64, u32)> = (0..self.len() as u32)
            .map(|id| (prefix_key(self.term(id)), id))
            .collect();
        keyed.sort_unstable();
        let mut start = 0;
        while start < keyed.len() {
            let key = keyed[start].0;
            let run = keyed[start..].iter().take_while(|(k, _)| *k == key).count();
            let end = start.saturating_add(run);
            if run > 1 {
                keyed[start..end].sort_unstable_by(|a, b| self.term(a.1).cmp(self.term(b.1)));
            }
            start = end;
        }
        keyed.into_iter().map(|(_, id)| id).collect()
    }
}

/// A term's first eight bytes as a big-endian integer, zero-padded: the
/// integers order as the prefixes do, and equal integers leave the order to
/// the bytes past them (or to the length, for a zero-padded short term).
fn prefix_key(term: &[u8]) -> u64 {
    let mut b = [0u8; 8];
    let n = term.len().min(8);
    b[..n].copy_from_slice(&term[..n]);
    u64::from_be_bytes(b)
}

/// One term's buffered postings.
#[derive(Debug, Default)]
struct TermPostingsBuf {
    /// `(doc, freq)`, doc ascending.
    docs: Vec<(i32, i32)>,
    /// Every position of every document, doc-major.
    positions: Vec<i32>,
    /// `(startOffset, endOffset)` parallel to `positions`.
    offsets: Vec<(i32, i32)>,
}

/// One field's inverted index under construction.
#[derive(Debug)]
pub struct FieldInverter {
    index_options: IndexOptions,
    terms: TermHash,
    postings: Vec<TermPostingsBuf>,
    /// Documents with at least one token -- the postings `docCount`.
    doc_count: i32,
    /// Field length (tokens) of every document inverted so far, `None` for a
    /// document with no value in this field: the norms input.
    lengths: Vec<Option<u32>>,
    /// Parallel to `lengths`: `FieldInvertState`'s `numOverlap` (tokens at a
    /// position increment of 0) and `uniqueTermCount` (distinct terms), the
    /// rest of what `Similarity.computeNorm` reads.
    overlaps_and_unique: Vec<(u32, u32)>,
    /// Parallel to `lengths`: the rest of `FieldInvertState` as the
    /// document's last value left it.
    ends: Vec<DocEnd>,
}

/// The part of one document's `FieldInvertState` the counts above do not
/// hold: where `PerField.invert` left `position` and `offset`, the
/// `maxTermFrequency` `FreqProxTermsWriterPerField` kept, and the last
/// value's `end()` increment and offset -- all its attribute source differs
/// by, since every value goes through the same standard chain, whose `end()`
/// clears every other attribute.
#[derive(Debug, Clone, Copy, Default)]
struct DocEnd {
    position: i32,
    offset: i32,
    max_term_frequency: u32,
    final_increment: i32,
    final_offset: i32,
}

impl FieldInverter {
    pub fn new(index_options: IndexOptions) -> Self {
        FieldInverter {
            index_options,
            terms: TermHash::new(),
            postings: Vec::new(),
            doc_count: 0,
            lengths: Vec::new(),
            overlaps_and_unique: Vec::new(),
            ends: Vec::new(),
        }
    }

    fn has_positions(&self) -> bool {
        matches!(
            self.index_options,
            IndexOptions::DocsAndFreqsAndPositions
                | IndexOptions::DocsAndFreqsAndPositionsAndOffsets
        )
    }

    fn has_offsets(&self) -> bool {
        self.index_options == IndexOptions::DocsAndFreqsAndPositionsAndOffsets
    }

    /// Inverts one document's values of this field. `doc` must be the next
    /// document id (ascending, no gaps: a document without the field passes
    /// no values). Positions and offsets across a multi-valued field follow
    /// `IndexingChain.PerField.invert`: after each value, the stream's final
    /// increment/offset, then the analyzer's position-increment and offset
    /// gaps (after the last value too: `FieldInvertState.getPosition()`
    /// counts them).
    pub fn add_document(&mut self, doc: i32, values: &[&str], analyzer: &Analyzer) {
        debug_assert_eq!(doc as usize, self.lengths.len());
        if values.is_empty() {
            self.lengths.push(None);
            self.overlaps_and_unique.push((0, 0));
            self.ends.push(DocEnd::default());
            return;
        }
        let with_positions = self.has_positions();
        let with_offsets = self.has_offsets();
        let mut position = -1i32;
        let mut offset = 0i32;
        let mut length = 0u32;
        let mut num_overlap = 0u32;
        let mut unique = 0u32;
        let mut max_freq = 0u32;
        let mut saw_new_term = false;
        let mut end = DocEnd::default();
        let gap = analyzer.position_increment_gap();
        let offset_gap = analyzer.offset_gap();
        for text in values {
            let base_offset = offset;
            let (final_inc, final_offset) =
                analyzer.for_each_token(text, |term, start, end, pos_inc| {
                    position = advance_position(position, pos_inc);
                    length = length.saturating_add(1);
                    // `IndexingChain.invert`: `if (posIncr == 0) numOverlap++`.
                    num_overlap = num_overlap.saturating_add(u32::from(pos_inc == 0));
                    let (id, is_new) = self.terms.add(term.as_bytes());
                    saw_new_term |= is_new;
                    if is_new {
                        self.postings.push(TermPostingsBuf::default());
                    }
                    let p = &mut self.postings[id as usize];
                    match p.docs.last_mut() {
                        Some((d, f)) if *d == doc => {
                            *f = f.saturating_add(1);
                            max_freq = max_freq.max(*f as u32);
                        }
                        _ => {
                            // The term's first occurrence in this document.
                            unique = unique.saturating_add(1);
                            max_freq = max_freq.max(1);
                            p.docs.push((doc, 1));
                        }
                    }
                    if with_positions {
                        p.positions.push(position);
                        if with_offsets {
                            p.offsets.push((
                                base_offset.saturating_add(start),
                                base_offset.saturating_add(end),
                            ));
                        }
                    }
                });
            position = advance_position(position, final_inc);
            offset = offset.saturating_add(final_offset);
            position = advance_position(position, gap);
            offset = offset.saturating_add(offset_gap);
            end.final_increment = final_inc;
            end.final_offset = final_offset;
        }
        if length > 0 {
            self.doc_count = self.doc_count.saturating_add(1);
        }
        end.position = position;
        end.offset = offset;
        // Without frequencies only `newTerm` -- a term new to the segment --
        // sets it, to 1.
        end.max_term_frequency = if self.index_options == IndexOptions::Docs {
            u32::from(saw_new_term)
        } else {
            max_freq
        };
        self.lengths.push(Some(length));
        self.overlaps_and_unique.push((num_overlap, unique));
        self.ends.push(end);
    }

    /// Documents with at least one token in this field.
    pub fn doc_count(&self) -> i32 {
        self.doc_count
    }

    /// Per-document token counts (`None`: no value), for norms.
    pub fn lengths(&self) -> &[Option<u32>] {
        &self.lengths
    }

    /// The `FieldInvertState` `Similarity.computeNorm` reads for document
    /// `doc`, or `None` when it has no value in this field.
    pub fn invert_state(&self, doc: usize) -> Option<crate::similarity::FieldInvertState> {
        invert_state(
            self.index_options,
            &self.lengths,
            &self.overlaps_and_unique,
            &self.ends,
            doc,
        )
    }

    /// Number of distinct terms buffered.
    pub fn num_terms(&self) -> usize {
        self.terms.len()
    }

    /// Heap bytes this inverter holds, for the flush trigger.
    // ARITH: a sum of live allocation sizes -- each bounded by `isize::MAX`
    // bytes, and far fewer of them than could overflow a `usize` -- the same
    // justification `VectorValue::ram_bytes` carries.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn ram_bytes_used(&self) -> usize {
        let per_term: usize = self
            .postings
            .iter()
            .map(|p| p.docs.capacity() * 8 + p.positions.capacity() * 4 + p.offsets.capacity() * 8)
            .sum();
        self.terms.arena.capacity()
            + self.terms.starts.capacity() * 4
            + self.terms.table.capacity() * 4
            + self.terms.hashes.capacity() * 8
            + self.postings.capacity() * std::mem::size_of::<TermPostingsBuf>()
            + per_term
            + self.lengths.capacity() * 8
            + self.overlaps_and_unique.capacity() * 8
            + self.ends.capacity() * std::mem::size_of::<DocEnd>()
    }

    /// Inverts `num_docs` documents, splitting them into contiguous runs that
    /// are inverted on rayon's pool, each into its own term hash, then merged
    /// in term order -- also on the pool ([`merge_runs`]). `values_of(doc,
    /// out)` fills `out` with document `doc`'s values of this field.
    ///
    /// Tokenizing and hashing are most of a flush, and nothing in them
    /// depends on another document, so a single-threaded writer's flush uses
    /// every core. The result is the one serial inversion builds -- asserted
    /// in this module's tests.
    pub fn invert_parallel<'a, F>(
        index_options: IndexOptions,
        num_docs: usize,
        values_of: F,
        analyzer: &Analyzer,
    ) -> InvertedField
    where
        F: Fn(usize, &mut Vec<&'a str>) + Sync,
    {
        use rayon::prelude::*;
        let invert_run = |run: std::ops::Range<usize>| {
            let mut inverter = FieldInverter::new(index_options);
            let mut values = Vec::new();
            for (local, doc) in run.enumerate() {
                values.clear();
                values_of(doc, &mut values);
                inverter.add_document(local as i32, &values, analyzer);
            }
            inverter
        };
        let threads = rayon::current_num_threads();
        // Below this a run's own term hash (and the merge that folds it in)
        // costs more than the parallelism saves.
        const MIN_RUN_DOCS: usize = 512;
        if threads <= 1 || num_docs < MIN_RUN_DOCS.saturating_mul(2) {
            return invert_run(0..num_docs).finish();
        }
        // Two runs per thread, so a run of long documents does not leave the
        // other threads idle at the end.
        let runs = threads
            .saturating_mul(2)
            .min(num_docs / MIN_RUN_DOCS)
            .max(1);
        let run_len = num_docs.div_ceil(runs);
        let parts: Vec<FieldInverter> = (0..runs)
            .into_par_iter()
            .map(|r| {
                let start = r.saturating_mul(run_len).min(num_docs);
                invert_run(start..start.saturating_add(run_len).min(num_docs))
            })
            .collect();
        merge_runs(index_options, parts)
    }

    /// This inversion, finished: terms sorted, positions split per document.
    pub fn finish(mut self) -> InvertedField {
        InvertedField {
            index_options: self.index_options,
            doc_count: self.doc_count,
            lengths: std::mem::take(&mut self.lengths),
            overlaps_and_unique: std::mem::take(&mut self.overlaps_and_unique),
            ends: std::mem::take(&mut self.ends),
            terms: self.into_term_postings(),
        }
    }

    /// The buffered terms as the postings writer takes them, sorted by term
    /// bytes, with positions and offsets split per document.
    ///
    /// Splitting a positions field's runs allocates per term and document,
    /// independently for every term, so the terms are converted on rayon's
    /// pool; the result keeps the sorted order.
    pub fn into_term_postings(mut self) -> Vec<TermPostings> {
        use rayon::prelude::*;
        let with_positions = self.has_positions();
        let with_offsets = self.has_offsets();
        let ids = self.terms.sorted_ids();
        let sorted: Vec<(u32, TermPostingsBuf)> = ids
            .into_iter()
            .map(|id| (id, std::mem::take(&mut self.postings[id as usize])))
            .collect();
        let terms = &self.terms;
        let convert = |(id, buf): (u32, TermPostingsBuf)| {
            let mut positions = Vec::new();
            let mut offsets = Vec::new();
            if with_positions {
                positions.reserve_exact(buf.docs.len());
                if with_offsets {
                    offsets.reserve_exact(buf.docs.len());
                }
                let mut at = 0usize;
                for &(_, freq) in &buf.docs {
                    let end = at.saturating_add(freq as usize);
                    positions.push(buf.positions[at..end].to_vec());
                    if with_offsets {
                        offsets.push(buf.offsets[at..end].to_vec());
                    }
                    at = end;
                }
            }
            TermPostings {
                term: terms.term(id).to_vec(),
                docs: buf.docs,
                positions,
                offsets,
                payload_bytes: Vec::new(),
                payload_lengths: Vec::new(),
            }
        };
        if with_positions {
            sorted.into_par_iter().map(convert).collect()
        } else {
            // Nothing to split: a move per term, cheaper than the pool.
            sorted.into_iter().map(convert).collect()
        }
    }
}

/// One field's finished inversion: its postings in term order, ready for
/// the postings writer, and every document's `FieldInvertState` for norms.
#[derive(Debug)]
pub struct InvertedField {
    index_options: IndexOptions,
    doc_count: i32,
    lengths: Vec<Option<u32>>,
    overlaps_and_unique: Vec<(u32, u32)>,
    ends: Vec<DocEnd>,
    terms: Vec<TermPostings>,
}

impl InvertedField {
    /// Documents with at least one token in this field.
    pub fn doc_count(&self) -> i32 {
        self.doc_count
    }

    /// Per-document token counts (`None`: no value), for norms.
    pub fn lengths(&self) -> &[Option<u32>] {
        &self.lengths
    }

    /// [`FieldInverter::invert_state`], for the finished field.
    pub fn invert_state(&self, doc: usize) -> Option<crate::similarity::FieldInvertState> {
        invert_state(
            self.index_options,
            &self.lengths,
            &self.overlaps_and_unique,
            &self.ends,
            doc,
        )
    }

    /// The postings, sorted by term bytes.
    pub fn into_term_postings(self) -> Vec<TermPostings> {
        self.terms
    }
}

/// `FieldInvertState` for document `doc` out of the per-document arrays.
fn invert_state(
    index_options: IndexOptions,
    lengths: &[Option<u32>],
    overlaps_and_unique: &[(u32, u32)],
    ends: &[DocEnd],
    doc: usize,
) -> Option<crate::similarity::FieldInvertState> {
    let length = lengths.get(doc).copied().flatten()?;
    let (num_overlap, unique) = overlaps_and_unique.get(doc).copied()?;
    let end = ends.get(doc).copied()?;
    let clamp = |n: u32| i32::try_from(n).unwrap_or(i32::MAX);
    Some(crate::similarity::FieldInvertState {
        docs_only: index_options == IndexOptions::Docs,
        position: end.position,
        length: clamp(length),
        num_overlap: clamp(num_overlap),
        offset: end.offset,
        max_term_frequency: clamp(end.max_term_frequency),
        unique_term_count: clamp(unique),
        attribute_source: Some(crate::similarity::end_attributes(
            end.final_increment,
            end.final_offset,
        )),
    })
}

/// Runs inverted independently (run `r`'s document `i` is document
/// `starts[r] + i` of the field), merged into one [`InvertedField`].
///
/// Each run's terms are sorted on its own thread; then the term space is
/// split at evenly spaced terms of the run with the most terms, and each
/// range is merged on the pool -- a k-way merge over the runs' sorted terms,
/// concatenating each term's postings in run order, so doc ids stay
/// ascending. Nothing is re-hashed and no range waits on another.
///
/// The only per-document state that depends on earlier runs is `DOCS`'
/// "a term new to the segment" flag (its `maxTermFrequency`): it is set
/// again here, on the first document of each term's first run.
fn merge_runs(index_options: IndexOptions, runs: Vec<FieldInverter>) -> InvertedField {
    use rayon::prelude::*;
    let with_positions = matches!(
        index_options,
        IndexOptions::DocsAndFreqsAndPositions | IndexOptions::DocsAndFreqsAndPositionsAndOffsets
    );
    let with_offsets = index_options == IndexOptions::DocsAndFreqsAndPositionsAndOffsets;
    let docs_only = index_options == IndexOptions::Docs;

    let sorted: Vec<Vec<u32>> = runs.par_iter().map(|run| run.terms.sorted_ids()).collect();
    let mut starts = Vec::with_capacity(runs.len());
    let mut next = 0usize;
    for run in &runs {
        starts.push(i32::try_from(next).unwrap_or(i32::MAX));
        next = next.saturating_add(run.lengths.len());
    }

    // Split terms: evenly spaced terms of the largest run.
    let largest = (0..runs.len())
        .max_by_key(|&r| sorted[r].len())
        .unwrap_or(0);
    let pieces = rayon::current_num_threads().saturating_mul(4).max(1);
    let step = sorted
        .get(largest)
        .map_or(0, |ids| ids.len().div_ceil(pieces))
        .max(1);
    let mut bounds: Vec<&[u8]> = sorted
        .get(largest)
        .map(|ids| {
            ids.iter()
                .skip(step)
                .step_by(step)
                .map(|&id| runs[largest].terms.term(id))
                .collect()
        })
        .unwrap_or_default();
    bounds.dedup();

    let merge_range = |lower: Option<&[u8]>, upper: Option<&[u8]>| {
        // Each run's slice of sorted ids inside `lower..upper`.
        let mut cursors: Vec<(usize, usize)> = runs
            .iter()
            .zip(&sorted)
            .map(|(run, ids)| {
                let at = |bound: Option<&[u8]>, default: usize| {
                    bound.map_or(default, |b| {
                        ids.partition_point(|&id| run.terms.term(id) < b)
                    })
                };
                (at(lower, 0), at(upper, ids.len()))
            })
            .collect();
        let mut out = Vec::new();
        let mut new_term_docs = Vec::new();
        loop {
            let mut smallest: Option<&[u8]> = None;
            for (r, &(at, end)) in cursors.iter().enumerate() {
                if at < end {
                    let term = runs[r].terms.term(sorted[r][at]);
                    if smallest.is_none_or(|s| term < s) {
                        smallest = Some(term);
                    }
                }
            }
            let Some(term) = smallest else { break };
            let mut merged = TermPostings {
                term: term.to_vec(),
                ..TermPostings::default()
            };
            let mut first_run = true;
            for (r, cursor) in cursors.iter_mut().enumerate() {
                let (at, end) = *cursor;
                if at >= end {
                    continue;
                }
                let id = sorted[r][at];
                if runs[r].terms.term(id) != term {
                    continue;
                }
                *cursor = (at.saturating_add(1), end);
                let buf = &runs[r].postings[id as usize];
                let base = starts[r];
                if first_run && docs_only {
                    if let Some(&(doc, _)) = buf.docs.first() {
                        new_term_docs.push(doc.saturating_add(base));
                    }
                }
                first_run = false;
                merged
                    .docs
                    .extend(buf.docs.iter().map(|&(d, f)| (d.saturating_add(base), f)));
                if with_positions {
                    let mut at = 0usize;
                    for &(_, freq) in &buf.docs {
                        let end = at.saturating_add(freq as usize);
                        merged.positions.push(buf.positions[at..end].to_vec());
                        if with_offsets {
                            merged.offsets.push(buf.offsets[at..end].to_vec());
                        }
                        at = end;
                    }
                }
            }
            out.push(merged);
        }
        (out, new_term_docs)
    };
    let pieces: Vec<(Vec<TermPostings>, Vec<i32>)> = (0..=bounds.len())
        .into_par_iter()
        .map(|i| {
            let lower = i.checked_sub(1).map(|j| bounds[j]);
            merge_range(lower, bounds.get(i).copied())
        })
        .collect();

    let mut doc_count = 0i32;
    let mut lengths = Vec::with_capacity(next);
    let mut overlaps_and_unique = Vec::with_capacity(next);
    let mut ends = Vec::with_capacity(next);
    for run in &runs {
        doc_count = doc_count.saturating_add(run.doc_count);
        lengths.extend_from_slice(&run.lengths);
        overlaps_and_unique.extend_from_slice(&run.overlaps_and_unique);
        ends.extend_from_slice(&run.ends);
    }
    drop(sorted);
    drop(runs);
    if docs_only {
        for end in &mut ends {
            end.max_term_frequency = 0;
        }
    }
    let mut terms = Vec::with_capacity(pieces.iter().map(|(t, _)| t.len()).sum());
    for (piece, new_term_docs) in pieces {
        terms.extend(piece);
        for doc in new_term_docs {
            if let Some(end) = ends.get_mut(doc as usize) {
                end.max_term_frequency = 1;
            }
        }
    }
    InvertedField {
        index_options,
        doc_count,
        lengths,
        overlaps_and_unique,
        ends,
        terms,
    }
}

/// `IndexingChain`'s position arithmetic: clamp at `IndexWriter.MAX_POSITION`
/// rather than overflow. The same rule `indexing_chain` applies.
#[inline]
fn advance_position(position: i32, increment: i32) -> i32 {
    match position.checked_add(increment) {
        Some(next) if next <= MAX_POSITION => next,
        _ => MAX_POSITION,
    }
}

#[cfg(test)]
mod tests {
    // Test code opts out of the arithmetic gate at the module boundary; see
    // `docs/arithmetic-gate.md`.
    #![allow(clippy::arithmetic_side_effects)]

    use super::*;
    use crate::indexing_chain::invert_documents;

    /// `FieldInvertState` as `computeNorm` reads it: tokens across every
    /// value, distinct terms, the `DOCS` flag; `None` without a value.
    #[test]
    fn invert_state_counts_length_and_distinct_terms() {
        let analyzer = Analyzer::standard(None);
        let mut inv = FieldInverter::new(IndexOptions::Docs);
        inv.add_document(0, &["a a b", "b c"], &analyzer);
        inv.add_document(1, &[], &analyzer);
        inv.add_document(2, &["..."], &analyzer);
        let s = inv.invert_state(0).unwrap();
        assert!(s.docs_only);
        assert_eq!((s.length, s.num_overlap, s.unique_term_count), (5, 0, 3));
        assert_eq!(inv.invert_state(1), None);
        let empty = inv.invert_state(2).unwrap();
        assert_eq!((empty.length, empty.unique_term_count), (0, 0));
        assert_eq!(inv.invert_state(3), None);
    }

    /// The rest of `FieldInvertState`: the position and offset after every
    /// value's gap, the largest frequency (for `DOCS`, 1 only while a term is
    /// new to the segment) and `end()`'s attributes.
    #[test]
    fn invert_state_positions_offsets_and_max_frequency() {
        let analyzer = Analyzer::standard(None)
            .with_position_increment_gap(10)
            .with_offset_gap(3);
        let mut inv = FieldInverter::new(IndexOptions::DocsAndFreqsAndPositions);
        inv.add_document(0, &["a b a", "c "], &analyzer);
        let s = inv.invert_state(0).unwrap();
        assert_eq!((s.position, s.offset, s.max_term_frequency), (23, 13, 2));
        let end = s.attribute_source.unwrap();
        assert_eq!((end.end_offset(), end.position_increment()), (2, 0));
        let mut docs = FieldInverter::new(IndexOptions::Docs);
        docs.add_document(0, &["x x"], &analyzer);
        docs.add_document(1, &["x"], &analyzer);
        assert_eq!(docs.invert_state(0).unwrap().max_term_frequency, 1);
        assert_eq!(docs.invert_state(1).unwrap().max_term_frequency, 0);
    }

    /// The flush trigger's inputs: the term count and the heap estimate, which
    /// grows with what is buffered; and positions clamp at `MAX_POSITION`
    /// rather than overflow, as `IndexingChain`'s do.
    #[test]
    fn term_count_ram_estimate_and_position_clamp() {
        let analyzer = Analyzer::standard(None);
        let mut inv = FieldInverter::new(IndexOptions::DocsAndFreqsAndPositions);
        let empty = inv.ram_bytes_used();
        assert_eq!(inv.num_terms(), 0);
        inv.add_document(0, &["alpha beta gamma alpha"], &analyzer);
        assert_eq!(inv.num_terms(), 3);
        let one_doc = inv.ram_bytes_used();
        assert!(one_doc >= empty);
        for d in 1..200 {
            inv.add_document(d, &[&format!("term{d} alpha")], &analyzer);
        }
        assert_eq!(inv.num_terms(), 202);
        assert!(inv.ram_bytes_used() > one_doc);

        assert_eq!(advance_position(3, 4), 7);
        assert_eq!(advance_position(MAX_POSITION, 1), MAX_POSITION);
        assert_eq!(advance_position(i32::MAX, 1), MAX_POSITION);
    }

    /// Short terms that differ only after their first byte or two -- the
    /// shape of a real vocabulary, and of the benchmark's `t0..t19999` --
    /// must spread over the table. Probing from the hash's low bits put them
    /// on a handful of slots (displacements in the thousands); this pins the
    /// longest probe run to something a well-mixed table actually produces.
    #[test]
    fn short_similar_terms_spread_over_the_table() {
        let mut h = TermHash::new();
        for i in 0..20_000 {
            let (_, added) = h.add(format!("t{i}").as_bytes());
            assert!(added);
        }
        assert_eq!(h.add(b"t123"), (123, false));
        let size = h.table.len();
        let mask = size - 1;
        let longest = h
            .table
            .iter()
            .enumerate()
            .filter(|(_, &e)| e != 0)
            .map(|(slot, &e)| {
                let home = home_slot(h.hashes[(e - 1) as usize], size);
                slot.wrapping_sub(home) & mask
            })
            .max()
            .unwrap();
        assert!(
            longest < 64,
            "longest probe run {longest} in a {size}-slot table"
        );
    }

    /// The general path's answer for the same input, reshaped the way
    /// `IndexWriter::build_postings_output` reshapes it.
    fn reference(
        docs: &[(i32, &str, &str)],
        analyzer: &Analyzer,
        field: &str,
        index_options: IndexOptions,
    ) -> Vec<TermPostings> {
        let inverted = invert_documents(docs, analyzer);
        let has_positions = matches!(
            index_options,
            IndexOptions::DocsAndFreqsAndPositions
                | IndexOptions::DocsAndFreqsAndPositionsAndOffsets
        );
        let has_offsets = index_options == IndexOptions::DocsAndFreqsAndPositionsAndOffsets;
        let mut out = Vec::new();
        for ((f, term), list) in &inverted.terms {
            if f != field {
                continue;
            }
            out.push(TermPostings {
                term: term.as_bytes().to_vec(),
                docs: list
                    .entries
                    .iter()
                    .map(|e| (e.doc_id, e.term_freq()))
                    .collect(),
                positions: if has_positions {
                    list.entries.iter().map(|e| e.positions()).collect()
                } else {
                    Vec::new()
                },
                offsets: if has_offsets {
                    list.entries.iter().map(|e| e.offsets()).collect()
                } else {
                    Vec::new()
                },
                payload_bytes: Vec::new(),
                payload_lengths: Vec::new(),
            });
        }
        out
    }

    fn corpus() -> Vec<(i32, Vec<&'static str>)> {
        vec![
            (0, vec!["The quick brown fox", "jumps over the lazy dog"]),
            (1, vec![]),
            (2, vec!["fox fox FOX; café, naïve résumé 3.14 1,000 U.S.A."]),
            (3, vec!["", "  ,, "]),
            (4, vec!["the the the"]),
            (5, vec!["don't stop O'Brien e-mail x_y"]),
        ]
    }

    #[test]
    fn matches_the_general_inverter_for_every_index_option_and_analyzer() {
        let stop = lucene_analysis::english_stop_words();
        let analyzers = [
            Analyzer::standard(None),
            Analyzer::standard(Some(&stop)),
            Analyzer::standard(Some(&stop))
                .with_position_increment_gap(7)
                .with_offset_gap(3),
            Analyzer::standard(None)
                .with_ascii_folding()
                .with_stemming(),
        ];
        for analyzer in &analyzers {
            for options in [
                IndexOptions::Docs,
                IndexOptions::DocsAndFreqs,
                IndexOptions::DocsAndFreqsAndPositions,
                IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
            ] {
                let mut inv = FieldInverter::new(options);
                let mut triples = Vec::new();
                for (doc, values) in corpus() {
                    inv.add_document(doc, &values, analyzer);
                    for v in values {
                        triples.push((doc, "body", v));
                    }
                }
                let want = reference(&triples, analyzer, "body", options);
                let lengths = inv.lengths().to_vec();
                let doc_count = inv.doc_count();
                let got = inv.into_term_postings();
                assert_eq!(got.len(), want.len(), "{options:?}");
                for (g, w) in got.iter().zip(&want) {
                    assert_eq!(g.term, w.term);
                    assert_eq!(g.docs, w.docs, "{:?}", String::from_utf8_lossy(&g.term));
                    assert_eq!(g.positions, w.positions);
                    assert_eq!(g.offsets, w.offsets);
                }
                // Doc count: documents that produced at least one token.
                let mut with_tokens: Vec<i32> = want
                    .iter()
                    .flat_map(|t| t.docs.iter().map(|d| d.0))
                    .collect();
                with_tokens.sort_unstable();
                with_tokens.dedup();
                assert_eq!(doc_count as usize, with_tokens.len());
                // Lengths: the sum of every term's freq per document.
                for (doc, values) in corpus() {
                    let expected: u32 = want
                        .iter()
                        .flat_map(|t| t.docs.iter())
                        .filter(|d| d.0 == doc)
                        .map(|d| d.1 as u32)
                        .sum();
                    let got_len = lengths[doc as usize];
                    if values.is_empty() {
                        assert_eq!(got_len, None);
                    } else {
                        assert_eq!(got_len, Some(expected), "doc {doc}");
                    }
                }
            }
        }
    }

    /// A corpus big enough for `invert_parallel` to split: repeated and
    /// unique terms, empty and missing values, multi-valued documents.
    fn big_corpus() -> Vec<Vec<String>> {
        let mut state = 0x9E37_79B9u32;
        (0..3001)
            .map(|doc| {
                let mut values = Vec::new();
                if doc % 97 == 0 {
                    return values;
                }
                for _ in 0..1 + doc % 3 {
                    let mut text = String::new();
                    for _ in 0..doc % 23 {
                        state ^= state << 13;
                        state ^= state >> 17;
                        state ^= state << 5;
                        text.push_str(&format!("w{} ", state % 700));
                    }
                    if doc % 11 == 0 {
                        text.push_str(&format!("only{doc}"));
                    }
                    values.push(text);
                }
                values
            })
            .collect()
    }

    fn assert_same_inversion(got: InvertedField, want: InvertedField, docs: usize) {
        assert_eq!(got.doc_count(), want.doc_count());
        assert_eq!(got.lengths(), want.lengths());
        for doc in 0..docs {
            assert_eq!(got.invert_state(doc), want.invert_state(doc), "doc {doc}");
        }
        let (got, want) = (got.into_term_postings(), want.into_term_postings());
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(&want) {
            assert_eq!(
                (&g.term, &g.docs, &g.positions, &g.offsets),
                (&w.term, &w.docs, &w.positions, &w.offsets)
            );
        }
    }

    /// `invert_parallel` (on a four-thread pool, so it really splits) and
    /// `merge_runs` over hand-made runs both build exactly the serial
    /// inversion, `DOCS`' new-term flag included.
    #[test]
    fn parallel_and_appended_runs_match_the_serial_inversion() {
        let corpus = big_corpus();
        let analyzer = Analyzer::standard(None);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        for options in [
            IndexOptions::Docs,
            IndexOptions::DocsAndFreqs,
            IndexOptions::DocsAndFreqsAndPositions,
            IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        ] {
            let serial = || {
                let mut inv = FieldInverter::new(options);
                for (doc, values) in corpus.iter().enumerate() {
                    let values: Vec<&str> = values.iter().map(String::as_str).collect();
                    inv.add_document(doc as i32, &values, &analyzer);
                }
                inv
            };
            let parallel = pool.install(|| {
                FieldInverter::invert_parallel(
                    options,
                    corpus.len(),
                    |doc, out: &mut Vec<&str>| out.extend(corpus[doc].iter().map(String::as_str)),
                    &analyzer,
                )
            });
            assert_same_inversion(parallel, serial().finish(), corpus.len());

            // Uneven runs, one of them empty, merged by hand.
            let mut runs = Vec::new();
            for run in [0..1, 1..1, 1..700, 700..2900, 2900..3001] {
                let mut part = FieldInverter::new(options);
                for (local, doc) in run.enumerate() {
                    let values: Vec<&str> = corpus[doc].iter().map(String::as_str).collect();
                    part.add_document(local as i32, &values, &analyzer);
                }
                runs.push(part);
            }
            let merged = pool.install(|| merge_runs(options, runs));
            assert_same_inversion(merged, serial().finish(), corpus.len());
        }
        // Too few documents to split: the serial path, same answer.
        let small = FieldInverter::invert_parallel(
            IndexOptions::DocsAndFreqs,
            10,
            |doc, out: &mut Vec<&str>| out.extend(corpus[doc].iter().map(String::as_str)),
            &analyzer,
        );
        assert_eq!(small.lengths().len(), 10);
    }

    #[test]
    fn the_term_hash_survives_growth_and_keeps_ids_stable() {
        let mut h = TermHash::new();
        let mut ids = Vec::new();
        for i in 0..5000u32 {
            let t = format!("t{i}");
            let (id, new) = h.add(t.as_bytes());
            assert!(new);
            assert_eq!(id, i);
            ids.push(t);
        }
        for (i, t) in ids.iter().enumerate() {
            assert_eq!(h.add(t.as_bytes()), (i as u32, false));
            assert_eq!(h.term(i as u32), t.as_bytes());
        }
        let sorted = h.sorted_ids();
        let mut want = ids.clone();
        want.sort();
        let got: Vec<&[u8]> = sorted.iter().map(|&id| h.term(id)).collect();
        let want: Vec<&[u8]> = want.iter().map(|s| s.as_bytes()).collect();
        assert_eq!(got, want);
    }

    /// The prefix-key sort against a plain byte sort where the eight-byte
    /// key alone cannot decide: the empty term, zero bytes (which the key's
    /// padding also produces), high bytes, and terms sharing a prefix of
    /// eight bytes or more.
    #[test]
    fn sorted_ids_order_by_bytes_past_the_prefix_key() {
        let terms: Vec<&[u8]> = vec![
            b"abcdefghij",
            b"abcdefgh",
            b"",
            b"abcdefgh\0",
            b"a",
            b"a\0",
            b"\0",
            b"\xff\xff",
            b"abcdefghi",
            b"abcdefgha",
            b"abcdefgh\xff",
            b"abcdefg",
            b"b",
        ];
        let mut h = TermHash::new();
        for t in &terms {
            h.add(t);
        }
        let got: Vec<&[u8]> = h.sorted_ids().iter().map(|&id| h.term(id)).collect();
        let mut want = terms.clone();
        want.sort();
        assert_eq!(got, want);
    }
}
