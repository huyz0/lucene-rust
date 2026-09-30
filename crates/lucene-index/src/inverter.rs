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
fn hash_bytes(bytes: &[u8]) -> u64 {
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
struct TermHash {
    arena: Vec<u8>,
    /// Term `i` is `arena[starts[i]..starts[i + 1]]`.
    starts: Vec<u32>,
    /// `id + 1`, or `0` for an empty slot. Power-of-two sized.
    table: Vec<u32>,
    hashes: Vec<u64>,
}

impl TermHash {
    fn new() -> Self {
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

    fn term(&self, id: u32) -> &[u8] {
        let (a, b) = (
            self.starts[id as usize],
            self.starts[(id as usize).wrapping_add(1)],
        );
        &self.arena[a as usize..b as usize]
    }

    /// The id of `bytes`, adding it if new; `true` when it was added.
    #[inline]
    fn add(&mut self, bytes: &[u8]) -> (u32, bool) {
        let h = hash_bytes(bytes);
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
        let size = self.table.len().saturating_mul(2);
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
    fn sorted_ids(&self) -> Vec<u32> {
        let mut ids: Vec<u32> = (0..self.len() as u32).collect();
        ids.sort_unstable_by(|&a, &b| self.term(a).cmp(self.term(b)));
        ids
    }
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
        let length = self.lengths.get(doc).copied().flatten()?;
        let (num_overlap, unique) = self.overlaps_and_unique.get(doc).copied()?;
        let end = self.ends.get(doc).copied()?;
        let clamp = |n: u32| i32::try_from(n).unwrap_or(i32::MAX);
        Some(crate::similarity::FieldInvertState {
            docs_only: self.index_options == IndexOptions::Docs,
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

    /// The buffered terms as the postings writer takes them, sorted by term
    /// bytes, with positions and offsets split per document.
    pub fn into_term_postings(mut self) -> Vec<TermPostings> {
        let with_positions = self.has_positions();
        let with_offsets = self.has_offsets();
        let ids = self.terms.sorted_ids();
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            let buf = std::mem::take(&mut self.postings[id as usize]);
            let mut positions = Vec::new();
            let mut offsets = Vec::new();
            if with_positions {
                positions.reserve_exact(buf.docs.len());
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
            out.push(TermPostings {
                term: self.terms.term(id).to_vec(),
                docs: buf.docs,
                positions,
                offsets,
                payload_bytes: Vec::new(),
                payload_lengths: Vec::new(),
            });
        }
        out
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
}
