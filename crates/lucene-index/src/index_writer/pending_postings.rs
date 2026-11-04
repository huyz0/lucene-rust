//! The buffered segment's postings, inverted as each document arrives --
//! `FreqProxTermsWriter`'s per-field `BytesRefHash` and
//! `ParallelPostingsArray`, filled by `IndexingChain.processDocument`.
//!
//! A term is hashed once, into its field's [`TermHash`], and each occurrence
//! is appended to the term's streams: `(doc, freq)` pairs, and for a field
//! that indexes them, positions and `(start, end)` offsets. Every stream of
//! a field lives in one [`IntPool`] as a chain of slices that grow by level,
//! as Lucene's `IntBlockPool` and `ByteSlicePool` keep them, so a term costs
//! no allocation of its own and the whole buffer is freed in a few large
//! blocks. A document's positions count equals its frequency, so a stream
//! of positions needs no per-document boundaries.
//!
//! A document (or block) that fails part-way leaves nothing behind:
//! [`PendingPostings::rollback`] cuts every stream back to a [`Checkpoint`]
//! taken before it. Java instead keeps the partial postings and marks the
//! document deleted; this port never buffered a failed document.
//!
//! The flush lays each field's streams out contiguously, terms in byte
//! order and documents renumbered into an index sort's order when there is
//! one ([`PendingPostings::flushed`]), and hands the postings writer a
//! [`TermView`] of each term, read in place.

use std::collections::BTreeMap;

use lucene_codecs::field_infos::IndexOptions;
use lucene_codecs::postings_writer::TermSource;

use super::explicit::{explicit_error, InvertedField};
use super::Result;
use crate::inverter::{hash_bytes, TermHash};

/// Ints per slice at each level, the last repeating: `ByteBlockPool`'s
/// `LEVEL_SIZE_ARRAY`. A slice's last int links the next slice.
const LEVELS: [u32; 10] = [5, 14, 20, 30, 40, 40, 80, 80, 120, 200];
const TOP_LEVEL: u8 = (LEVELS.len() - 1) as u8;

/// A stream with no slice yet.
const NONE: u32 = u32::MAX;

fn level_size(level: u8) -> u32 {
    LEVELS[usize::from(level.min(TOP_LEVEL))]
}

fn to_u32(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

/// Where one stream's chain of slices starts and where its next int goes.
#[derive(Debug, Clone, Copy)]
struct Stream {
    start: u32,
    write: u32,
    /// The current slice's link slot: the stream is full at it.
    limit: u32,
    level: u8,
}

impl Stream {
    const EMPTY: Stream = Stream {
        start: NONE,
        write: NONE,
        limit: NONE,
        level: 0,
    };
}

/// One field's streams: chained slices in one growable block of ints.
#[derive(Debug, Default)]
struct IntPool {
    buf: Vec<i32>,
}

impl IntPool {
    /// Appends `v` to `s`; returns where it went.
    #[inline]
    fn push(&mut self, s: &mut Stream, v: i32) -> u32 {
        if s.write == s.limit {
            self.next_slice(s);
        }
        let at = s.write;
        self.buf[at as usize] = v;
        s.write = at.saturating_add(1);
        at
    }

    /// Starts `s`'s chain, or links a slice one level larger to its end.
    #[cold]
    fn next_slice(&mut self, s: &mut Stream) {
        let level = if s.start == NONE {
            0
        } else {
            s.level.saturating_add(1).min(TOP_LEVEL)
        };
        let size = level_size(level);
        let at = to_u32(self.buf.len());
        self.buf
            .resize(self.buf.len().saturating_add(size as usize), 0);
        if s.start == NONE {
            s.start = at;
        } else {
            self.buf[s.limit as usize] = i32::from_ne_bytes(at.to_ne_bytes());
        }
        s.write = at;
        s.limit = at.saturating_add(size).saturating_sub(1);
        s.level = level;
    }

    /// The first `n` ints of `s`.
    fn read(&self, s: &Stream, n: u32) -> StreamReader<'_> {
        StreamReader {
            buf: &self.buf,
            at: s.start,
            limit: s.start.saturating_add(level_size(0)).saturating_sub(1),
            level: 0,
            left: if s.start == NONE { 0 } else { n },
        }
    }

    /// `s` cut back to its first `n` ints.
    fn cut(&self, s: &Stream, n: u32) -> Stream {
        if s.start == NONE {
            return *s;
        }
        let mut r = self.read(s, n);
        while r.next().is_some() {}
        Stream {
            start: s.start,
            write: r.at,
            limit: r.limit,
            level: r.level,
        }
    }
}

/// Reads a stream's ints in order, following its slice links.
struct StreamReader<'a> {
    buf: &'a [i32],
    at: u32,
    limit: u32,
    level: u8,
    left: u32,
}

impl Iterator for StreamReader<'_> {
    type Item = i32;

    #[inline]
    fn next(&mut self) -> Option<i32> {
        if self.left == 0 {
            return None;
        }
        if self.at == self.limit {
            let link = self.buf[self.limit as usize];
            self.at = u32::from_ne_bytes(link.to_ne_bytes());
            self.level = self.level.saturating_add(1).min(TOP_LEVEL);
            self.limit = self
                .at
                .saturating_add(level_size(self.level))
                .saturating_sub(1);
        }
        let v = self.buf[self.at as usize];
        self.at = self.at.saturating_add(1);
        self.left = self.left.saturating_sub(1);
        Some(v)
    }
}

/// One term's streams and its newest document.
#[derive(Debug, Clone, Copy)]
struct TermRun {
    /// `(doc, freq)` pairs, doc ascending.
    docs: Stream,
    positions: Stream,
    /// `(start, end)` pairs, one per position.
    offsets: Stream,
    doc_count: u32,
    /// Positions written: the sum of the documents' frequencies.
    occurrences: u32,
    last_doc: i32,
    /// `last_doc`'s frequency so far, and where the pool holds it.
    freq: i32,
    freq_at: u32,
}

impl TermRun {
    const NEW: TermRun = TermRun {
        docs: Stream::EMPTY,
        positions: Stream::EMPTY,
        offsets: Stream::EMPTY,
        doc_count: 0,
        occurrences: 0,
        last_doc: -1,
        freq: 0,
        freq_at: 0,
    };
}

/// A term's payloads once its field has seen one: a length per position,
/// zero where an occurrence has none, and their bytes concatenated.
#[derive(Debug, Default)]
struct TermPayloads {
    lengths: Vec<u32>,
    bytes: Vec<u8>,
}

/// What [`FieldRuns::occurrence`] found for a term in the current document.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TermSlot {
    id: u32,
    /// The term was new to the buffered segment: `newTerm`.
    pub(crate) new_to_segment: bool,
    /// The term's frequency in the document so far; `None` before its
    /// first occurrence there.
    pub(crate) freq: Option<i32>,
}

/// One field's dictionary under construction.
#[derive(Debug)]
pub(crate) struct FieldRuns {
    positions: bool,
    offsets: bool,
    terms: TermHash,
    /// By term id (`TermHash` ids are dense, in insertion order).
    runs: Vec<TermRun>,
    pool: IntPool,
    /// By term id, once the field has seen a payload:
    /// `FreqProxTermsWriterPerField.sawPayloads`.
    payloads: Option<Vec<TermPayloads>>,
    /// Documents with at least one term in the field: the postings
    /// `docCount`.
    doc_count: i32,
    last_doc: i32,
    /// What the runs and dictionary grew by, as `bytesUsed` counts it: the
    /// RAM buffer's share of this field.
    bytes: usize,
    /// Scratch for [`PendingPostings::add_field`]: the document's term hashes.
    hashes: Vec<u64>,
}

/// How many terms ahead of the one being added [`FieldRuns::prefetch_ahead`]
/// starts loading the likely term's bytes and run, and (half as far) the pool
/// ints that run appends to.
const PREFETCH_TERMS: usize = 8;

/// A new term's fixed cost as `bytesUsed` counts it: its postings-array
/// entries, its hash slot share, start and hash.
const TERM_BYTES: usize = 24 + 4 + 4 + 8;

impl FieldRuns {
    fn new(index_options: IndexOptions) -> Self {
        FieldRuns {
            positions: index_options.subsumes_positions(),
            offsets: index_options.subsumes_offsets(),
            terms: TermHash::new(),
            runs: Vec::new(),
            pool: IntPool::default(),
            payloads: None,
            doc_count: 0,
            last_doc: -1,
            bytes: 0,
            hashes: Vec::new(),
        }
    }

    /// Starts loading what adding the terms after `i` will read: for the
    /// term [`PREFETCH_TERMS`] ahead, the bytes and run of the id its first
    /// table slot names (that slot was prefetched with every hash); for the
    /// term half as far ahead, whose run is in cache by now, the pool ints
    /// its documents and positions go to next.
    #[inline]
    fn prefetch_ahead(&self, hashes: &[u64], i: usize) {
        if let Some(id) = hashes
            .get(i.saturating_add(PREFETCH_TERMS))
            .and_then(|&h| self.terms.likely_id(h))
        {
            self.terms.prefetch_term(id);
            lucene_util::simd::prefetch(&self.runs, id as usize);
        }
        if let Some(run) = hashes
            .get(i.saturating_add(PREFETCH_TERMS / 2))
            .and_then(|&h| self.terms.likely_id(h))
            .and_then(|id| self.runs.get(id as usize))
        {
            lucene_util::simd::prefetch(&self.pool.buf, run.docs.write as usize);
            if self.positions {
                lucene_util::simd::prefetch(&self.pool.buf, run.positions.write as usize);
            }
        }
    }

    /// The first non-empty payload: every occurrence buffered so far gets
    /// its zero length, as `FreqProxTermsWriterPerField.writeProx` writes a
    /// length for every occurrence of a field that stores payloads.
    #[cold]
    fn start_payloads(&mut self) {
        self.payloads = Some(
            self.runs
                .iter()
                .map(|r| TermPayloads {
                    lengths: vec![0; r.occurrences as usize],
                    bytes: Vec::new(),
                })
                .collect(),
        );
    }

    /// `TermsHashPerField.add`'s lookup: `term`'s id, added to the
    /// dictionary if new, and its frequency in `doc` so far.
    #[inline]
    pub(crate) fn occurrence(&mut self, doc: i32, term: &[u8]) -> TermSlot {
        self.occurrence_hashed(doc, term, hash_bytes(term))
    }

    /// [`Self::occurrence`] for a term whose [`hash_bytes`] is known.
    #[inline]
    fn occurrence_hashed(&mut self, doc: i32, term: &[u8], h: u64) -> TermSlot {
        let (id, new_to_segment) = self.terms.add_hashed(term, h);
        if new_to_segment {
            self.runs.push(TermRun::NEW);
            if let Some(p) = &mut self.payloads {
                p.push(TermPayloads::default());
            }
            self.bytes = self
                .bytes
                .saturating_add(TERM_BYTES.saturating_add(term.len()));
        }
        let run = &self.runs[id as usize];
        TermSlot {
            id,
            new_to_segment,
            freq: (run.last_doc == doc).then_some(run.freq),
        }
    }

    /// The term's first occurrence in `doc`, at `freq`: `newTerm` or
    /// `addTerm`'s new-document half.
    #[inline]
    pub(crate) fn first_in_doc(&mut self, slot: TermSlot, doc: i32, freq: i32) {
        if self.last_doc != doc {
            self.last_doc = doc;
            self.doc_count = self.doc_count.saturating_add(1);
        }
        let run = &mut self.runs[slot.id as usize];
        self.pool.push(&mut run.docs, doc);
        run.freq_at = self.pool.push(&mut run.docs, freq);
        run.last_doc = doc;
        run.freq = freq;
        run.doc_count = run.doc_count.saturating_add(1);
        self.bytes = self
            .bytes
            .saturating_add(if self.positions { 12 } else { 8 });
    }

    /// Sets the term's frequency in the current document, past its first
    /// occurrence.
    #[inline]
    pub(crate) fn set_freq(&mut self, slot: TermSlot, freq: i32) {
        let run = &mut self.runs[slot.id as usize];
        if run.doc_count > 0 {
            run.freq = freq;
            self.pool.buf[run.freq_at as usize] = freq;
        }
    }

    /// One occurrence's position, offsets (kept when the field indexes them)
    /// and payload (empty is none): `writeProx` and `writeOffsets`. Only for
    /// a field that indexes positions.
    #[inline]
    pub(crate) fn prox(
        &mut self,
        slot: TermSlot,
        position: i32,
        offsets: (i32, i32),
        payload: &[u8],
    ) {
        // `writeProx` runs only for a field with positions: only there can a
        // payload switch `storePayloads` on.
        if !self.positions {
            return;
        }
        if !payload.is_empty() && self.payloads.is_none() {
            self.start_payloads();
        }
        let id = slot.id as usize;
        let run = &mut self.runs[id];
        self.pool.push(&mut run.positions, position);
        run.occurrences = run.occurrences.saturating_add(1);
        let mut bytes = 4usize;
        if self.offsets {
            self.pool.push(&mut run.offsets, offsets.0);
            self.pool.push(&mut run.offsets, offsets.1);
            bytes = bytes.saturating_add(8);
        }
        if let Some(p) = &mut self.payloads {
            let p = &mut p[id];
            p.bytes.extend_from_slice(payload);
            p.lengths.push(to_u32(payload.len()));
            bytes = bytes.saturating_add(4).saturating_add(payload.len());
        }
        self.bytes = self.bytes.saturating_add(bytes);
    }

    /// Term `id`'s documents, positions, offsets and payloads, appended to
    /// `out`'s columns with each document renumbered by `old_to_new` (and
    /// the documents put back in doc order) when given.
    fn lay_out(
        &self,
        id: u32,
        old_to_new: Option<&[usize]>,
        scratch: &mut Scratch,
        out: &mut FlushedField,
    ) -> Result<()> {
        let run = &self.runs[id as usize];
        let payloads = self.payloads.as_ref().map(|p| &p[id as usize]);
        let docs_from = out.docs.len();
        let pos_from = out.positions.len();
        let len_from = out.payload_lengths.len();
        let bytes_from = out.payload_bytes.len();
        let mut pairs = self.pool.read(&run.docs, run.doc_count.saturating_mul(2));
        let mut freqs = 0u32;
        match old_to_new {
            None => {
                while let (Some(doc), Some(freq)) = (pairs.next(), pairs.next()) {
                    out.docs.push((doc, freq));
                    if self.positions {
                        freqs = freqs.saturating_add(u32::try_from(freq).unwrap_or(0));
                        out.ends.push(freqs);
                    }
                }
                if self.positions {
                    out.positions
                        .extend(self.pool.read(&run.positions, run.occurrences));
                }
                if self.offsets {
                    let mut o = self
                        .pool
                        .read(&run.offsets, run.occurrences.saturating_mul(2));
                    while let (Some(s), Some(e)) = (o.next(), o.next()) {
                        out.offsets.push((s, e));
                    }
                }
                if let Some(p) = payloads {
                    out.payload_lengths.extend_from_slice(&p.lengths);
                    out.payload_bytes.extend_from_slice(&p.bytes);
                }
            }
            Some(map) => {
                scratch.clear();
                while let (Some(doc), Some(freq)) = (pairs.next(), pairs.next()) {
                    let new = usize::try_from(doc)
                        .ok()
                        .and_then(|d| map.get(d))
                        .and_then(|&n| i32::try_from(n).ok())
                        .ok_or_else(|| {
                            explicit_error("a buffered document outside the sort map")
                        })?;
                    let i = scratch.docs.len();
                    scratch.docs.push((doc, freq));
                    scratch.order.push((new, i));
                    let n = if self.positions {
                        u32::try_from(freq).unwrap_or(0)
                    } else {
                        0
                    };
                    scratch.starts.push((freqs, scratch.payload_at));
                    if let Some(p) = payloads {
                        let lengths = p
                            .lengths
                            .get(freqs as usize..freqs.saturating_add(n) as usize)
                            .unwrap_or_default();
                        scratch.payload_at = lengths
                            .iter()
                            .fold(scratch.payload_at, |a, &l| a.saturating_add(l as usize));
                    }
                    freqs = freqs.saturating_add(n);
                }
                if self.positions {
                    scratch
                        .positions
                        .extend(self.pool.read(&run.positions, run.occurrences));
                }
                if self.offsets {
                    let mut o = self
                        .pool
                        .read(&run.offsets, run.occurrences.saturating_mul(2));
                    while let (Some(s), Some(e)) = (o.next(), o.next()) {
                        scratch.offsets.push((s, e));
                    }
                }
                scratch.order.sort_unstable();
                let mut written = 0u32;
                for &(new, i) in &scratch.order {
                    out.docs.push((new, scratch.docs[i].1));
                    if !self.positions {
                        continue;
                    }
                    let (from, bytes_at) = scratch.starts[i];
                    let to = scratch
                        .starts
                        .get(i.saturating_add(1))
                        .map_or(freqs, |&(s, _)| s);
                    let span = from as usize..to as usize;
                    written = written.saturating_add(to.saturating_sub(from));
                    out.ends.push(written);
                    out.positions
                        .extend_from_slice(scratch.positions.get(span.clone()).unwrap_or_default());
                    if self.offsets {
                        out.offsets.extend_from_slice(
                            scratch.offsets.get(span.clone()).unwrap_or_default(),
                        );
                    }
                    if let Some(p) = payloads {
                        let lengths = p.lengths.get(span).unwrap_or_default();
                        let end = lengths
                            .iter()
                            .fold(bytes_at, |a, &l| a.saturating_add(l as usize));
                        out.payload_lengths.extend_from_slice(lengths);
                        out.payload_bytes
                            .extend_from_slice(p.bytes.get(bytes_at..end).unwrap_or_default());
                    }
                }
            }
        }
        // A document's positions are as many as its frequency.
        if self.positions && freqs != run.occurrences {
            return Err(explicit_error(
                "buffered positions disagree with the term frequencies",
            ));
        }
        out.term_bytes.extend_from_slice(self.terms.term(id));
        out.terms.push(TermSpan {
            term_end: to_u32(out.term_bytes.len()),
            docs: (to_u32(docs_from), to_u32(out.docs.len())),
            positions: (to_u32(pos_from), to_u32(out.positions.len())),
            payload_lengths: (to_u32(len_from), to_u32(out.payload_lengths.len())),
            payload_bytes: (to_u32(bytes_from), to_u32(out.payload_bytes.len())),
        });
        Ok(())
    }

    /// The field laid out for the postings writer: [`PendingPostings::flushed`].
    fn flushed(&self, number: i32, old_to_new: Option<&[usize]>) -> Result<FlushedField> {
        let ids: Vec<u32> = self
            .terms
            .sorted_ids()
            .into_iter()
            .filter(|&id| self.runs[id as usize].doc_count > 0)
            .collect();
        let (docs, occurrences) = ids.iter().fold((0usize, 0usize), |(d, o), &id| {
            let r = &self.runs[id as usize];
            (
                d.saturating_add(r.doc_count as usize),
                o.saturating_add(r.occurrences as usize),
            )
        });
        let term_bytes = ids
            .iter()
            .map(|&id| self.terms.term(id).len())
            .fold(0usize, usize::saturating_add);
        let mut out = FlushedField {
            number,
            doc_count: self.doc_count,
            payloads: self.payloads.is_some(),
            term_bytes: Vec::with_capacity(term_bytes),
            terms: Vec::with_capacity(ids.len()),
            docs: Vec::with_capacity(docs),
            ends: Vec::with_capacity(if self.positions { docs } else { 0 }),
            positions: Vec::with_capacity(occurrences),
            offsets: Vec::with_capacity(if self.offsets { occurrences } else { 0 }),
            payload_lengths: Vec::new(),
            payload_bytes: Vec::new(),
        };
        let mut scratch = Scratch::default();
        for id in ids {
            self.lay_out(id, old_to_new, &mut scratch, &mut out)?;
        }
        Ok(out)
    }

    /// `document`'s occurrences of `id`, as [`InvertedField`] holds them:
    /// what tests read back after inverting a document.
    #[cfg(test)]
    fn doc_term(&self, id: u32, doc: i32) -> Option<super::explicit::InvertedTerm> {
        let run = &self.runs[id as usize];
        let pairs: Vec<i32> = self
            .pool
            .read(&run.docs, run.doc_count.saturating_mul(2))
            .collect();
        let i = pairs.chunks(2).position(|c| c[0] == doc)?;
        let freq = pairs[i.saturating_mul(2).saturating_add(1)];
        let (from, n) = if self.positions {
            let from: usize = pairs.chunks(2).take(i).map(|c| c[1] as usize).sum();
            (from, freq as usize)
        } else {
            (0, 0)
        };
        let positions: Vec<i32> = self
            .pool
            .read(&run.positions, run.occurrences)
            .skip(from)
            .take(n)
            .collect();
        let offsets: Vec<i32> = if self.offsets {
            self.pool
                .read(&run.offsets, run.occurrences.saturating_mul(2))
                .skip(from.saturating_mul(2))
                .take(n.saturating_mul(2))
                .collect()
        } else {
            Vec::new()
        };
        let payloads = self.payloads.as_ref().map_or_else(Vec::new, |p| {
            let p = &p[id as usize];
            let mut at: usize = p.lengths[..from].iter().map(|&l| l as usize).sum();
            p.lengths[from..from.saturating_add(n)]
                .iter()
                .map(|&l| {
                    let end = at.saturating_add(l as usize);
                    let b = p.bytes[at..end].to_vec();
                    at = end;
                    b
                })
                .collect()
        });
        Some(super::explicit::InvertedTerm {
            term: self.terms.term(id).to_vec(),
            freq,
            positions,
            offsets: offsets.chunks(2).map(|c| (c[0], c[1])).collect(),
            payloads,
        })
    }
}

/// Reused per term by [`FieldRuns::lay_out`] under an index sort.
#[derive(Default)]
struct Scratch {
    docs: Vec<(i32, i32)>,
    /// `(new doc, index into docs)`, sorted into the new order.
    order: Vec<(i32, usize)>,
    /// Per document: where its positions and payload bytes start.
    starts: Vec<(u32, usize)>,
    payload_at: usize,
    positions: Vec<i32>,
    offsets: Vec<(i32, i32)>,
}

impl Scratch {
    fn clear(&mut self) {
        self.docs.clear();
        self.order.clear();
        self.starts.clear();
        self.payload_at = 0;
        self.positions.clear();
        self.offsets.clear();
    }
}

/// Where the buffered postings stood before a document or block: see
/// [`PendingPostings::rollback`].
#[derive(Debug)]
pub(crate) struct Checkpoint {
    doc: i32,
    fields: Vec<(i32, FieldMark)>,
}

/// One field's counts at a [`Checkpoint`].
#[derive(Debug, Clone, Copy)]
struct FieldMark {
    terms: usize,
    doc_count: i32,
    last_doc: i32,
    payloads: bool,
    bytes: usize,
}

/// Every field's buffered postings: see the module doc.
#[derive(Debug, Default)]
pub(crate) struct PendingPostings {
    fields: BTreeMap<i32, FieldRuns>,
}

impl PendingPostings {
    /// Field `number`'s dictionary, created with the registered field's
    /// `index_options` on its first term.
    pub(crate) fn field(&mut self, number: i32, index_options: IndexOptions) -> &mut FieldRuns {
        self.fields
            .entry(number)
            .or_insert_with(|| FieldRuns::new(index_options))
    }

    /// Inverts an explicit document's `field` for document `doc`, the
    /// buffer's newest. `index_options` is the registered field's.
    pub(crate) fn add_field(
        &mut self,
        doc: i32,
        field: &InvertedField,
        index_options: IndexOptions,
    ) -> Result<()> {
        if field.terms.is_empty() {
            return Ok(());
        }
        let f = self.field(field.field_number, index_options);
        // Every term's hash first, starting each one's table slot loading:
        // the table outgrows the caches, and lookups waiting on their slots
        // were most of the cost of buffering a document. Then, a few terms
        // ahead of the one being added, the likely term's bytes and run, and
        // a little later the pool ints its run appends to.
        let mut hashes = std::mem::take(&mut f.hashes);
        hashes.clear();
        for t in field.terms.iter() {
            f.terms.prefetch_slot(t.hash);
            hashes.push(t.hash);
        }
        for (i, t) in field.terms.iter().enumerate() {
            f.prefetch_ahead(&hashes, i);
            // every term once: `validate_explicit` refused a repeat
            let slot = f.occurrence_hashed(doc, t.term, hashes[i]);
            f.first_in_doc(slot, doc, t.freq);
            // the postings writer reads positions and offsets only for a
            // field that indexes positions
            if f.positions {
                for (i, &position) in t.positions.iter().enumerate() {
                    let offsets = t.offsets.get(i).copied().unwrap_or_default();
                    f.prox(slot, position, offsets, t.payloads.get(i));
                }
            }
        }
        f.hashes = hashes;
        Ok(())
    }

    /// Marks where the postings stand before document `doc` (a block's
    /// first) is inverted.
    pub(crate) fn checkpoint(&self, doc: i32) -> Checkpoint {
        Checkpoint {
            doc,
            fields: self
                .fields
                .iter()
                .map(|(&n, f)| {
                    let mark = FieldMark {
                        terms: f.runs.len(),
                        doc_count: f.doc_count,
                        last_doc: f.last_doc,
                        payloads: f.payloads.is_some(),
                        bytes: f.bytes,
                    };
                    (n, mark)
                })
                .collect(),
        }
    }

    /// Undoes everything since `checkpoint`: fields it did not know are
    /// dropped, terms it did not know leave the dictionary, and every run
    /// loses its documents from the checkpoint's on. It walks every buffered
    /// term, where Java marks the failed document deleted in O(1): a stream
    /// of refused documents against a large buffer pays per refusal.
    pub(crate) fn rollback(&mut self, checkpoint: Checkpoint) {
        let known: BTreeMap<i32, FieldMark> = checkpoint.fields.into_iter().collect();
        let cut_at = checkpoint.doc;
        self.fields.retain(|n, f| {
            let Some(&mark) = known.get(n) else {
                return false;
            };
            if f.runs.len() > mark.terms {
                f.runs.truncate(mark.terms);
                f.terms.truncate(mark.terms);
                if let Some(p) = &mut f.payloads {
                    p.truncate(mark.terms);
                }
            }
            if !mark.payloads {
                f.payloads = None;
            }
            for (id, run) in f.runs.iter_mut().enumerate() {
                if run.last_doc < cut_at {
                    continue;
                }
                let mut pairs = f.pool.read(&run.docs, run.doc_count.saturating_mul(2));
                let (mut keep, mut occurrences, mut last) = (0u32, 0u32, None);
                while let (Some(doc), Some(freq)) = (pairs.next(), pairs.next()) {
                    if doc >= cut_at {
                        break;
                    }
                    keep = keep.saturating_add(1);
                    occurrences = occurrences.saturating_add(u32::try_from(freq).unwrap_or(0));
                    last = Some((doc, freq));
                }
                run.docs = f.pool.cut(&run.docs, keep.saturating_mul(2));
                run.doc_count = keep;
                (run.last_doc, run.freq, run.freq_at) = match last {
                    Some((doc, freq)) => (doc, freq, run.docs.write.saturating_sub(1)),
                    None => (-1, 0, 0),
                };
                if f.positions {
                    run.positions = f.pool.cut(&run.positions, occurrences);
                    run.offsets = f.pool.cut(&run.offsets, occurrences.saturating_mul(2));
                    run.occurrences = occurrences;
                    if let Some(p) = f.payloads.as_mut().and_then(|p| p.get_mut(id)) {
                        p.lengths.truncate(occurrences as usize);
                        let bytes = p
                            .lengths
                            .iter()
                            .fold(0usize, |a, &l| a.saturating_add(l as usize));
                        p.bytes.truncate(bytes);
                    }
                }
            }
            (f.doc_count, f.last_doc, f.bytes) = (mark.doc_count, mark.last_doc, mark.bytes);
            true
        });
    }

    /// The fields that saw a non-empty payload: `storePayloads`.
    #[cfg(test)]
    fn payload_fields(&self) -> impl Iterator<Item = i32> + '_ {
        self.fields
            .iter()
            .filter(|(_, f)| f.payloads.is_some())
            .map(|(&n, _)| n)
    }

    /// Every field's postings laid out for the postings writer, documents
    /// renumbered by `old_to_new` (buffered doc id to sorted) under an index
    /// sort. The buffer is left as it was.
    pub(crate) fn flushed(&self, old_to_new: Option<&[usize]>) -> Result<FlushedPostings> {
        let fields = self
            .fields
            .iter()
            .filter(|(_, f)| f.doc_count > 0)
            .map(|(&n, f)| f.flushed(n, old_to_new))
            .collect::<Result<_>>()?;
        Ok(FlushedPostings { fields })
    }

    /// [`Self::flushed`], freeing each field's buffer once it is laid out.
    pub(crate) fn into_flushed(self, old_to_new: Option<&[usize]>) -> Result<FlushedPostings> {
        let fields = self
            .fields
            .into_iter()
            .filter(|(_, f)| f.doc_count > 0)
            .map(|(n, f)| f.flushed(n, old_to_new))
            .collect::<Result<_>>()?;
        Ok(FlushedPostings { fields })
    }

    /// What the buffered postings count toward the RAM buffer.
    pub(crate) fn bytes(&self) -> usize {
        self.fields
            .values()
            .map(|f| f.bytes)
            .fold(0, usize::saturating_add)
    }

    pub(crate) fn clear(&mut self) {
        self.fields.clear();
    }

    /// Field `number`'s terms of document `doc`, in no particular order.
    #[cfg(test)]
    pub(crate) fn doc_terms(&self, number: i32, doc: i32) -> Vec<super::explicit::InvertedTerm> {
        self.fields.get(&number).map_or_else(Vec::new, |f| {
            (0..f.runs.len() as u32)
                .filter_map(|id| f.doc_term(id, doc))
                .collect()
        })
    }
}

/// One term's place in a [`FlushedField`]'s columns.
#[derive(Debug, Clone, Copy)]
struct TermSpan {
    term_end: u32,
    /// Its documents, and their `ends`.
    docs: (u32, u32),
    /// Its positions, and their offsets when the field has them.
    positions: (u32, u32),
    payload_lengths: (u32, u32),
    payload_bytes: (u32, u32),
}

/// One field's postings as the writer reads them: every term's documents,
/// positions, offsets and payloads in contiguous columns, terms in byte
/// order.
#[derive(Debug)]
struct FlushedField {
    number: i32,
    doc_count: i32,
    payloads: bool,
    term_bytes: Vec<u8>,
    terms: Vec<TermSpan>,
    docs: Vec<(i32, i32)>,
    /// Per document, where its positions end within its term's: parallel
    /// to `docs` for a field with positions, else empty.
    ends: Vec<u32>,
    positions: Vec<i32>,
    offsets: Vec<(i32, i32)>,
    payload_lengths: Vec<u32>,
    payload_bytes: Vec<u8>,
}

/// The buffered postings laid out for a flush: [`PendingPostings::flushed`].
#[derive(Debug)]
pub(crate) struct FlushedPostings {
    fields: Vec<FlushedField>,
}

impl FlushedPostings {
    /// The fields that store payloads.
    pub(crate) fn payload_fields(&self) -> impl Iterator<Item = i32> + '_ {
        self.fields.iter().filter(|f| f.payloads).map(|f| f.number)
    }

    /// Every field's postings as the writer takes them, `(field, docCount,
    /// terms in byte order)`, field ascending.
    pub(crate) fn term_lists(&self) -> Vec<(i32, i32, Vec<TermView<'_>>)> {
        self.fields
            .iter()
            .map(|f| {
                let mut term_start = 0usize;
                let terms = f
                    .terms
                    .iter()
                    .map(|s| {
                        let range = |(a, b): (u32, u32)| a as usize..b as usize;
                        let term = &f.term_bytes[term_start..s.term_end as usize];
                        term_start = s.term_end as usize;
                        let docs = range(s.docs);
                        let positions = range(s.positions);
                        TermView {
                            term,
                            docs: &f.docs[docs.clone()],
                            ends: f.ends.get(docs).unwrap_or_default(),
                            positions: &f.positions[positions.clone()],
                            offsets: f.offsets.get(positions).unwrap_or_default(),
                            payload_lengths: &f.payload_lengths[range(s.payload_lengths)],
                            payload_bytes: &f.payload_bytes[range(s.payload_bytes)],
                        }
                    })
                    .collect();
                (f.number, f.doc_count, terms)
            })
            .collect()
    }

    /// Heap bytes held.
    pub(crate) fn ram_bytes(&self) -> usize {
        self.fields
            .iter()
            .map(|f| {
                [
                    f.term_bytes.capacity(),
                    f.terms
                        .capacity()
                        .saturating_mul(std::mem::size_of::<TermSpan>()),
                    f.docs.capacity().saturating_mul(8),
                    f.ends.capacity().saturating_mul(4),
                    f.positions.capacity().saturating_mul(4),
                    f.offsets.capacity().saturating_mul(8),
                    f.payload_lengths.capacity().saturating_mul(4),
                    f.payload_bytes.capacity(),
                ]
                .into_iter()
                .fold(0, usize::saturating_add)
            })
            .fold(0, usize::saturating_add)
    }
}

/// One term's flushed postings, read in place by the postings writer.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TermView<'a> {
    term: &'a [u8],
    docs: &'a [(i32, i32)],
    /// Parallel to `docs` for a field with positions, else empty.
    ends: &'a [u32],
    positions: &'a [i32],
    offsets: &'a [(i32, i32)],
    payload_lengths: &'a [u32],
    payload_bytes: &'a [u8],
}

impl TermView<'_> {
    /// Document `i`'s span `[start, end)` in the positions.
    fn span(&self, i: usize) -> std::ops::Range<usize> {
        let start = i
            .checked_sub(1)
            .and_then(|p| self.ends.get(p))
            .map_or(0, |&e| e as usize);
        start..self.ends.get(i).map_or(start, |&e| e as usize)
    }
}

impl TermSource for TermView<'_> {
    fn term(&self) -> &[u8] {
        self.term
    }
    fn docs(&self) -> &[(i32, i32)] {
        self.docs
    }
    fn position_lists(&self) -> usize {
        self.ends.len()
    }
    fn positions(&self, doc: usize) -> &[i32] {
        self.positions.get(self.span(doc)).unwrap_or_default()
    }
    fn offset_lists(&self) -> usize {
        if self.offsets.is_empty() {
            0
        } else {
            self.ends.len()
        }
    }
    fn offsets(&self, doc: usize) -> &[(i32, i32)] {
        self.offsets.get(self.span(doc)).unwrap_or_default()
    }
    fn payload_bytes(&self) -> &[u8] {
        self.payload_bytes
    }
    fn payload_lengths(&self) -> &[u32] {
        self.payload_lengths
    }
}

#[cfg(test)]
mod tests {
    use super::super::explicit::InvertedTerm;
    use super::*;

    const FULL: IndexOptions = IndexOptions::DocsAndFreqsAndPositionsAndOffsets;

    /// A term at `positions`, each at offsets `(2p, 2p + 1)`.
    fn term(t: &str, positions: &[i32], payloads: &[&[u8]]) -> InvertedTerm {
        InvertedTerm {
            term: t.as_bytes().to_vec(),
            freq: positions.len().max(1) as i32,
            positions: positions.to_vec(),
            offsets: positions
                .iter()
                .map(|&p| (p.saturating_mul(2), p.saturating_mul(2).saturating_add(1)))
                .collect(),
            payloads: payloads.iter().map(|p| p.to_vec()).collect(),
        }
    }

    fn add(
        p: &mut PendingPostings,
        field: i32,
        doc: i32,
        opts: IndexOptions,
        terms: Vec<InvertedTerm>,
    ) {
        let field = InvertedField {
            field_number: field,
            terms: terms.into(),
            norm: None,
        };
        p.add_field(doc, &field, opts).unwrap();
    }

    fn get(p: &PendingPostings, field: i32, doc: i32, t: &str) -> Option<InvertedTerm> {
        p.doc_terms(field, doc)
            .into_iter()
            .find(|x| x.term == t.as_bytes())
    }

    fn view<'a>(lists: &[(i32, i32, Vec<TermView<'a>>)], field: i32, t: &str) -> TermView<'a> {
        let (_, _, terms) = lists.iter().find(|(n, _, _)| *n == field).unwrap();
        *terms.iter().find(|v| v.term() == t.as_bytes()).unwrap()
    }

    /// Streams interleave in one pool across many slice levels, read back
    /// in order, and a cut at any length resumes appending from there.
    #[test]
    fn pool_streams_chain_slices_and_cut_anywhere() {
        let mut pool = IntPool::default();
        let (mut a, mut b) = (Stream::EMPTY, Stream::EMPTY);
        for i in 0..2_000 {
            pool.push(&mut a, i);
            pool.push(&mut b, -i);
        }
        assert!(pool.read(&a, 2_000).eq(0..2_000));
        assert!(pool.read(&b, 2_000).eq((0..2_000).map(|i: i32| -i)));
        // longest first: a cut overwrites what followed it
        for keep in [2_000u32, 777, 19, 18, 5, 4, 1, 0] {
            let mut c = pool.cut(&a, keep);
            pool.push(&mut c, 9_999);
            let got: Vec<i32> = pool.read(&c, keep + 1).collect();
            assert_eq!(got.len() as u32, keep + 1);
            assert!(got[..keep as usize].iter().copied().eq(0..keep as i32));
            assert_eq!(got[keep as usize], 9_999);
        }
        assert_eq!(pool.read(&Stream::EMPTY, 3).count(), 0);
    }

    /// A rollback across slice boundaries leaves every stream ready to
    /// take the next document.
    #[test]
    fn rollback_past_many_documents_keeps_appending() {
        let mut p = PendingPostings::default();
        for doc in 0..50 {
            add(&mut p, 0, doc, FULL, vec![term("a", &[doc, doc + 1], &[])]);
        }
        let cp = p.checkpoint(50);
        for doc in 50..80 {
            add(&mut p, 0, doc, FULL, vec![term("a", &[1, 2, 3], &[b"x"])]);
        }
        p.rollback(cp);
        add(&mut p, 0, 50, FULL, vec![term("a", &[7], &[])]);
        let flushed = p.flushed(None).unwrap();
        let lists = flushed.term_lists();
        let a = view(&lists, 0, "a");
        assert_eq!(a.docs().len(), 51);
        assert_eq!(a.docs()[49], (49, 2));
        assert_eq!(a.docs()[50], (50, 1));
        assert_eq!(a.positions(49), [49, 50]);
        assert_eq!(a.positions(50), [7]);
        assert_eq!(a.offsets(50), [(14, 15)]);
        assert!(a.payload_lengths().is_empty());
    }

    /// A rollback takes back a document's new terms, its occurrences of old
    /// ones, and the payloads it was the first to bring -- the zero lengths
    /// they backfilled into earlier documents included.
    #[test]
    fn rollback_undoes_a_document_down_to_its_payloads() {
        let mut p = PendingPostings::default();
        add(
            &mut p,
            0,
            0,
            FULL,
            vec![term("a", &[0, 1], &[]), term("b", &[2], &[])],
        );
        let (cp, bytes) = (p.checkpoint(1), p.bytes());
        add(
            &mut p,
            0,
            1,
            FULL,
            vec![term("a", &[0], &[b"xy"]), term("c", &[1], &[])],
        );
        // the first payload backfills a zero length for every occurrence
        assert_eq!(get(&p, 0, 0, "a").unwrap().payloads, [b"", b""]);
        assert_eq!(get(&p, 0, 1, "a").unwrap().payloads, [b"xy"]);
        assert_eq!(p.payload_fields().collect::<Vec<_>>(), [0]);
        p.rollback(cp);
        assert!(p.doc_terms(0, 1).is_empty());
        let a = get(&p, 0, 0, "a").unwrap();
        assert_eq!((a.positions, a.payloads), (vec![0, 1], vec![]));
        assert_eq!(p.payload_fields().count(), 0);
        assert_eq!(p.bytes(), bytes);
        let flushed = p.flushed(None).unwrap();
        let lists = flushed.term_lists();
        assert_eq!(lists.len(), 1);
        assert_eq!(lists[0].1, 1, "docCount");
        let terms: Vec<&[u8]> = lists[0].2.iter().map(TermSource::term).collect();
        assert_eq!(terms, [b"a", b"b"]);
        assert!(view(&lists, 0, "a").payload_bytes().is_empty());
    }

    /// Payloads the checkpoint already had survive, cut to its documents.
    #[test]
    fn rollback_keeps_the_payloads_before_the_checkpoint() {
        let mut p = PendingPostings::default();
        add(&mut p, 0, 0, FULL, vec![term("a", &[0], &[b"p"])]);
        let cp = p.checkpoint(1);
        add(&mut p, 0, 1, FULL, vec![term("a", &[0, 1], &[b"qq", b""])]);
        p.rollback(cp);
        let flushed = p.flushed(None).unwrap();
        let lists = flushed.term_lists();
        let a = view(&lists, 0, "a");
        assert_eq!(
            (a.payload_bytes(), a.payload_lengths()),
            (&b"p"[..], &[1u32][..])
        );
        assert_eq!(a.docs(), [(0, 1)]);
        // a field the checkpoint never saw goes altogether
        let cp = p.checkpoint(1);
        add(&mut p, 5, 1, IndexOptions::Docs, vec![term("k", &[], &[])]);
        p.rollback(cp);
        assert!(p.doc_terms(5, 1).is_empty());
        assert_eq!(p.flushed(None).unwrap().term_lists().len(), 1);
    }

    /// An index sort's remap moves each document's positions, offsets and
    /// payloads with it.
    #[test]
    fn remap_carries_each_documents_occurrences_into_sort_order() {
        let mut p = PendingPostings::default();
        add(&mut p, 0, 0, FULL, vec![term("a", &[0], &[b"a0"])]);
        add(&mut p, 1, 0, IndexOptions::Docs, vec![term("k", &[], &[])]);
        add(&mut p, 0, 1, FULL, vec![term("a", &[0, 3], &[b"", b"b"])]);
        add(&mut p, 0, 2, FULL, vec![term("a", &[5], &[b"cc"])]);
        add(&mut p, 1, 2, IndexOptions::Docs, vec![term("k", &[], &[])]);
        let remapped = p.flushed(Some(&[2, 0, 1])).unwrap();
        let lists = remapped.term_lists();
        let a = view(&lists, 0, "a");
        assert_eq!(a.docs(), [(0, 2), (1, 1), (2, 1)]);
        assert_eq!(a.position_lists(), 3);
        assert_eq!(a.positions(0), [0, 3]);
        assert_eq!(a.positions(1), [5]);
        assert_eq!(a.positions(2), [0]);
        assert_eq!(a.offset_lists(), 3);
        assert_eq!(a.offsets(0), [(0, 1), (6, 7)]);
        assert_eq!(a.offsets(2), [(0, 1)]);
        assert_eq!(a.payload_lengths(), [0, 1, 2, 2]);
        assert_eq!(a.payload_bytes(), b"bcca0");
        // a docs-only field has no occurrences to move
        let k = view(&lists, 1, "k");
        assert_eq!(k.docs(), [(1, 1), (2, 1)]);
        assert_eq!((k.position_lists(), k.offset_lists()), (0, 0));
        assert!(k.positions(0).is_empty() && k.offsets(0).is_empty());
        assert!(k.payload_bytes().is_empty() && k.payload_lengths().is_empty());
        // the buffered order is untouched
        let flushed = p.flushed(None).unwrap();
        assert_eq!(view(&flushed.term_lists(), 0, "a").docs()[0], (0, 1));
        let e = p.flushed(Some(&[0])).unwrap_err().to_string();
        assert!(e.contains("outside the sort map"), "{e}");
    }

    /// The document API's side of a run: frequencies counted per document,
    /// occurrences kept only where the field indexes them.
    #[test]
    fn occurrences_are_kept_as_the_field_indexes_them() {
        let mut p = PendingPostings::default();
        add(&mut p, 0, 0, FULL, Vec::new());
        assert!(
            p.flushed(None).unwrap().term_lists().is_empty(),
            "an empty field adds nothing"
        );
        let f = p.field(2, IndexOptions::DocsAndFreqsAndPositions);
        let slot = f.occurrence(0, b"x");
        assert!(slot.new_to_segment && slot.freq.is_none());
        f.first_in_doc(slot, 0, 1);
        f.prox(slot, 4, (9, 9), b"");
        let slot = f.occurrence(0, b"x");
        assert_eq!((slot.new_to_segment, slot.freq), (false, Some(1)));
        f.set_freq(slot, 2);
        f.prox(slot, 6, (9, 9), b"");
        let docs_only = p.field(3, IndexOptions::Docs);
        let slot = docs_only.occurrence(0, b"y");
        docs_only.first_in_doc(slot, 0, 1);
        docs_only.prox(slot, 1, (0, 1), b"ignored");
        let flushed = p.flushed(None).unwrap();
        let lists = flushed.term_lists();
        let x = view(&lists, 2, "x");
        assert_eq!((x.docs(), x.positions(0)), (&[(0, 2)][..], &[4, 6][..]));
        assert_eq!(x.offset_lists(), 0, "no offsets indexed");
        assert!(x.offsets(0).is_empty());
        assert!(get(&p, 3, 0, "y").unwrap().positions.is_empty());
        assert_eq!(
            p.payload_fields().count(),
            0,
            "a docs-only field stores no payloads"
        );
        assert!(p.bytes() > 0);
        p.clear();
        assert_eq!(p.bytes(), 0);
    }
}
