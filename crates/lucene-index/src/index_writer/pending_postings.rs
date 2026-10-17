//! The buffered segment's postings, inverted as each document arrives --
//! `FreqProxTermsWriter`'s per-field `BytesRefHash` and
//! `ParallelPostingsArray`, filled by `IndexingChain.processDocument`.
//!
//! [`super::IndexWriter`] used to keep every buffered document's
//! [`InvertedField`] terms (a `Vec<u8>` and a positions `Vec` per term per
//! document) until flush, then hash them all again into one dictionary per
//! field. Allocating, re-hashing and finally dropping those per-document
//! vectors was most of the document API's indexing cost. Here a term is
//! hashed once, into its field's [`TermHash`], and its occurrence appended
//! to flat per-term runs: the document API inverts each token straight into
//! [`FieldRuns`], and an explicit document's terms are moved in as it is
//! buffered ([`PendingPostings::add_field`]). The document keeps its
//! field's norm; its `InvertedField` keeps no terms.
//!
//! A term's run is its `(doc, freq)` list plus, only for a field that
//! indexes positions, a [`TermProx`] -- so a docs-only field (a keyword
//! field, the commonest) costs one list per term.
//!
//! A document (or block) that fails part-way leaves nothing behind:
//! [`PendingPostings::rollback`] truncates every run back to a
//! [`Checkpoint`] taken before it. Java instead keeps the partial postings
//! and marks the document deleted; this port never buffered a failed
//! document, and still does not.
//!
//! The flush hands the postings writer a [`TermView`] of every run, read in
//! place. Doc ids are insertion order; a flush under an index sort first
//! remaps every term's documents into sort order
//! ([`PendingPostings::remap`]), as `FreqProxTermsWriter.flush` wraps its
//! `FreqProxFields` in `SortingTermsEnum`.

use std::collections::BTreeMap;

use lucene_codecs::field_infos::IndexOptions;
use lucene_codecs::postings_writer::TermSource;

use super::explicit::{explicit_error, InvertedField};
use super::Result;
use crate::inverter::TermHash;

/// One term's `(doc, freq)` across the buffered documents, doc ascending.
type DocList = Vec<(i32, i32)>;

/// One term's occurrences, for a field that indexes positions: parallel to
/// its [`DocList`].
#[derive(Debug, Default)]
struct TermProx {
    /// Where each document's positions end in `positions` (and `offsets`,
    /// when the field has them).
    ends: Vec<u32>,
    positions: Vec<i32>,
    offsets: Vec<(i32, i32)>,
    /// One length per position once the field has seen a payload, zero
    /// where an occurrence has none; their bytes concatenated.
    payload_lengths: Vec<u32>,
    payload_bytes: Vec<u8>,
}

impl TermProx {
    /// Document `i`'s span `[start, end)` in the positions.
    fn span(&self, i: usize) -> (usize, usize) {
        let start = i.checked_sub(1).map_or(0, |prev| self.ends[prev] as usize);
        (start, self.ends[i] as usize)
    }

    /// Keeps the first `keep` documents.
    fn truncate(&mut self, keep: usize) {
        if keep >= self.ends.len() {
            return;
        }
        let end = keep
            .checked_sub(1)
            .map_or(0, |last| self.ends[last] as usize);
        self.ends.truncate(keep);
        self.positions.truncate(end);
        self.offsets.truncate(end);
        if !self.payload_lengths.is_empty() {
            self.payload_lengths.truncate(end);
            let bytes = self
                .payload_lengths
                .iter()
                .map(|&l| l as usize)
                .fold(0, usize::saturating_add);
            self.payload_bytes.truncate(bytes);
        }
    }
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
    docs: Vec<DocList>,
    /// By term id when the field indexes positions; empty otherwise.
    prox: Vec<TermProx>,
    /// Documents with at least one term in the field: the postings
    /// `docCount`.
    doc_count: i32,
    last_doc: i32,
    /// `FreqProxTermsWriterPerField.sawPayloads`.
    payloads: bool,
    /// What the runs and dictionary grew by, as `bytesUsed` counts it: the
    /// RAM buffer's share of this field.
    bytes: usize,
}

/// A new term's fixed cost: its lists, its hash slot share, start and hash.
const TERM_BYTES: usize = std::mem::size_of::<DocList>() + 4 + 4 + 8;

impl FieldRuns {
    fn new(index_options: IndexOptions) -> Self {
        FieldRuns {
            positions: index_options.subsumes_positions(),
            offsets: index_options.subsumes_offsets(),
            terms: TermHash::new(),
            docs: Vec::new(),
            prox: Vec::new(),
            doc_count: 0,
            last_doc: -1,
            payloads: false,
            bytes: 0,
        }
    }

    /// The first non-empty payload: every occurrence buffered so far gets
    /// its zero length, as `FreqProxTermsWriterPerField.writeProx` writes a
    /// length for every occurrence of a field that stores payloads.
    fn start_payloads(&mut self) {
        self.payloads = true;
        for prox in &mut self.prox {
            prox.payload_lengths.resize(prox.positions.len(), 0);
        }
    }

    /// `TermsHashPerField.add`'s lookup: `term`'s id, added to the
    /// dictionary if new, and its frequency in `doc` so far.
    #[inline]
    pub(crate) fn occurrence(&mut self, doc: i32, term: &[u8]) -> TermSlot {
        let (id, new_to_segment) = self.terms.add(term);
        if new_to_segment {
            self.docs.push(Vec::new());
            let mut bytes = TERM_BYTES.saturating_add(term.len());
            if self.positions {
                self.prox.push(TermProx::default());
                bytes = bytes.saturating_add(std::mem::size_of::<TermProx>());
            }
            self.bytes = self.bytes.saturating_add(bytes);
        }
        let freq = match self.docs[id as usize].last() {
            Some(&(d, f)) if d == doc => Some(f),
            _ => None,
        };
        TermSlot {
            id,
            new_to_segment,
            freq,
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
        self.docs[slot.id as usize].push((doc, freq));
        self.bytes = self.bytes.saturating_add(8);
        if let Some(prox) = self.prox.get_mut(slot.id as usize) {
            prox.ends
                .push(u32::try_from(prox.positions.len()).unwrap_or(u32::MAX));
            self.bytes = self.bytes.saturating_add(4);
        }
    }

    /// Sets the term's frequency in the current document, past its first
    /// occurrence.
    #[inline]
    pub(crate) fn set_freq(&mut self, slot: TermSlot, freq: i32) {
        if let Some(last) = self.docs[slot.id as usize].last_mut() {
            last.1 = freq;
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
        if !payload.is_empty() && !self.payloads {
            self.start_payloads();
        }
        let (with_offsets, with_payloads) = (self.offsets, self.payloads);
        let Some(prox) = self.prox.get_mut(slot.id as usize) else {
            return;
        };
        prox.positions.push(position);
        let mut bytes = 4usize;
        if with_offsets {
            prox.offsets.push(offsets);
            bytes = bytes.saturating_add(8);
        }
        if with_payloads {
            prox.payload_bytes.extend_from_slice(payload);
            prox.payload_lengths
                .push(u32::try_from(payload.len()).unwrap_or(u32::MAX));
            bytes = bytes.saturating_add(4).saturating_add(payload.len());
        }
        self.bytes = self.bytes.saturating_add(bytes);
        if let Some(end) = prox.ends.last_mut() {
            *end = u32::try_from(prox.positions.len()).unwrap_or(u32::MAX);
        }
    }

    /// `document`'s occurrences of `id`, as [`InvertedField`] holds them:
    /// what tests read back after inverting a document.
    #[cfg(test)]
    fn doc_term(&self, id: u32, doc: i32) -> Option<super::explicit::InvertedTerm> {
        let docs = &self.docs[id as usize];
        let i = docs.iter().position(|&(d, _)| d == doc)?;
        let mut out = super::explicit::InvertedTerm {
            term: self.terms.term(id).to_vec(),
            freq: docs[i].1,
            positions: Vec::new(),
            offsets: Vec::new(),
            payloads: Vec::new(),
        };
        let Some(prox) = self.prox.get(id as usize) else {
            return Some(out);
        };
        let (start, end) = prox.span(i);
        out.positions = prox.positions[start..end].to_vec();
        if self.offsets {
            out.offsets = prox.offsets[start..end].to_vec();
        }
        if !prox.payload_lengths.is_empty() {
            let mut at: usize = prox.payload_lengths[..start]
                .iter()
                .map(|&l| l as usize)
                .sum();
            out.payloads = prox.payload_lengths[start..end]
                .iter()
                .map(|&l| {
                    let next = at.saturating_add(l as usize);
                    let p = prox.payload_bytes[at..next].to_vec();
                    at = next;
                    p
                })
                .collect();
        }
        Some(out)
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
        for t in &field.terms {
            // every term once: `validate_explicit` refused a repeat
            let slot = f.occurrence(doc, &t.term);
            f.first_in_doc(slot, doc, t.freq);
            // the postings writer reads positions and offsets only for a
            // field that indexes positions
            if f.positions {
                for (i, &position) in t.positions.iter().enumerate() {
                    let offsets = t.offsets.get(i).copied().unwrap_or_default();
                    let payload = t.payloads.get(i).map_or(&[][..], Vec::as_slice);
                    f.prox(slot, position, offsets, payload);
                }
            }
        }
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
                        terms: f.docs.len(),
                        doc_count: f.doc_count,
                        last_doc: f.last_doc,
                        payloads: f.payloads,
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
        self.fields.retain(|n, f| {
            let Some(&mark) = known.get(n) else {
                return false;
            };
            if f.docs.len() > mark.terms {
                f.docs.truncate(mark.terms);
                f.prox.truncate(mark.terms);
                f.terms.truncate(mark.terms);
            }
            for (id, docs) in f.docs.iter_mut().enumerate() {
                let keep = docs.partition_point(|&(d, _)| d < checkpoint.doc);
                docs.truncate(keep);
                if let Some(prox) = f.prox.get_mut(id) {
                    prox.truncate(keep);
                    if !mark.payloads {
                        prox.payload_lengths = Vec::new();
                        prox.payload_bytes = Vec::new();
                    }
                }
            }
            (f.doc_count, f.last_doc, f.payloads, f.bytes) =
                (mark.doc_count, mark.last_doc, mark.payloads, mark.bytes);
            true
        });
    }

    /// The fields that saw a non-empty payload: `storePayloads`.
    pub(crate) fn payload_fields(&self) -> impl Iterator<Item = i32> + '_ {
        self.fields
            .iter()
            .filter(|(_, f)| f.payloads)
            .map(|(&n, _)| n)
    }

    /// Every field's postings as the writer takes them, `(field, docCount,
    /// terms in byte order)`, field ascending. `remapped`, under an index
    /// sort, holds the runs in sorted doc order ([`Self::remap`]).
    pub(crate) fn term_lists<'a>(
        &'a self,
        remapped: Option<&'a RemappedRuns>,
    ) -> Vec<(i32, i32, Vec<TermView<'a>>)> {
        self.fields
            .iter()
            .filter(|(_, f)| f.doc_count > 0)
            .map(|(&n, f)| {
                let (docs, prox) = remapped
                    .and_then(|r| r.fields.get(&n))
                    .map_or((f.docs.as_slice(), f.prox.as_slice()), |(d, p)| {
                        (d.as_slice(), p.as_slice())
                    });
                let terms = f
                    .terms
                    .sorted_ids()
                    .into_iter()
                    .filter_map(|id| {
                        let id_docs = docs.get(id as usize)?;
                        (!id_docs.is_empty()).then(|| TermView {
                            term: f.terms.term(id),
                            docs: id_docs,
                            prox: prox.get(id as usize),
                        })
                    })
                    .collect();
                (n, f.doc_count, terms)
            })
            .collect()
    }

    /// Every run with its documents renumbered by `old_to_new` (buffered doc
    /// id to sorted) and put back in doc order, with their positions,
    /// offsets and payloads.
    pub(crate) fn remap(&self, old_to_new: &[usize]) -> Result<RemappedRuns> {
        let fields = self
            .fields
            .iter()
            .map(|(&n, f)| {
                let mut docs = Vec::with_capacity(f.docs.len());
                let mut prox = Vec::with_capacity(f.prox.len());
                for (id, id_docs) in f.docs.iter().enumerate() {
                    let (d, p) = remap_run(id_docs, f.prox.get(id), old_to_new)?;
                    docs.push(d);
                    prox.extend(p);
                }
                Ok((n, (docs, prox)))
            })
            .collect::<Result<_>>()?;
        Ok(RemappedRuns { fields })
    }

    /// What the buffered postings count toward the RAM buffer.
    pub(crate) fn bytes(&self) -> usize {
        self.fields
            .values()
            .map(|f| f.bytes)
            .fold(0, usize::saturating_add)
    }

    /// Heap bytes held, for the flush's segment-size estimate.
    pub(crate) fn ram_bytes(&self) -> usize {
        self.fields
            .values()
            .map(|f| {
                let lists = f
                    .docs
                    .capacity()
                    .saturating_mul(std::mem::size_of::<DocList>())
                    .saturating_add(
                        f.prox
                            .capacity()
                            .saturating_mul(std::mem::size_of::<TermProx>()),
                    );
                let docs = f
                    .docs
                    .iter()
                    .map(|d| d.capacity().saturating_mul(8))
                    .fold(lists, usize::saturating_add);
                f.prox
                    .iter()
                    .map(|p| {
                        p.ends
                            .capacity()
                            .saturating_mul(4)
                            .saturating_add(p.positions.capacity().saturating_mul(4))
                            .saturating_add(p.offsets.capacity().saturating_mul(8))
                            .saturating_add(p.payload_lengths.capacity().saturating_mul(4))
                            .saturating_add(p.payload_bytes.capacity())
                    })
                    .fold(
                        docs.saturating_add(f.terms.ram_bytes()),
                        usize::saturating_add,
                    )
            })
            .fold(0, usize::saturating_add)
    }

    pub(crate) fn clear(&mut self) {
        self.fields.clear();
    }

    /// Field `number`'s terms of document `doc`, in no particular order.
    #[cfg(test)]
    pub(crate) fn doc_terms(&self, number: i32, doc: i32) -> Vec<super::explicit::InvertedTerm> {
        self.fields.get(&number).map_or_else(Vec::new, |f| {
            (0..f.docs.len() as u32)
                .filter_map(|id| f.doc_term(id, doc))
                .collect()
        })
    }
}

/// The buffered runs in an index sort's doc order: [`PendingPostings::remap`].
#[derive(Debug)]
pub(crate) struct RemappedRuns {
    fields: BTreeMap<i32, (Vec<DocList>, Vec<TermProx>)>,
}

/// One term's buffered postings, read in place by the postings writer.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TermView<'a> {
    term: &'a [u8],
    docs: &'a [(i32, i32)],
    prox: Option<&'a TermProx>,
}

impl TermSource for TermView<'_> {
    fn term(&self) -> &[u8] {
        self.term
    }
    fn docs(&self) -> &[(i32, i32)] {
        self.docs
    }
    fn position_lists(&self) -> usize {
        self.prox.map_or(0, |p| p.ends.len())
    }
    fn positions(&self, doc: usize) -> &[i32] {
        self.prox.map_or(&[], |p| {
            let (start, end) = p.span(doc);
            p.positions.get(start..end).unwrap_or_default()
        })
    }
    fn offset_lists(&self) -> usize {
        self.prox
            .filter(|p| !p.offsets.is_empty())
            .map_or(0, |p| p.ends.len())
    }
    fn offsets(&self, doc: usize) -> &[(i32, i32)] {
        self.prox.map_or(&[], |p| {
            let (start, end) = p.span(doc);
            p.offsets.get(start..end).unwrap_or_default()
        })
    }
    fn payload_bytes(&self) -> &[u8] {
        self.prox.map_or(&[], |p| &p.payload_bytes)
    }
    fn payload_lengths(&self) -> &[u32] {
        self.prox.map_or(&[], |p| &p.payload_lengths)
    }
}

/// A term's `docs` (and `prox`, for a field with positions) with every
/// document renumbered by `old_to_new` and the documents, with their
/// positions, offsets and payloads, put back in doc order.
fn remap_run(
    docs: &[(i32, i32)],
    prox: Option<&TermProx>,
    old_to_new: &[usize],
) -> Result<(DocList, Option<TermProx>)> {
    let new_doc = |old: i32| -> Result<i32> {
        usize::try_from(old)
            .ok()
            .and_then(|o| old_to_new.get(o))
            .and_then(|&n| i32::try_from(n).ok())
            .ok_or_else(|| explicit_error("a buffered document outside the sort map"))
    };
    let mut order: Vec<(i32, usize)> = docs
        .iter()
        .enumerate()
        .map(|(i, &(doc, _))| Ok((new_doc(doc)?, i)))
        .collect::<Result<_>>()?;
    order.sort_unstable();
    let out_docs = order.iter().map(|&(doc, i)| (doc, docs[i].1)).collect();
    let Some(prox) = prox else {
        return Ok((out_docs, None));
    };
    let payloads = !prox.payload_lengths.is_empty();
    // where each document's payload bytes start, in buffered order
    let mut byte_starts = Vec::new();
    if payloads {
        let mut at = 0usize;
        for i in 0..prox.ends.len() {
            byte_starts.push(at);
            let (start, end) = prox.span(i);
            at = prox.payload_lengths[start..end]
                .iter()
                .map(|&l| l as usize)
                .fold(at, usize::saturating_add);
        }
    }
    let mut out = TermProx {
        ends: Vec::with_capacity(prox.ends.len()),
        positions: Vec::with_capacity(prox.positions.len()),
        offsets: Vec::with_capacity(prox.offsets.len()),
        payload_lengths: Vec::with_capacity(prox.payload_lengths.len()),
        payload_bytes: Vec::with_capacity(prox.payload_bytes.len()),
    };
    for &(_, i) in &order {
        let (start, end) = prox.span(i);
        out.positions.extend_from_slice(&prox.positions[start..end]);
        out.ends
            .push(u32::try_from(out.positions.len()).unwrap_or(u32::MAX));
        if !prox.offsets.is_empty() {
            out.offsets.extend_from_slice(&prox.offsets[start..end]);
        }
        if payloads {
            let lengths = &prox.payload_lengths[start..end];
            let from = byte_starts[i];
            let to = lengths
                .iter()
                .map(|&l| l as usize)
                .fold(from, usize::saturating_add);
            out.payload_lengths.extend_from_slice(lengths);
            out.payload_bytes
                .extend_from_slice(&prox.payload_bytes[from..to]);
        }
    }
    Ok((out_docs, Some(out)))
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
            terms,
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
        let lists = p.term_lists(None);
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
        let lists = p.term_lists(None);
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
        assert_eq!(p.term_lists(None).len(), 1);
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
        let remapped = p.remap(&[2, 0, 1]).unwrap();
        let lists = p.term_lists(Some(&remapped));
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
        assert_eq!(view(&p.term_lists(None), 0, "a").docs()[0], (0, 1));
        let e = p.remap(&[0]).unwrap_err().to_string();
        assert!(e.contains("outside the sort map"), "{e}");
    }

    /// The document API's side of a run: frequencies counted per document,
    /// occurrences kept only where the field indexes them.
    #[test]
    fn occurrences_are_kept_as_the_field_indexes_them() {
        let mut p = PendingPostings::default();
        add(&mut p, 0, 0, FULL, Vec::new());
        assert!(p.term_lists(None).is_empty(), "an empty field adds nothing");
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
        let lists = p.term_lists(None);
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
        assert!(p.ram_bytes() > 0 && p.bytes() > 0);
        p.clear();
        assert_eq!((p.bytes(), p.ram_bytes()), (0, 0));
    }
}
