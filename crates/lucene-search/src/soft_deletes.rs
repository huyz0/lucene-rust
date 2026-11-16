//! Task #48: soft-delete **visibility** — real Lucene's `IndexWriterConfig
//! .setSoftDeletesField` convention, where a document is invisible to search
//! if *either* its hard-delete bit is cleared in `.liv` *or* its
//! soft-deletes-field doc-values value is present for that doc.
//!
//! # Scope: what this is
//!
//! Real Lucene's soft-delete check
//! (`SoftDeletesDirectoryReaderWrapper`/`PendingSoftDeletes`) is exactly a
//! `DocValuesFieldExistsQuery` over the configured soft-deletes field: a doc
//! is soft-deleted if that field has *any* value for it in the current
//! (newest) version of the document — not a specific sentinel value, mere
//! presence. [`is_soft_deleted`]/[`effective_live_docs`] use that same
//! presence-based rule against an already-opened
//! [`lucene_codecs::doc_values::NumericEntry`] (see
//! `crates/lucene-codecs/src/doc_values.rs`).
//!
//! # Scope: what this deliberately is not
//!
//! **Incremental marking, through an update overlay.**
//! Real Lucene's soft-delete *write* path (`IndexWriter.softUpdateDocument`)
//! relies on `NumericDocValuesFieldUpdates`: an existing segment's doc-values
//! file gets a small delta record appended (a per-doc-values- generation "diff"
//! file), not rewritten from scratch, so marking one doc soft-deleted is cheap
//! even on a huge segment. [`lucene_codecs::doc_values_updates`] provides this
//! primitive in single-generation form: [`mark_soft_deleted_via_overlay`]
//! writes a tiny standalone overlay file recording just the one doc's
//! soft-deletes-field value (any value — presence is all that matters, see
//! [`is_soft_deleted`]'s doc comment) without touching the base `.dvd`/`.dvm`
//! bytes at all, and [`is_soft_deleted_with_overlay`]/
//! [`effective_live_docs_with_overlay`] check that overlay before falling back
//! to the base decode. This is a **single overlay round** (one file, one batch
//! of updates); it does not implement stacking many sequential update
//! generations with newest-wins semantics across rounds, nor
//! `SegmentCommitInfo`/`.si` `docValuesGen` metadata wiring (see that module's
//! own doc comment for the full scope statement). A caller needing a second
//! independent update round has to compose a fresh merged overlay itself rather
//! than relying on multi-generation stacking.
//!
//! The plain (non-overlay) [`is_soft_deleted`]/[`effective_live_docs`]
//! functions below are the right choice whenever no
//! overlay is in play (the common case — most segments never get an
//! incremental doc-values update at all).
//!
//! **No cross-segment orchestration.** Like [`crate::directory_reader`],
//! this operates on one already-opened segment's already-decoded
//! `NumericEntry` + `.dvd` bytes; a caller wires it in per segment.
//!
//! # Why a plain [`FixedBitSet`] output, not a new parameter everywhere
//!
//! Every scored-query function in `crate::lib`/`crate::multi_segment` already
//! takes `live_docs: Option<&FixedBitSet>` and treats `None` as "every doc is
//! live" (see e.g. `search_term_query`). [`effective_live_docs`] computes a
//! single combined bitset — hard-live AND NOT soft-deleted — that a caller
//! can pass into any of those *existing* parameters unchanged: no query
//! function's signature needs to grow an extra "and also check this
//! doc-values field" parameter, and every existing single-hard-delete-only
//! caller is completely unaffected (they simply never call
//! [`effective_live_docs`] and keep passing their own `live_docs` straight
//! through, exactly as before).

use std::collections::HashMap;

use lucene_codecs::doc_values::{self, NumericEntry};
use lucene_codecs::doc_values_updates;
use lucene_store::codec_util::ID_LENGTH;
use lucene_util::fixed_bit_set::FixedBitSet;

pub type Error = doc_values::Error;
pub type Result<T> = std::result::Result<T, Error>;

/// An already-opened numeric doc-values field configured as a segment's
/// soft-deletes field (`FieldInfo.soft_deletes_field == true`,
/// `doc_values_type == Numeric`): the decoded [`NumericEntry`] plus its
/// `.dvd` data bytes, exactly the two pieces
/// [`lucene_codecs::doc_values::numeric_value`] needs.
#[derive(Debug, Clone, Copy)]
pub struct SoftDeletesField<'a> {
    pub data: &'a [u8],
    pub entry: &'a NumericEntry,
}

/// Every document the soft-deletes field carries a value for, resolved in a
/// **single pass** over the field's presence encoding.
///
/// This is `FieldExistsQuery.getDocValuesDocIdSetIterator(field, reader)`, the
/// iterator `PendingSoftDeletes.applySoftDeletes` drains exactly once. Asking
/// [`is_soft_deleted`] per document instead re-derives that presence encoding
/// from the start of the region on every call -- for a sparse field that is a
/// fresh `IndexedDISI` walk per document, so building a live-docs bitset that
/// way costs `O(maxDoc * blocks)` where Java pays `O(maxDoc + cardinality)`.
enum PresentDocs {
    /// The field has no values at all (`docsWithFieldOffset == -2`).
    Nothing,
    /// Every document in `0..count` has a value (`docsWithFieldOffset == -1`).
    Every(usize),
    /// Exactly these documents, ascending (`IndexedDISI`).
    These(Vec<i32>),
}

fn corrupt(what: &str) -> doc_values::Error {
    doc_values::Error::Store(lucene_store::Error::Corrupted(format!(
        "soft-deletes field: {what}"
    )))
}

fn present_docs(field: &SoftDeletesField<'_>, max_doc: usize) -> Result<PresentDocs> {
    let entry = field.entry;
    if entry.is_empty_field() {
        return Ok(PresentDocs::Nothing);
    }
    if entry.is_dense() {
        // A dense entry's `numValues` is the field's document count. A negative
        // one is corrupt metadata, and must not silently become "every document
        // is soft-deleted" -- that would empty the segment.
        let count = usize::try_from(entry.num_values)
            .map_err(|_| corrupt(&format!("negative numValues {}", entry.num_values)))?;
        return Ok(PresentDocs::Every(count));
    }
    let start = usize::try_from(entry.docs_with_field_offset).map_err(|_| {
        corrupt(&format!(
            "negative docsWithFieldOffset {}",
            entry.docs_with_field_offset
        ))
    })?;
    let len = usize::try_from(entry.docs_with_field_length).map_err(|_| {
        corrupt(&format!(
            "negative docsWithFieldLength {}",
            entry.docs_with_field_length
        ))
    })?;
    let region = field
        .data
        .get(start..start.saturating_add(len))
        .ok_or_else(|| {
            corrupt(&format!(
                "presence region {start}..{} runs past the end of the {}-byte .dvd",
                start.saturating_add(len),
                field.data.len()
            ))
        })?;
    // Bounded by the segment as it decodes, so a corrupt region cannot
    // grow the list past `max_doc` entries.
    Ok(PresentDocs::These(
        lucene_codecs::indexed_disi::decode_doc_ids_below(
            region,
            entry.dense_rank_power,
            i32::try_from(max_doc).unwrap_or(i32::MAX),
        )?,
    ))
}

/// A fully-live bitset over `0..max_doc`, or a clone of the caller's hard
/// live-docs -- the starting point both `effective_live_docs` variants clear
/// bits out of, matching `PendingDeletes.getMutableBits()`.
///
/// **The returned bitset's length is `live_docs`' when there is one, and
/// `max_doc` only when there is not.** The two are the same number for every
/// caller in this port, but they arrive as separate parameters, so everything
/// downstream bounds against the *returned bitset's* own `len()` rather than
/// against `max_doc` -- see [`clear_present`].
fn hard_live_bits(live_docs: Option<&FixedBitSet>, max_doc: usize) -> FixedBitSet {
    match live_docs {
        Some(existing) => existing.clone(),
        None => {
            let mut all_live = FixedBitSet::new(max_doc);
            for i in 0..max_doc {
                // FBS: `all_live` is `FixedBitSet::new(max_doc)` two lines
                // above, so `max_doc` is its `len()`.
                all_live.set(i);
            }
            all_live
        }
    }
}

/// Clears every document `present` covers, skipping any the caller says is
/// shadowed by an update overlay. `PendingSoftDeletes.applySoftDeletes`'s
/// `bits.getAndClear(docID)` loop.
fn clear_present(
    bits: &mut FixedBitSet,
    present: &PresentDocs,
    max_doc: usize,
    skip: &dyn Fn(i32) -> bool,
) {
    // Bound every doc against **the bitset's own length**, not against
    // `max_doc`. `bits` comes from [`hard_live_bits`], which clones the
    // caller's `live_docs` when there is one -- a bitset whose length is the
    // caller's, not `max_doc`. They agree for every caller in this port, and
    // that is exactly the shape `docs/arithmetic-gate.md` names: an index
    // bounded against something other than the bitset it indexes is one
    // caller away from a ghost bit (a silently *live* document that should be
    // dead) or an index panic. Hoisted: one load, not one per doc.
    let num_bits = bits.len().min(max_doc);
    match present {
        PresentDocs::Nothing => {}
        PresentDocs::Every(count) => {
            let end = num_bits.min(*count);
            for doc in 0..end {
                if !skip(doc as i32) {
                    bits.clear(doc);
                }
            }
        }
        PresentDocs::These(docs) => {
            for &doc in docs {
                let doc_usize = doc as usize;
                if doc >= 0 && doc_usize < num_bits && !skip(doc) {
                    bits.clear(doc_usize);
                }
            }
        }
    }
}

/// A single doc's soft-delete state: real Lucene's rule is *presence*, not
/// value equality — `DocValuesFieldExistsQuery`, not a marker-value compare.
/// A doc the field has no value for at all (sparse encoding, `numeric_value`
/// returning `None`) is not soft-deleted; any value present — including `0`
/// — means it is.
pub fn is_soft_deleted(field: &SoftDeletesField<'_>, doc: i32) -> Result<bool> {
    Ok(doc_values::numeric_value(field.data, field.entry, doc)?.is_some())
}

/// The single per-doc visibility check: NOT
/// live if hard-deleted (`live_docs` bit cleared) OR soft-deleted (soft-
/// deletes field has a value for this doc) — an OR of "is invisible", i.e.
/// an AND of "is visible": `live(doc) == live_docs.get(doc) &&
/// !soft_deleted(doc)`.
///
/// `soft_deletes: None` means this segment has no soft-deletes field
/// configured at all (the common case) — falls back to pure hard-delete
/// visibility.
pub fn is_live(
    live_docs: Option<&FixedBitSet>,
    soft_deletes: Option<&SoftDeletesField<'_>>,
    doc: i32,
) -> Result<bool> {
    // `doc` is a caller-supplied `i32` and `live_docs` a caller-supplied
    // bitset: nothing here relates the two. A negative `doc` sign-extends
    // through `as usize` to `usize::MAX`, and a
    // `doc` merely past the end reads a ghost bit -- a document reported
    // *live* that the bitset never covered. A doc outside the live-docs
    // bitset is not live, which is the same answer `vector_query`'s accept
    // filter already gives.
    let hard_live = live_docs.is_none_or(|bits| bits.get_doc(doc));
    if !hard_live {
        return Ok(false);
    }
    if let Some(field) = soft_deletes {
        if is_soft_deleted(field, doc)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Computes a single combined live-docs bitset — hard-live AND NOT
/// soft-deleted, for every doc `0..max_doc` — suitable to pass directly into
/// any existing `live_docs: Option<&FixedBitSet>` parameter in this crate
/// (see this module's doc comment for why that's the chosen integration
/// point rather than a new parameter on every query function).
///
/// Returns `live_docs` unchanged (cloned) when `soft_deletes` is `None` — no
/// new allocation-shape surprises for the (still overwhelmingly common)
/// case of a segment with no soft-deletes field configured at all.
pub fn effective_live_docs(
    live_docs: Option<&FixedBitSet>,
    soft_deletes: Option<&SoftDeletesField<'_>>,
    max_doc: usize,
) -> Result<Option<FixedBitSet>> {
    let Some(field) = soft_deletes else {
        return Ok(live_docs.cloned());
    };

    let present = present_docs(field, max_doc)?;
    let mut bits = hard_live_bits(live_docs, max_doc);
    clear_present(&mut bits, &present, max_doc, &|_| false);
    Ok(Some(bits))
}

/// Task #54's concrete unblocking of this module's write-side gap: marks a
/// single document soft-deleted by writing **only** a tiny update-overlay
/// file (`lucene_codecs::doc_values_updates::write_numeric_updates`) — the
/// base segment's `.dvd`/`.dvm` bytes are never read, decoded, or rewritten
/// by this function at all. The overlay records `(doc, marker_value)`; per
/// [`is_soft_deleted`]'s presence-based rule, `marker_value`'s actual number
/// is irrelevant (any value in the overlay means "soft-deleted"), so `0` is
/// used here as an arbitrary constant. Composing this with any *other*
/// already-buffered updates for the same segment (multiple docs marked
/// before one flush) is the caller's responsibility — pass every
/// `(doc, value)` pair accumulated so far; this is still exactly one overlay
/// generation, not a
/// stacked-generations API.
pub fn mark_soft_deleted_via_overlay(
    docs: &[i32],
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> Vec<u8> {
    let updates: Vec<(i32, Option<i64>)> = docs.iter().map(|&doc| (doc, Some(0i64))).collect();
    doc_values_updates::write_numeric_updates(&updates, segment_id, segment_suffix)
}

/// The overlay-aware counterpart to [`is_soft_deleted`]: a doc is
/// soft-deleted if it has a value in the overlay (marked via
/// [`mark_soft_deleted_via_overlay`] or any other overlay write) OR the base
/// doc-values field has a value for it — the same presence-based OR
/// [`is_soft_deleted`] already implements, just with an extra source of
/// presence to check first.
pub fn is_soft_deleted_with_overlay(
    field: &SoftDeletesField<'_>,
    overlay: &HashMap<i32, Option<i64>>,
    doc: i32,
) -> Result<bool> {
    match overlay.get(&doc) {
        // A value in the overlay means "soft-deleted", same presence-based
        // rule the base check uses.
        Some(Some(_)) => return Ok(true),
        // A `reset` entry (`DocValuesFieldUpdates.reset`) removes the doc's
        // soft-deletes value, so the doc is *not* soft-deleted -- and the
        // removal shadows whatever the base holds, so don't fall through.
        Some(None) => return Ok(false),
        None => {}
    }
    is_soft_deleted(field, doc)
}

/// The overlay-aware counterpart to [`effective_live_docs`]: identical
/// hard-live-AND-NOT-soft-deleted computation, except a doc's soft-delete
/// state is resolved through the overlay first (see
/// [`is_soft_deleted_with_overlay`]) rather than the base doc-values decode
/// alone — this is what lets a doc marked soft-deleted purely via
/// [`mark_soft_deleted_via_overlay`] (no base rewrite) actually become
/// invisible to search.
pub fn effective_live_docs_with_overlay(
    live_docs: Option<&FixedBitSet>,
    soft_deletes: Option<&SoftDeletesField<'_>>,
    overlay: &HashMap<i32, Option<i64>>,
    max_doc: usize,
) -> Result<Option<FixedBitSet>> {
    let present = match soft_deletes {
        Some(field) => present_docs(field, max_doc)?,
        None if overlay.is_empty() => return Ok(live_docs.cloned()),
        // No base soft-deletes field configured, but the overlay itself
        // carries soft-delete marks (e.g. a soft-deletes field introduced
        // only via updates, never present in the base segment) -- still
        // apply them.
        None => PresentDocs::Nothing,
    };

    let mut bits = hard_live_bits(live_docs, max_doc);
    // The base pass skips any document the overlay speaks for: an overlay
    // entry -- a value *or* a `reset` removal -- shadows the base entirely,
    // matching [`is_soft_deleted_with_overlay`].
    clear_present(&mut bits, &present, max_doc, &|doc| {
        overlay.contains_key(&doc)
    });
    // Same rule as `clear_present`: the bound is the bitset's, not `max_doc`.
    let num_bits = bits.len().min(max_doc);
    for (&doc, value) in overlay {
        if value.is_some() && doc >= 0 && (doc as usize) < num_bits {
            bits.clear(doc as usize);
        }
    }

    Ok(Some(bits))
}

impl crate::directory_reader::DirectoryReader {
    /// `new SoftDeletesDirectoryReaderWrapper(reader, field)`: this reader
    /// with every document that has a doc value for `field` deleted, and the
    /// segments left without a live document dropped. A segment with no
    /// such document (or no such field) is shared as it is; one with some
    /// gets new live documents (hard and soft) under a new reader cache key,
    /// its core shared. Reopening the result wraps again.
    ///
    /// # Errors
    /// [`crate::directory_reader::Error::SoftDeletes`] when a segment's
    /// docs-with-field region for `field` does not decode.
    pub fn soft_deletes_wrapper(&self, field: &str) -> crate::directory_reader::Result<Self> {
        let mut segments = Vec::with_capacity(self.segment_readers().len());
        for reader in self.segment_readers() {
            let wrapped = soft_deletes_leaf(reader, field)
                .map_err(|e| crate::directory_reader::Error::SoftDeletes(Box::new(e)))?;
            let wrapped = wrapped.unwrap_or_else(|| reader.clone());
            if wrapped.num_docs() != 0 {
                segments.push(wrapped);
            }
        }
        Ok(self.with_segments(segments, field.into()))
    }
}

/// `SoftDeletesDirectoryReaderWrapper.wrap(reader, field)`: `reader` with
/// its soft deletes applied, or `None` when it has none to apply (`wrap`
/// returning the reader itself).
fn soft_deletes_leaf(
    reader: &crate::directory_reader::SegmentReader,
    field: &str,
) -> crate::Result<Option<crate::directory_reader::SegmentReader>> {
    use crate::directory_reader::ExistsDocs;
    let Some(exists) = reader.doc_values_exists_docs(field)? else {
        return Ok(None);
    };
    let max_doc = usize::try_from(reader.max_doc).unwrap_or(0);
    let mut bits = match reader.live_docs() {
        Some(live) => live.clone(),
        None => {
            let mut all = FixedBitSet::new(max_doc);
            all.set_range(0, max_doc);
            all
        }
    };
    // `PendingSoftDeletes.applySoftDeletes`: the live documents cleared.
    let mut num_soft_deletes = 0usize;
    let mut clear = |bits: &mut FixedBitSet, doc: usize| {
        if doc < bits.len() && bits.get(doc) {
            bits.clear(doc);
            num_soft_deletes += 1;
        }
    };
    match exists {
        ExistsDocs::None => {}
        ExistsDocs::All => (0..max_doc).for_each(|doc| clear(&mut bits, doc)),
        ExistsDocs::Bits(docs) => {
            let mut from = 0;
            while let Some(doc) = docs.next_set_bit(from).filter(|&d| d < max_doc) {
                clear(&mut bits, doc);
                from = doc + 1;
            }
        }
    }
    if num_soft_deletes == 0 {
        return Ok(None);
    }
    Ok(Some(reader.with_live_docs(bits)))
}

#[cfg(test)]
mod tests;
