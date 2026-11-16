//! Resolved document deletes for an already-flushed segment.
//!
//! Java: PendingDeletes.writeLiveDocs and ReadersAndUpdates.writeLiveDocs.
//! Term/query resolution belongs to the writer; this module clears the resolved
//! IDs, writes the next live-docs generation (unsynced: the commit that names
//! it syncs it, as Java's `IndexWriter.startCommit` does), and updates segment
//! metadata only after success. The optional codec writer supplies native
//! Lucene90 bytes; the index retains generation and cleanup ownership.

use lucene_codecs::codec::Lucene104Codec;
use lucene_codecs::live_docs;
use lucene_codecs::live_docs_writer::LiveDocsWriteState;
use lucene_store::directory::Directory;
use lucene_util::fixed_bit_set::FixedBitSet;

use crate::segment_infos::SegmentCommitInfo;

use crate::created_outputs::CreatedOutputs;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    #[error(transparent)]
    LiveDocs(#[from] live_docs::Error),
    #[error("doc id {doc_id} out of range for max_doc={max_doc}")]
    DocOutOfRange { doc_id: i32, max_doc: usize },
    #[error("invalid deletion count: {del_count} vs maxDoc={max_doc} (segment={segment})")]
    DelCountExceedsMaxDoc {
        segment: String,
        del_count: i32,
        max_doc: usize,
    },
}

pub type Result<T> = std::result::Result<T, Error>;

/// The real `IndexFileNames.fileNameFromGeneration(segment, "liv", delGen)`
/// convention for a segment's live-docs file: `_<segment>_<delGen in base
/// 36>.liv` (e.g. segment `_0`, delGen 1 -> `_0_1.liv`; delGen 36 ->
/// `_0_10.liv`). Real Lucene encodes the generation suffix in base 36 (see
/// `Long.toString(gen, 36)` in `IndexFileNames.fileNameFromGeneration`), the
/// same convention `lucene_util::base36`/`live_docs.rs`'s own index-header
/// suffix already use in this port -- reusing `to_base36` here instead of
/// plain decimal formatting keeps this new filename convention consistent
/// with the rest of the port rather than inventing a diverging one. Nothing
/// in this port established this filename shape before this module --
/// `SegmentCommitInfo.del_gen` was tracked purely as metadata (see
/// `segment_infos.rs`/`segment_info.rs`, both of which read/write `del_gen`
/// as an integer with no filename derived from it).
pub fn liv_file_name(segment_name: &str, del_gen: i64) -> String {
    if del_gen == 0 {
        // `IndexFileNames.fileNameFromGeneration`'s generation-0 branch:
        // `segmentFileName(base, "", ext)`, i.e. no generation suffix at all.
        // Real Lucene never reaches it for `.liv` (`getNextDelGen()` goes
        // -1 -> 1), but the naming rule is the naming rule: emitting
        // `_0_0.liv` here would send every caller looking for a file no
        // Lucene writer would ever produce.
        return format!("{segment_name}.liv");
    }
    format!(
        "{segment_name}_{}.liv",
        lucene_util::base36::to_base36(del_gen)
    )
}

/// The mechanical "mark these doc IDs deleted" primitive: given a segment's
/// current live-docs state (`None` means "all `max_doc` docs are live",
/// matching a `SegmentCommitInfo` with `del_gen == -1`; `Some(bits)` means an
/// existing, already-partially-deleted bitset) and a batch of doc IDs to
/// delete, returns a new bitset with exactly those bits cleared, plus the
/// count of doc IDs that were *newly* turned non-live this call (deleting an
/// already-deleted doc is idempotent and does not double-count).
///
/// Returns [`Error::DocOutOfRange`] for any `doc_id` outside `0..max_doc`
/// rather than silently ignoring it or panicking -- an out-of-range doc ID
/// means the caller and this segment disagree about `max_doc`, which is a
/// caller bug worth surfacing, not a case to paper over.
pub fn mark_deleted(
    live_docs: Option<&FixedBitSet>,
    max_doc: usize,
    doc_ids: impl IntoIterator<Item = i32>,
) -> Result<(FixedBitSet, usize)> {
    let mut bits = match live_docs {
        Some(existing) => existing.clone(),
        // `FixedBitSet.set(0, numBits)` -- fill whole words at once instead
        // of `max_doc` individual read-modify-write bit sets (which is what
        // a per-bit loop compiles to). The final partial word is masked so
        // no bit past `max_doc` is ever set, matching `FixedBitSet`'s
        // invariant that trailing bits stay zero (`cardinality` counts raw
        // words).
        None => {
            let words = lucene_util::fixed_bit_set::bits2words(max_doc);
            let mut all_ones = vec![u64::MAX; words];
            let tail = max_doc & 63;
            if tail != 0 {
                if let Some(last) = all_ones.last_mut() {
                    // ARITH: `tail` is `max_doc & 63`, so `1..=63` inside this
                    // branch; `1u64 << 63` is `2^63` and `- 1` cannot
                    // underflow (the shift result is at least 2).
                    #[allow(clippy::arithmetic_side_effects)]
                    let mask = (1u64 << tail) - 1;
                    *last = mask;
                }
            }
            FixedBitSet::from_words(all_ones, max_doc)
        }
    };

    // Bound every doc ID against **the bitset's own length**, never against
    // `max_doc`. The two are the same number for every caller in this port,
    // but `live_docs` is a caller-supplied `&FixedBitSet` and `max_doc` a
    // separate caller-supplied `usize`: if they ever disagree, bounding on
    // `max_doc` and then indexing `bits` is precisely the shape
    // `FixedBitSet::get` turns into a panic, in release as well as debug.
    // This function's own doc comment promises `DocOutOfRange` "rather than ... panicking", so the
    // bound has to be the one that actually governs the indexing. Hoisted:
    // one load, not one per doc ID.
    let num_bits = bits.len();
    let mut newly_deleted = 0usize;
    for doc_id in doc_ids {
        let idx = match usize::try_from(doc_id) {
            Ok(idx) if idx < num_bits && idx < max_doc => idx,
            _ => return Err(Error::DocOutOfRange { doc_id, max_doc }),
        };
        if bits.get(idx) {
            bits.clear(idx);
            // ARITH: the bit is cleared in the same breath, so each of the
            // `max_doc` doc IDs can increment this at most once however many
            // times `doc_ids` repeats it -- `newly_deleted <= max_doc`, and
            // `max_doc` is a `usize` this bitset was sized from.
            #[allow(clippy::arithmetic_side_effects)]
            {
                newly_deleted += 1;
            }
        }
        // Already-deleted: idempotent no-op, not double-counted.
    }

    Ok((bits, newly_deleted))
}

/// Applies a batch of newly-deleted doc IDs to `sci` (an already-flushed
/// segment's current [`SegmentCommitInfo`]): resolves the updated live-docs
/// bitset via [`mark_deleted`], writes it as that segment's next-generation
/// `.liv` file (via [`liv_file_name`] + [`lucene_codecs::live_docs::write`]),
/// and returns a new `SegmentCommitInfo` with
/// `del_gen` incremented (starting at `1` the first time a segment gets any
/// deletions, matching real `SegmentCommitInfo.getNextDelGen()`: `delGen ==
/// -1` -> next is `1`, otherwise `delGen + 1`) and `del_count` increased by
/// the number of *newly* deleted docs this call (previously deleted docs from
/// an earlier generation stay deleted and are not re-counted).
///
/// `max_doc` is the segment's total doc count (from its `.si`), needed to
/// size a from-scratch "all live" bitset when `sci.del_gen == -1` and to
/// bounds-check `doc_ids`. `current_live_docs` is `None` if `sci.del_gen ==
/// -1` (no `.liv` file exists yet), or the already-parsed bitset from that
/// segment's current-generation `.liv` file otherwise -- the caller is
/// expected to have read it via `live_docs::parse` beforehand (this module
/// doesn't re-derive it from `sci` alone, since reading the current `.liv`
/// file is the caller's I/O to do, matching how `merge.rs`'s `MergeSource`
/// takes an already-parsed `live_docs` rather than re-opening it itself).
pub fn apply_deletes(
    dir: &dyn Directory,
    sci: &SegmentCommitInfo,
    current_live_docs: Option<&FixedBitSet>,
    max_doc: usize,
    doc_ids: impl IntoIterator<Item = i32>,
) -> Result<SegmentCommitInfo> {
    apply_deletes_with_codec(dir, sci, current_live_docs, max_doc, doc_ids, None)
}

/// `PendingDeletes.writeLiveDocs`: the configured same-wire writer, with
/// generation publication and failed-output cleanup retained by the index.
pub(crate) fn apply_deletes_with_codec(
    dir: &dyn Directory,
    sci: &SegmentCommitInfo,
    current_live_docs: Option<&FixedBitSet>,
    max_doc: usize,
    doc_ids: impl IntoIterator<Item = i32>,
    codec: Option<&dyn Lucene104Codec>,
) -> Result<SegmentCommitInfo> {
    let (new_bits, newly_deleted) = mark_deleted(current_live_docs, max_doc, doc_ids)?;

    // `SegmentCommitInfo.getNextDelGen()`, not a fresh `del_gen + 1`
    // derivation: after a crash, `SegmentInfos.inflateGens` pushes
    // `next_write_del_gen` past every `.liv` generation left in the directory,
    // and honouring that is the whole point of the field (see
    // `crate::index_file_deleter::inflate_gens`).
    let next_del_gen = sci.next_write_del_gen();
    // `SegmentInfos.write` throws `IllegalStateException` for a `delCount`
    // past `maxDoc`, so producing one here would build a commit this port's
    // own writer would then have to reject (and real Lucene's reader would
    // reject on the way back in). Catching it at the source names the
    // segment; catching it at write time would not.
    //
    // The addition is checked because `sci.del_count` came off `segments_N`,
    // where the only bound this layer can apply is `>= 0` (the `maxDoc` half
    // of Java's check lives in the `.si`, which `segment_infos` deliberately
    // does not read). A `del_count` near `i32::MAX` plus one newly deleted doc
    // overflows -- a panic in a debug build, and in a release build a *negative*
    // `del_count` that sails past the `> max_doc` test below and gets written
    // into the commit.
    let computed = i32::try_from(newly_deleted)
        .ok()
        .filter(|_| sci.del_count >= 0)
        .and_then(|n| sci.del_count.checked_add(n));
    let new_del_count = match computed {
        Some(n) if n >= 0 && n as usize <= max_doc => n,
        // On overflow there is no representable count to name, so report the
        // saturated one: a `delCount` of `i32::MAX` against any real `maxDoc`
        // is a visible absurdity, which is what the caller needs to see.
        other => {
            return Err(Error::DelCountExceedsMaxDoc {
                segment: sci.segment_name.clone(),
                del_count: other.unwrap_or(i32::MAX),
                max_doc,
            });
        }
    };

    let file_name = liv_file_name(&sci.segment_name, next_del_gen);
    if let Some(writer) = codec.and_then(Lucene104Codec::live_docs_writer) {
        // Java: PendingDeletes.writeLiveDocs uses TrackingDirectoryWrapper so
        // a callback that fails before creating an output cannot delete an
        // existing file. Keep this operation's created files local as well.
        let tracked = CreatedOutputs::new(dir);
        let written = writer
            .write_live_docs(
                &new_bits,
                LiveDocsWriteState {
                    directory: &tracked,
                    segment: &sci.segment_name,
                    id: &sci.segment_id,
                    max_doc,
                    del_gen: sci.del_gen,
                    next_del_gen,
                    // `new_del_count` already proves the previous count is
                    // nonnegative and representable above.
                    del_count: usize::try_from(sci.del_count).unwrap_or(0),
                },
                newly_deleted,
            )
            .map_err(lucene_store::Error::from);
        if let Err(error) = written {
            let created = std::mem::take(&mut *tracked.files());
            for name in created {
                let _ = dir.delete_file(&name);
            }
            return Err(error.into());
        }
    } else {
        let liv_bytes = live_docs::write(
            &new_bits,
            &sci.segment_id,
            next_del_gen,
            new_del_count as usize,
        )?;
        let mut out = dir.create_output(&file_name)?;
        use lucene_store::data_output::DataOutput;
        out.write_bytes(&liv_bytes);
        // Not synced: the commit that names this generation syncs it
        // (`IndexWriter.startCommit`); a generation superseded before any
        // commit names it is never synced at all.
        let written = out.close();
        if let Err(e) = written {
            // `PendingDeletes.writeLiveDocs`' `finally`: on failure,
            // `IOUtils.deleteFilesIgnoringExceptions(dir, createdFiles)`. A partial
            // `.liv` must not remain available to a later commit after a failure.
            // (Java also calls `advanceNextWriteDelGen()`; callers holding the
            // live `SegmentCommitInfo` do that, see `IndexWriter`.)
            let _ = dir.delete_file(&file_name);
            return Err(e.into());
        }
    }

    // `ReadersAndUpdates.writeLiveDocs`: the same `SegmentCommitInfo`, with
    // its delete count raised and `advanceDelGen()` called on it -- not a
    // hand-rebuilt copy. Going through the mutator is what keeps this path
    // on Java's `generationAdvanced()`, which re-derives
    // `SegmentCommitInfo.id`: the id is documented as a change token
    // ("changes each time the segment changes due to a delete, doc-value or
    // field update"), so a delete round that left it alone would tell every
    // reader that keys reuse or replication off it that nothing changed.
    let mut updated = sci.clone();
    updated.del_count = new_del_count;
    updated.advance_del_gen();
    debug_assert_eq!(updated.del_gen, next_del_gen);
    Ok(updated)
}

#[cfg(test)]
mod tests;
