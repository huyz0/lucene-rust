//! Real Lucene's `IndexWriter.updateDocument(Term, doc)`: an atomic
//! delete-by-term + add-document, composed entirely out of already-built
//! primitives — [`crate::term_delete::resolve_and_apply_term_delete`] for the
//! delete half and [`crate::segment_writer::flush_stored_only_segment`]
//! for the add half, committed together via one
//! [`crate::segment_infos::write`] call.
//!
//! # Why this is new work, not a rebuild
//!
//! `term_delete.rs`'s and `segment_writer.rs`'s own module docs both say the
//! same thing: delete-by-term-scoped-to-one-segment and flush-one-segment
//! already exist as separate primitives; what was explicitly deferred is
//! *wiring them into one atomic operation* ("a first-class `updateDocument`
//! wrapper is left for when multi-segment resolution exists, so it composes
//! correctly rather than silently only covering one segment" —
//! `term_delete.rs`). This module is that wrapper: it fans the delete out
//! over every segment in a [`SegmentInfos`] that the caller has an opened
//! delete source for, flushes the replacement document as a new segment, and
//! commits the whole updated segment list in a single `segments_N` write.
//!
//! # Atomicity guarantee
//!
//! A reader only ever observes a specific `segments_N` generation (opened by
//! name/generation number) — never a partially-written one, because
//! [`crate::segment_infos::write`] writes the whole file to a fresh output
//! and `Directory.sync`s it before returning (see that function's own doc
//! comment). [`update_document`] performs exactly **one** such write, at the
//! very end, after every fallible step (delete resolution/application to
//! each affected segment, and flushing the new segment) has already
//! succeeded:
//!
//! 1. Resolve-and-apply the term delete against each segment with a supplied
//!    [`SegmentDeleteSource`] (writing an updated `.liv` file per affected
//!    segment, exactly as [`crate::term_delete::resolve_and_apply_term_delete`]
//!    already does standalone).
//! 2. Flush the new document(s) to a brand-new segment (writing its
//!    `.fdt`/`.fdx`/`.fdm`/`.fnm`/`.si`, exactly as
//!    [`crate::segment_writer::flush_stored_only_segment`] already does
//!    standalone).
//! 3. Sync every file steps 1 and 2 wrote (neither primitive syncs; Java
//!    leaves that to `IndexWriter.startCommit` too), then commit the updated
//!    segment list (old segments with bumped `del_gen`s + the one new
//!    segment) as the next `segments_N` generation.
//!
//! If step 1 or 2 fails, [`update_document`] returns `Err` before step 3 ever
//! runs: no `segments_N` is written, so the previous commit is still the
//! current one and still completely valid — the newly written `.liv`/segment
//! files are simply unreferenced orphans (the same crash-safety shape real
//! Lucene relies on: a commit either fully lands as a new generation or it
//! doesn't happen at all). No reader can ever open a `segments_N` that
//! reflects only the delete or only the add, because no such file is ever
//! written — the two halves become visible in the same fsync'd write.
//!
//! # A known, inherited limitation (not new to this module)
//!
//! [`crate::segment_writer::flush_stored_only_segment`] only ever writes
//! stored-fields-only segments (this port has no write-side postings format
//! yet — see `PLAN.md` Phase 5 item 2). That means a segment this port wrote
//! itself has no `.tim`/`.tip`/`.doc` files, so a delete-by-term issued
//! against *only self-written segments* can never actually match anything —
//! there is nothing to resolve against. This is not a bug in
//! [`update_document`]; it is the same "mergeable/deletable if a caller has
//! the data" scope line `merge.rs` and `term_delete.rs` already draw. Tests
//! below exercise the delete half against the same checked-in real-Lucene
//! fixture `term_delete.rs`'s own tests use, which does have real postings.

use lucene_codecs::blocktree::BlockTreeFields;
use lucene_codecs::field_infos::FieldInfo;
use lucene_codecs::postings::DocInput;
use lucene_codecs::stored_fields::Document;
use lucene_store::codec_util::ID_LENGTH;
use lucene_store::directory::Directory;
use lucene_util::fixed_bit_set::FixedBitSet;

use crate::segment_info::LuceneVersion;
use crate::segment_infos::{self, SegmentInfos};
use crate::segment_writer::{self, flush_stored_only_segment};
use crate::term_delete::{self, resolve_and_apply_term_delete};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    TermDelete(#[from] term_delete::Error),
    #[error(transparent)]
    SegmentWriter(#[from] segment_writer::Error),
    #[error(transparent)]
    SegmentInfos(#[from] segment_infos::Error),
    /// Syncing the written files before the commit failed.
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Everything needed to resolve `(field, term)` against one already-open
/// segment, for one call into [`update_document`]. Mirrors the parameters
/// [`crate::term_delete::resolve_and_apply_term_delete`] already takes for a
/// single segment — this port has no reader pool/`ReaderPool` that opens and
/// caches these automatically, so (as with that function) the caller
/// supplies whatever it already has open.
pub struct SegmentDeleteSource<'a> {
    /// Must match a [`crate::segment_infos::SegmentCommitInfo::segment_name`]
    /// in the [`SegmentInfos`] passed to [`update_document`]; segments with
    /// no matching source are left untouched (no delete resolved against
    /// them — e.g. because the caller hasn't opened them, or because they
    /// have no postings for `field` at all).
    pub segment_name: &'a str,
    pub fields: &'a BlockTreeFields,
    pub doc_in: Option<&'a DocInput<'a>>,
    pub live_docs: Option<&'a FixedBitSet>,
    pub max_doc: usize,
}

/// The atomic `deleteDocuments(Term) + addDocument(doc)` real Lucene calls
/// `updateDocument`. See the module doc for the exact atomicity guarantee.
///
/// `segment_infos` is the writer's current view of the index (its most
/// recently committed segment list); `delete_sources` supplies opened
/// postings for whichever of those segments the caller wants the term delete
/// resolved against (see [`SegmentDeleteSource`]). The replacement document
/// is flushed as a brand-new segment named `new_segment_name`/
/// `new_segment_id` via [`flush_stored_only_segment`] — same stored-fields-
/// only scope that function already has.
///
/// Returns the new, committed [`SegmentInfos`] (next generation, `version`
/// bumped by one) on success. On `Err`, nothing has been committed: the
/// previous `segments_N` is still current (see module doc).
#[allow(clippy::too_many_arguments)]
pub fn update_document(
    dir: &dyn Directory,
    segment_infos: &SegmentInfos,
    delete_sources: &[SegmentDeleteSource],
    field: &str,
    term: &[u8],
    new_segment_name: &str,
    new_segment_id: [u8; ID_LENGTH],
    codec_name: &str,
    lucene_version: LuceneVersion,
    new_fields: &[FieldInfo],
    new_docs: &[Document],
) -> Result<SegmentInfos> {
    let created = crate::created_outputs::CreatedOutputs::new(dir);
    let updated_segments = update_segments(
        &created,
        segment_infos,
        delete_sources,
        field,
        term,
        NewSegment {
            name: new_segment_name,
            id: new_segment_id,
            codec_name,
            lucene_version,
            fields: new_fields,
            docs: new_docs,
            use_compound_file: false,
        },
    )?;

    // Step 3: the single atomic commit point. Everything above has already
    // succeeded, so this is the only write that can make the update visible
    // -- and it makes both halves visible together, in one fsync'd file,
    // after the files it names were synced.
    let mut new_segment_infos = segment_infos.clone();
    // ARITH: both counters are capped at `segment_infos::MAX_GENERATION`
    // (`i64::MAX / 2`) on the way in (`segment_infos::parse`) and on the way
    // out (`check_writable_generations`), precisely so this one-per-commit
    // step cannot overflow -- see that constant. Without the cap a flipped
    // byte in the 8-byte `version` field turns the *next* commit into a panic
    // (debug) or a negative commit version (release), which a reader would
    // then rank below every older commit.
    #[allow(clippy::arithmetic_side_effects)]
    {
        new_segment_infos.generation += 1;
        new_segment_infos.version += 1;
    }
    new_segment_infos.segments = updated_segments;

    // The files it names are synced first (`startCommit`).
    let written: Vec<String> = created.files().iter().cloned().collect();
    dir.sync(&written)?;
    segment_infos::write(&new_segment_infos, dir)?;

    Ok(new_segment_infos)
}

/// The replacement segment [`update_segments`] flushes.
pub struct NewSegment<'a> {
    pub name: &'a str,
    pub id: [u8; ID_LENGTH],
    pub codec_name: &'a str,
    pub lucene_version: LuceneVersion,
    pub fields: &'a [FieldInfo],
    pub docs: &'a [Document],
    pub use_compound_file: bool,
}

/// Steps 1 and 2 of [`update_document`]: the term delete resolved against
/// every segment with a source, then the replacement document flushed as a
/// new segment. Returns the segment list the commit should publish; nothing
/// is committed, so a caller with its own commit path (the writer's, which
/// also keeps versions and soft-delete counts) publishes it.
pub fn update_segments(
    dir: &dyn Directory,
    segment_infos: &SegmentInfos,
    delete_sources: &[SegmentDeleteSource],
    field: &str,
    term: &[u8],
    new_segment: NewSegment<'_>,
) -> Result<Vec<crate::segment_infos::SegmentCommitInfo>> {
    // Step 1: resolve-and-apply the term delete against every segment we
    // have an opened source for. Fallible; nothing has been committed yet if
    // this returns `Err` partway through (the `.liv` files already written
    // for earlier segments in this loop are orphaned, not referenced by any
    // `segments_N` — see module doc).
    // ARITH: an in-memory `Vec` length plus the one new segment this call
    // flushes; `segments.len()` is at most `isize::MAX`.
    #[allow(clippy::arithmetic_side_effects)]
    let capacity = segment_infos.segments.len() + 1;
    let mut updated_segments = Vec::with_capacity(capacity);
    for sci in &segment_infos.segments {
        match delete_sources
            .iter()
            .find(|src| src.segment_name == sci.segment_name)
        {
            Some(src) => {
                let updated = resolve_and_apply_term_delete(
                    dir,
                    sci,
                    src.fields,
                    src.doc_in,
                    src.live_docs,
                    src.max_doc,
                    field,
                    term,
                )?;
                updated_segments.push(updated);
            }
            None => updated_segments.push(sci.clone()),
        }
    }

    // Step 2: flush the replacement document(s) as a new segment. Fallible;
    // still nothing committed if this fails (the delete's `.liv` files from
    // step 1 remain orphaned, harmlessly, since the previous `segments_N`
    // never referenced them).
    // Compound-file packing is the caller's decision
    // (`NewSegment::use_compound_file`); `update_document` itself keeps the
    // loose-file layout.
    let new_sci = flush_stored_only_segment(
        dir,
        new_segment.name,
        new_segment.id,
        new_segment.codec_name,
        new_segment.lucene_version,
        new_segment.fields,
        new_segment.docs,
        new_segment.use_compound_file,
    )?;
    updated_segments.push(new_sci);
    Ok(updated_segments)
}

#[cfg(test)]
mod tests;
