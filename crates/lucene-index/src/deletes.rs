//! Doc-ID-level delete mechanics for one already-flushed segment: "mark these
//! doc IDs deleted, write the updated `.liv` file, bump `del_gen`/`del_count`
//! on its [`SegmentCommitInfo`]".
//!
//! # Scope: what this is
//!
//! Real Lucene's delete path has two halves: (1) `BufferedUpdates` +
//! `ReaderPool` resolve *which* doc IDs a delete-by-term/delete-by-query
//! actually names, per segment, by running the query against each segment's
//! live postings/doc-values reader; (2) `ReadersAndUpdates.writeLiveDocs`
//! takes that resolved set of doc IDs and turns it into an updated live-docs
//! bitset, a new `_<segment>_<delGen>.liv` file, and a bumped
//! `SegmentCommitInfo.delGen`/`delCount`. This module is only half (2).
//!
//! # Scope: what this deliberately is not
//!
//! **No delete-by-term/delete-by-query resolution.** This port has no
//! `IndexWriter` with a live, per-segment postings/doc-values reader wired to
//! a query executor across all open segments (real `TestDeletes`-style
//! `writer.deleteDocuments(new Term("id", "1"))` needs exactly that: search
//! every segment for docs matching the term, union the resulting doc IDs).
//! That is a genuinely separate, larger feature -- it depends on a live
//! multi-segment index reader + query execution being wired into the write
//! path, which nothing in this port does yet (search and index/write are
//! still separate, unconnected halves). Building a fake version of it here
//! (e.g. a linear scan over in-memory `Document`s) would not match real
//! Lucene's `BufferedUpdates` semantics (generation-ordered resolution against
//! whatever segments existed *at delete time*, not at flush time) and would
//! have no real caller to prove it against. Deferred; see `docs/parity.md`.
//!
//! **No `updateDocument`.** Real `IndexWriter.updateDocument(Term, doc)` is
//! defined as delete-by-term (see above, not in scope) followed by
//! `addDocument`. Since delete-by-term isn't here, a faithful
//! `updateDocument` can't be either -- an "update" that instead took a raw
//! doc ID would silently diverge from real semantics (a caller must already
//! know which doc ID currently holds that logical document, which is exactly
//! the mapping `updateDocument`'s `Term` lookup exists to avoid requiring).
//! Rather than force a misleading abstraction, this module exposes the two
//! primitives a caller already has enough context to use correctly by hand:
//! delete the old doc ID via [`apply_deletes`], then add the replacement doc
//! via a separate [`crate::segment_writer::flush_stored_only_segment`] or
//! merge call. Revisit real `updateDocument` once delete-by-term exists.

use lucene_codecs::live_docs;
use lucene_store::directory::Directory;
use lucene_util::fixed_bit_set::FixedBitSet;

use crate::segment_infos::SegmentCommitInfo;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    #[error(transparent)]
    LiveDocs(#[from] live_docs::Error),
    #[error("doc id {doc_id} out of range for max_doc={max_doc}")]
    DocOutOfRange { doc_id: i32, max_doc: usize },
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
        None => {
            let mut all_live = FixedBitSet::new(max_doc);
            for i in 0..max_doc {
                all_live.set(i);
            }
            all_live
        }
    };

    let mut newly_deleted = 0usize;
    for doc_id in doc_ids {
        if doc_id < 0 || doc_id as usize >= max_doc {
            return Err(Error::DocOutOfRange { doc_id, max_doc });
        }
        let idx = doc_id as usize;
        if bits.get(idx) {
            bits.clear(idx);
            newly_deleted += 1;
        }
        // Already-deleted: idempotent no-op, not double-counted.
    }

    Ok((bits, newly_deleted))
}

/// Applies a batch of newly-deleted doc IDs to `sci` (an already-flushed
/// segment's current [`SegmentCommitInfo`]): resolves the updated live-docs
/// bitset via [`mark_deleted`], writes it as that segment's next-generation
/// `.liv` file (via [`liv_file_name`] + [`lucene_codecs::live_docs::write`]),
/// syncs it through `dir`, and returns a new `SegmentCommitInfo` with
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
    let (new_bits, newly_deleted) = mark_deleted(current_live_docs, max_doc, doc_ids)?;

    let next_del_gen = if sci.del_gen < 0 { 1 } else { sci.del_gen + 1 };
    let new_del_count = sci.del_count + newly_deleted as i32;

    let liv_bytes = live_docs::write(
        &new_bits,
        &sci.segment_id,
        next_del_gen,
        new_del_count as usize,
    )?;
    let file_name = liv_file_name(&sci.segment_name, next_del_gen);
    let mut out = dir.create_output(&file_name)?;
    {
        use lucene_store::data_output::DataOutput;
        out.write_bytes(&liv_bytes);
    }
    out.close()?;
    dir.sync(std::slice::from_ref(&file_name))?;

    Ok(SegmentCommitInfo {
        segment_name: sci.segment_name.clone(),
        segment_id: sci.segment_id,
        codec_name: sci.codec_name.clone(),
        del_gen: next_del_gen,
        del_count: new_del_count,
        field_infos_gen: sci.field_infos_gen,
        doc_values_gen: sci.doc_values_gen,
        soft_del_count: sci.soft_del_count,
        sci_id: sci.sci_id,
        field_infos_files: sci.field_infos_files.clone(),
        dv_update_files: sci.dv_update_files.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_store::codec_util::ID_LENGTH;
    use lucene_store::directory::FsDirectory;

    fn sci(segment_name: &str, del_gen: i64, del_count: i32) -> SegmentCommitInfo {
        SegmentCommitInfo {
            segment_name: segment_name.to_string(),
            segment_id: [7u8; ID_LENGTH],
            codec_name: "Lucene104".to_string(),
            del_gen,
            del_count,
            field_infos_gen: -1,
            doc_values_gen: -1,
            soft_del_count: 0,
            sci_id: None,
            field_infos_files: vec![],
            dv_update_files: vec![],
        }
    }

    // --- liv_file_name ---

    #[test]
    fn liv_file_name_matches_ifn_convention() {
        assert_eq!(liv_file_name("_0", 1), "_0_1.liv");
        assert_eq!(liv_file_name("_3", 0), "_3_0.liv");
        // Generation is base-36, not decimal -- this is the case decimal and
        // base36 diverge, so it's the one that actually proves the encoding.
        assert_eq!(liv_file_name("_0", 36), "_0_10.liv");
        assert_eq!(liv_file_name("_0", 100), "_0_2s.liv");
    }

    // --- mark_deleted ---

    #[test]
    fn deleting_from_all_live_none_state() {
        let (bits, newly) = mark_deleted(None, 5, [1, 3]).unwrap();
        assert_eq!(newly, 2);
        assert!(bits.get(0));
        assert!(!bits.get(1));
        assert!(bits.get(2));
        assert!(!bits.get(3));
        assert!(bits.get(4));
        assert_eq!(bits.cardinality(), 3);
    }

    #[test]
    fn deleting_from_existing_partially_deleted_bitset() {
        let mut existing = FixedBitSet::new(4);
        existing.set(0);
        existing.set(1);
        existing.set(2);
        existing.set(3);
        existing.clear(2); // doc 2 already deleted from a prior generation

        let (bits, newly) = mark_deleted(Some(&existing), 4, [0]).unwrap();
        assert_eq!(newly, 1);
        assert!(!bits.get(0));
        assert!(bits.get(1));
        assert!(!bits.get(2)); // stays deleted
        assert!(bits.get(3));
    }

    #[test]
    fn deleting_an_already_deleted_doc_is_idempotent() {
        let mut existing = FixedBitSet::new(3);
        existing.set(0);
        existing.set(1);
        existing.set(2);
        existing.clear(1);

        let (bits, newly) = mark_deleted(Some(&existing), 3, [1, 1]).unwrap();
        assert_eq!(newly, 0); // doc 1 was already deleted, both calls no-op
        assert!(!bits.get(1));
    }

    #[test]
    fn boundary_doc_ids_zero_and_max_doc_minus_one() {
        let (bits, newly) = mark_deleted(None, 5, [0, 4]).unwrap();
        assert_eq!(newly, 2);
        assert!(!bits.get(0));
        assert!(bits.get(1));
        assert!(!bits.get(4));
    }

    #[test]
    fn out_of_range_doc_id_is_an_error_not_silent_or_panic() {
        let result = mark_deleted(None, 5, [5]);
        assert!(matches!(
            result,
            Err(Error::DocOutOfRange {
                doc_id: 5,
                max_doc: 5
            })
        ));
    }

    #[test]
    fn negative_doc_id_is_an_error() {
        let result = mark_deleted(None, 5, [-1]);
        assert!(matches!(
            result,
            Err(Error::DocOutOfRange {
                doc_id: -1,
                max_doc: 5
            })
        ));
    }

    #[test]
    fn empty_doc_ids_from_none_state_is_a_no_op() {
        let (bits, newly) = mark_deleted(None, 3, []).unwrap();
        assert_eq!(newly, 0);
        assert_eq!(bits.cardinality(), 3);
    }

    // --- apply_deletes: full round-trip via real Directory I/O ---

    fn tempdir() -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "lucene-rust-deletes-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn first_delete_round_writes_gen_one_and_bumps_del_count() {
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let info = sci("_0", -1, 0);

        let updated = apply_deletes(&dir, &info, None, 4, [1, 3]).unwrap();
        assert_eq!(updated.del_gen, 1);
        assert_eq!(updated.del_count, 2);

        let bytes = std::fs::read(tmp.join("_0_1.liv")).unwrap();
        let parsed = live_docs::parse(&bytes, &info.segment_id, 1, 4, 2).unwrap();
        assert!(parsed.get(0));
        assert!(!parsed.get(1));
        assert!(parsed.get(2));
        assert!(!parsed.get(3));
    }

    #[test]
    fn second_delete_round_increments_gen_and_unions_with_first() {
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let info = sci("_0", -1, 0);

        let after_first = apply_deletes(&dir, &info, None, 5, [1]).unwrap();
        assert_eq!(after_first.del_gen, 1);
        assert_eq!(after_first.del_count, 1);

        let first_liv = std::fs::read(tmp.join("_0_1.liv")).unwrap();
        let first_bits = live_docs::parse(&first_liv, &info.segment_id, 1, 5, 1).unwrap();

        let after_second = apply_deletes(&dir, &after_first, Some(&first_bits), 5, [3]).unwrap();
        assert_eq!(after_second.del_gen, 2);
        assert_eq!(after_second.del_count, 2);

        let second_liv = std::fs::read(tmp.join("_0_2.liv")).unwrap();
        let second_bits = live_docs::parse(&second_liv, &info.segment_id, 2, 5, 2).unwrap();
        // Union of both rounds: doc 1 (first round) and doc 3 (second round)
        // both stay deleted; everything else stays live.
        assert!(second_bits.get(0));
        assert!(!second_bits.get(1));
        assert!(second_bits.get(2));
        assert!(!second_bits.get(3));
        assert!(second_bits.get(4));
    }

    #[test]
    fn redeleting_same_doc_in_a_later_generation_does_not_double_count() {
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let info = sci("_0", -1, 0);

        let after_first = apply_deletes(&dir, &info, None, 3, [0]).unwrap();
        let first_liv = std::fs::read(tmp.join("_0_1.liv")).unwrap();
        let first_bits = live_docs::parse(&first_liv, &info.segment_id, 1, 3, 1).unwrap();

        let after_second = apply_deletes(&dir, &after_first, Some(&first_bits), 3, [0]).unwrap();
        assert_eq!(after_second.del_gen, 2);
        assert_eq!(after_second.del_count, 1); // not double-counted
    }

    #[test]
    fn out_of_range_doc_id_propagates_as_error_without_writing_a_file() {
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let info = sci("_0", -1, 0);

        let result = apply_deletes(&dir, &info, None, 3, [9]);
        assert!(result.is_err());
        assert!(!tmp.join("_0_1.liv").exists());
    }
}
