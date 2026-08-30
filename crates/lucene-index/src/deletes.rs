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
    // `FixedBitSet::get` turns into a panic (it indexes `words[index >> 6]`
    // and only `debug_assert`s the bound, so a release build either panics or
    // silently reads a ghost bit past `num_bits`). This function's own doc
    // comment promises `DocOutOfRange` "rather than ... panicking", so the
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
            })
        }
    };

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
        // `SegmentCommitInfo.advanceDelGen()`: the generation just consumed is
        // now the current one, and the next write goes one past it.
        // ARITH: `next_del_gen` is derived from `del_gen`, which is capped at
        // `segment_infos::MAX_GENERATION` (`i64::MAX / 2`) on every path that
        // can set it from outside this process -- `segment_infos::parse`, and
        // `index_file_deleter`'s `usable_generation` for a value carried in by
        // a file name -- and which `segment_infos::check_writable_generations`
        // refuses to serialize above the cap. See that constant.
        #[allow(clippy::arithmetic_side_effects)]
        next_write_del_gen: next_del_gen + 1,
        next_write_field_infos_gen: sci.next_write_field_infos_gen,
        next_write_doc_values_gen: sci.next_write_doc_values_gen,
        buffered_deletes_gen: sci.buffered_deletes_gen,
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
            ..Default::default()
        }
    }

    // --- liv_file_name ---

    #[test]
    fn liv_file_name_matches_ifn_convention() {
        assert_eq!(liv_file_name("_0", 1), "_0_1.liv");
        // Generation 0 has no suffix at all -- `IndexFileNames
        // .fileNameFromGeneration`'s dedicated `gen == 0` branch, which
        // returns `segmentFileName(base, "", ext)`.
        assert_eq!(liv_file_name("_3", 0), "_3.liv");
        // Generation is base-36, not decimal -- this is the case decimal and
        // base36 diverge, so it's the one that actually proves the encoding.
        assert_eq!(liv_file_name("_0", 36), "_0_10.liv");
        assert_eq!(liv_file_name("_0", 100), "_0_2s.liv");
    }

    /// The all-live bitset is built by filling whole `u64` words and masking
    /// the tail, not by setting `max_doc` bits one at a time. A `max_doc`
    /// that is not a multiple of 64 is the case that catches a wrong mask --
    /// `cardinality()` counts raw words, so a stray set bit past `max_doc`
    /// would show up as a live doc that does not exist.
    #[test]
    fn all_live_bitset_masks_bits_past_max_doc() {
        for max_doc in [1usize, 63, 64, 65, 130, 256] {
            let (bits, newly) = mark_deleted(None, max_doc, []).unwrap();
            assert_eq!(newly, 0);
            assert_eq!(bits.cardinality(), max_doc, "max_doc={max_doc}");
            assert!((0..max_doc).all(|i| bits.get(i)), "max_doc={max_doc}");
        }
    }

    /// A commit can never record more deletions than the segment has docs --
    /// `SegmentInfos.write` refuses to serialize one, so building it is a
    /// bug worth surfacing where it happens.
    #[test]
    fn del_count_past_max_doc_is_rejected() {
        let tmp = lucene_util::test_support::TempDir::new("deletes-del-count-past-max-doc");
        let dir = FsDirectory::open(&tmp);
        // A segment of 3 docs that already claims 3 deletions; deleting one
        // more would make 4.
        let info = sci("_0", 1, 3);
        let mut live = FixedBitSet::new(3);
        live.set(0);
        let err = apply_deletes(&dir, &info, Some(&live), 3, [0]).unwrap_err();
        assert!(
            matches!(
                &err,
                Error::DelCountExceedsMaxDoc {
                    del_count: 4,
                    max_doc: 3,
                    ..
                }
            ),
            "unexpected error: {err}"
        );
        std::fs::remove_dir_all(&tmp).ok();
    }

    /// `del_count` comes off `segments_N`, where the only bound this port can
    /// apply is `>= 0` — the `maxDoc` half of Java's check lives in the `.si`,
    /// which `segment_infos` deliberately does not read. So `del_count +
    /// newly_deleted` can overflow `i32`: a **panic** in a debug build, and in
    /// a release build a *negative* count that sails straight past the
    /// `> max_doc` test and into the commit.
    #[test]
    fn del_count_overflow_is_an_error_not_a_wrap() {
        let tmp = lucene_util::test_support::TempDir::new("deletes-del-count-overflow");
        let dir = FsDirectory::open(&tmp);
        let info = sci("_0", 1, i32::MAX);
        let mut live = FixedBitSet::new(3);
        live.set(0);
        let err = apply_deletes(&dir, &info, Some(&live), 3, [0]).unwrap_err();
        assert!(
            matches!(
                &err,
                Error::DelCountExceedsMaxDoc {
                    del_count: i32::MAX,
                    max_doc: 3,
                    ..
                }
            ),
            "unexpected error: {err}"
        );
        std::fs::remove_dir_all(&tmp).ok();
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

    use lucene_util::test_support::TempDir;

    /// A scratch directory that removes itself when the test ends -- unless
    /// the test is panicking, in which case its bytes stay for inspection.
    fn tempdir() -> TempDir {
        TempDir::new("deletes")
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
