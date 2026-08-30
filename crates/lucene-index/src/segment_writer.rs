//! A minimal, honest slice of what real Lucene's `DocumentsWriterPerThread`
//! (buffer documents) + `IndexWriter.commit()` (describe N segments in one
//! `segments_N`) do together -- scoped down to exactly what this port's
//! existing write-side primitives support today.
//!
//! # What this is
//!
//! [`flush_stored_only_segment`] takes an in-memory batch of already-built
//! [`Document`]s and [`FieldInfo`]s, "flushes" them to a brand-new segment
//! (`.fdt`/`.fdx`/`.fdm` stored fields + `.fnm` field infos + `.si` segment
//! info, all written and synced through a real [`Directory`]), and returns
//! the [`SegmentCommitInfo`] that describes it -- ready to push into a
//! [`SegmentInfos::segments`] list. Call it more than once against the same
//! `Directory` with distinct segment names, collect the resulting
//! [`SegmentCommitInfo`]s, and pass all of them to one [`segment_infos::write`]
//! call: that produces a single commit (`segments_N`) that lists multiple,
//! independently-flushed segments -- exactly what `IndexWriter.commit()`
//! does after several `DocumentsWriterPerThread.flush()` calls, minus
//! everything this port hasn't earned yet (see "What this deliberately is
//! not" below).
//!
//! [`segment_infos::write`] itself already generalizes to any number of
//! segments (`SegmentInfos::segments: Vec<SegmentCommitInfo>`, with a plain
//! loop over them in both `parse` and `write`) -- that part of a
//! multi-segment commit was *already* mechanical, not new work. What was
//! missing, and what this module adds, is the reusable "flush one batch of
//! documents to one new segment" building block, so a caller doesn't have to
//! hand-copy the `.fdt`/`.fnm`/`.si`-writing boilerplate (previously
//! duplicated across `write_segment_info_fixture.rs` and
//! `write_segment_infos_fixture.rs`) once per segment.
//!
//! # What this deliberately is not
//!
//! This is **not** an `IndexWriter`. In particular, on purpose, it has:
//! - no RAM accounting or automatic flush-triggering (the caller decides
//!   when to call [`flush_stored_only_segment`], there's no
//!   `ramBufferSizeMB`-style threshold),
//! - no merging (`TieredMergePolicy`/`ConcurrentMergeScheduler` equivalents),
//! - no deletes/updates during indexing (`BufferedUpdates`),
//! - no NRT reopen,
//! - no concurrency (`DocumentsWriterPerThread`-per-thread pooling) -- one
//!   caller, one directory, sequential calls,
//! - and no indexed fields at all yet: like the single-segment fixture it
//!   generalizes, every field is stored-only (`IndexOptions::None`, no doc
//!   values/points/vectors/term vectors), because this port has no write-side
//!   postings/doc-values/points/vectors format built into a reusable form
//!   yet. `SegmentCoreReaders` only opens those producers when
//!   `FieldInfos.hasPostings()`/`hasDocValues()`/etc. are true (see
//!   `org.apache.lucene.index.SegmentCoreReaders`), so a segment with zero
//!   indexed fields needs none of those files -- a real constraint, not a
//!   shortcut in this module.
//!
//! See `docs/parity.md` and `PLAN.md`'s Phase 5 section for the exact,
//! currently-true scope line.
//!
//! # Why a plain function, not a stateful writer/builder object
//!
//! Two shapes were weighed for this slice: (a) exactly what's here -- a
//! free function taking an already-built batch of documents and producing
//! one segment -- versus (b) a stateful `IndexWriter`-shaped builder with an
//! `add_document`/`commit()` API that internally buffers documents across
//! calls. (b) was rejected for now: it would still cap out at one segment
//! per `commit()` (this port has no RAM-threshold/flush-triggering logic to
//! decide *when* to start a second segment), so the extra stateful API
//! surface wouldn't unlock anything this module's callers can't already do
//! by calling [`flush_stored_only_segment`] more than once themselves (see
//! `write_multi_segment_commit_fixture.rs`). Revisit (b) once a real
//! flush-trigger policy (even a trivial "every N documents" one) gives a
//! builder object something genuine to own as internal state --
//! introducing it earlier would be state management with no real caller
//! yet.

use crate::segment_info::{self, IndexSortField, LuceneVersion, SegmentInfo, SortKeyComparator};
use crate::segment_infos::SegmentCommitInfo;
use lucene_codecs::compound_format;
use lucene_codecs::field_infos::{self, FieldInfo};
use lucene_codecs::stored_fields::{self, Document};
use lucene_store::codec_util::ID_LENGTH;
use lucene_store::data_output::DataOutput;
use lucene_store::directory::Directory;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    #[error(transparent)]
    CompoundFormat(#[from] compound_format::Error),
    #[error(transparent)]
    SegmentInfo(#[from] segment_info::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Flushes `docs` (already fully built -- no analysis chain here, see
/// `PLAN.md` Phase 5 item 1) to a brand-new, stored-fields-only segment named
/// `segment_name` inside `dir`: writes and syncs `<name>.fdt`/`.fdx`/`.fdm`
/// (stored fields), `<name>.fnm` (field infos), and `<name>.si` (segment
/// info), then returns the [`SegmentCommitInfo`] a caller can push into a
/// [`crate::segment_infos::SegmentInfos::segments`] list.
///
/// `segment_id` must be unique per segment in a commit (mirrors real
/// Lucene's `StringHelper.randomId()` per flushed segment -- the caller
/// picks it since this module has no random-id policy of its own yet).
/// `codec_name` is recorded in `segments_N` as-is; this port only ever
/// writes fresh segments so it's the caller's job to pass the name of
/// whatever codec produced the referenced files (matches
/// `crate::segment_info::write`'s existing stance of never round-tripping
/// an old format).
///
/// `use_compound_file` chooses the on-disk layout: when `false` (the
/// original behavior, unchanged), the segment's `.fdt`/`.fdx`/`.fdm`/`.fnm`
/// are written as independent loose files. When `true`, those same four
/// already-complete codec files are packed into one `<segment_name>.cfs`
/// (data) + `<segment_name>.cfe` (entries) pair via
/// [`compound_format::write`] instead -- mirroring real Lucene's
/// `SegmentInfo.setUseCompoundFile(true)` /
/// `Lucene90CompoundFormat.write(...)`, called from `IndexWriter` once a
/// flushed segment's size falls under `TieredMergePolicy`'s
/// `noCFSRatio`/`maxCFSSegmentSizeMB` threshold. This port has no merge
/// policy or segment-size accounting yet (see `PLAN.md` Phase 5), so rather
/// than fake a size heuristic that has nothing real to compare against, the
/// caller decides directly with this boolean -- simpler, and just as
/// correct for every caller this port has today (both `update_document.rs`
/// and this module's own tests pass a literal `true`/`false`).
#[allow(clippy::too_many_arguments)]
pub fn flush_stored_only_segment(
    dir: &dyn Directory,
    segment_name: &str,
    segment_id: [u8; ID_LENGTH],
    codec_name: &str,
    lucene_version: LuceneVersion,
    fields: &[FieldInfo],
    docs: &[Document],
    use_compound_file: bool,
) -> Result<SegmentCommitInfo> {
    flush_stored_only_segment_with_blocks(
        dir,
        segment_name,
        segment_id,
        codec_name,
        lucene_version,
        fields,
        docs,
        use_compound_file,
        false,
    )
}

/// [`flush_stored_only_segment`] plus real Lucene's
/// `SegmentInfo.setHasBlocks()`.
///
/// `has_blocks` records that this segment contains at least one **document
/// block** -- a run of documents added together by
/// `IndexWriter.addDocuments`/`updateDocuments` and guaranteed to occupy
/// contiguous doc IDs, which is what parent-field join queries rely on.
/// `DocumentsWriterPerThread.updateDocuments` sets it whenever a single call
/// indexed more than one document (`if (numDocs > 1) segmentInfo.setHasBlocks()`),
/// and `IndexWriter.mergeMiddle` ORs it across the merged readers.
///
/// It is a separate entry point rather than a ninth parameter on
/// [`flush_stored_only_segment`] because Java's is a *mutator* on an
/// already-built `SegmentInfo`, and because every existing caller of this
/// module writes single-document adds only -- they all mean `false`, and
/// making them say so adds nothing.
#[allow(clippy::too_many_arguments)]
pub fn flush_stored_only_segment_with_blocks(
    dir: &dyn Directory,
    segment_name: &str,
    segment_id: [u8; ID_LENGTH],
    codec_name: &str,
    lucene_version: LuceneVersion,
    fields: &[FieldInfo],
    docs: &[Document],
    use_compound_file: bool,
    has_blocks: bool,
) -> Result<SegmentCommitInfo> {
    let flushed = write_stored_only_segment_files(
        dir,
        segment_name,
        segment_id,
        codec_name,
        lucene_version,
        fields,
        docs,
        use_compound_file,
        has_blocks,
    )?;
    seal_flushed_segment(dir, segment_name, flushed)
}

/// A segment whose codec files are on disk but whose `.si` is **not yet
/// written**: the in-memory `SegmentInfo` that will become it, the
/// `SegmentCommitInfo` that will describe it in `segments_N`, and the file
/// names still to be fsynced.
///
/// # Why this exists
///
/// Java's `IndexWriter.sealFlushedSegment` writes a segment's `.si` **once**,
/// at the end, out of an in-memory `SegmentInfo` that every format's writer
/// has already added its files to (`SegmentInfo.files` accumulates through
/// the `TrackingDirectoryWrapper`). This port had no way to say "the segment
/// is not finished yet", so each of
/// [`crate::index_writer::IndexWriter`]'s five per-format file groups --
/// postings, term vectors, doc values, norms, vectors -- plus the index-sort
/// descriptor re-opened, re-parsed, extended and re-wrote the `.si` and
/// fsynced it again: up to seven `.si` writes and six parses per commit, of
/// a file whose whole content was already in memory.
///
/// A caller that writes more than stored fields therefore takes
/// [`write_stored_only_segment_files`], pushes its own file names into
/// [`Self::info`]`.files` and [`Self::pending_sync`], and finishes with
/// [`seal_flushed_segment`].
pub struct FlushedSegment {
    /// The segment's `.si` content, still only in memory. A caller extends
    /// `files` (and may set `index_sort`) before sealing.
    pub info: SegmentInfo,
    /// The `segments_N` entry for this segment.
    pub commit: SegmentCommitInfo,
    /// Every file written for this segment so far, to be fsynced together
    /// with the `.si` by [`seal_flushed_segment`]. A caller appends the names
    /// it writes itself.
    pub pending_sync: Vec<String>,
}

/// [`flush_stored_only_segment_with_blocks`] up to but **not including** the
/// `.si`: writes the stored fields and field infos (or the compound pair),
/// and returns the [`FlushedSegment`] a caller extends with its own formats'
/// files before calling [`seal_flushed_segment`].
///
/// The returned `info.files` already lists the stored-fields/field-infos
/// files *and* the `.si` itself, because `Lucene99SegmentInfoFormat.write`
/// does `si.addFile(fileName)` before encoding and every consumer that walks
/// `SegmentInfo.files` -- `IndexFileDeleter`, `CheckIndex`,
/// `checksum_verify` -- reference-counts from exactly that set.
#[allow(clippy::too_many_arguments)]
pub fn write_stored_only_segment_files(
    dir: &dyn Directory,
    segment_name: &str,
    segment_id: [u8; ID_LENGTH],
    codec_name: &str,
    lucene_version: LuceneVersion,
    fields: &[FieldInfo],
    docs: &[Document],
    use_compound_file: bool,
    has_blocks: bool,
) -> Result<FlushedSegment> {
    let doc_count = docs.len() as i32;

    let (fdt, fdx, fdm) = stored_fields::write_best_speed(docs, &segment_id, "");
    let fnm = field_infos::write(fields, &segment_id, "");

    let fdt_name = format!("{segment_name}.fdt");
    let fdx_name = format!("{segment_name}.fdx");
    let fdm_name = format!("{segment_name}.fdm");
    let fnm_name = format!("{segment_name}.fnm");

    let (files, written_names) = if use_compound_file {
        let sub_files = vec![
            (".fdt".to_string(), fdt),
            (".fdx".to_string(), fdx),
            (".fdm".to_string(), fdm),
            (".fnm".to_string(), fnm),
        ];
        let (cfs, cfe) = compound_format::write(&segment_id, &sub_files)?;
        let cfs_name = format!("{segment_name}.cfs");
        let cfe_name = format!("{segment_name}.cfe");
        write_file(dir, &cfs_name, &cfs)?;
        write_file(dir, &cfe_name, &cfe)?;
        (
            vec![cfs_name.clone(), cfe_name.clone()],
            vec![cfs_name, cfe_name],
        )
    } else {
        for (name, bytes) in [(&fdt_name, &fdt), (&fdx_name, &fdx), (&fdm_name, &fdm)] {
            write_file(dir, name, bytes)?;
        }
        write_file(dir, &fnm_name, &fnm)?;
        (
            vec![
                fdt_name.clone(),
                fdx_name.clone(),
                fdm_name.clone(),
                fnm_name.clone(),
            ],
            vec![fdt_name, fdx_name, fdm_name, fnm_name],
        )
    };

    // `Lucene99SegmentInfoFormat.write` calls `si.addFile(fileName)` before
    // encoding, so a segment's own `.si` is always a member of the file set it
    // records. Leaving it out makes every consumer that walks `SegmentInfo.files`
    // -- `IndexFileDeleter`, `CheckIndex`, our own `checksum_verify` -- blind to
    // the one file that names all the others.
    let si_name = format!("{segment_name}.si");
    let mut files = files;
    files.push(si_name.clone());

    let si = SegmentInfo {
        id: segment_id,
        version: lucene_version,
        min_version: Some(lucene_version),
        doc_count,
        is_compound_file: use_compound_file,
        has_blocks,
        diagnostics: vec![
            ("source".to_string(), "flush".to_string()),
            (
                "lucene.version".to_string(),
                format!(
                    "{}.{}.{}",
                    lucene_version.major, lucene_version.minor, lucene_version.bugfix
                ),
            ),
        ],
        files,
        attributes: vec![(
            "Lucene90StoredFieldsFormat.mode".to_string(),
            "BEST_SPEED".to_string(),
        )],
        index_sort: None,
    };
    Ok(FlushedSegment {
        info: si,
        commit: SegmentCommitInfo {
            segment_name: segment_name.to_string(),
            segment_id,
            codec_name: codec_name.to_string(),
            del_gen: -1,
            del_count: 0,
            field_infos_gen: -1,
            doc_values_gen: -1,
            soft_del_count: 0,
            // `DocumentsWriterPerThread.flush`: every freshly flushed segment
            // gets its own `StringHelper.randomId()`. Derived from the segment
            // id rather than drawn from a CSPRNG: distinctness is the only
            // property anything reads it for, and the segment id is already
            // distinct per segment.
            sci_id: Some(derive_sci_id(&segment_id)),
            field_infos_files: vec![],
            dv_update_files: vec![],
            ..Default::default()
        },
        pending_sync: written_names,
    })
}

/// `IndexWriter.sealFlushedSegment`'s tail: writes the segment's `.si` from
/// the in-memory [`SegmentInfo`] -- **once** -- and fsyncs it together with
/// every file the segment's formats wrote.
///
/// `flushed.info.files` is the authority on what the `.si` records;
/// `flushed.pending_sync` is what actually gets fsynced. They differ by
/// exactly the `.si` itself, which is in the first (a `.si` lists itself) and
/// is appended to the second here.
pub fn seal_flushed_segment(
    dir: &dyn Directory,
    segment_name: &str,
    flushed: FlushedSegment,
) -> Result<SegmentCommitInfo> {
    let FlushedSegment {
        info,
        commit,
        mut pending_sync,
    } = flushed;
    let si_name = format!("{segment_name}.si");
    debug_assert!(
        info.files.contains(&si_name),
        "a segment's own `.si` must be a member of the file set it records"
    );
    write_file(dir, &si_name, &segment_info::write(&info, ""))?;
    pending_sync.push(si_name);
    dir.sync(&pending_sync)?;
    Ok(commit)
}

/// `StringHelper.randomId()`'s role for a freshly created
/// [`SegmentCommitInfo`]: an id that distinguishes this segment-commit from
/// every other one. Java draws 16 random bytes; this derives them from the
/// segment's own (already unique) id, which satisfies the only property any
/// consumer relies on — `SegmentInfos.readCommit` accepts any 16 bytes and
/// nothing in Lucene validates them — without pulling in a CSPRNG.
pub(crate) fn derive_sci_id(segment_id: &[u8; ID_LENGTH]) -> [u8; ID_LENGTH] {
    let mut out = [0u8; ID_LENGTH];
    for (i, slot) in out.iter_mut().enumerate() {
        // A byte-wise involution-free mix: distinct segment ids stay
        // distinct, and the result never equals the segment id itself.
        *slot = segment_id[i] ^ (0xA5u8.wrapping_add(i as u8));
    }
    out
}

/// One field of a multi-field index sort passed to
/// [`flush_sorted_stored_only_segment`]: the `SortField` itself, plus that
/// field's per-doc sort key (parallel to the flush's `docs`). Real Lucene's
/// `Sort` is an array of `SortField`s applied in priority order -- this
/// struct is one array element together with the column it reads.
#[derive(Debug, Clone)]
pub struct SortKeySpec<'a> {
    pub sort: &'a IndexSortField,
    /// `keys[i]` is doc `i`'s (pre-sort) value for this field, or `None` if
    /// doc `i` has no value for it. Must have exactly one entry per doc.
    ///
    /// One `i64` per document in the doc-values column's own encoding, which
    /// is what every [`SortKeyComparator`]-supported sort reduces to: the raw
    /// long for INT/LONG, the raw float/double bits for FLOAT/DOUBLE, the
    /// selector's chosen value for a SORTED_NUMERIC column, the term ordinal
    /// for a STRING or SORTED_SET one.
    pub keys: &'a [Option<i64>],
}

/// Like [`flush_stored_only_segment`], but physically reorders `docs` by a
/// (possibly multi-field) index sort before writing anything, and records
/// that sort in the resulting `.si` (real Lucene's
/// `IndexWriterConfig.setIndexSort` + `DocumentsWriterPerThread`'s
/// sort-on-flush behavior -- see PLAN.md's index-sort task).
///
/// # Scope (see `docs/parity.md` for the full write-up)
///
/// - **One or more fields, priority-ordered.** `sort_fields[0]` is the
///   primary sort key; `sort_fields[1]` breaks ties in `sort_fields[0]`, and
///   so on -- mirroring real Lucene's `Sort` being an ordered array of
///   `SortField`s. Each field's own `reverse`/missing-value policy applies
///   independently at its own tier. This port has no write-side doc-values
///   format that could feed the sort, so each field's keys must already be in
///   memory, parallel to `docs`, which is exactly what [`SortKeySpec::keys`]
///   is. `sort_fields` must be non-empty.
/// - **Every sort a [`SortKeyComparator`] exists for**: the four numeric
///   `SortField.Type`s with any missing value (or none), both
///   `SortedNumericSelector`s, and term-ordinal sorts. A `BinarySortField`
///   has no single-`i64` key and is rejected by `IndexWriter::set_index_sort`
///   before it can reach here; this function asserts rather than silently
///   sorting by nothing.
/// - **Missing values**: `keys[i] == None` means doc `i` has no value for
///   that sort field. It is substituted with that field's sentinel and then
///   compared like any other value, so `reverse` applies to it too -- see
///   [`SortKeyComparator`], which cites Java's own comparator.
/// - **Stable sort**: docs that compare equal across every field (including
///   both-missing at every tier) keep their original relative order,
///   matching `Vec::sort_by`'s stability guarantee and real Lucene's own
///   stable-merge-sort-based flush sort.
#[allow(clippy::too_many_arguments)]
pub fn flush_sorted_stored_only_segment(
    dir: &dyn Directory,
    segment_name: &str,
    segment_id: [u8; ID_LENGTH],
    codec_name: &str,
    lucene_version: LuceneVersion,
    fields: &[FieldInfo],
    docs: &[Document],
    sort_fields: &[SortKeySpec<'_>],
) -> Result<SegmentCommitInfo> {
    assert!(
        !sort_fields.is_empty(),
        "sort_fields must contain at least one sort key"
    );
    for spec in sort_fields {
        assert_eq!(
            docs.len(),
            spec.keys.len(),
            "sort_keys must have exactly one entry per doc for field {:?}",
            spec.sort.field
        );
    }

    // The one permutation `IndexWriter::flush`'s index-sorted path also uses
    // (see [`sort_permutation`]), so the two orders cannot drift.
    let order = sort_permutation(docs.len(), sort_fields);

    let sorted_docs: Vec<Document> = order.iter().map(|&i| docs[i].clone()).collect();

    // The index-sort descriptor goes into the `SegmentInfo` *before* the `.si`
    // is written, not into a second copy of it afterwards:
    // `write_stored_only_segment_files` hands back the in-memory
    // `SegmentInfo`, so there is nothing to re-read, re-parse, rewrite and
    // re-fsync.
    let mut flushed = write_stored_only_segment_files(
        dir,
        segment_name,
        segment_id,
        codec_name,
        lucene_version,
        fields,
        &sorted_docs,
        false,
        false,
    )?;
    flushed.info.index_sort = Some(sort_fields.iter().map(|spec| spec.sort.clone()).collect());
    seal_flushed_segment(dir, segment_name, flushed)
}

/// The comparators `sort_fields` induces, in priority order.
///
/// Panics if any of them has none ([`SortKeyComparator::new`] returning
/// `None`, i.e. a `BinarySortField`). That is a programming error, not a
/// data error: `IndexWriter::set_index_sort` refuses such a sort, so no
/// buffer can be permuted by one. Sorting by "always equal" instead would
/// produce a segment whose `.si` claims an order it does not have -- valid
/// files, clean checksums, wrong index.
fn comparators(sort_fields: &[SortKeySpec<'_>]) -> Vec<SortKeyComparator> {
    sort_fields
        .iter()
        .map(|spec| {
            SortKeyComparator::new(spec.sort).unwrap_or_else(|| {
                panic!(
                    "sort field {:?} has no single-i64 comparator; \
                     IndexWriter::set_index_sort must refuse it before a flush sees it",
                    spec.sort.field
                )
            })
        })
        .collect()
}

/// The permutation a (possibly multi-field) index sort imposes on a batch of
/// `doc_count` buffered documents: entry `i` of the returned vector is the
/// **pre-sort** index of the document that becomes doc id `i` in the flushed
/// segment (real Lucene's `Sorter.DocMap.newToOld`).
///
/// `sort_fields` is applied in priority order -- `sort_fields[0]` is the
/// primary key, `sort_fields[1]` breaks its ties, and so on -- and the final
/// tie-break is the original index, so the permutation is a *stable* sort and
/// therefore a deterministic function of its input (Java's
/// `Sorter.sortAndLeaveUnpacked` likewise falls back to doc id, via
/// `TimSorter` over a stable base order).
///
/// Shared by [`flush_sorted_stored_only_segment`] and by
/// `IndexWriter::flush`'s index-sorted path, so that the order the writer
/// physically imposes and the order this module's own primitive imposes can
/// never drift apart.
pub fn sort_permutation(doc_count: usize, sort_fields: &[SortKeySpec<'_>]) -> Vec<usize> {
    // Resolved once, outside the O(n log n) comparison loop, rather than
    // re-derived from the `SortField` on every comparison.
    let cmps = comparators(sort_fields);
    let mut order: Vec<usize> = (0..doc_count).collect();
    order.sort_by(|&a, &b| {
        sort_fields
            .iter()
            .zip(&cmps)
            .fold(std::cmp::Ordering::Equal, |acc, (spec, cmp)| {
                acc.then_with(|| cmp.compare(spec.keys[a], spec.keys[b]))
            })
            .then(a.cmp(&b))
    });
    order
}

/// Applies `new_to_old` (a [`sort_permutation`] result) to `items` in place:
/// afterwards `items[i]` is what `items[new_to_old[i]]` was.
///
/// Cycle-following rather than "build a permuted copy": a flush buffer holds
/// whole `Document`s (and, in `IndexWriter`, parallel vectors of vectors), so
/// the copy would clone or move every one of them into a second buffer that
/// is live at the same time as the first -- doubling the peak footprint of
/// exactly the structure a flush exists to get rid of. This moves each
/// element at most twice and allocates one `u32` per document.
pub fn permute_in_place<T>(items: &mut [T], new_to_old: &[usize]) {
    debug_assert_eq!(items.len(), new_to_old.len());
    // The cycle walk below moves "the element at `i` belongs at `dest[i]`",
    // so it needs the *inverse* of `new_to_old` (`Sorter.DocMap.oldToNew`).
    // Inverting is one pass over a `u32` array; getting the direction wrong
    // is a permutation that is plausible, self-consistent and wrong (it
    // applies the inverse ordering), which is exactly the kind of defect a
    // sorted flush hides.
    let mut dest: Vec<u32> = vec![0; items.len()];
    for (new, &old) in new_to_old.iter().enumerate() {
        dest[old] = new as u32;
    }
    for i in 0..items.len() {
        while dest[i] as usize != i {
            let j = dest[i] as usize;
            items.swap(i, j);
            dest.swap(i, j);
        }
    }
}

fn write_file(dir: &dyn Directory, name: &str, bytes: &[u8]) -> Result<()> {
    let mut out = dir.create_output(name)?;
    out.write_bytes(bytes);
    out.close()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_codecs::field_infos::{
        DocValuesSkipIndexType, DocValuesType, IndexOptions, VectorEncoding,
        VectorSimilarityFunction,
    };
    use lucene_codecs::stored_fields::{FieldValue, StoredField};
    use lucene_store::directory::FsDirectory;

    fn version() -> LuceneVersion {
        LuceneVersion {
            major: 10,
            minor: 0,
            bugfix: 0,
        }
    }

    fn stored_only_field(name: &str, number: i32) -> FieldInfo {
        FieldInfo {
            name: name.to_string(),
            number,
            store_term_vectors: false,
            omit_norms: false,
            store_payloads: false,
            soft_deletes_field: false,
            parent_field: false,
            index_options: IndexOptions::None,
            doc_values_type: DocValuesType::None,
            doc_values_skip_index_type: DocValuesSkipIndexType::None,
            doc_values_gen: -1,
            attributes: vec![],
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
            vector_dimension: 0,
            vector_encoding: VectorEncoding::Float32,
            vector_similarity_function: VectorSimilarityFunction::Euclidean,
        }
    }

    fn doc(id: &str) -> Document {
        Document {
            fields: vec![StoredField {
                field_number: 0,
                value: FieldValue::String(id.to_string()),
            }],
        }
    }

    #[test]
    fn flushes_a_segment_with_the_expected_files_and_doc_count() {
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let docs = vec![doc("1"), doc("2")];

        let sci = flush_stored_only_segment(
            &dir,
            "_0",
            [7u8; ID_LENGTH],
            "Lucene104",
            version(),
            &fields,
            &docs,
            false,
        )
        .unwrap();

        assert_eq!(sci.segment_name, "_0");
        assert_eq!(sci.segment_id, [7u8; ID_LENGTH]);
        assert_eq!(sci.codec_name, "Lucene104");
        assert_eq!(sci.del_count, 0);
        for ext in ["fdt", "fdx", "fdm", "fnm", "si"] {
            assert!(
                std::path::Path::new(&tmp)
                    .join(format!("_0.{ext}"))
                    .exists(),
                "missing _0.{ext}"
            );
        }

        // The .si file must claim the same doc count we flushed -- cross-check
        // against segment_info::parse the same way the real fixture does.
        let si_bytes = std::fs::read(tmp.join("_0.si")).unwrap();
        let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
        assert_eq!(si.doc_count, docs.len() as i32);
    }

    #[test]
    fn two_flushes_produce_two_independent_segments() {
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];

        let sci0 = flush_stored_only_segment(
            &dir,
            "_0",
            [1u8; ID_LENGTH],
            "Lucene104",
            version(),
            &fields,
            &[doc("1")],
            false,
        )
        .unwrap();
        let sci1 = flush_stored_only_segment(
            &dir,
            "_1",
            [2u8; ID_LENGTH],
            "Lucene104",
            version(),
            &fields,
            &[doc("2"), doc("3")],
            false,
        )
        .unwrap();

        assert_ne!(sci0.segment_name, sci1.segment_name);
        assert_ne!(sci0.segment_id, sci1.segment_id);
        for ext in ["fdt", "fdx", "fdm", "fnm", "si"] {
            assert!(std::path::Path::new(&tmp)
                .join(format!("_0.{ext}"))
                .exists());
            assert!(std::path::Path::new(&tmp)
                .join(format!("_1.{ext}"))
                .exists());
        }
    }

    #[test]
    fn flush_surfaces_directory_io_error_rather_than_panicking() {
        // A directory that doesn't exist makes the very first create_output
        // (the .fdt file) fail -- confirms Error::Store's #[from] wrapping
        // actually propagates a real Directory I/O failure as an Err rather
        // than panicking or silently losing the error, the one path this
        // module's own error type exists to cover.
        let dir = FsDirectory::open("/nonexistent-lucene-rust-segment-writer-test-dir");
        let fields = vec![stored_only_field("id", 0)];
        let docs = vec![doc("1")];

        let result = flush_stored_only_segment(
            &dir,
            "_0",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
            &fields,
            &docs,
            false,
        );

        assert!(matches!(result, Err(Error::Store(_))));
    }

    /// `use_compound_file: true` must produce a `.cfs`/`.cfe` pair instead of
    /// loose `.fdt`/`.fdx`/`.fdm`/`.fnm`, and the `.si` must record
    /// `is_compound_file: true` -- the two facts a reader relies on to know
    /// which layout to open.
    #[test]
    fn compound_flush_writes_cfs_cfe_pair_and_marks_si_compound() {
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let docs = vec![doc("1"), doc("2"), doc("3")];

        let sci = flush_stored_only_segment(
            &dir,
            "_0",
            [3u8; ID_LENGTH],
            "Lucene104",
            version(),
            &fields,
            &docs,
            true,
        )
        .unwrap();

        for ext in ["cfs", "cfe", "si"] {
            assert!(
                std::path::Path::new(&tmp)
                    .join(format!("_0.{ext}"))
                    .exists(),
                "missing _0.{ext}"
            );
        }
        for ext in ["fdt", "fdx", "fdm", "fnm"] {
            assert!(
                !std::path::Path::new(&tmp)
                    .join(format!("_0.{ext}"))
                    .exists(),
                "loose _0.{ext} should not exist in compound mode"
            );
        }

        let si_bytes = std::fs::read(tmp.join("_0.si")).unwrap();
        let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
        assert!(si.is_compound_file);
        // The `.si` lists itself, as `Lucene99SegmentInfoFormat.write` does.
        assert_eq!(
            si.files,
            vec![
                "_0.cfs".to_string(),
                "_0.cfe".to_string(),
                "_0.si".to_string()
            ]
        );
    }

    /// The meaningful end-to-end check: flush with `use_compound_file: true`,
    /// then recover the original `.fdt`/`.fdx`/`.fdm`/`.fnm` sub-files
    /// byte-for-byte via the already-verified `compound_format` reader
    /// (`compound_format::parse_entries` + `open_input`), and confirm
    /// `stored_fields::open` can read documents back out *through* those
    /// recovered slices -- not by re-deriving from the original in-memory
    /// buffers, so a byte-offset bug in the new wiring would show up here.
    #[test]
    fn compound_flush_round_trips_through_compound_reader_and_stored_fields() {
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let segment_id = [5u8; ID_LENGTH];
        let fields = vec![stored_only_field("id", 0)];
        let docs = vec![doc("alpha"), doc("beta"), doc("gamma")];

        flush_stored_only_segment(
            &dir,
            "_0",
            segment_id,
            "Lucene104",
            version(),
            &fields,
            &docs,
            true,
        )
        .unwrap();

        let cfs = std::fs::read(tmp.join("_0.cfs")).unwrap();
        let cfe = std::fs::read(tmp.join("_0.cfe")).unwrap();

        let entries = compound_format::parse_entries(&cfe, &segment_id).unwrap();
        compound_format::check_data_header_footer(&cfs, &segment_id, &entries).unwrap();

        let fdt = compound_format::open_input(&cfs, &entries, ".fdt")
            .unwrap()
            .as_slice();
        let fdx = compound_format::open_input(&cfs, &entries, ".fdx")
            .unwrap()
            .as_slice();
        let fdm = compound_format::open_input(&cfs, &entries, ".fdm")
            .unwrap()
            .as_slice();
        let fnm = compound_format::open_input(&cfs, &entries, ".fnm")
            .unwrap()
            .as_slice();

        // Field infos recovered through the compound reader must still parse
        // and describe the one stored-only field we flushed.
        let parsed_fields = field_infos::parse(fnm, &segment_id, "").unwrap();
        assert_eq!(parsed_fields.fields.len(), 1);
        assert_eq!(parsed_fields.fields[0].name, "id");

        // Stored fields recovered through the compound reader must still
        // open and yield the exact documents flushed, in order.
        let reader = stored_fields::open(fdt, fdx, fdm, &segment_id, "").unwrap();
        for (i, expected) in docs.iter().enumerate() {
            let got = reader.document(i as i32).unwrap();
            assert_eq!(got.fields.len(), expected.fields.len());
            let expected_value = match &expected.fields[0].value {
                FieldValue::String(s) => s.clone(),
                other => panic!("unexpected fixture field value shape: {other:?}"),
            };
            match &got.fields[0].value {
                FieldValue::String(s) => assert_eq!(*s, expected_value),
                other => panic!("unexpected recovered field value shape: {other:?}"),
            }
        }
    }

    use lucene_util::test_support::TempDir;

    /// A scratch directory that removes itself when the test ends -- unless the
    /// test is panicking, in which case its bytes stay for inspection.
    fn tempdir() -> TempDir {
        TempDir::new("segment-writer")
    }

    /// `comparators`' guard, which is unreachable through `IndexWriter`
    /// (`set_index_sort` refuses a `BinarySortField`) but is the difference
    /// between a loud failure and a segment whose `.si` claims an order its
    /// bytes do not have: with no comparator, every pair would compare equal
    /// and the "sort" would be the identity permutation.
    #[test]
    #[should_panic(expected = "has no single-i64 comparator")]
    fn a_sort_with_no_comparator_panics_rather_than_sorting_by_nothing() {
        let sort = IndexSortField {
            field: "bytes".to_string(),
            reverse: false,
            kind: crate::segment_info::IndexSortKind::Binary(
                crate::segment_info::StringMissingValue::Last,
            ),
        };
        let keys = [Some(1), Some(0)];
        sort_permutation(
            2,
            &[SortKeySpec {
                sort: &sort,
                keys: &keys,
            }],
        );
    }

    fn doc_ids(
        dir: &FsDirectory,
        segment_name: &str,
        segment_id: [u8; ID_LENGTH],
        n: usize,
    ) -> Vec<String> {
        let fdt = dir.open(&format!("{segment_name}.fdt")).unwrap().to_vec();
        let fdx = dir.open(&format!("{segment_name}.fdx")).unwrap().to_vec();
        let fdm = dir.open(&format!("{segment_name}.fdm")).unwrap().to_vec();
        let reader = stored_fields::open(&fdt, &fdx, &fdm, &segment_id, "").unwrap();
        (0..n as i32)
            .map(|i| {
                let d = reader.document(i).unwrap();
                match &d.fields[0].value {
                    FieldValue::String(s) => s.clone(),
                    other => panic!("unexpected field value shape: {other:?}"),
                }
            })
            .collect()
    }

    /// Reads back the `.si`'s index-sort descriptor for a flushed segment.
    fn read_index_sort(
        dir: &FsDirectory,
        segment_name: &str,
        segment_id: [u8; ID_LENGTH],
    ) -> Option<Vec<crate::segment_info::IndexSortField>> {
        let si_bytes = dir.open(&format!("{segment_name}.si")).unwrap().to_vec();
        segment_info::parse(&si_bytes, &segment_id)
            .unwrap()
            .index_sort
    }

    #[test]
    fn sorted_flush_reorders_docs_ascending_by_numeric_key() {
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let segment_id = [11u8; ID_LENGTH];
        // Insertion order: c(30), a(10), b(20) -- must come out a, b, c.
        let docs = vec![doc("c"), doc("a"), doc("b")];
        let sort_keys = vec![Some(30), Some(10), Some(20)];

        flush_sorted_stored_only_segment(
            &dir,
            "_0",
            segment_id,
            "Lucene104",
            version(),
            &fields,
            &docs,
            &[SortKeySpec {
                sort: &IndexSortField::long("num", false, Some(i64::MAX)),
                keys: &sort_keys,
            }],
        )
        .unwrap();

        assert_eq!(doc_ids(&dir, "_0", segment_id, 3), vec!["a", "b", "c"]);
        let fields_sort = read_index_sort(&dir, "_0", segment_id).unwrap();
        assert_eq!(fields_sort.len(), 1);
        assert_eq!(
            fields_sort[0],
            IndexSortField::long("num", false, Some(i64::MAX))
        );
    }

    #[test]
    fn sorted_flush_reorders_docs_descending_by_numeric_key() {
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let segment_id = [12u8; ID_LENGTH];
        let docs = vec![doc("a"), doc("b"), doc("c")];
        let sort_keys = vec![Some(10), Some(20), Some(30)];

        flush_sorted_stored_only_segment(
            &dir,
            "_0",
            segment_id,
            "Lucene104",
            version(),
            &fields,
            &docs,
            &[SortKeySpec {
                sort: &IndexSortField::long("num", true, Some(i64::MIN)),
                keys: &sort_keys,
            }],
        )
        .unwrap();

        assert_eq!(doc_ids(&dir, "_0", segment_id, 3), vec!["c", "b", "a"]);
        let fields_sort = read_index_sort(&dir, "_0", segment_id).unwrap();
        assert!(fields_sort[0].reverse);
    }

    #[test]
    fn sorted_flush_places_missing_values_first() {
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let segment_id = [13u8; ID_LENGTH];
        // b has no value; with MissingValue::First it must land before a/c
        // regardless of its insertion position.
        let docs = vec![doc("a"), doc("b"), doc("c")];
        let sort_keys = vec![Some(10), None, Some(20)];

        flush_sorted_stored_only_segment(
            &dir,
            "_0",
            segment_id,
            "Lucene104",
            version(),
            &fields,
            &docs,
            &[SortKeySpec {
                sort: &IndexSortField::long("num", false, Some(i64::MIN)),
                keys: &sort_keys,
            }],
        )
        .unwrap();

        assert_eq!(doc_ids(&dir, "_0", segment_id, 3), vec!["b", "a", "c"]);
    }

    #[test]
    fn sorted_flush_places_missing_values_last() {
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let segment_id = [14u8; ID_LENGTH];
        let docs = vec![doc("a"), doc("b"), doc("c")];
        let sort_keys = vec![Some(10), None, Some(20)];

        flush_sorted_stored_only_segment(
            &dir,
            "_0",
            segment_id,
            "Lucene104",
            version(),
            &fields,
            &docs,
            &[SortKeySpec {
                sort: &IndexSortField::long("num", false, Some(i64::MAX)),
                keys: &sort_keys,
            }],
        )
        .unwrap();

        assert_eq!(doc_ids(&dir, "_0", segment_id, 3), vec!["a", "c", "b"]);
    }

    /// The missing-value **sentinel is reversed along with every other
    /// value**, which is what real Lucene does and what the `.si` this flush
    /// writes actually says.
    ///
    /// `segment_info::write_sort_field` emits `SortField(field, LONG,
    /// reverse)` with `missingValue = Long.MAX_VALUE` for
    /// missing-last, and Java's reader-side comparator for that
    /// is `reverseMul * Long.compare(values[d1], values[d2])` over an array
    /// pre-filled with the sentinel (`IndexSorter.LongSorter`). So with
    /// `reverse: true` the `Long.MAX_VALUE` doc compares **greatest** and,
    /// after `reverseMul`, lands **first**.
    ///
    /// This test used to assert the opposite ("missing-last stays last even
    /// when reversed"), which made the physical order disagree with the `.si`
    /// describing it -- a disagreement real Lucene's `CheckIndex.testSort`
    /// rejects outright.
    #[test]
    fn sorted_flush_missing_value_sentinel_is_reversed_like_any_other_value() {
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let segment_id = [16u8; ID_LENGTH];
        let docs = vec![doc("a"), doc("b"), doc("c")];
        let sort_keys = vec![Some(10), None, Some(20)];

        flush_sorted_stored_only_segment(
            &dir,
            "_0",
            segment_id,
            "Lucene104",
            version(),
            &fields,
            &docs,
            &[SortKeySpec {
                // descending, missing-last
                sort: &IndexSortField::long("num", true, Some(i64::MAX)),
                keys: &sort_keys,
            }],
        )
        .unwrap();

        // Values as Lucene sees them: a=10, b=Long.MAX_VALUE (missing,
        // `Last`), c=20. Ascending that is a, c, b; reversed it is b, c, a.
        assert_eq!(doc_ids(&dir, "_0", segment_id, 3), vec!["b", "c", "a"]);
    }

    #[test]
    fn sorted_flush_is_stable_for_equal_keys() {
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let segment_id = [15u8; ID_LENGTH];
        // a and b tie at key 10 -- must keep original relative order (a
        // before b), with c(5) placed first.
        let docs = vec![doc("a"), doc("b"), doc("c")];
        let sort_keys = vec![Some(10), Some(10), Some(5)];

        flush_sorted_stored_only_segment(
            &dir,
            "_0",
            segment_id,
            "Lucene104",
            version(),
            &fields,
            &docs,
            &[SortKeySpec {
                sort: &IndexSortField::long("num", false, Some(i64::MAX)),
                keys: &sort_keys,
            }],
        )
        .unwrap();

        assert_eq!(doc_ids(&dir, "_0", segment_id, 3), vec!["c", "a", "b"]);
    }

    #[test]
    #[should_panic(expected = "sort_keys must have exactly one entry per doc")]
    fn sorted_flush_panics_on_mismatched_sort_keys_length() {
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];

        let _ = flush_sorted_stored_only_segment(
            &dir,
            "_0",
            [16u8; ID_LENGTH],
            "Lucene104",
            version(),
            &fields,
            &[doc("a"), doc("b")],
            &[SortKeySpec {
                sort: &IndexSortField::long("num", false, Some(i64::MAX)),
                keys: &[Some(1)],
            }],
        );
    }

    #[test]
    #[should_panic(expected = "sort_fields must contain at least one sort key")]
    fn sorted_flush_panics_on_empty_sort_fields() {
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];

        let _ = flush_sorted_stored_only_segment(
            &dir,
            "_0",
            [17u8; ID_LENGTH],
            "Lucene104",
            version(),
            &fields,
            &[doc("a")],
            &[],
        );
    }

    /// The core new multi-field behavior: field 1 has ties (all docs share
    /// key `1`), so field 2 must break them. Without priority-ordered
    /// comparison this would either stay in insertion order (ignoring field
    /// 2 entirely) or crash.
    #[test]
    fn sorted_flush_second_field_breaks_ties_in_first() {
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let segment_id = [20u8; ID_LENGTH];
        // Every doc ties on `primary`; `secondary` must decide the order.
        let docs = vec![doc("a"), doc("b"), doc("c")];
        let primary = vec![Some(1), Some(1), Some(1)];
        let secondary = vec![Some(30), Some(10), Some(20)];

        flush_sorted_stored_only_segment(
            &dir,
            "_0",
            segment_id,
            "Lucene104",
            version(),
            &fields,
            &docs,
            &[
                SortKeySpec {
                    sort: &IndexSortField::long("primary", false, Some(i64::MAX)),
                    keys: &primary,
                },
                SortKeySpec {
                    sort: &IndexSortField::long("secondary", false, Some(i64::MAX)),
                    keys: &secondary,
                },
            ],
        )
        .unwrap();

        // Tied on `primary`; ascending `secondary` breaks the tie: b(10),
        // c(20), a(30).
        assert_eq!(doc_ids(&dir, "_0", segment_id, 3), vec!["b", "c", "a"]);
        let fields_sort = read_index_sort(&dir, "_0", segment_id).unwrap();
        assert_eq!(fields_sort.len(), 2);
        assert_eq!(fields_sort[0].field, "primary");
        assert_eq!(fields_sort[1].field, "secondary");
    }

    /// The primary field actually differs here, so ties never reach the
    /// second field's comparator -- confirms the primary field, not the
    /// secondary, decides the order whenever it can.
    #[test]
    fn sorted_flush_first_field_wins_when_it_differs() {
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let segment_id = [21u8; ID_LENGTH];
        let docs = vec![doc("a"), doc("b"), doc("c")];
        // Distinct primary values; secondary is reversed from primary order
        // to prove primary wins outright.
        let primary = vec![Some(30), Some(10), Some(20)];
        let secondary = vec![Some(1), Some(2), Some(3)];

        flush_sorted_stored_only_segment(
            &dir,
            "_0",
            segment_id,
            "Lucene104",
            version(),
            &fields,
            &docs,
            &[
                SortKeySpec {
                    sort: &IndexSortField::long("primary", false, Some(i64::MAX)),
                    keys: &primary,
                },
                SortKeySpec {
                    sort: &IndexSortField::long("secondary", false, Some(i64::MAX)),
                    keys: &secondary,
                },
            ],
        )
        .unwrap();

        assert_eq!(doc_ids(&dir, "_0", segment_id, 3), vec!["b", "c", "a"]);
    }

    /// Each field's `reverse` flag is independent: field 1 descending while
    /// field 2 ascending, with field 1 tied so field 2 must decide -- proves
    /// one field's reverse setting doesn't leak into another's comparison.
    #[test]
    fn sorted_flush_second_field_reverse_is_independent_of_first() {
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let segment_id = [22u8; ID_LENGTH];
        let docs = vec![doc("a"), doc("b"), doc("c")];
        let primary = vec![Some(1), Some(1), Some(1)];
        let secondary = vec![Some(10), Some(30), Some(20)];

        flush_sorted_stored_only_segment(
            &dir,
            "_0",
            segment_id,
            "Lucene104",
            version(),
            &fields,
            &docs,
            &[
                SortKeySpec {
                    // descending, but tied -- irrelevant here
                    sort: &IndexSortField::long("primary", true, Some(i64::MAX)),
                    keys: &primary,
                },
                SortKeySpec {
                    // descending: b(30), c(20), a(10)
                    sort: &IndexSortField::long("secondary", true, Some(i64::MAX)),
                    keys: &secondary,
                },
            ],
        )
        .unwrap();

        assert_eq!(doc_ids(&dir, "_0", segment_id, 3), vec!["b", "c", "a"]);
        let fields_sort = read_index_sort(&dir, "_0", segment_id).unwrap();
        assert!(fields_sort[0].reverse);
        assert!(fields_sort[1].reverse);
    }

    /// `permute_in_place` must implement `newToOld` -- `items[i]` becomes
    /// what `items[new_to_old[i]]` was. The inverse permutation is a
    /// plausible, self-consistent, wrong answer (it is the same permutation
    /// for every involution, so a two-element or reversed test cannot tell
    /// them apart): this uses a 3-cycle, which can.
    #[test]
    fn permute_in_place_applies_new_to_old_not_its_inverse() {
        let mut items = vec!["a", "b", "c"];
        // doc 0 <- old 2, doc 1 <- old 0, doc 2 <- old 1.
        permute_in_place(&mut items, &[2, 0, 1]);
        assert_eq!(items, vec!["c", "a", "b"]);

        // An involution, to show it is right there too, and the identity.
        let mut items = vec![1, 2, 3, 4];
        permute_in_place(&mut items, &[3, 1, 2, 0]);
        assert_eq!(items, vec![4, 2, 3, 1]);
        let mut items = vec![1, 2, 3];
        permute_in_place(&mut items, &[0, 1, 2]);
        assert_eq!(items, vec![1, 2, 3]);
        let mut empty: Vec<u8> = Vec::new();
        permute_in_place(&mut empty, &[]);
        assert!(empty.is_empty());
    }

    /// The permutation is the same one `flush_sorted_stored_only_segment`
    /// and `IndexWriter::flush` both apply, so it is worth pinning
    /// separately: multi-tier priority, stability, and the missing-value
    /// sentinel reversing with everything else.
    #[test]
    fn sort_permutation_is_priority_ordered_stable_and_sentinel_reversing() {
        let primary = vec![Some(1), Some(1), Some(2), Some(1)];
        let secondary = vec![Some(30), Some(10), Some(0), Some(10)];
        let order = sort_permutation(
            4,
            &[
                SortKeySpec {
                    sort: &IndexSortField::long("p", false, Some(i64::MAX)),
                    keys: &primary,
                },
                SortKeySpec {
                    sort: &IndexSortField::long("s", false, Some(i64::MAX)),
                    keys: &secondary,
                },
            ],
        );
        // p ascending: {1,3} tie at (1,10) -- kept in insertion order -- then
        // 0 at (1,30), then 2 at p=2.
        assert_eq!(order, vec![1, 3, 0, 2]);

        // One tier, reversed, with a missing value: `Last` is Long.MAX_VALUE,
        // so reversed it sorts first.
        let keys = vec![Some(5), None, Some(9)];
        let order = sort_permutation(
            3,
            &[SortKeySpec {
                sort: &IndexSortField::long("p", true, Some(i64::MAX)),
                keys: &keys,
            }],
        );
        assert_eq!(order, vec![1, 2, 0]);
    }
}
