//! Flushes a batch of stored-only documents to a new segment, the building
//! block under the fixtures and the stored-fields-only writer paths.
//!
//! [`flush_stored_only_segment`] takes an in-memory batch of already-built
//! [`Document`]s and [`FieldInfo`]s, writes them to a brand-new segment
//! (`.fdt`/`.fdx`/`.fdm` stored fields, `.fnm` field infos and `.si` segment
//! info, all through a [`Directory`]), and returns the [`SegmentCommitInfo`]
//! that describes it, ready for [`crate::segment_infos::SegmentInfos::segments`].
//! Calling it more than once against one `Directory` with distinct segment
//! names and passing all the results to one [`crate::segment_infos::write()`]
//! produces a multi-segment commit.
//!
//! It is not an `IndexWriter`: the caller decides when to flush, and every
//! field is stored-only (`IndexOptions::None`, no doc values, points, vectors
//! or term vectors), so the segment needs none of those producers' files.
//! Indexed documents go through [`crate::index_writer`].

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
    #[error(transparent)]
    StoredFields(#[from] stored_fields::Error),
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
    // The convenience wrappers keep their `SegmentCommitInfo`-only shape: they
    // have no `IndexFileDeleter` to hand the file list to.
    seal_flushed_segment(dir, segment_name, flushed).map(|(sci, _)| sci)
}

/// A segment whose codec files are on disk but whose `.si` is **not yet
/// written**: the in-memory `SegmentInfo` that will become it, the
/// `SegmentCommitInfo` that will describe it in `segments_N`.
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
/// descriptor re-opened, re-parsed, extended and re-wrote the `.si`: up to
/// seven `.si` writes and six parses per commit, of
/// a file whose whole content was already in memory.
///
/// A caller that writes more than stored fields therefore takes
/// [`write_stored_only_segment_files`], pushes its own file names into
/// [`Self::info`]`.files`, and finishes with [`seal_flushed_segment`].
pub struct FlushedSegment {
    /// The segment's `.si` content, still only in memory. A caller extends
    /// `files` (and may set `index_sort`) before sealing.
    pub info: SegmentInfo,
    /// The `segments_N` entry for this segment.
    pub commit: SegmentCommitInfo,
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
///
/// A non-compound call whose documents hold at least
/// [`stored_fields::SPILL_BYTES`] of stored values streams the `.fdt` to its
/// file as it compresses (the bytes are the same); a compound call packs it
/// from memory.
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
    write_stored_only_segment_files_inner(
        dir,
        segment_name,
        segment_id,
        codec_name,
        lucene_version,
        fields,
        docs,
        use_compound_file,
        has_blocks,
        None,
        None,
    )
}

/// `StoredFieldsFormat.fieldsWriter` for an IndexWriter flush, before its
/// separately assembled compound file and segment info are sealed.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_stored_only_segment_files_with_codec(
    dir: &dyn Directory,
    segment_name: &str,
    segment_id: [u8; ID_LENGTH],
    codec_name: &str,
    lucene_version: LuceneVersion,
    fields: &[FieldInfo],
    docs: &[Document],
    has_blocks: bool,
    codec: Option<&dyn lucene_codecs::codec::Lucene104Codec>,
    prebuilt: Option<StoredFieldsBytes>,
) -> Result<FlushedSegment> {
    write_stored_only_segment_files_inner(
        dir,
        segment_name,
        segment_id,
        codec_name,
        lucene_version,
        fields,
        docs,
        false,
        has_blocks,
        codec,
        prebuilt,
    )
}

/// The native stored-fields writer's `(.fdt, .fdx, .fdm)` bytes; an empty
/// `.fdt` is already in its file.
pub(crate) type StoredFieldsBytes = (Vec<u8>, Vec<u8>, Vec<u8>);

/// The bytes [`write_stored_only_segment_files_with_codec`] would build with
/// the native writer for `docs`, or `None` when `codec` installs its own
/// stored-fields writer (which writes through the directory itself).
///
/// Pure computation -- compressing every document -- with no directory
/// access, so a flush runs it on another thread alongside inversion and
/// hands the result back as `prebuilt`.
pub(crate) fn build_native_stored_fields(
    docs: &[Document],
    segment_id: &[u8; ID_LENGTH],
    codec: Option<&dyn lucene_codecs::codec::Lucene104Codec>,
) -> Option<StoredFieldsBytes> {
    if codec.is_some_and(|codec| codec.stored_fields_writer_factory().is_some()) {
        return None;
    }
    let mode = codec.map_or(stored_fields::Mode::BestSpeed, |codec| {
        codec.stored_fields_mode()
    });
    let mut writer =
        stored_fields::StoredFieldsWriter::new(mode, segment_id, mode.segment_suffix());
    for doc in docs {
        writer.add_document(doc);
    }
    Some(writer.finish())
}

/// [`build_native_stored_fields`] for a flush that writes the `.fdt` itself:
/// when `docs` hold at least [`stored_fields::SPILL_BYTES`] of stored values,
/// the writer streams the `.fdt` to `fdt_name` in `dir` as it compresses, so
/// a flush of a few very large stored values holds about a chunk of
/// compressed bytes instead of the whole file. The bytes written are the
/// ones the buffered path writes; the returned `.fdt` is then empty, as
/// [`StoredFieldsBytes`] says an empty one is already in its file.
///
/// # Errors
/// Creating or closing the `.fdt`.
pub(crate) fn build_native_stored_fields_to(
    dir: &dyn Directory,
    fdt_name: &str,
    docs: &[Document],
    segment_id: &[u8; ID_LENGTH],
    codec: Option<&dyn lucene_codecs::codec::Lucene104Codec>,
) -> Result<Option<StoredFieldsBytes>> {
    if stored_bytes(docs) < stored_fields::SPILL_BYTES
        || codec.is_some_and(|codec| codec.stored_fields_writer_factory().is_some())
    {
        return Ok(build_native_stored_fields(docs, segment_id, codec));
    }
    let mode = codec.map_or(stored_fields::Mode::BestSpeed, |codec| {
        codec.stored_fields_mode()
    });
    let mut writer =
        stored_fields::StoredFieldsWriter::new(mode, segment_id, mode.segment_suffix());
    writer.stream_to(dir.create_output(fdt_name)?);
    for doc in docs {
        writer.add_document(doc);
    }
    let (out, fdx, fdm) = writer.finish_streamed();
    if let Some(out) = out {
        out.close()?;
    }
    Ok(Some((Vec::new(), fdx, fdm)))
}

/// The bytes of the string and binary values `docs` store, saturating.
fn stored_bytes(docs: &[Document]) -> usize {
    use stored_fields::FieldValue;
    docs.iter()
        .flat_map(|doc| &doc.fields)
        .map(|f| match &f.value {
            FieldValue::String(s) => s.len(),
            FieldValue::Shared(s) => s.len(),
            FieldValue::Binary(b) => b.len(),
            _ => 0,
        })
        .fold(0usize, usize::saturating_add)
}

/// The `.si` attribute a segment whose `BEST_SPEED` stored fields were all
/// written by the high-compression LZ4 match finder records (value
/// [`LZ4_HC_VALUE`]). Readers ignore it; a merge that writes the same way
/// copies such a segment's chunks instead of compressing them again.
pub const LZ4_HC_ATTRIBUTE: &str = "accelerated.stored_fields.lz4_hc";
/// [`LZ4_HC_ATTRIBUTE`]'s value.
pub const LZ4_HC_VALUE: &str = "hc";

/// Whether `attributes` carry [`LZ4_HC_ATTRIBUTE`].
pub(crate) fn has_lz4_hc_attribute(attributes: &[(String, String)]) -> bool {
    attributes
        .iter()
        .any(|(k, v)| k == LZ4_HC_ATTRIBUTE && v == LZ4_HC_VALUE)
}

pub use lucene_codecs::stored_fields::{
    ZSTD_PAGED_MODE_ATTRIBUTE as ZSTD_PAGED_ATTRIBUTE, ZSTD_PAGED_MODE_VALUE as ZSTD_PAGED_VALUE,
};

/// `Lucene90StoredFieldsFormat.fieldsWriter`'s `.si` attribute naming its
/// mode, or OpenSearch custom-codecs' `Lucene104CustomStoredFieldsFormat`
/// one; a `CompressingCodec` format records none.
pub(crate) fn stored_fields_mode_attribute(mode: stored_fields::Mode) -> Vec<(String, String)> {
    let value = match mode {
        stored_fields::Mode::BestSpeed => "BEST_SPEED",
        stored_fields::Mode::BestCompression => "BEST_COMPRESSION",
        stored_fields::Mode::Compressing(_) => return Vec::new(),
        stored_fields::Mode::Zstd(z) => {
            return vec![(
                "Lucene104CustomStoredFieldsFormat.mode".to_string(),
                z.mode_name().to_string(),
            )]
        }
        stored_fields::Mode::ZstdPaged(_) => {
            return vec![(
                ZSTD_PAGED_ATTRIBUTE.to_string(),
                ZSTD_PAGED_VALUE.to_string(),
            )]
        }
    };
    vec![(
        "Lucene90StoredFieldsFormat.mode".to_string(),
        value.to_string(),
    )]
}

#[allow(clippy::too_many_arguments)]
fn write_stored_only_segment_files_inner(
    dir: &dyn Directory,
    segment_name: &str,
    segment_id: [u8; ID_LENGTH],
    codec_name: &str,
    lucene_version: LuceneVersion,
    fields: &[FieldInfo],
    docs: &[Document],
    use_compound_file: bool,
    has_blocks: bool,
    codec: Option<&dyn lucene_codecs::codec::Lucene104Codec>,
    prebuilt: Option<StoredFieldsBytes>,
) -> Result<FlushedSegment> {
    let doc_count = docs.len() as i32;
    let factory = codec.and_then(|codec| codec.stored_fields_writer_factory());
    let mode = codec.map_or(stored_fields::Mode::BestSpeed, |codec| {
        codec.stored_fields_mode()
    });
    let suffix = mode.segment_suffix();

    let native = if let Some(factory) = factory {
        use lucene_codecs::stored_fields_writer::{with_writer, StoredFieldsWriteState};
        // Only the non-compound IndexWriter entry point supplies a factory.
        // Its files are packed later, together with the other segment formats.
        with_writer(
            factory,
            StoredFieldsWriteState {
                directory: dir,
                segment: segment_name,
                id: &segment_id,
                suffix,
                mode,
            },
            |writer| -> stored_fields::Result<()> {
                for doc in docs {
                    writer.start_document()?;
                    for field in &doc.fields {
                        writer.write_field(field.field_number, &field.value)?;
                    }
                    writer.finish_document()?;
                }
                writer.finish(doc_count)
            },
        )?;
        None
    } else if let Some(prebuilt) = prebuilt {
        Some(prebuilt)
    } else if use_compound_file {
        // The compound file packs the `.fdt` from memory.
        build_native_stored_fields(docs, &segment_id, codec)
    } else {
        let fdt_name = crate::index_writer::segment_file_name(segment_name, suffix, "fdt");
        build_native_stored_fields_to(dir, &fdt_name, docs, &segment_id, codec)?
    };
    let fnm = field_infos::write(fields, &segment_id, "");

    let stored_name = |ext| crate::index_writer::segment_file_name(segment_name, suffix, ext);
    let (fdt_name, fdx_name, fdm_name) =
        (stored_name("fdt"), stored_name("fdx"), stored_name("fdm"));
    let fnm_name = format!("{segment_name}.fnm");

    let files = if let Some((fdt, fdx, fdm)) = native {
        if use_compound_file {
            debug_assert!(!fdt.is_empty(), "a compound flush builds the .fdt itself");
            // Compound entries are file names without the segment name.
            let member = |ext| crate::index_writer::segment_file_name("", suffix, ext);
            let sub_files = vec![
                (member("fdt"), fdt),
                (member("fdx"), fdx),
                (member("fdm"), fdm),
                (".fnm".to_string(), fnm),
            ];
            let (cfs, cfe) = compound_format::write(&segment_id, &sub_files)?;
            let cfs_name = format!("{segment_name}.cfs");
            let cfe_name = format!("{segment_name}.cfe");
            write_file(dir, &cfs_name, &cfs)?;
            write_file(dir, &cfe_name, &cfe)?;
            vec![cfs_name, cfe_name]
        } else {
            // An empty `.fdt` was written to its file as the buffer filled.
            if !fdt.is_empty() {
                write_file(dir, &fdt_name, &fdt)?;
            }
            for (name, bytes) in [(&fdx_name, &fdx), (&fdm_name, &fdm)] {
                write_file(dir, name, bytes)?;
            }
            write_file(dir, &fnm_name, &fnm)?;
            vec![fdt_name, fdx_name, fdm_name, fnm_name]
        }
    } else {
        write_file(dir, &fnm_name, &fnm)?;
        vec![fdt_name, fdx_name, fdm_name, fnm_name]
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
        // `DocumentsWriterPerThread.sealFlushedSegment`.
        diagnostics: segment_info::writer_diagnostics(
            segment_info::SOURCE_FLUSH,
            lucene_version,
            &[],
        ),
        files,
        attributes: stored_fields_mode_attribute(mode),
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
    })
}

/// `IndexWriter.sealFlushedSegment`'s tail: writes the segment's `.si` from
/// the in-memory [`SegmentInfo`] -- **once**.
///
/// Nothing is fsynced here, as in Java: a flushed segment's files become
/// durable when the first commit that names them syncs them
/// ([`crate::index_file_deleter::IndexFileDeleter::sync_commit_files`]). A
/// caller that writes its own `segments_N` over this segment without an
/// `IndexWriter` must sync `info.files` first.
pub fn seal_flushed_segment(
    dir: &dyn Directory,
    segment_name: &str,
    flushed: FlushedSegment,
) -> Result<(SegmentCommitInfo, Vec<String>)> {
    let FlushedSegment { info, commit } = flushed;
    let si_name = format!("{segment_name}.si");
    debug_assert!(
        info.files.contains(&si_name),
        "a segment's own `.si` must be a member of the file set it records"
    );
    write_file(dir, &si_name, &segment_info::write(&info, ""))?;
    // The file list travels out with the commit so
    // `IndexFileDeleter::record_segment_files` can reference-count the segment
    // without opening and re-parsing the `.si` this function just wrote --
    // which is what Java gets for free from `SegmentCommitInfo` holding its
    // `SegmentInfo`.
    Ok((commit, info.files))
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
    // `SegmentInfo`, so there is nothing to re-read, re-parse and rewrite.
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
    seal_flushed_segment(dir, segment_name, flushed).map(|(sci, _)| sci)
}

/// The comparators `sort_fields` induces, in priority order.
fn comparators(sort_fields: &[SortKeySpec<'_>]) -> Vec<SortKeyComparator> {
    sort_fields
        .iter()
        .map(|spec| SortKeyComparator::new(spec.sort))
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

/// `IndexWriter.createCompoundFile`: packs every file of the segment but its
/// `.si` into `<segment>.cfs`/`.cfe` with [`compound_format::write`] (each
/// entry named by `IndexFileNames.stripSegmentName`: `.fdt`,
/// `_Lucene104_0.doc`, ...), deletes the loose files, and records the new
/// layout in `info` (`setFiles`, `setUseCompoundFile(true)`) -- the step both
/// `DocumentsWriterPerThread.sealFlushedSegment` and `IndexWriter.mergeMiddle`
/// take before the `.si` is written.
///
/// `.liv` and doc-values/field-infos update files are never packed: they are
/// written after the `.si`, per generation, outside `info.files`.
pub fn create_compound_file(
    dir: &dyn Directory,
    segment_name: &str,
    info: &mut SegmentInfo,
) -> Result<()> {
    let si_name = format!("{segment_name}.si");
    let mut sub_files = Vec::with_capacity(info.files.len());
    let mut packed = Vec::with_capacity(info.files.len());
    for file in &info.files {
        if *file == si_name {
            continue;
        }
        let bytes = dir.open(file)?.to_vec();
        let entry = file.strip_prefix(segment_name).unwrap_or(file).to_string();
        sub_files.push((entry, bytes));
        packed.push(file.clone());
    }
    let (cfs, cfe) = compound_format::write(&info.id, &sub_files)?;
    let cfs_name = format!("{segment_name}.cfs");
    let cfe_name = format!("{segment_name}.cfe");
    write_file(dir, &cfs_name, &cfs)?;
    write_file(dir, &cfe_name, &cfe)?;
    for file in &packed {
        dir.delete_file(file)?;
    }
    let lists_si = info.files.contains(&si_name);
    info.files = vec![cfs_name.clone(), cfe_name.clone()];
    if lists_si {
        info.files.push(si_name);
    }
    info.is_compound_file = true;
    Ok(())
}

/// [`create_compound_file`] for a segment whose `.si` is already on disk (a
/// merge writes it before the writer's `useCompoundFile` decision): packs,
/// then rewrites the `.si` so it records the compound layout --
/// the same bytes Java's single, later `.si` write produces. Returns the
/// segment's new file list.
pub fn convert_to_compound(
    dir: &dyn Directory,
    segment_name: &str,
    segment_id: &[u8; ID_LENGTH],
) -> Result<Vec<String>> {
    let si_name = format!("{segment_name}.si");
    let si_bytes = dir.open(&si_name)?.to_vec();
    let mut info = segment_info::parse(&si_bytes, segment_id)?;
    create_compound_file(dir, segment_name, &mut info)?;
    dir.delete_file(&si_name)?;
    write_file(dir, &si_name, &segment_info::write(&info, ""))?;
    Ok(info.files)
}

fn write_file(dir: &dyn Directory, name: &str, bytes: &[u8]) -> Result<()> {
    let mut out = dir.create_output(name)?;
    out.write_bytes(bytes);
    out.close()?;
    Ok(())
}

#[cfg(test)]
mod tests;
