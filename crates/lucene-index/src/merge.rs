//! Port of `org.apache.lucene.index.SegmentMerger` (plus the field-numbering
//! half of `FieldInfos.FieldNumbers`) -- merges N already-flushed segments
//! into one new segment, dropping deleted docs and renumbering doc ids to be
//! contiguous (`0..mergedDocCount`). Stored fields are always merged; doc
//! values, norms, and term vectors are merged too whenever a source supplies
//! them (see "Doc values / norms / term vectors" below for the honest scope
//! of that part).
//!
//! # What this is
//!
//! [`merge_stored_only_segments`] takes, for each source segment, its already
//! read-back [`FieldInfos`](field_infos::FieldInfos), a [`Document`] reader
//! (via this port's stored-fields reader, [`stored_fields::open`] +
//! [`stored_fields::StoredFieldsReader::document`]), an optional per-source
//! live-docs bitset (via [`live_docs::parse`], or `None` if the source has no
//! deletions), and optional per-source doc-values/norms/term-vectors data
//! (see [`MergeSource`]), and:
//! 1. reconciles field numbering across sources by field name (see
//!    [`reconcile_field_numbers`]) -- real Lucene's `FieldInfos.FieldNumbers`
//!    does the same job (a global, writer-wide field-number authority so the
//!    same field name gets the same number everywhere), scoped down here to
//!    exactly the merge-time case: two segments naming the same field
//!    differently, or a field only some segments have;
//! 2. filters out non-live docs per source, remaps each surviving doc's
//!    field numbers to the merged numbering, and renumbers docs contiguously
//!    by simply concatenating surviving docs in source order (matches real
//!    `SegmentMerger`'s `MergeState.docMaps`, minus any doc-ID-remapping
//!    policy fancier than "keep source order, drop gaps" -- this port has no
//!    index sort or other doc-reordering merge policy yet);
//! 3. merges any supplied doc-values/norms/term-vectors data the same way
//!    (drop deleted docs, renumber contiguously, remap field numbers), then
//!    writes stored fields, field infos, segment info, and whichever of
//!    `.dvm`/`.dvd`/`.dvs`, `.nvm`/`.nvd`, `.tvd`/`.tvx`/`.tvm` the merge
//!    produced, directly through `dir` -- mirroring exactly the write-side
//!    work [`crate::segment_writer::flush_stored_only_segment`] does for a
//!    stored-fields-only flush, generalized to the extra formats.
//!
//! # What this deliberately is not
//!
//! - **Not a merge policy.** No `TieredMergePolicy`-style "which segments
//!   should merge, and when" decision -- the caller picks the sources.
//! - **Not concurrent/background.** One synchronous call, like
//!   `flush_stored_only_segment`.
//! - **No merge-time codec upgrade.** The merged segment's codec/version are
//!   caller-supplied, same stance as `flush_stored_only_segment`.
//! - **No `FieldInfos.FieldNumbers`-style full schema-consistency check.**
//!   Real Lucene's field-number authority also verifies that two segments
//!   agreeing on a field name agree on its indexing options, doc-values
//!   type, etc. (`verifySameSchema`). This port's reconciliation only unifies
//!   field *numbers* by name; it does not check that two sources agree on
//!   every other `FieldInfo` attribute. Revisit if that ever bites.
//!
//! # Doc values / norms / term vectors: mergeable, but not from a real flush
//!
//! [`segment_writer::flush_stored_only_segment`] -- the only write-side path
//! that produces a full segment in this port -- still only ever writes
//! stored-fields-only segments; nothing in this port's normal flush path
//! ever produces a segment with doc values, norms, or term vectors. So this
//! module cannot (yet) be exercised end-to-end from "flush two real segments,
//! merge them": there is no real caller that hands it doc-values/norms/
//! term-vectors *sources* today.
//!
//! What *is* real: the write-side encoders for these formats already exist
//! as standalone functions ([`lucene_codecs::doc_values::write_single_dense_numeric_field`],
//! [`lucene_codecs::norms::write_single_dense_field`],
//! [`lucene_codecs::term_vectors::write_best_speed`]), and their read-side
//! counterparts can decode arbitrary per-source data (including data written
//! by a test, or by some future caller once a real per-field flush path
//! exists). [`merge_stored_only_segments`] therefore accepts, per source,
//! *optional* already-decoded doc-values/norms/term-vectors data (see
//! [`MergeSource`]) and, if supplied, merges it the same way stored fields
//! are merged (drop deleted docs, renumber contiguously, reconcile field
//! numbers) and re-encodes it with the existing write functions. This makes
//! the merge logic real and testable without requiring a new flush path --
//! but until a caller exists that can *produce* per-field doc-values/norms/
//! term-vectors data for a real segment, nothing in this port actually
//! drives this code outside of its own tests.
//!
//! ## Scope of the doc-values/norms merge
//!
//! [`lucene_codecs::doc_values::write_single_dense_numeric_field`] and
//! [`lucene_codecs::norms::write_single_dense_field`] each write a complete,
//! self-contained `.dvm`/`.dvd`/`.dvs` (or `.nvm`/`.nvd`) file pair/triple
//! for exactly **one field** -- multi-field `.dvd`/`.nvd` files (the real
//! on-disk shape, where every field's data shares one file) aren't
//! supported by this port's write side yet. This merge inherits that same
//! limit: at most one numeric-doc-values field and at most one norms field
//! may be merged per call ([`Error::TooManyNumericDocValuesFields`] /
//! [`Error::TooManyNormsFields`] otherwise). Term vectors have no such limit
//! (`write_best_speed` already handles any number of fields per doc).
//!
//! ## The "sparse across sources" rule
//!
//! Real Lucene requires every doc in a merged segment to either uniformly
//! have or uniformly lack doc-values/norms for a field, per that field's
//! `FieldInfos` declaration -- a field can't have doc values for some docs
//! and not others within one segment (`DocValuesType.NONE` vs. non-`NONE` is
//! segment-wide per field). This port's write functions go further: they
//! only support the fully **dense** case (every doc 0..max_doc has a value).
//! So a doc-values/norms field can only be merged here if **every source
//! that contributes at least one live doc** supplies decodable data for that
//! field for **every one of its live docs** -- if any live-doc-contributing
//! source is missing the field entirely, or has it only sparsely, this
//! returns [`Error::DocValuesFieldMissingInSource`] /
//! [`Error::NormsFieldMissingInSource`] rather than silently dropping the
//! field or a doc's value.
//!
//! Term vectors have no such constraint: a source with no term-vectors
//! reader for a doc, or a doc with none, simply contributes an empty
//! [`lucene_codecs::term_vectors::TermVectorsDocument`] (matches the real
//! per-doc "this doc has none" case `write_best_speed` already handles).
//!
//! See `docs/parity.md` and `PLAN.md`'s Phase 5 section for the exact,
//! currently-true scope line.

use std::collections::HashMap;

use crate::segment_info::{self, IndexSortField, LuceneVersion, SegmentInfo, SortMissingValue};
use crate::segment_infos::SegmentCommitInfo;
use lucene_codecs::doc_values::{self, NumericEntry};
use lucene_codecs::field_infos::{self, FieldInfo};
use lucene_codecs::norms::{self, NormsEntry};
use lucene_codecs::stored_fields::{self, Document};
use lucene_codecs::term_vectors::{self, TermVectorsDocument, TermVectorsReader};
use lucene_store::codec_util::ID_LENGTH;
use lucene_store::data_output::DataOutput;
use lucene_store::directory::Directory;
use lucene_util::fixed_bit_set::FixedBitSet;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    #[error(transparent)]
    StoredFields(#[from] lucene_codecs::stored_fields::Error),
    #[error(transparent)]
    DocValues(#[from] lucene_codecs::doc_values::Error),
    #[error(transparent)]
    Norms(#[from] lucene_codecs::norms::Error),
    #[error(transparent)]
    TermVectors(#[from] lucene_codecs::term_vectors::Error),
    #[error(transparent)]
    DocValuesWrite(#[from] lucene_codecs::doc_values::WriteError),
    #[error(transparent)]
    NormsWrite(#[from] lucene_codecs::norms::WriteError),
    /// A `MergeSource`'s stored fields referenced a field number absent from
    /// that same source's own `field_infos` -- an inconsistent/malformed
    /// `MergeSource` (its `reader` and `field_infos` don't actually describe
    /// the same segment), not something a well-formed caller can trigger.
    #[error(
        "source segment's stored field number {field_number} has no entry in that source's own field_infos"
    )]
    UnknownSourceFieldNumber { field_number: i32 },
    /// More than one field across the merged sources has numeric doc-values
    /// data -- unsupported, see this module's doc comment on the
    /// single-field limit of `write_single_dense_numeric_field`.
    #[error(
        "merging numeric doc values for more than one field per call isn't supported yet (found fields {0:?})"
    )]
    TooManyNumericDocValuesFields(Vec<i32>),
    /// Same limit as [`Error::TooManyNumericDocValuesFields`], for norms.
    #[error(
        "merging norms for more than one field per call isn't supported yet (found fields {0:?})"
    )]
    TooManyNormsFields(Vec<i32>),
    /// A field has numeric doc-values data in at least one source that
    /// contributes live docs, but not in every such source (or not for
    /// every one of that source's live docs) -- see this module's doc
    /// comment on the "sparse across sources" rule.
    #[error(
        "merged field number {merged_field_number} has numeric doc values in some sources but not in every source that contributes live docs (or not for every one of that source's live docs)"
    )]
    DocValuesFieldMissingInSource { merged_field_number: i32 },
    /// Same as [`Error::DocValuesFieldMissingInSource`], for norms.
    #[error(
        "merged field number {merged_field_number} has norms in some sources but not in every source that contributes live docs (or not for every one of that source's live docs)"
    )]
    NormsFieldMissingInSource { merged_field_number: i32 },
}

pub type Result<T> = std::result::Result<T, Error>;

/// One source's numeric doc-values data for a single field: the whole
/// source segment's `.dvd` bytes plus the parsed [`NumericEntry`] describing
/// that field within it (`entry.field_number` is that source's *original*
/// field number, before merge-time renumbering).
pub struct SourceNumericDocValues<'a> {
    pub data: &'a [u8],
    pub entry: NumericEntry,
}

/// One source's norms data for a single field -- same shape as
/// [`SourceNumericDocValues`], for [`NormsEntry`]/`.nvd` instead.
pub struct SourceNorms<'a> {
    pub data: &'a [u8],
    pub entry: NormsEntry,
}

/// One source segment's already-decoded input to a merge: its field infos
/// (from `.fnm`, via [`lucene_codecs::field_infos::parse`]), a stored-fields
/// reader over its `.fdt`/`.fdx`/`.fdm` (via [`stored_fields::open`]), an
/// optional live-docs bitset (`None` means "no deletions -- every doc up to
/// `reader.max_doc()` is live", matching a segment whose `SegmentCommitInfo`
/// has `del_gen == -1`), and optional per-field doc-values/norms/
/// term-vectors data (all empty/`None` by default -- a source with none of
/// these contributes only stored fields, same as before this module gained
/// them).
pub struct MergeSource<'a> {
    pub field_infos: &'a [FieldInfo],
    pub reader: &'a lucene_codecs::stored_fields::StoredFieldsReader<'a>,
    pub live_docs: Option<&'a FixedBitSet>,
    /// This source's numeric doc-values fields, if any (see this module's
    /// doc comment: at most one distinct field across *all* sources may
    /// have numeric doc-values data in one merge call).
    pub numeric_doc_values: &'a [SourceNumericDocValues<'a>],
    /// This source's norms fields, if any (same one-field-across-all-sources
    /// limit as `numeric_doc_values`).
    pub norms: &'a [SourceNorms<'a>],
    /// This source's term-vectors reader, or `None` if this source has no
    /// term vectors at all (every doc then contributes an empty
    /// [`TermVectorsDocument`]).
    pub term_vectors: Option<&'a TermVectorsReader<'a>>,
}

impl<'a> MergeSource<'a> {
    /// Convenience constructor for the common "stored fields only" case
    /// (matches this module's original, pre-doc-values/norms/term-vectors
    /// shape) -- avoids every existing caller having to spell out three new
    /// empty/`None` fields.
    pub fn stored_only(
        field_infos: &'a [FieldInfo],
        reader: &'a lucene_codecs::stored_fields::StoredFieldsReader<'a>,
        live_docs: Option<&'a FixedBitSet>,
    ) -> Self {
        Self {
            field_infos,
            reader,
            live_docs,
            numeric_doc_values: &[],
            norms: &[],
            term_vectors: None,
        }
    }
}

/// Reconciles field numbering across `sources_fields` (one source's
/// [`FieldInfos`](field_infos::FieldInfos)-equivalent field list per entry):
/// assigns every distinct field *name* a single, contiguous merged field
/// number, in first-seen order across sources (source 0's fields first, then
/// any new names introduced by source 1, etc.) -- mirrors real Lucene's
/// `FieldInfos.FieldNumbers.addOrGet`, which hands out a process-wide number
/// per name and reuses it for every segment that has that field, regardless
/// of what number that segment originally used.
///
/// Returns the merged field list (one [`FieldInfo`] per distinct name, using
/// the *first* source's metadata for that name -- see this module's "what
/// this deliberately is not" note on schema consistency) and, per source, a
/// map from that source's original field number to the merged number.
pub fn reconcile_field_numbers(
    sources_fields: &[&[FieldInfo]],
) -> (Vec<FieldInfo>, Vec<HashMap<i32, i32>>) {
    let mut merged_fields: Vec<FieldInfo> = Vec::new();
    let mut name_to_merged_number: HashMap<String, i32> = HashMap::new();
    let mut per_source_maps: Vec<HashMap<i32, i32>> = Vec::with_capacity(sources_fields.len());

    for fields in sources_fields {
        let mut map = HashMap::with_capacity(fields.len());
        for f in *fields {
            let merged_number = *name_to_merged_number
                .entry(f.name.clone())
                .or_insert_with(|| {
                    let number = merged_fields.len() as i32;
                    let mut renumbered = f.clone();
                    renumbered.number = number;
                    merged_fields.push(renumbered);
                    number
                });
            map.insert(f.number, merged_number);
        }
        per_source_maps.push(map);
    }

    (merged_fields, per_source_maps)
}

/// Merges `sources` (already-opened, in source order) into one brand-new
/// stored-fields-only segment named `merged_segment_name` inside `dir`,
/// exactly as [`crate::segment_writer::flush_stored_only_segment`] writes a
/// freshly-flushed one -- deleted docs (per each source's `live_docs`) are
/// dropped, surviving docs are renumbered contiguously by concatenating
/// sources in order, and field numbers are reconciled by name (see
/// [`reconcile_field_numbers`]).
///
/// A source with `live_docs` fully cleared (every doc deleted) naturally
/// contributes zero docs to the merge -- this port merges it anyway rather
/// than requiring the caller to have already dropped it (real Lucene's
/// `IndexWriter` drops a 100%-deleted segment before a merge is even
/// scheduled, purely as a merge-policy optimization; skipping that
/// optimization here costs nothing but a no-op source pass).
pub fn merge_stored_only_segments(
    dir: &dyn Directory,
    sources: &[MergeSource],
    merged_segment_name: &str,
    merged_segment_id: [u8; ID_LENGTH],
    codec_name: &str,
    lucene_version: LuceneVersion,
) -> Result<SegmentCommitInfo> {
    let sources_fields: Vec<&[FieldInfo]> = sources.iter().map(|s| s.field_infos).collect();
    let (merged_fields, per_source_maps) = reconcile_field_numbers(&sources_fields);

    // Concatenate surviving docs in source order, remapping field numbers,
    // and remember each source's list of surviving (pre-merge) doc ids --
    // needed below to walk the same docs again for doc values/norms/term
    // vectors without recomputing liveness.
    let mut merged_docs: Vec<Document> = Vec::new();
    let mut per_source_live_ids: Vec<Vec<i32>> = Vec::with_capacity(sources.len());
    for (source, field_number_map) in sources.iter().zip(per_source_maps.iter()) {
        let max_doc = source.reader.max_doc();
        let mut live_ids = Vec::new();
        for doc_id in 0..max_doc {
            let is_live = source
                .live_docs
                .map(|bits| bits.get(doc_id as usize))
                .unwrap_or(true);
            if !is_live {
                continue;
            }
            live_ids.push(doc_id);
            let mut doc = source.reader.document(doc_id)?;
            for field in &mut doc.fields {
                field.field_number = *field_number_map.get(&field.field_number).ok_or(
                    Error::UnknownSourceFieldNumber {
                        field_number: field.field_number,
                    },
                )?;
            }
            merged_docs.push(doc);
        }
        per_source_live_ids.push(live_ids);
    }
    let doc_count = merged_docs.len() as i32;

    let numeric_dv = merge_numeric_doc_values(sources, &per_source_maps, &per_source_live_ids)?;
    let merged_norms = merge_norms(sources, &per_source_maps, &per_source_live_ids)?;
    let tv_docs = merge_term_vectors(sources, &per_source_maps, &per_source_live_ids)?;

    let mut files: Vec<String> = Vec::new();

    let (fdt, fdx, fdm) = stored_fields::write_best_speed(&merged_docs, &merged_segment_id, "");
    let fdt_name = format!("{merged_segment_name}.fdt");
    let fdx_name = format!("{merged_segment_name}.fdx");
    let fdm_name = format!("{merged_segment_name}.fdm");
    for (name, bytes) in [(&fdt_name, &fdt), (&fdx_name, &fdx), (&fdm_name, &fdm)] {
        write_file(dir, name, bytes)?;
        files.push(name.clone());
    }

    let fnm_name = format!("{merged_segment_name}.fnm");
    let fnm = field_infos::write(&merged_fields, &merged_segment_id, "");
    write_file(dir, &fnm_name, &fnm)?;
    files.push(fnm_name);

    if let Some((field_number, values)) = numeric_dv {
        let (dvm, dvd, dvs) = doc_values::write_single_dense_numeric_field(
            field_number,
            &values,
            doc_count,
            &merged_segment_id,
            "",
        )?;
        for (ext, bytes) in [("dvm", &dvm), ("dvd", &dvd), ("dvs", &dvs)] {
            let name = format!("{merged_segment_name}.{ext}");
            write_file(dir, &name, bytes)?;
            files.push(name);
        }
    }

    if let Some((field_number, values)) = merged_norms {
        let (nvm, nvd) = norms::write_single_dense_field(
            field_number,
            &values,
            doc_count,
            &merged_segment_id,
            "",
        )?;
        for (ext, bytes) in [("nvm", &nvm), ("nvd", &nvd)] {
            let name = format!("{merged_segment_name}.{ext}");
            write_file(dir, &name, bytes)?;
            files.push(name);
        }
    }

    if let Some(tv_docs) = tv_docs {
        let (tvd, tvx, tvm) = term_vectors::write_best_speed(&tv_docs, &merged_segment_id, "");
        for (ext, bytes) in [("tvd", &tvd), ("tvx", &tvx), ("tvm", &tvm)] {
            let name = format!("{merged_segment_name}.{ext}");
            write_file(dir, &name, bytes)?;
            files.push(name);
        }
    }

    let si = SegmentInfo {
        id: merged_segment_id,
        version: lucene_version,
        min_version: Some(lucene_version),
        doc_count,
        is_compound_file: false,
        has_blocks: false,
        diagnostics: vec![
            ("source".to_string(), "merge".to_string()),
            (
                "lucene.version".to_string(),
                format!(
                    "{}.{}.{}",
                    lucene_version.major, lucene_version.minor, lucene_version.bugfix
                ),
            ),
        ],
        files: files.clone(),
        attributes: vec![(
            "Lucene90StoredFieldsFormat.mode".to_string(),
            "BEST_SPEED".to_string(),
        )],
        // Merges never re-sort by an index-sort key in this port (a real,
        // documented gap: see docs/parity.md and PLAN.md's index-sort entry)
        // -- a merged segment is never sort-order-preserving, so it must not
        // claim an index sort in its `.si` regardless of what the input
        // segments declared.
        index_sort: None,
    };
    let si_name = format!("{merged_segment_name}.si");
    let si_bytes = segment_info::write(&si, "");
    write_file(dir, &si_name, &si_bytes)?;
    files.push(si_name);

    dir.sync(&files)?;

    Ok(SegmentCommitInfo {
        segment_name: merged_segment_name.to_string(),
        segment_id: merged_segment_id,
        codec_name: codec_name.to_string(),
        del_gen: -1,
        del_count: 0,
        field_infos_gen: -1,
        doc_values_gen: -1,
        soft_del_count: 0,
        sci_id: None,
        field_infos_files: vec![],
        dv_update_files: vec![],
    })
}

/// A single field's priority tier for [`merge_sorted_stored_only_segments`]'s
/// k-way merge -- the cross-source analogue of
/// [`crate::segment_writer::SortKeySpec`]: `per_source_keys[i][d]` is source
/// `i`'s original (pre-merge) doc `d`'s value for this field, or `None` if
/// that doc has no value. `per_source_keys` must have exactly one entry per
/// source, and each source's slice must have exactly one entry per doc in
/// that source (`source.reader.max_doc()` entries).
pub struct MergeSortKeySpec<'a> {
    pub field: &'a str,
    pub reverse: bool,
    pub missing: SortMissingValue,
    pub per_source_keys: &'a [&'a [Option<i64>]],
}

/// Merges `sources` into one brand-new stored-fields-only segment whose
/// documents are produced in **global** sort order across all sources -- a
/// genuine k-way merge by sort key (at each step, take whichever source's
/// current head doc has the smallest key, in `sort_fields` priority order),
/// not a concatenation of source 0's docs, then source 1's, etc. the way
/// [`merge_stored_only_segments`] works. This is the real behavior of
/// merging index-sorted segments in Lucene: since every source is already
/// internally sorted by the same key, the merged segment can be produced by
/// a single forward pass over all sources at once.
///
/// # Precondition (caller-guaranteed, not re-checked here)
///
/// Real Lucene requires every segment being merged to share the exact same
/// index sort -- merging segments with different (or absent-vs-present)
/// index sorts is a hard error in `SegmentInfos`/`IndexWriter`, not something
/// this port tries to detect or repair. This function takes that as a
/// precondition: `sort_fields` is the *one* shared sort every source is
/// already ordered by (each source's own doc 0, 1, 2, ... must already be
/// non-decreasing by this exact key -- true for any segment written by
/// [`crate::segment_writer::flush_sorted_stored_only_segment`] or produced
/// by a previous call to this same function). It is the caller's job to
/// have verified this, e.g. by comparing each source's own
/// `SegmentInfo.index_sort` against `sort_fields` for equality before
/// calling; this function does not re-verify it or attempt to detect an
/// out-of-order source (`crate::merge` deliberately has no dependency on
/// walking a whole `SegmentInfo` to do that check, keeping this function
/// usable from a plain in-memory source list, same as
/// [`merge_stored_only_segments`]). Passing sources that are not actually
/// sorted by `sort_fields` silently produces a merged segment that is not
/// sorted either -- garbage in, garbage out, exactly like the analogous
/// precondition on `flush_sorted_stored_only_segment`'s caller-supplied
/// `SortKeySpec::keys`.
///
/// # Scope
///
/// Only stored fields are reordered by sort key here -- this port's write
/// side has no path that reorders doc-values/norms/term-vectors data during
/// a merge (see this module's top doc comment). Any doc-values/norms/term-
/// vectors data attached to a `MergeSource` is silently ignored by this
/// function; use [`merge_stored_only_segments`] instead if that data needs
/// to be merged (concatenation order, no re-sort -- and no index-sort
/// metadata in the resulting `.si`, since that merge doesn't preserve sort
/// order).
pub fn merge_sorted_stored_only_segments(
    dir: &dyn Directory,
    sources: &[MergeSource],
    sort_fields: &[MergeSortKeySpec<'_>],
    merged_segment_name: &str,
    merged_segment_id: [u8; ID_LENGTH],
    codec_name: &str,
    lucene_version: LuceneVersion,
) -> Result<SegmentCommitInfo> {
    assert!(
        !sort_fields.is_empty(),
        "sort_fields must contain at least one sort key"
    );
    for spec in sort_fields {
        assert_eq!(
            spec.per_source_keys.len(),
            sources.len(),
            "per_source_keys must have exactly one entry per source for field {:?}",
            spec.field
        );
        for (source, keys) in sources.iter().zip(spec.per_source_keys.iter()) {
            assert_eq!(
                keys.len(),
                source.reader.max_doc() as usize,
                "per_source_keys must have exactly one entry per doc in that source for field {:?}",
                spec.field
            );
        }
    }

    let sources_fields: Vec<&[FieldInfo]> = sources.iter().map(|s| s.field_infos).collect();
    let (merged_fields, per_source_maps) = reconcile_field_numbers(&sources_fields);

    // Per-source live (pre-merge) doc ids, ascending -- unlike
    // merge_stored_only_segments this is NOT concatenated: a k-way merge
    // walks each source's list via its own cursor, always advancing
    // whichever source currently has the globally-smallest head key.
    let mut per_source_live_ids: Vec<Vec<i32>> = Vec::with_capacity(sources.len());
    for source in sources {
        let max_doc = source.reader.max_doc();
        let mut live_ids = Vec::new();
        for doc_id in 0..max_doc {
            let is_live = source
                .live_docs
                .map(|bits| bits.get(doc_id as usize))
                .unwrap_or(true);
            if is_live {
                live_ids.push(doc_id);
            }
        }
        per_source_live_ids.push(live_ids);
    }

    let mut cursors = vec![0usize; sources.len()];
    let mut merged_docs: Vec<Document> = Vec::new();
    loop {
        // Find the source whose current head doc has the smallest sort key,
        // in `sort_fields` priority order -- a linear scan across sources
        // per step (this port's scale has typically few concurrent merge
        // sources, so a min-heap would be unneeded complexity here; see the
        // module-level task note this function was built from).
        let mut best: Option<usize> = None;
        for (src_idx, live_ids) in per_source_live_ids.iter().enumerate() {
            let cursor = cursors[src_idx];
            if cursor >= live_ids.len() {
                continue;
            }
            best = Some(match best {
                None => src_idx,
                Some(current_best) => {
                    let ord = compare_heads(
                        sort_fields,
                        current_best,
                        per_source_live_ids[current_best][cursors[current_best]],
                        src_idx,
                        live_ids[cursor],
                    );
                    if ord == std::cmp::Ordering::Greater {
                        src_idx
                    } else {
                        current_best
                    }
                }
            });
        }
        let Some(src_idx) = best else {
            break;
        };
        let doc_id = per_source_live_ids[src_idx][cursors[src_idx]];
        cursors[src_idx] += 1;

        let mut doc = sources[src_idx].reader.document(doc_id)?;
        let field_number_map = &per_source_maps[src_idx];
        for field in &mut doc.fields {
            field.field_number = *field_number_map.get(&field.field_number).ok_or(
                Error::UnknownSourceFieldNumber {
                    field_number: field.field_number,
                },
            )?;
        }
        merged_docs.push(doc);
    }
    let doc_count = merged_docs.len() as i32;

    let mut files: Vec<String> = Vec::new();

    let (fdt, fdx, fdm) = stored_fields::write_best_speed(&merged_docs, &merged_segment_id, "");
    let fdt_name = format!("{merged_segment_name}.fdt");
    let fdx_name = format!("{merged_segment_name}.fdx");
    let fdm_name = format!("{merged_segment_name}.fdm");
    for (name, bytes) in [(&fdt_name, &fdt), (&fdx_name, &fdx), (&fdm_name, &fdm)] {
        write_file(dir, name, bytes)?;
        files.push(name.clone());
    }

    let fnm_name = format!("{merged_segment_name}.fnm");
    let fnm = field_infos::write(&merged_fields, &merged_segment_id, "");
    write_file(dir, &fnm_name, &fnm)?;
    files.push(fnm_name);

    let si = SegmentInfo {
        id: merged_segment_id,
        version: lucene_version,
        min_version: Some(lucene_version),
        doc_count,
        is_compound_file: false,
        has_blocks: false,
        diagnostics: vec![
            ("source".to_string(), "merge".to_string()),
            (
                "lucene.version".to_string(),
                format!(
                    "{}.{}.{}",
                    lucene_version.major, lucene_version.minor, lucene_version.bugfix
                ),
            ),
        ],
        files: files.clone(),
        attributes: vec![(
            "Lucene90StoredFieldsFormat.mode".to_string(),
            "BEST_SPEED".to_string(),
        )],
        // Unlike merge_stored_only_segments, this merge genuinely preserves
        // (global) sort order across sources, so -- matching real Lucene --
        // the merged segment correctly keeps claiming the same index sort
        // its inputs had, rather than being forced to `None`.
        index_sort: Some(
            sort_fields
                .iter()
                .map(|spec| IndexSortField {
                    field: spec.field.to_string(),
                    reverse: spec.reverse,
                    missing: spec.missing,
                })
                .collect(),
        ),
    };
    let si_name = format!("{merged_segment_name}.si");
    let si_bytes = segment_info::write(&si, "");
    write_file(dir, &si_name, &si_bytes)?;
    files.push(si_name);

    dir.sync(&files)?;

    Ok(SegmentCommitInfo {
        segment_name: merged_segment_name.to_string(),
        segment_id: merged_segment_id,
        codec_name: codec_name.to_string(),
        del_gen: -1,
        del_count: 0,
        field_infos_gen: -1,
        doc_values_gen: -1,
        soft_del_count: 0,
        sci_id: None,
        field_infos_files: vec![],
        dv_update_files: vec![],
    })
}

/// Multi-tier comparator for the k-way merge: folds `sort_fields` in
/// priority order using [`crate::segment_writer::sort_key_rank`] (the exact
/// same per-tier comparator [`crate::segment_writer::flush_sorted_stored_only_segment`]
/// uses within one batch -- reused here, not reimplemented), then breaks any
/// remaining tie first by source index and finally by original doc id,
/// giving a fully deterministic total order.
fn compare_heads(
    sort_fields: &[MergeSortKeySpec<'_>],
    src_a: usize,
    doc_a: i32,
    src_b: usize,
    doc_b: i32,
) -> std::cmp::Ordering {
    sort_fields
        .iter()
        .fold(std::cmp::Ordering::Equal, |acc, spec| {
            acc.then_with(|| {
                let key_a = spec.per_source_keys[src_a][doc_a as usize];
                let key_b = spec.per_source_keys[src_b][doc_b as usize];
                crate::segment_writer::sort_key_rank(key_a, key_b, spec.reverse, spec.missing)
            })
        })
        .then_with(|| src_a.cmp(&src_b))
        .then_with(|| doc_a.cmp(&doc_b))
}

fn write_file(dir: &dyn Directory, name: &str, bytes: &[u8]) -> Result<()> {
    let mut out = dir.create_output(name)?;
    out.write_bytes(bytes);
    out.close()?;
    Ok(())
}

/// Merges numeric doc-values data across `sources` into one `(merged_field_
/// number, per_doc_values)` pair, contiguous in the same doc order
/// `merged_docs` was built in -- or `Ok(None)` if no source has any numeric
/// doc-values data at all. See this module's doc comment for the
/// single-field limit and the "sparse across sources" rule this enforces.
fn merge_numeric_doc_values(
    sources: &[MergeSource],
    per_source_maps: &[HashMap<i32, i32>],
    per_source_live_ids: &[Vec<i32>],
) -> Result<Option<(i32, Vec<i64>)>> {
    let mut candidates: Vec<i32> = Vec::new();
    for ((source, map), live_ids) in sources.iter().zip(per_source_maps).zip(per_source_live_ids) {
        if live_ids.is_empty() {
            // A fully-deleted source contributes no docs, so whatever
            // doc-values fields it happens to carry can't affect the merged
            // output -- skip it, consistent with the same exemption applied
            // when checking for a field missing from a source below.
            continue;
        }
        for nf in source.numeric_doc_values {
            if let Some(&merged_number) = map.get(&nf.entry.field_number) {
                if !candidates.contains(&merged_number) {
                    candidates.push(merged_number);
                }
            }
        }
    }
    if candidates.len() > 1 {
        return Err(Error::TooManyNumericDocValuesFields(candidates));
    }
    let Some(merged_field_number) = candidates.into_iter().next() else {
        return Ok(None);
    };

    let mut values: Vec<i64> = Vec::new();
    for ((source, map), live_ids) in sources.iter().zip(per_source_maps).zip(per_source_live_ids) {
        if live_ids.is_empty() {
            continue;
        }
        let original_number = map
            .iter()
            .find(|&(_, &merged)| merged == merged_field_number)
            .map(|(&orig, _)| orig);
        let Some(original_number) = original_number else {
            return Err(Error::DocValuesFieldMissingInSource {
                merged_field_number,
            });
        };
        let Some(entry) = source
            .numeric_doc_values
            .iter()
            .find(|nf| nf.entry.field_number == original_number)
        else {
            return Err(Error::DocValuesFieldMissingInSource {
                merged_field_number,
            });
        };
        for &doc_id in live_ids {
            let value = doc_values::numeric_value(entry.data, &entry.entry, doc_id)?.ok_or(
                Error::DocValuesFieldMissingInSource {
                    merged_field_number,
                },
            )?;
            values.push(value);
        }
    }
    Ok(Some((merged_field_number, values)))
}

/// Same shape and same rules as [`merge_numeric_doc_values`], for norms.
fn merge_norms(
    sources: &[MergeSource],
    per_source_maps: &[HashMap<i32, i32>],
    per_source_live_ids: &[Vec<i32>],
) -> Result<Option<(i32, Vec<i64>)>> {
    let mut candidates: Vec<i32> = Vec::new();
    for ((source, map), live_ids) in sources.iter().zip(per_source_maps).zip(per_source_live_ids) {
        if live_ids.is_empty() {
            // Same "fully-deleted source can't affect the merged output"
            // exemption as merge_numeric_doc_values.
            continue;
        }
        for nf in source.norms {
            if let Some(&merged_number) = map.get(&nf.entry.field_number) {
                if !candidates.contains(&merged_number) {
                    candidates.push(merged_number);
                }
            }
        }
    }
    if candidates.len() > 1 {
        return Err(Error::TooManyNormsFields(candidates));
    }
    let Some(merged_field_number) = candidates.into_iter().next() else {
        return Ok(None);
    };

    let mut values: Vec<i64> = Vec::new();
    for ((source, map), live_ids) in sources.iter().zip(per_source_maps).zip(per_source_live_ids) {
        if live_ids.is_empty() {
            continue;
        }
        let original_number = map
            .iter()
            .find(|&(_, &merged)| merged == merged_field_number)
            .map(|(&orig, _)| orig);
        let Some(original_number) = original_number else {
            return Err(Error::NormsFieldMissingInSource {
                merged_field_number,
            });
        };
        let Some(entry) = source
            .norms
            .iter()
            .find(|nf| nf.entry.field_number == original_number)
        else {
            return Err(Error::NormsFieldMissingInSource {
                merged_field_number,
            });
        };
        for &doc_id in live_ids {
            let value = norms::norm_value(entry.data, &entry.entry, doc_id)?.ok_or(
                Error::NormsFieldMissingInSource {
                    merged_field_number,
                },
            )?;
            values.push(value);
        }
    }
    Ok(Some((merged_field_number, values)))
}

/// Merges term-vectors data across `sources`, contiguous in the same doc
/// order `merged_docs` was built in, remapping every merged doc's field
/// numbers -- or `Ok(None)` if no source has a term-vectors reader at all
/// (distinguishing "nobody supplied term vectors" from "every doc has an
/// empty term-vectors document" isn't needed by `write_best_speed`, but
/// `None` lets a caller skip writing `.tvd`/`.tvx`/`.tvm` entirely when
/// nothing in the merge has term vectors).
fn merge_term_vectors(
    sources: &[MergeSource],
    per_source_maps: &[HashMap<i32, i32>],
    per_source_live_ids: &[Vec<i32>],
) -> Result<Option<Vec<TermVectorsDocument>>> {
    if sources.iter().all(|s| s.term_vectors.is_none()) {
        return Ok(None);
    }

    let mut merged_docs: Vec<TermVectorsDocument> = Vec::new();
    for ((source, map), live_ids) in sources.iter().zip(per_source_maps).zip(per_source_live_ids) {
        for &doc_id in live_ids {
            let mut doc = match source.term_vectors {
                Some(reader) => reader.document(doc_id)?.unwrap_or_default(),
                None => TermVectorsDocument::default(),
            };
            for field in &mut doc.fields {
                field.field_number =
                    *map.get(&field.field_number)
                        .ok_or(Error::UnknownSourceFieldNumber {
                            field_number: field.field_number,
                        })?;
            }
            merged_docs.push(doc);
        }
    }
    Ok(Some(merged_docs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segment_writer;
    use lucene_codecs::field_infos::{
        DocValuesSkipIndexType, DocValuesType, IndexOptions, VectorEncoding,
        VectorSimilarityFunction,
    };
    use lucene_codecs::stored_fields::{self, FieldValue, StoredField};
    use lucene_codecs::term_vectors::{TermVectorField, TermVectorTerm};
    use lucene_store::directory::FsDirectory;

    fn version() -> LuceneVersion {
        LuceneVersion {
            major: 10,
            minor: 0,
            bugfix: 0,
        }
    }

    fn field(name: &str, number: i32) -> FieldInfo {
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

    fn doc_with(field_number: i32, value: &str) -> Document {
        Document {
            fields: vec![StoredField {
                field_number,
                value: FieldValue::String(value.to_string()),
            }],
        }
    }

    // --- reconcile_field_numbers ---

    #[test]
    fn single_source_keeps_its_own_numbering_shape() {
        let fields = vec![field("id", 0), field("body", 1)];
        let sources: Vec<&[FieldInfo]> = vec![&fields];
        let (merged, maps) = reconcile_field_numbers(&sources);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].name, "id");
        assert_eq!(merged[0].number, 0);
        assert_eq!(merged[1].name, "body");
        assert_eq!(merged[1].number, 1);
        assert_eq!(maps[0].get(&0), Some(&0));
        assert_eq!(maps[0].get(&1), Some(&1));
    }

    #[test]
    fn same_name_different_numbers_across_sources_unify() {
        // Source 0 has "id"=0, "body"=1; source 1 has "body"=0, "id"=1 --
        // opposite numbering for the exact same two field names.
        let fields0 = vec![field("id", 0), field("body", 1)];
        let fields1 = vec![field("body", 0), field("id", 1)];
        let sources: Vec<&[FieldInfo]> = vec![&fields0, &fields1];
        let (merged, maps) = reconcile_field_numbers(&sources);

        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].name, "id");
        assert_eq!(merged[1].name, "body");

        // Source 0: id(0)->0, body(1)->1
        assert_eq!(maps[0][&0], 0);
        assert_eq!(maps[0][&1], 1);
        // Source 1: body(0)->1, id(1)->0
        assert_eq!(maps[1][&0], 1);
        assert_eq!(maps[1][&1], 0);
    }

    #[test]
    fn field_present_in_only_some_sources_gets_its_own_merged_number() {
        let fields0 = vec![field("id", 0)];
        let fields1 = vec![field("id", 0), field("extra", 1)];
        let sources: Vec<&[FieldInfo]> = vec![&fields0, &fields1];
        let (merged, maps) = reconcile_field_numbers(&sources);

        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].name, "id");
        assert_eq!(merged[1].name, "extra");
        assert_eq!(maps[0].len(), 1);
        assert_eq!(maps[1][&1], 1);
    }

    #[test]
    fn empty_sources_produce_empty_merged_fields() {
        let sources: Vec<&[FieldInfo]> = vec![];
        let (merged, maps) = reconcile_field_numbers(&sources);
        assert!(merged.is_empty());
        assert!(maps.is_empty());
    }

    #[test]
    fn merged_field_keeps_first_sources_metadata() {
        let mut fields0 = vec![field("id", 0)];
        fields0[0].doc_values_gen = 99;
        let fields1 = vec![field("id", 5)];
        let sources: Vec<&[FieldInfo]> = vec![&fields0, &fields1];
        let (merged, _maps) = reconcile_field_numbers(&sources);
        assert_eq!(merged[0].doc_values_gen, 99);
    }

    // --- merge_stored_only_segments (full round-trip via real Directory I/O) ---

    fn tempdir() -> String {
        let dir = std::env::temp_dir().join(format!(
            "lucene-rust-merge-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir.to_str().unwrap().to_string()
    }

    /// Flushes one stored-fields-only segment (via `flush_stored_only_segment`)
    /// and returns its bytes plus field list, ready to be re-opened as a merge
    /// source -- mirrors how a real caller would read a segment's files off
    /// disk before merging it.
    struct FlushedSegment {
        fdt: Vec<u8>,
        fdx: Vec<u8>,
        fdm: Vec<u8>,
        fields: Vec<FieldInfo>,
        segment_id: [u8; ID_LENGTH],
    }

    fn flush(
        dir: &FsDirectory,
        tmp: &str,
        name: &str,
        segment_id: [u8; ID_LENGTH],
        fields: &[FieldInfo],
        docs: &[Document],
    ) -> FlushedSegment {
        segment_writer::flush_stored_only_segment(
            dir,
            name,
            segment_id,
            "Lucene104",
            version(),
            fields,
            docs,
            false,
        )
        .unwrap();
        FlushedSegment {
            fdt: std::fs::read(std::path::Path::new(tmp).join(format!("{name}.fdt"))).unwrap(),
            fdx: std::fs::read(std::path::Path::new(tmp).join(format!("{name}.fdx"))).unwrap(),
            fdm: std::fs::read(std::path::Path::new(tmp).join(format!("{name}.fdm"))).unwrap(),
            fields: fields.to_vec(),
            segment_id,
        }
    }

    fn open_reader(seg: &FlushedSegment) -> stored_fields::StoredFieldsReader<'_> {
        stored_fields::open(&seg.fdt, &seg.fdx, &seg.fdm, &seg.segment_id, "").unwrap()
    }

    #[test]
    fn two_segments_no_deletions_merge_with_contiguous_doc_ids() {
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let fields = vec![field("id", 0)];

        let seg0 = flush(
            &dir,
            &tmp,
            "_0",
            [1u8; ID_LENGTH],
            &fields,
            &[doc_with(0, "a"), doc_with(0, "b")],
        );
        let seg1 = flush(
            &dir,
            &tmp,
            "_1",
            [2u8; ID_LENGTH],
            &fields,
            &[doc_with(0, "c")],
        );

        let reader0 = open_reader(&seg0);
        let reader1 = open_reader(&seg1);
        let sources = vec![
            MergeSource::stored_only(&seg0.fields, &reader0, None),
            MergeSource::stored_only(&seg1.fields, &reader1, None),
        ];

        let sci = merge_stored_only_segments(
            &dir,
            &sources,
            "_merged",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();
        assert_eq!(sci.segment_name, "_merged");

        let merged_fdt = std::fs::read(std::path::Path::new(&tmp).join("_merged.fdt")).unwrap();
        let merged_fdx = std::fs::read(std::path::Path::new(&tmp).join("_merged.fdx")).unwrap();
        let merged_fdm = std::fs::read(std::path::Path::new(&tmp).join("_merged.fdm")).unwrap();
        let merged_reader =
            stored_fields::open(&merged_fdt, &merged_fdx, &merged_fdm, &[9u8; ID_LENGTH], "")
                .unwrap();
        assert_eq!(merged_reader.max_doc(), 3);
        let vals: Vec<String> = (0..3)
            .map(
                |i| match &merged_reader.document(i).unwrap().fields[0].value {
                    FieldValue::String(s) => s.clone(),
                    _ => unreachable!(),
                },
            )
            .collect();
        assert_eq!(vals, vec!["a", "b", "c"]);
    }

    #[test]
    fn some_docs_deleted_in_each_source_are_dropped() {
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let fields = vec![field("id", 0)];

        let seg0 = flush(
            &dir,
            &tmp,
            "_0",
            [1u8; ID_LENGTH],
            &fields,
            &[doc_with(0, "a"), doc_with(0, "b")],
        );
        let seg1 = flush(
            &dir,
            &tmp,
            "_1",
            [2u8; ID_LENGTH],
            &fields,
            &[doc_with(0, "c"), doc_with(0, "d")],
        );

        let reader0 = open_reader(&seg0);
        let reader1 = open_reader(&seg1);

        let mut live0 = FixedBitSet::new(2);
        live0.set(0); // keep "a", drop "b"
        let mut live1 = FixedBitSet::new(2);
        live1.set(1); // drop "c", keep "d"

        let sources = vec![
            MergeSource::stored_only(&seg0.fields, &reader0, Some(&live0)),
            MergeSource::stored_only(&seg1.fields, &reader1, Some(&live1)),
        ];

        let dir2 = FsDirectory::open(&tmp);
        merge_stored_only_segments(
            &dir2,
            &sources,
            "_merged2",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();

        let merged_fdt = std::fs::read(std::path::Path::new(&tmp).join("_merged2.fdt")).unwrap();
        let merged_fdx = std::fs::read(std::path::Path::new(&tmp).join("_merged2.fdx")).unwrap();
        let merged_fdm = std::fs::read(std::path::Path::new(&tmp).join("_merged2.fdm")).unwrap();
        let merged_reader =
            stored_fields::open(&merged_fdt, &merged_fdx, &merged_fdm, &[9u8; ID_LENGTH], "")
                .unwrap();
        assert_eq!(merged_reader.max_doc(), 2);
        let vals: Vec<String> = (0..2)
            .map(
                |i| match &merged_reader.document(i).unwrap().fields[0].value {
                    FieldValue::String(s) => s.clone(),
                    _ => unreachable!(),
                },
            )
            .collect();
        assert_eq!(vals, vec!["a", "d"]);
    }

    #[test]
    fn fully_deleted_source_contributes_zero_docs() {
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let fields = vec![field("id", 0)];

        let seg0 = flush(
            &dir,
            &tmp,
            "_0",
            [1u8; ID_LENGTH],
            &fields,
            &[doc_with(0, "a")],
        );
        let seg1 = flush(
            &dir,
            &tmp,
            "_1",
            [2u8; ID_LENGTH],
            &fields,
            &[doc_with(0, "b"), doc_with(0, "c")],
        );

        let reader0 = open_reader(&seg0);
        let reader1 = open_reader(&seg1);
        let live1 = FixedBitSet::new(2); // all deleted, nothing set

        let sources = vec![
            MergeSource::stored_only(&seg0.fields, &reader0, None),
            MergeSource::stored_only(&seg1.fields, &reader1, Some(&live1)),
        ];

        merge_stored_only_segments(
            &dir,
            &sources,
            "_merged3",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();

        let merged_fdt = std::fs::read(std::path::Path::new(&tmp).join("_merged3.fdt")).unwrap();
        let merged_fdx = std::fs::read(std::path::Path::new(&tmp).join("_merged3.fdx")).unwrap();
        let merged_fdm = std::fs::read(std::path::Path::new(&tmp).join("_merged3.fdm")).unwrap();
        let merged_reader =
            stored_fields::open(&merged_fdt, &merged_fdx, &merged_fdm, &[9u8; ID_LENGTH], "")
                .unwrap();
        assert_eq!(merged_reader.max_doc(), 1);
        match &merged_reader.document(0).unwrap().fields[0].value {
            FieldValue::String(s) => assert_eq!(s, "a"),
            _ => unreachable!(),
        }
    }

    #[test]
    fn field_number_mismatch_across_sources_is_reconciled_during_merge() {
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        // Source 0: "id"=0, "body"=1. Source 1: "body"=0, "id"=1 (opposite).
        let fields0 = vec![field("id", 0), field("body", 1)];
        let fields1 = vec![field("body", 0), field("id", 1)];

        let doc0 = Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("id0".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::String("body0".to_string()),
                },
            ],
        };
        let doc1 = Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("body1".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::String("id1".to_string()),
                },
            ],
        };

        let seg0 = flush(&dir, &tmp, "_0", [1u8; ID_LENGTH], &fields0, &[doc0]);
        let seg1 = flush(&dir, &tmp, "_1", [2u8; ID_LENGTH], &fields1, &[doc1]);

        let reader0 = open_reader(&seg0);
        let reader1 = open_reader(&seg1);
        let sources = vec![
            MergeSource::stored_only(&seg0.fields, &reader0, None),
            MergeSource::stored_only(&seg1.fields, &reader1, None),
        ];

        merge_stored_only_segments(
            &dir,
            &sources,
            "_merged4",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();

        let merged_fdt = std::fs::read(std::path::Path::new(&tmp).join("_merged4.fdt")).unwrap();
        let merged_fdx = std::fs::read(std::path::Path::new(&tmp).join("_merged4.fdx")).unwrap();
        let merged_fdm = std::fs::read(std::path::Path::new(&tmp).join("_merged4.fdm")).unwrap();
        let merged_fnm = std::fs::read(std::path::Path::new(&tmp).join("_merged4.fnm")).unwrap();
        let merged_fields =
            lucene_codecs::field_infos::parse(&merged_fnm, &[9u8; ID_LENGTH], "").unwrap();
        let id_number = merged_fields
            .fields
            .iter()
            .find(|f| f.name == "id")
            .unwrap()
            .number;
        let body_number = merged_fields
            .fields
            .iter()
            .find(|f| f.name == "body")
            .unwrap()
            .number;

        let merged_reader =
            stored_fields::open(&merged_fdt, &merged_fdx, &merged_fdm, &[9u8; ID_LENGTH], "")
                .unwrap();
        assert_eq!(merged_reader.max_doc(), 2);

        let doc0 = merged_reader.document(0).unwrap();
        let id0 = doc0
            .fields
            .iter()
            .find(|f| f.field_number == id_number)
            .unwrap();
        assert_eq!(id0.value, FieldValue::String("id0".to_string()));
        let body0 = doc0
            .fields
            .iter()
            .find(|f| f.field_number == body_number)
            .unwrap();
        assert_eq!(body0.value, FieldValue::String("body0".to_string()));

        let doc1 = merged_reader.document(1).unwrap();
        let id1 = doc1
            .fields
            .iter()
            .find(|f| f.field_number == id_number)
            .unwrap();
        assert_eq!(id1.value, FieldValue::String("id1".to_string()));
        let body1 = doc1
            .fields
            .iter()
            .find(|f| f.field_number == body_number)
            .unwrap();
        assert_eq!(body1.value, FieldValue::String("body1".to_string()));
    }

    #[test]
    fn no_sources_produces_an_empty_segment() {
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let sources: Vec<MergeSource> = vec![];
        let sci = merge_stored_only_segments(
            &dir,
            &sources,
            "_empty",
            [3u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();
        assert_eq!(sci.segment_name, "_empty");

        // Re-open the actually-written files rather than trusting the
        // returned `SegmentCommitInfo` alone -- confirms a zero-doc merge
        // produces a genuinely well-formed, re-openable segment (max_doc ==
        // 0, no documents iterable), not just a struct that claims success.
        let fdt = std::fs::read(std::path::Path::new(&tmp).join("_empty.fdt")).unwrap();
        let fdx = std::fs::read(std::path::Path::new(&tmp).join("_empty.fdx")).unwrap();
        let fdm = std::fs::read(std::path::Path::new(&tmp).join("_empty.fdm")).unwrap();
        let reader = stored_fields::open(&fdt, &fdx, &fdm, &sci.segment_id, "").unwrap();
        assert_eq!(reader.max_doc(), 0);
    }

    #[test]
    fn stored_field_number_absent_from_its_own_source_field_infos_is_an_error() {
        // A malformed `MergeSource`: its stored fields reference field number
        // 7, but its own `field_infos` never declares that number. Real
        // callers can't construct this from `flush_stored_only_segment` +
        // `field_infos::parse`, but merge_stored_only_segments should still
        // surface it as an `Err`, not panic, per this port's stance of never
        // trusting a caller-supplied invariant with an `unwrap`/`expect` when
        // an `Err` is easy to return instead.
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let fields = vec![field("id", 0)];
        let docs = vec![doc_with(7, "orphaned")];
        let seg = flush(&dir, &tmp, "_0", [1u8; ID_LENGTH], &fields, &docs);
        let reader = open_reader(&seg);

        let sources = vec![MergeSource::stored_only(&seg.fields, &reader, None)];
        let result = merge_stored_only_segments(
            &dir,
            &sources,
            "_merged",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        );
        assert!(matches!(
            result,
            Err(Error::UnknownSourceFieldNumber { field_number: 7 })
        ));
    }

    #[test]
    fn full_round_trip_through_a_real_written_and_reparsed_liv_file() {
        // End-to-end: flush 2 segments, write a real `.liv` for one of them
        // via `lucene_codecs::live_docs::write`, read it back via `parse`
        // (not just constructed in memory), merge, then confirm the merged
        // segment's stored fields match exactly the surviving docs.
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let fields = vec![field("id", 0)];

        let seg0 = flush(
            &dir,
            &tmp,
            "_0",
            [1u8; ID_LENGTH],
            &fields,
            &[doc_with(0, "a"), doc_with(0, "b"), doc_with(0, "c")],
        );
        let seg1 = flush(
            &dir,
            &tmp,
            "_1",
            [2u8; ID_LENGTH],
            &fields,
            &[doc_with(0, "d")],
        );

        // Delete doc 1 ("b") from _0 via a real .liv file, round-tripped
        // through the actual write/parse pair.
        let mut live0 = FixedBitSet::new(3);
        live0.set(0);
        live0.set(2);
        let liv_bytes = lucene_codecs::live_docs::write(&live0, &seg0.segment_id, 1, 1).unwrap();
        let parsed_live0 =
            lucene_codecs::live_docs::parse(&liv_bytes, &seg0.segment_id, 1, 3, 1).unwrap();

        let reader0 = open_reader(&seg0);
        let reader1 = open_reader(&seg1);
        let sources = vec![
            MergeSource::stored_only(&seg0.fields, &reader0, Some(&parsed_live0)),
            MergeSource::stored_only(&seg1.fields, &reader1, None),
        ];

        merge_stored_only_segments(
            &dir,
            &sources,
            "_merged5",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();

        let merged_fdt = std::fs::read(std::path::Path::new(&tmp).join("_merged5.fdt")).unwrap();
        let merged_fdx = std::fs::read(std::path::Path::new(&tmp).join("_merged5.fdx")).unwrap();
        let merged_fdm = std::fs::read(std::path::Path::new(&tmp).join("_merged5.fdm")).unwrap();
        let merged_reader =
            stored_fields::open(&merged_fdt, &merged_fdx, &merged_fdm, &[9u8; ID_LENGTH], "")
                .unwrap();
        assert_eq!(merged_reader.max_doc(), 3);
        let vals: Vec<String> = (0..3)
            .map(
                |i| match &merged_reader.document(i).unwrap().fields[0].value {
                    FieldValue::String(s) => s.clone(),
                    _ => unreachable!(),
                },
            )
            .collect();
        assert_eq!(vals, vec!["a", "c", "d"]);
    }

    #[test]
    fn stored_fields_error_wraps_into_this_modules_error_type() {
        // Confirms `Error::StoredFields`'s `#[from]` wrapping actually
        // propagates a real `stored_fields::Error` (the kind
        // `reader.document()` can return mid-merge, e.g. a corrupted chunk)
        // as an `Err` through this module's own error type, rather than
        // requiring a full corrupt-fixture integration setup to exercise the
        // conversion.
        let source_err = stored_fields::Error::DocOutOfRange(5, 3);
        let wrapped: Error = source_err.into();
        assert!(matches!(wrapped, Error::StoredFields(_)));
    }

    // --- doc values / norms / term vectors merging ---

    fn numeric_field(name: &str, number: i32) -> FieldInfo {
        let mut f = field(name, number);
        f.doc_values_type = DocValuesType::Numeric;
        f
    }

    fn norms_field(name: &str, number: i32) -> FieldInfo {
        let mut f = field(name, number);
        f.omit_norms = false;
        f
    }

    fn tv_field(name: &str, number: i32) -> FieldInfo {
        let mut f = field(name, number);
        f.store_term_vectors = true;
        f
    }

    /// A test-owned numeric doc-values field: writes it via the real
    /// write-side encoder, then re-parses the meta via the real read-side
    /// decoder to get a genuine [`NumericEntry`] -- exercises the same
    /// encode/decode round trip a real caller would, rather than hand
    /// building a `NumericEntry`.
    struct FlushedNumericDv {
        data: Vec<u8>,
        entry: NumericEntry,
    }

    fn flush_numeric_dv(
        field_number: i32,
        values: &[i64],
        segment_id: [u8; ID_LENGTH],
    ) -> FlushedNumericDv {
        let max_doc = values.len() as i32;
        let (meta, data, _skip) = doc_values::write_single_dense_numeric_field(
            field_number,
            values,
            max_doc,
            &segment_id,
            "",
        )
        .unwrap();
        let field_infos = field_infos::FieldInfos {
            fields: vec![numeric_field("x", field_number)],
        };
        let (_version, parsed) =
            doc_values::parse_meta(&meta, &segment_id, "", &field_infos).unwrap();
        let entry = parsed.numeric_entry(field_number).unwrap().clone();
        FlushedNumericDv { data, entry }
    }

    impl FlushedNumericDv {
        fn source(&self) -> SourceNumericDocValues<'_> {
            SourceNumericDocValues {
                data: &self.data,
                entry: self.entry.clone(),
            }
        }
    }

    struct FlushedNorms {
        data: Vec<u8>,
        entry: NormsEntry,
    }

    fn flush_norms(field_number: i32, values: &[i64], segment_id: [u8; ID_LENGTH]) -> FlushedNorms {
        let max_doc = values.len() as i32;
        let (meta, data) =
            norms::write_single_dense_field(field_number, values, max_doc, &segment_id, "")
                .unwrap();
        let (_version, parsed) = norms::parse_meta(&meta, &segment_id, "").unwrap();
        let entry = *parsed.entry(field_number).unwrap();
        FlushedNorms { data, entry }
    }

    impl FlushedNorms {
        fn source(&self) -> SourceNorms<'_> {
            SourceNorms {
                data: &self.data,
                entry: self.entry,
            }
        }
    }

    struct FlushedTermVectors {
        tvd: Vec<u8>,
        tvx: Vec<u8>,
        tvm: Vec<u8>,
        segment_id: [u8; ID_LENGTH],
    }

    fn flush_term_vectors(
        docs: &[TermVectorsDocument],
        segment_id: [u8; ID_LENGTH],
    ) -> FlushedTermVectors {
        let (tvd, tvx, tvm) = term_vectors::write_best_speed(docs, &segment_id, "");
        FlushedTermVectors {
            tvd,
            tvx,
            tvm,
            segment_id,
        }
    }

    impl FlushedTermVectors {
        fn reader(&self) -> TermVectorsReader<'_> {
            term_vectors::open(&self.tvd, &self.tvx, &self.tvm, &self.segment_id, "").unwrap()
        }
    }

    fn tv_doc(field_number: i32, terms: &[(&str, i32)]) -> TermVectorsDocument {
        TermVectorsDocument {
            fields: vec![TermVectorField {
                field_number,
                has_positions: true,
                has_offsets: false,
                has_payloads: false,
                terms: terms
                    .iter()
                    .map(|(t, pos)| TermVectorTerm {
                        term: t.as_bytes().to_vec(),
                        freq: 1,
                        positions: Some(vec![*pos]),
                        start_offsets: None,
                        end_offsets: None,
                        payloads: None,
                    })
                    .collect(),
            }],
        }
    }

    #[test]
    fn numeric_doc_values_merge_across_two_sources_with_deletions() {
        let seg0_id = [1u8; ID_LENGTH];
        let seg1_id = [2u8; ID_LENGTH];
        // Source 0: 2 docs, doc 1 deleted -> only doc "10" survives.
        let dv0 = flush_numeric_dv(0, &[10, 20], seg0_id);
        // Source 1: 1 doc, no deletions -> "30" survives.
        let dv1 = flush_numeric_dv(0, &[30], seg1_id);

        let fields = vec![numeric_field("num", 0)];
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let stored0 = flush(
            &dir,
            &tmp,
            "_0",
            seg0_id,
            &fields,
            &[doc_with(0, "a"), doc_with(0, "b")],
        );
        let stored1 = flush(&dir, &tmp, "_1", seg1_id, &fields, &[doc_with(0, "c")]);
        let reader0 = open_reader(&stored0);
        let reader1 = open_reader(&stored1);

        let mut live0 = FixedBitSet::new(2);
        live0.set(0); // keep doc 0 ("a"/10), drop doc 1 ("b"/20)

        let dv0_source = [dv0.source()];
        let dv1_source = [dv1.source()];
        let source0 = MergeSource {
            field_infos: &stored0.fields,
            reader: &reader0,
            live_docs: Some(&live0),
            numeric_doc_values: &dv0_source,
            norms: &[],
            term_vectors: None,
        };
        let source1 = MergeSource {
            field_infos: &stored1.fields,
            reader: &reader1,
            live_docs: None,
            numeric_doc_values: &dv1_source,
            norms: &[],
            term_vectors: None,
        };

        let sci = merge_stored_only_segments(
            &dir,
            &[source0, source1],
            "_merged_dv",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();
        assert_eq!(sci.segment_name, "_merged_dv");

        let dvd = std::fs::read(std::path::Path::new(&tmp).join("_merged_dv.dvd")).unwrap();
        let dvm = std::fs::read(std::path::Path::new(&tmp).join("_merged_dv.dvm")).unwrap();
        let merged_field_infos = field_infos::FieldInfos {
            fields: vec![numeric_field("num", 0)],
        };
        let (_v, meta) =
            doc_values::parse_meta(&dvm, &[9u8; ID_LENGTH], "", &merged_field_infos).unwrap();
        let entry = meta.numeric_entry(0).unwrap();
        let values: Vec<i64> = (0..2)
            .map(|d| doc_values::numeric_value(&dvd, entry, d).unwrap().unwrap())
            .collect();
        assert_eq!(values, vec![10, 30]);
    }

    #[test]
    fn numeric_doc_values_missing_in_a_live_contributing_source_is_an_error() {
        let seg0_id = [1u8; ID_LENGTH];
        let dv0 = flush_numeric_dv(0, &[10], seg0_id);
        let fields = vec![numeric_field("num", 0)];
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let stored0 = flush(&dir, &tmp, "_0", seg0_id, &fields, &[doc_with(0, "a")]);
        let stored1 = flush(
            &dir,
            &tmp,
            "_1",
            [2u8; ID_LENGTH],
            &fields,
            &[doc_with(0, "b")],
        );
        let reader0 = open_reader(&stored0);
        let reader1 = open_reader(&stored1);

        let dv0_source = [dv0.source()];
        let source0 = MergeSource {
            field_infos: &stored0.fields,
            reader: &reader0,
            live_docs: None,
            numeric_doc_values: &dv0_source,
            norms: &[],
            term_vectors: None,
        };
        // Source 1 has live docs but no numeric doc-values entry at all for
        // field "num" -- the sparse-across-sources case this port refuses
        // to silently drop.
        let source1 = MergeSource::stored_only(&stored1.fields, &reader1, None);

        let result = merge_stored_only_segments(
            &dir,
            &[source0, source1],
            "_merged_dv_err",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        );
        assert!(matches!(
            result,
            Err(Error::DocValuesFieldMissingInSource {
                merged_field_number: 0
            })
        ));
    }

    #[test]
    fn more_than_one_numeric_doc_values_field_is_rejected() {
        let seg0_id = [1u8; ID_LENGTH];
        let dv_a = flush_numeric_dv(0, &[1], seg0_id);
        let dv_b = flush_numeric_dv(1, &[2], seg0_id);
        let fields = vec![numeric_field("a", 0), numeric_field("b", 1)];
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let stored0 = flush(
            &dir,
            &tmp,
            "_0",
            seg0_id,
            &fields,
            &[Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String("x".to_string()),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::String("y".to_string()),
                    },
                ],
            }],
        );
        let reader0 = open_reader(&stored0);
        let sources_a = dv_a.source();
        let sources_b = dv_b.source();
        let numeric = vec![
            SourceNumericDocValues {
                data: sources_a.data,
                entry: sources_a.entry.clone(),
            },
            SourceNumericDocValues {
                data: sources_b.data,
                entry: sources_b.entry.clone(),
            },
        ];
        let source0 = MergeSource {
            field_infos: &stored0.fields,
            reader: &reader0,
            live_docs: None,
            numeric_doc_values: &numeric,
            norms: &[],
            term_vectors: None,
        };

        let result = merge_stored_only_segments(
            &dir,
            &[source0],
            "_merged_dv_toomany",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        );
        assert!(matches!(
            result,
            Err(Error::TooManyNumericDocValuesFields(_))
        ));
    }

    #[test]
    fn a_fully_deleted_sources_unrelated_numeric_field_does_not_trigger_too_many_fields() {
        // Source 0 (live) has numeric-dv field "a"; source 1 is 100% deleted
        // but happens to carry an unrelated numeric-dv field "junk" -- since
        // source 1 contributes zero docs to the merge, its doc-values field
        // must not count toward the "more than one field" limit (regression
        // for a bug where the too-many-fields check ran before the
        // zero-live-docs exemption already applied elsewhere in this
        // module).
        let seg0_id = [1u8; ID_LENGTH];
        let seg1_id = [2u8; ID_LENGTH];
        let dv_a = flush_numeric_dv(0, &[10, 20], seg0_id);
        let dv_junk = flush_numeric_dv(0, &[99], seg1_id);
        let fields0 = vec![numeric_field("a", 0)];
        let fields1 = vec![numeric_field("junk", 0)];
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let stored0 = flush(
            &dir,
            &tmp,
            "_0",
            seg0_id,
            &fields0,
            &[
                Document {
                    fields: vec![StoredField {
                        field_number: 0,
                        value: FieldValue::String("x".to_string()),
                    }],
                },
                Document {
                    fields: vec![StoredField {
                        field_number: 0,
                        value: FieldValue::String("y".to_string()),
                    }],
                },
            ],
        );
        let stored1 = flush(
            &dir,
            &tmp,
            "_1",
            seg1_id,
            &fields1,
            &[Document {
                fields: vec![StoredField {
                    field_number: 0,
                    value: FieldValue::String("z".to_string()),
                }],
            }],
        );
        let reader0 = open_reader(&stored0);
        let reader1 = open_reader(&stored1);
        let numeric0 = vec![dv_a.source()];
        let numeric1 = vec![dv_junk.source()];
        let all_deleted = FixedBitSet::new(1); // source 1: nothing live
        let source0 = MergeSource {
            field_infos: &stored0.fields,
            reader: &reader0,
            live_docs: None,
            numeric_doc_values: &numeric0,
            norms: &[],
            term_vectors: None,
        };
        let source1 = MergeSource {
            field_infos: &stored1.fields,
            reader: &reader1,
            live_docs: Some(&all_deleted),
            numeric_doc_values: &numeric1,
            norms: &[],
            term_vectors: None,
        };

        let sci = merge_stored_only_segments(
            &dir,
            &[source0, source1],
            "_merged_dv_deleted_unrelated",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();

        let merged_reader =
            std::fs::read(std::path::Path::new(&tmp).join(format!("{}.fdt", sci.segment_name)));
        assert!(
            merged_reader.is_ok(),
            "merge should succeed, not reject on source 1's unrelated deleted-only field"
        );
    }

    #[test]
    fn norms_merge_across_two_sources_with_deletions() {
        let seg0_id = [1u8; ID_LENGTH];
        let seg1_id = [2u8; ID_LENGTH];
        let norms0 = flush_norms(0, &[1, 2], seg0_id);
        let norms1 = flush_norms(0, &[3], seg1_id);

        let fields = vec![norms_field("body", 0)];
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let stored0 = flush(
            &dir,
            &tmp,
            "_0",
            seg0_id,
            &fields,
            &[doc_with(0, "a"), doc_with(0, "b")],
        );
        let stored1 = flush(&dir, &tmp, "_1", seg1_id, &fields, &[doc_with(0, "c")]);
        let reader0 = open_reader(&stored0);
        let reader1 = open_reader(&stored1);

        let mut live0 = FixedBitSet::new(2);
        live0.set(1); // drop doc 0 ("a"/1), keep doc 1 ("b"/2)

        let norms0_source = [norms0.source()];
        let norms1_source = [norms1.source()];
        let source0 = MergeSource {
            field_infos: &stored0.fields,
            reader: &reader0,
            live_docs: Some(&live0),
            numeric_doc_values: &[],
            norms: &norms0_source,
            term_vectors: None,
        };
        let source1 = MergeSource {
            field_infos: &stored1.fields,
            reader: &reader1,
            live_docs: None,
            numeric_doc_values: &[],
            norms: &norms1_source,
            term_vectors: None,
        };

        merge_stored_only_segments(
            &dir,
            &[source0, source1],
            "_merged_norms",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();

        let nvd = std::fs::read(std::path::Path::new(&tmp).join("_merged_norms.nvd")).unwrap();
        let nvm = std::fs::read(std::path::Path::new(&tmp).join("_merged_norms.nvm")).unwrap();
        let (_v, parsed) = norms::parse_meta(&nvm, &[9u8; ID_LENGTH], "").unwrap();
        let entry = parsed.entry(0).unwrap();
        let values: Vec<i64> = (0..2)
            .map(|d| norms::norm_value(&nvd, entry, d).unwrap().unwrap())
            .collect();
        assert_eq!(values, vec![2, 3]);
    }

    #[test]
    fn term_vectors_merge_across_two_sources_with_deletions_and_a_source_with_none() {
        let seg0_id = [1u8; ID_LENGTH];
        // Source 0: 2 docs, both with a term-vectors field 0 ("id"->0).
        let tv0 = flush_term_vectors(&[tv_doc(0, &[("a", 0)]), tv_doc(0, &[("b", 0)])], seg0_id);
        // Source 1: 1 doc, no term-vectors reader at all -- contributes an
        // empty term-vectors document for its live doc.
        let fields0 = vec![tv_field("id", 0)];
        let fields1 = vec![tv_field("id", 0)];
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let stored0 = flush(
            &dir,
            &tmp,
            "_0",
            seg0_id,
            &fields0,
            &[doc_with(0, "a"), doc_with(0, "b")],
        );
        let stored1 = flush(
            &dir,
            &tmp,
            "_1",
            [2u8; ID_LENGTH],
            &fields1,
            &[doc_with(0, "c")],
        );
        let reader0 = open_reader(&stored0);
        let reader1 = open_reader(&stored1);
        let tv0_reader = tv0.reader();

        let mut live0 = FixedBitSet::new(2);
        live0.set(1); // drop doc 0, keep doc 1 ("b")

        let source0 = MergeSource {
            field_infos: &stored0.fields,
            reader: &reader0,
            live_docs: Some(&live0),
            numeric_doc_values: &[],
            norms: &[],
            term_vectors: Some(&tv0_reader),
        };
        let source1 = MergeSource::stored_only(&stored1.fields, &reader1, None);

        merge_stored_only_segments(
            &dir,
            &[source0, source1],
            "_merged_tv",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();

        let tvd = std::fs::read(std::path::Path::new(&tmp).join("_merged_tv.tvd")).unwrap();
        let tvx = std::fs::read(std::path::Path::new(&tmp).join("_merged_tv.tvx")).unwrap();
        let tvm = std::fs::read(std::path::Path::new(&tmp).join("_merged_tv.tvm")).unwrap();
        let merged_reader = term_vectors::open(&tvd, &tvx, &tvm, &[9u8; ID_LENGTH], "").unwrap();
        assert_eq!(merged_reader.max_doc(), 2);

        let doc0 = merged_reader.document(0).unwrap().unwrap();
        assert_eq!(doc0.fields[0].terms[0].term, b"b");

        // Source 1's doc contributed no term vectors at all.
        assert!(merged_reader.document(1).unwrap().is_none());
    }

    #[test]
    fn full_round_trip_merges_stored_fields_doc_values_norms_and_term_vectors_together() {
        let seg0_id = [1u8; ID_LENGTH];
        let seg1_id = [2u8; ID_LENGTH];

        let dv0 = flush_numeric_dv(0, &[100, 200], seg0_id);
        let dv1 = flush_numeric_dv(0, &[300], seg1_id);
        let norms0 = flush_norms(0, &[1, 2], seg0_id);
        let norms1 = flush_norms(0, &[3], seg1_id);
        let tv0 = flush_term_vectors(&[tv_doc(0, &[("x", 0)]), tv_doc(0, &[("y", 0)])], seg0_id);
        let tv1 = flush_term_vectors(&[tv_doc(0, &[("z", 0)])], seg1_id);

        let mut field0 = numeric_field("body", 0);
        field0.store_term_vectors = true;
        field0.omit_norms = false;
        let fields = vec![field0];

        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let stored0 = flush(
            &dir,
            &tmp,
            "_0",
            seg0_id,
            &fields,
            &[doc_with(0, "a"), doc_with(0, "b")],
        );
        let stored1 = flush(&dir, &tmp, "_1", seg1_id, &fields, &[doc_with(0, "c")]);
        let reader0 = open_reader(&stored0);
        let reader1 = open_reader(&stored1);
        let tv0_reader = tv0.reader();
        let tv1_reader = tv1.reader();

        let dv0_source = [dv0.source()];
        let dv1_source = [dv1.source()];
        let norms0_source = [norms0.source()];
        let norms1_source = [norms1.source()];
        let source0 = MergeSource {
            field_infos: &stored0.fields,
            reader: &reader0,
            live_docs: None,
            numeric_doc_values: &dv0_source,
            norms: &norms0_source,
            term_vectors: Some(&tv0_reader),
        };
        let source1 = MergeSource {
            field_infos: &stored1.fields,
            reader: &reader1,
            live_docs: None,
            numeric_doc_values: &dv1_source,
            norms: &norms1_source,
            term_vectors: Some(&tv1_reader),
        };

        let merged_id = [9u8; ID_LENGTH];
        merge_stored_only_segments(
            &dir,
            &[source0, source1],
            "_merged_all",
            merged_id,
            "Lucene104",
            version(),
        )
        .unwrap();

        // Stored fields.
        let merged_fdt = std::fs::read(std::path::Path::new(&tmp).join("_merged_all.fdt")).unwrap();
        let merged_fdx = std::fs::read(std::path::Path::new(&tmp).join("_merged_all.fdx")).unwrap();
        let merged_fdm = std::fs::read(std::path::Path::new(&tmp).join("_merged_all.fdm")).unwrap();
        let stored_reader =
            stored_fields::open(&merged_fdt, &merged_fdx, &merged_fdm, &merged_id, "").unwrap();
        assert_eq!(stored_reader.max_doc(), 3);
        let vals: Vec<String> = (0..3)
            .map(
                |i| match &stored_reader.document(i).unwrap().fields[0].value {
                    FieldValue::String(s) => s.clone(),
                    _ => unreachable!(),
                },
            )
            .collect();
        assert_eq!(vals, vec!["a", "b", "c"]);

        // Doc values.
        let dvd = std::fs::read(std::path::Path::new(&tmp).join("_merged_all.dvd")).unwrap();
        let dvm = std::fs::read(std::path::Path::new(&tmp).join("_merged_all.dvm")).unwrap();
        let merged_field_infos = field_infos::FieldInfos {
            fields: vec![numeric_field("body", 0)],
        };
        let (_v, dv_meta) =
            doc_values::parse_meta(&dvm, &merged_id, "", &merged_field_infos).unwrap();
        let dv_entry = dv_meta.numeric_entry(0).unwrap();
        let dv_values: Vec<i64> = (0..3)
            .map(|d| {
                doc_values::numeric_value(&dvd, dv_entry, d)
                    .unwrap()
                    .unwrap()
            })
            .collect();
        assert_eq!(dv_values, vec![100, 200, 300]);

        // Norms.
        let nvd = std::fs::read(std::path::Path::new(&tmp).join("_merged_all.nvd")).unwrap();
        let nvm = std::fs::read(std::path::Path::new(&tmp).join("_merged_all.nvm")).unwrap();
        let (_v, norms_meta) = norms::parse_meta(&nvm, &merged_id, "").unwrap();
        let norms_entry = norms_meta.entry(0).unwrap();
        let norms_values: Vec<i64> = (0..3)
            .map(|d| norms::norm_value(&nvd, norms_entry, d).unwrap().unwrap())
            .collect();
        assert_eq!(norms_values, vec![1, 2, 3]);

        // Term vectors.
        let tvd = std::fs::read(std::path::Path::new(&tmp).join("_merged_all.tvd")).unwrap();
        let tvx = std::fs::read(std::path::Path::new(&tmp).join("_merged_all.tvx")).unwrap();
        let tvm = std::fs::read(std::path::Path::new(&tmp).join("_merged_all.tvm")).unwrap();
        let tv_reader = term_vectors::open(&tvd, &tvx, &tvm, &merged_id, "").unwrap();
        assert_eq!(tv_reader.max_doc(), 3);
        let terms: Vec<Vec<u8>> = (0..3)
            .map(|d| {
                tv_reader.document(d).unwrap().unwrap().fields[0].terms[0]
                    .term
                    .clone()
            })
            .collect();
        assert_eq!(terms, vec![b"x".to_vec(), b"y".to_vec(), b"z".to_vec()]);

        // Segment info lists every file.
        let si_bytes = std::fs::read(std::path::Path::new(&tmp).join("_merged_all.si")).unwrap();
        let si = segment_info::parse(&si_bytes, &merged_id).unwrap();
        for ext in [
            "fdt", "fdx", "fdm", "fnm", "dvm", "dvd", "dvs", "nvm", "nvd", "tvd", "tvx", "tvm",
        ] {
            let name = format!("_merged_all.{ext}");
            assert!(si.files.contains(&name), "missing {name} in .si files list");
        }
    }

    // --- merge_sorted_stored_only_segments (k-way sort-preserving merge) ---

    /// Reads back the merged segment's stored "id" field (a String) for every
    /// doc, in doc order -- the assertion helper every k-way-merge test below
    /// uses to confirm both order and content.
    fn read_merged_ids(tmp: &str, segment_name: &str, segment_id: [u8; ID_LENGTH]) -> Vec<String> {
        let fdt =
            std::fs::read(std::path::Path::new(tmp).join(format!("{segment_name}.fdt"))).unwrap();
        let fdx =
            std::fs::read(std::path::Path::new(tmp).join(format!("{segment_name}.fdx"))).unwrap();
        let fdm =
            std::fs::read(std::path::Path::new(tmp).join(format!("{segment_name}.fdm"))).unwrap();
        let reader = stored_fields::open(&fdt, &fdx, &fdm, &segment_id, "").unwrap();
        (0..reader.max_doc())
            .map(|d| match &reader.document(d).unwrap().fields[0].value {
                FieldValue::String(s) => s.clone(),
                _ => unreachable!(),
            })
            .collect()
    }

    #[test]
    fn two_sources_with_interleaved_keys_produce_globally_sorted_output_not_concatenation() {
        // Source 0 (already sorted by "num" ascending): 10, 30, 50.
        // Source 1 (already sorted by "num" ascending): 20, 40.
        // Naive concatenation would yield 10,30,50,20,40 -- visibly
        // out-of-order at the 50->20 boundary. The real k-way merge must
        // interleave to 10,20,30,40,50.
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let fields = vec![field("id", 0)];

        let seg0 = flush(
            &dir,
            &tmp,
            "_0",
            [1u8; ID_LENGTH],
            &fields,
            &[doc_with(0, "10"), doc_with(0, "30"), doc_with(0, "50")],
        );
        let seg1 = flush(
            &dir,
            &tmp,
            "_1",
            [2u8; ID_LENGTH],
            &fields,
            &[doc_with(0, "20"), doc_with(0, "40")],
        );

        let reader0 = open_reader(&seg0);
        let reader1 = open_reader(&seg1);
        let sources = vec![
            MergeSource::stored_only(&seg0.fields, &reader0, None),
            MergeSource::stored_only(&seg1.fields, &reader1, None),
        ];

        let keys0: Vec<Option<i64>> = vec![Some(10), Some(30), Some(50)];
        let keys1: Vec<Option<i64>> = vec![Some(20), Some(40)];
        let per_source_keys: Vec<&[Option<i64>]> = vec![&keys0, &keys1];
        let sort_fields = vec![MergeSortKeySpec {
            field: "num",
            reverse: false,
            missing: SortMissingValue::Last,
            per_source_keys: &per_source_keys,
        }];

        let merged_id = [9u8; ID_LENGTH];
        let sci = merge_sorted_stored_only_segments(
            &dir,
            &sources,
            &sort_fields,
            "_merged_sorted",
            merged_id,
            "Lucene104",
            version(),
        )
        .unwrap();
        assert_eq!(sci.segment_name, "_merged_sorted");

        let ids = read_merged_ids(&tmp, "_merged_sorted", merged_id);
        assert_eq!(ids, vec!["10", "20", "30", "40", "50"]);

        // Confirm this is NOT what naive concatenation would produce.
        assert_ne!(ids, vec!["10", "30", "50", "20", "40"]);

        // The merged .si must keep the same index-sort descriptor, not lose
        // or null it out.
        let si_bytes = std::fs::read(std::path::Path::new(&tmp).join("_merged_sorted.si")).unwrap();
        let si = segment_info::parse(&si_bytes, &merged_id).unwrap();
        let sort = si.index_sort.unwrap();
        assert_eq!(sort.len(), 1);
        assert_eq!(sort[0].field, "num");
        assert!(!sort[0].reverse);
        assert_eq!(sort[0].missing, SortMissingValue::Last);
    }

    #[test]
    fn three_sources_k_way_merge_by_sort_key() {
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let fields = vec![field("id", 0)];

        let seg0 = flush(
            &dir,
            &tmp,
            "_0",
            [1u8; ID_LENGTH],
            &fields,
            &[doc_with(0, "1"), doc_with(0, "9")],
        );
        let seg1 = flush(
            &dir,
            &tmp,
            "_1",
            [2u8; ID_LENGTH],
            &fields,
            &[doc_with(0, "4"), doc_with(0, "6")],
        );
        let seg2 = flush(
            &dir,
            &tmp,
            "_2",
            [3u8; ID_LENGTH],
            &fields,
            &[doc_with(0, "2"), doc_with(0, "5"), doc_with(0, "8")],
        );

        let reader0 = open_reader(&seg0);
        let reader1 = open_reader(&seg1);
        let reader2 = open_reader(&seg2);
        let sources = vec![
            MergeSource::stored_only(&seg0.fields, &reader0, None),
            MergeSource::stored_only(&seg1.fields, &reader1, None),
            MergeSource::stored_only(&seg2.fields, &reader2, None),
        ];

        let keys0: Vec<Option<i64>> = vec![Some(1), Some(9)];
        let keys1: Vec<Option<i64>> = vec![Some(4), Some(6)];
        let keys2: Vec<Option<i64>> = vec![Some(2), Some(5), Some(8)];
        let per_source_keys: Vec<&[Option<i64>]> = vec![&keys0, &keys1, &keys2];
        let sort_fields = vec![MergeSortKeySpec {
            field: "num",
            reverse: false,
            missing: SortMissingValue::Last,
            per_source_keys: &per_source_keys,
        }];

        let merged_id = [9u8; ID_LENGTH];
        merge_sorted_stored_only_segments(
            &dir,
            &sources,
            &sort_fields,
            "_merged_three",
            merged_id,
            "Lucene104",
            version(),
        )
        .unwrap();

        let ids = read_merged_ids(&tmp, "_merged_three", merged_id);
        assert_eq!(ids, vec!["1", "2", "4", "5", "6", "8", "9"]);
    }

    #[test]
    fn tie_on_primary_field_across_sources_is_broken_by_secondary_field() {
        // Both sources' first doc ties on "num"=5; the secondary field "tie"
        // must break it (source 0's doc has tie=1, source 1's has tie=0, so
        // source 1's doc must come first despite arriving from the
        // "later" source).
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let fields = vec![field("id", 0)];

        let seg0 = flush(
            &dir,
            &tmp,
            "_0",
            [1u8; ID_LENGTH],
            &fields,
            &[doc_with(0, "A"), doc_with(0, "C")],
        );
        let seg1 = flush(
            &dir,
            &tmp,
            "_1",
            [2u8; ID_LENGTH],
            &fields,
            &[doc_with(0, "B")],
        );

        let reader0 = open_reader(&seg0);
        let reader1 = open_reader(&seg1);
        let sources = vec![
            MergeSource::stored_only(&seg0.fields, &reader0, None),
            MergeSource::stored_only(&seg1.fields, &reader1, None),
        ];

        // Primary "num": source0 = [5, 8], source1 = [5].
        let num0: Vec<Option<i64>> = vec![Some(5), Some(8)];
        let num1: Vec<Option<i64>> = vec![Some(5)];
        let num_keys: Vec<&[Option<i64>]> = vec![&num0, &num1];
        // Secondary "tie": source0's doc "A" has tie=1, source1's doc "B" has
        // tie=0 -- ascending means "B" (tie=0) must sort before "A" (tie=1).
        let tie0: Vec<Option<i64>> = vec![Some(1), Some(0)];
        let tie1: Vec<Option<i64>> = vec![Some(0)];
        let tie_keys: Vec<&[Option<i64>]> = vec![&tie0, &tie1];

        let sort_fields = vec![
            MergeSortKeySpec {
                field: "num",
                reverse: false,
                missing: SortMissingValue::Last,
                per_source_keys: &num_keys,
            },
            MergeSortKeySpec {
                field: "tie",
                reverse: false,
                missing: SortMissingValue::Last,
                per_source_keys: &tie_keys,
            },
        ];

        let merged_id = [9u8; ID_LENGTH];
        merge_sorted_stored_only_segments(
            &dir,
            &sources,
            &sort_fields,
            "_merged_tiebreak",
            merged_id,
            "Lucene104",
            version(),
        )
        .unwrap();

        let ids = read_merged_ids(&tmp, "_merged_tiebreak", merged_id);
        // "B" (num=5,tie=0) before "A" (num=5,tie=1) before "C" (num=8).
        assert_eq!(ids, vec!["B", "A", "C"]);
    }

    #[test]
    fn stored_field_content_stays_attached_to_the_right_doc_after_reordering() {
        // Multi-field docs where the field content itself encodes the sort
        // key, confirming the whole Document (not just a scalar) travels
        // with its doc through the k-way merge -- a shuffle bug that
        // permuted docs independently of their sort key would show up here
        // as mismatched (key, payload) pairs.
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let fields = vec![field("key", 0), field("payload", 1)];

        fn doc(key: &str, payload: &str) -> Document {
            Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String(key.to_string()),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::String(payload.to_string()),
                    },
                ],
            }
        }

        let seg0 = flush(
            &dir,
            &tmp,
            "_0",
            [1u8; ID_LENGTH],
            &fields,
            &[doc("3", "three"), doc("7", "seven")],
        );
        let seg1 = flush(
            &dir,
            &tmp,
            "_1",
            [2u8; ID_LENGTH],
            &fields,
            &[doc("1", "one"), doc("5", "five")],
        );

        let reader0 = open_reader(&seg0);
        let reader1 = open_reader(&seg1);
        let sources = vec![
            MergeSource::stored_only(&seg0.fields, &reader0, None),
            MergeSource::stored_only(&seg1.fields, &reader1, None),
        ];

        let keys0: Vec<Option<i64>> = vec![Some(3), Some(7)];
        let keys1: Vec<Option<i64>> = vec![Some(1), Some(5)];
        let per_source_keys: Vec<&[Option<i64>]> = vec![&keys0, &keys1];
        let sort_fields = vec![MergeSortKeySpec {
            field: "key",
            reverse: false,
            missing: SortMissingValue::Last,
            per_source_keys: &per_source_keys,
        }];

        let merged_id = [9u8; ID_LENGTH];
        merge_sorted_stored_only_segments(
            &dir,
            &sources,
            &sort_fields,
            "_merged_payload",
            merged_id,
            "Lucene104",
            version(),
        )
        .unwrap();

        let fdt = std::fs::read(std::path::Path::new(&tmp).join("_merged_payload.fdt")).unwrap();
        let fdx = std::fs::read(std::path::Path::new(&tmp).join("_merged_payload.fdx")).unwrap();
        let fdm = std::fs::read(std::path::Path::new(&tmp).join("_merged_payload.fdm")).unwrap();
        let reader = stored_fields::open(&fdt, &fdx, &fdm, &merged_id, "").unwrap();
        assert_eq!(reader.max_doc(), 4);

        let expected = [("1", "one"), ("3", "three"), ("5", "five"), ("7", "seven")];
        for (i, (key, payload)) in expected.iter().enumerate() {
            let d = reader.document(i as i32).unwrap();
            let got_key = match &d.fields[0].value {
                FieldValue::String(s) => s.clone(),
                _ => unreachable!(),
            };
            let got_payload = match &d.fields[1].value {
                FieldValue::String(s) => s.clone(),
                _ => unreachable!(),
            };
            assert_eq!(&got_key, key, "doc {i} key mismatch");
            assert_eq!(&got_payload, payload, "doc {i} payload mismatch");
        }
    }

    #[test]
    fn deleted_docs_are_dropped_before_the_k_way_merge_walks_a_source() {
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let fields = vec![field("id", 0)];

        let seg0 = flush(
            &dir,
            &tmp,
            "_0",
            [1u8; ID_LENGTH],
            &fields,
            &[doc_with(0, "10"), doc_with(0, "20"), doc_with(0, "30")],
        );
        let seg1 = flush(
            &dir,
            &tmp,
            "_1",
            [2u8; ID_LENGTH],
            &fields,
            &[doc_with(0, "15")],
        );

        let reader0 = open_reader(&seg0);
        let reader1 = open_reader(&seg1);

        // Drop doc 1 ("20") from source 0.
        let mut live0 = FixedBitSet::new(3);
        live0.set(0);
        live0.set(2);

        let sources = vec![
            MergeSource::stored_only(&seg0.fields, &reader0, Some(&live0)),
            MergeSource::stored_only(&seg1.fields, &reader1, None),
        ];

        let keys0: Vec<Option<i64>> = vec![Some(10), Some(20), Some(30)];
        let keys1: Vec<Option<i64>> = vec![Some(15)];
        let per_source_keys: Vec<&[Option<i64>]> = vec![&keys0, &keys1];
        let sort_fields = vec![MergeSortKeySpec {
            field: "num",
            reverse: false,
            missing: SortMissingValue::Last,
            per_source_keys: &per_source_keys,
        }];

        let merged_id = [9u8; ID_LENGTH];
        merge_sorted_stored_only_segments(
            &dir,
            &sources,
            &sort_fields,
            "_merged_deleted",
            merged_id,
            "Lucene104",
            version(),
        )
        .unwrap();

        let ids = read_merged_ids(&tmp, "_merged_deleted", merged_id);
        assert_eq!(ids, vec!["10", "15", "30"]);
    }

    #[test]
    fn no_sources_produces_an_empty_sorted_segment() {
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let sources: Vec<MergeSource> = vec![];
        let per_source_keys: Vec<&[Option<i64>]> = vec![];
        let sort_fields = vec![MergeSortKeySpec {
            field: "num",
            reverse: false,
            missing: SortMissingValue::Last,
            per_source_keys: &per_source_keys,
        }];

        let merged_id = [3u8; ID_LENGTH];
        let sci = merge_sorted_stored_only_segments(
            &dir,
            &sources,
            &sort_fields,
            "_merged_empty_sorted",
            merged_id,
            "Lucene104",
            version(),
        )
        .unwrap();
        assert_eq!(sci.segment_name, "_merged_empty_sorted");

        let fdt =
            std::fs::read(std::path::Path::new(&tmp).join("_merged_empty_sorted.fdt")).unwrap();
        let fdx =
            std::fs::read(std::path::Path::new(&tmp).join("_merged_empty_sorted.fdx")).unwrap();
        let fdm =
            std::fs::read(std::path::Path::new(&tmp).join("_merged_empty_sorted.fdm")).unwrap();
        let reader = stored_fields::open(&fdt, &fdx, &fdm, &merged_id, "").unwrap();
        assert_eq!(reader.max_doc(), 0);
    }

    #[test]
    fn field_number_reconciliation_still_applies_during_the_k_way_merge() {
        // Same field-name-vs-number-mismatch setup as the concatenation
        // merge's own test, confirming the k-way merge path also reconciles
        // field numbers by name rather than trusting per-source numbering.
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let fields0 = vec![field("num", 0), field("id", 1)];
        let fields1 = vec![field("id", 0), field("num", 1)];

        let doc0 = Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("10".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::String("first".to_string()),
                },
            ],
        };
        let doc1 = Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("second".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::String("5".to_string()),
                },
            ],
        };

        let seg0 = flush(&dir, &tmp, "_0", [1u8; ID_LENGTH], &fields0, &[doc0]);
        let seg1 = flush(&dir, &tmp, "_1", [2u8; ID_LENGTH], &fields1, &[doc1]);

        let reader0 = open_reader(&seg0);
        let reader1 = open_reader(&seg1);
        let sources = vec![
            MergeSource::stored_only(&seg0.fields, &reader0, None),
            MergeSource::stored_only(&seg1.fields, &reader1, None),
        ];

        let keys0: Vec<Option<i64>> = vec![Some(10)];
        let keys1: Vec<Option<i64>> = vec![Some(5)];
        let per_source_keys: Vec<&[Option<i64>]> = vec![&keys0, &keys1];
        let sort_fields = vec![MergeSortKeySpec {
            field: "num",
            reverse: false,
            missing: SortMissingValue::Last,
            per_source_keys: &per_source_keys,
        }];

        let merged_id = [9u8; ID_LENGTH];
        merge_sorted_stored_only_segments(
            &dir,
            &sources,
            &sort_fields,
            "_merged_reconcile",
            merged_id,
            "Lucene104",
            version(),
        )
        .unwrap();

        let merged_fnm =
            std::fs::read(std::path::Path::new(&tmp).join("_merged_reconcile.fnm")).unwrap();
        let merged_fields = lucene_codecs::field_infos::parse(&merged_fnm, &merged_id, "").unwrap();
        let id_number = merged_fields
            .fields
            .iter()
            .find(|f| f.name == "id")
            .unwrap()
            .number;

        let fdt = std::fs::read(std::path::Path::new(&tmp).join("_merged_reconcile.fdt")).unwrap();
        let fdx = std::fs::read(std::path::Path::new(&tmp).join("_merged_reconcile.fdx")).unwrap();
        let fdm = std::fs::read(std::path::Path::new(&tmp).join("_merged_reconcile.fdm")).unwrap();
        let reader = stored_fields::open(&fdt, &fdx, &fdm, &merged_id, "").unwrap();
        assert_eq!(reader.max_doc(), 2);

        // Sorted by num: doc1 (num=5, "second"/"5") comes first, doc0
        // (num=10, "10"/"first") comes second -- confirm the "id" field's
        // content followed its own doc through the reordering.
        let d0 = reader.document(0).unwrap();
        let id0 = d0
            .fields
            .iter()
            .find(|f| f.field_number == id_number)
            .unwrap();
        assert_eq!(id0.value, FieldValue::String("second".to_string()));

        let d1 = reader.document(1).unwrap();
        let id1 = d1
            .fields
            .iter()
            .find(|f| f.field_number == id_number)
            .unwrap();
        assert_eq!(id1.value, FieldValue::String("first".to_string()));
    }
}
