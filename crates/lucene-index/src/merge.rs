//! Port of `org.apache.lucene.index.SegmentMerger` (plus the field-numbering
//! half of `FieldInfos.FieldNumbers`) -- merges N already-flushed segments
//! into one new segment, dropping deleted docs and renumbering doc ids to be
//! contiguous (`0..mergedDocCount`). Stored fields are always merged; doc
//! values, norms, term vectors, postings, BKD points and KNN vectors are
//! merged too whenever a source supplies them (see "Doc values / norms /
//! term vectors", "Postings", "Points" and "Vectors" below for the honest
//! scope of each part).
//!
//! # One merge, two document orders
//!
//! [`merge_segments`] is the single implementation. `sort_fields` chooses
//! **only** the order documents come out in -- `MergeState.docMaps`:
//! concatenation by source (`buildDeletionDocMaps`) or a k-way merge on the
//! shared index sort (`MultiSorter.sort`). Every format is merged through
//! that one order, so no format can end up in a different order from
//! another, and no format can be written by one entry point and forgotten by
//! the other. [`merge_stored_only_segments`] and
//! [`merge_sorted_stored_only_segments`] are two thin wrappers over it.
//!
//! # What this is
//!
//! [`merge_stored_only_segments`] takes, for each source segment, its already
//! read-back [`FieldInfos`](field_infos::FieldInfos), a
//! [`Document`](stored_fields::Document) reader
//! (via this port's stored-fields reader, [`stored_fields::open`] +
//! [`stored_fields::StoredFieldsReader::document`]), an optional per-source
//! live-docs bitset (via [`lucene_codecs::live_docs::parse`], or `None` if the source has no
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
//!    either by concatenating surviving docs in source order or by k-way
//!    merging them on the shared index sort -- exactly the two shapes real
//!    `SegmentMerger`'s `MergeState.docMaps` takes;
//! 3. merges any supplied doc-values/norms/term-vectors/postings/points data
//!    the same way (drop deleted docs, renumber contiguously, remap field
//!    numbers), then writes stored fields, field infos, segment info, and
//!    whichever of `.dvm`/`.dvd`/`.dvs`, `.nvm`/`.nvd`, `.tvd`/`.tvx`/`.tvm`,
//!    `.doc`/`.tim`/`.tip`/`.tmd`, `.kdm`/`.kdi`/`.kdd`, and
//!    `.vec`/`.vemf`/`.vem`/`.vex` the merge produced,
//!    directly through `dir` -- mirroring exactly the write-side work
//!    [`crate::segment_writer::flush_stored_only_segment`] does for a
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
//! - **No merge-time doc-id re-sorting in [`merge_stored_only_segments`].**
//!   Real Lucene's `MergeState` re-sorts by an index sort only when the
//!   merged segment has one; this entry point is the no-sort half and
//!   deliberately concatenates. [`merge_sorted_stored_only_segments`] is the
//!   other half. Both are [`merge_segments`].
//!
//! (Schema consistency across sources *is* now checked -- see
//! [`reconcile_field_numbers`], which ports `FieldInfos.Builder.add`'s
//! `verifySameSchema` + `setStorePayloads` behaviour.)
//!
//! # Doc values / norms / term vectors
//!
//! Buffered and explicit IndexWriter flushes supply multiple fields per
//! segment. MergeSource accepts their decoded doc-values, norms and term-vector
//! columns, reconciles field numbers, drops deleted documents and writes the
//! resulting columns in merged physical document order.
//!
//! ## Scope of the doc-values/norms merge
//!
//! **Doc values have no per-merge field limit.** Every merged field of every
//! type is written into one `.dvm`/`.dvd`/`.dvs` triple through
//! [`lucene_codecs::doc_values::write_dense_fields`], which is the real
//! on-disk shape of a `Lucene90DocValuesFormat` segment (`numFields`
//! interleaved meta entries over shared data). A **multi-tier index sort**
//! needs exactly that: a second sort tier is a second NUMERIC column, so the
//! merge carries several fields per type and several types per call.
//!
//! **Norms have no per-merge field limit either**: every merged norms
//! field goes into one `.nvm`/`.nvd` pair through
//! [`lucene_codecs::norms::write_fields`], which is the real on-disk shape of
//! a `Lucene90NormsFormat` segment (`Lucene90NormsConsumer` gets one
//! `addNormsField` call per field into one pair). Term vectors and postings have never had a limit.
//!
//! ## The "sparse across sources" rule
//!
//! Real Lucene requires every doc in a merged segment to either uniformly
//! have or uniformly lack doc-values/norms for a field, per that field's
//! `FieldInfos` declaration -- a field can't have doc values for some docs
//! and not others within one segment (`DocValuesType.NONE` vs. non-`NONE` is
//! segment-wide per field). Within that, **per-document** sparsity is
//! ordinary: `SortField.setMissingValue` exists precisely for it.
//!
//! - **Doc values of every type** may be sparse, as
//!   `DocValuesConsumer.getMerged*DocValues` makes them: a source whose
//!   `FieldInfo` has no column for the field is simply not a sub, so its
//!   documents come out of the merge without a value, and so does any
//!   document its column skips. A merged column only some documents have a
//!   value for is written through the same `IndexedDISI` + values body
//!   `Lucene90DocValuesConsumer` uses, and a column **no** merged document
//!   has a value for is still written, as its all-missing form
//!   (`docsWithFieldOffset = -2`).
//! - **Norms** may be absent in a whole source or individual documents. Their
//!   gaps remain absent on disk; a present zero remains a value. A field with
//!   no surviving values is written as an empty sparse column.
//!
//! Term vectors have no such constraint: a source with no term-vectors
//! reader at all, or a doc with none, simply contributes an empty
//! [`lucene_codecs::term_vectors::TermVectorsDocument`] -- exactly what
//! `TermVectorsWriter.merge` does when `mergeState.termVectorsReaders[i]`
//! is null (`addAllDocVectors(null, ...)`), which is the normal state of
//! affairs when term vectors get turned on for an index that already has
//! segments.
//! `write_best_speed` supports offsets and payloads as well as positions,
//! and [`write_merged_term_vectors`] passes them through unchanged: unlike postings
//! (whose per-term doc/freq/position lists from different sources have to be
//! concatenated into one ascending-by-docID stream per term), a term
//! vector's positions/offsets/payloads are entirely doc-local -- each
//! source's [`TermVectorsReader::document`] call already returns one fully
//! self-contained [`lucene_codecs::term_vectors::TermVectorsDocument`], so
//! merging is just "read the doc, remap its field numbers, append it to the
//! merged doc list" with no cross-source term-level combination step at all.
//! This applies uniformly regardless of `has_offsets`/`has_payloads`: those
//! flags and their per-term data ride along with the rest of the field
//! untouched. Positions-only, offsets-only, payloads-only, and
//! offsets+payloads term vectors all merge and round-trip correctly through
//! the real reader/writer stack.
//!
//! ## Doc-values type scope
//!
//! **NUMERIC**, **BINARY**, **SORTED**, **SORTED_NUMERIC**, and
//! **SORTED_SET** doc-values are all merged here ([`merge_numeric_doc_values`],
//! [`merge_binary_doc_values`], [`merge_sorted_doc_values`],
//! [`merge_sorted_numeric_doc_values`], [`merge_sorted_set_doc_values`]) --
//! BINARY needed no ordinal remapping, so it was a straightforward mirror of
//! the NUMERIC logic (same per-source concatenation, same "sparse across
//! sources" rule, same single-field-per-call limit). SORTED is genuinely
//! different: each source's term dictionary is built independently, so
//! ordinal `N` in source A's dictionary is generally a *different term* than
//! ordinal `N` in source B's (real Lucene's `OrdinalMap` exists to solve
//! exactly this during a merge). [`merge_sorted_doc_values`] resolves each
//! live doc's ordinal straight to term bytes via that doc's *own source's*
//! dictionary ([`lucene_codecs::terms_dict::decode_all_terms`]), then hands
//! the merge's full per-doc *term-bytes* list (not ordinals) to
//! [`lucene_codecs::doc_values::write_single_dense_sorted_field`], which
//! rebuilds the merged, deduplicated, sorted dictionary itself -- so there
//! is no separate ordinal-remapping table to get wrong; two sources' docs
//! that share a term land on the same merged dictionary entry purely
//! because the merged dictionary is deduplicated by term bytes.
//!
//! SORTED_NUMERIC is multi-valued NUMERIC with no shared dictionary at all:
//! [`merge_sorted_numeric_doc_values`] simply concatenates each live doc's
//! own `Vec<i64>` of values, generalizing
//! [`merge_numeric_doc_values`]'s one-value-per-doc concatenation to a list
//! per doc. SORTED_SET is multi-valued SORTED, so it reuses
//! [`merge_sorted_doc_values`]'s exact "resolve to bytes, let the writer
//! dedupe" approach, just per-*value* instead of per-doc:
//! [`merge_sorted_set_doc_values`] resolves each of a live doc's own source's
//! ordinals to term bytes via that source's own dictionary, producing a
//! `Vec<Vec<u8>>` per doc, and hands the whole thing to
//! [`lucene_codecs::doc_values::write_single_dense_sorted_set_field`], which
//! rebuilds the merged, deduplicated dictionary itself -- same
//! no-ordinal-remapping-table-to-get-wrong property as SORTED.
//!
//! # Postings
//!
//! [`merge_term_range`] walks each source's term dictionary with [`TermCursor`]
//! and selects the smallest current term across sources. It maps each source's
//! doc/frequency pairs through [`build_doc_id_maps`], drops deleted documents,
//! and sorts mapped documents when the merged index sort interleaves sources.
//! A source with no terms for a field contributes nothing, matching
//! `FieldsConsumer.merge`. [`check_format_coverage`] rejects an unopened
//! postings format rather than silently omitting its data.
//!
//! All [`IndexOptions`] are supported, including positions, offsets and
//! payloads. The current term's metadata and doc/frequency stream are decoded
//! once; [`lucene_codecs::postings::read_occurrences_flat`] validates its whole
//! occurrence stream before live-document filtering. This keeps corruption in
//! a deleted document from being hidden when its source still contributes.
//! [`MergedRun`] owns one term range's flat occurrence buffers and document
//! boundaries, and its [`MergedTermView`]s implement [`TermSource`] for the
//! native writer. Sorted merges apply
//! the same permutation to documents, positions, offsets and payload runs.
//!
//! [`reconcile_field_numbers`] requires all sources sharing a field name to
//! agree on `index_options` ([`Error::PostingsIndexOptionsDisagreement`]). It
//! ORs `store_payloads` as `FieldInfos.Builder.add` does: a payload-free source
//! contributes a zero payload length for every occurrence when another source
//! enables payloads for the merged field.
//!
//! Both buffered and explicit [`crate::index_writer::IndexWriter`] flushes
//! produce the native postings groups consumed here. The merge publishes them
//! through [`lucene_codecs::postings_writer::write_fields`] or optional installed
//! consumers via [`crate::postings_writer::write_installed`]. The enclosing
//! [`merge_segments_mapped`] tracks successful file creations and removes its
//! own outputs on returned errors, including files completed before a later
//! occurrence decoder or consumer fails.
//!
//! # Points
//!
//! [`merge_points`] merges each source's BKD points data (`.kdm`/`.kdi`/
//! `.kdd`) for every field any source declares points for ([`SourcePoints`],
//! attached per source via [`MergeSource::points`]), re-encoding the result
//! with [`lucene_codecs::points::write_packed`]. Unlike SORTED doc values or
//! postings, a points field has no shared term dictionary to resolve
//! ordinals against -- it's a flat, per-doc set of fixed-width packed values
//! (closer in spirit to SORTED_NUMERIC doc values than to postings), so this
//! reads back every live doc's points via that source's own already-opened
//! [`lucene_codecs::points::PointsReader`] (the *exact same* reader
//! `lucene_search`'s points range query already uses -- no new BKD decode
//! logic was written for this merge), drops non-live docs and remaps
//! surviving doc ids to the merged id space (reusing
//! [`build_doc_id_maps`], the same mechanism [`merge_term_range`] uses), and
//! hands the per-source streams to [`lucene_codecs::points::write_merged_with_mode`]:
//! a one-dimensional field is k-way merged by value when every source has a
//! native points reader, as `BKDWriter.merge` does. A point-free source selects
//! the generic add/finish path, as does every other dimensionality -- through the
//! byte-identical `BKDWriter` port, with the merged segment's `maxDoc`.
//!
//! A source whose own `FieldInfos` never saw the points field simply
//! contributes no points, matching `PointsWriter.merge`'s `if
//! (readerFieldInfo == null) continue;`; a source that declares the field
//! but whose caller supplied no [`SourcePoints`] for it is still a hard
//! error ([`Error::PointsFieldMissingInSource`]), same caller-wiring-bug
//! distinction postings draws. A field has no per-doc denseness
//! requirement of its own here either: a live doc contributing zero points for a field (or a field
//! that ends up with zero surviving points overall, e.g. every
//! contributing doc's point belonged to a deleted doc) is not an error --
//! points are naturally sparse the same way postings are. A field that ends
//! up with zero points after the merge has no BKD entry, matching Java's
//! `finish()` returning a null finalizer. Its declared dimensions remain in
//! FieldInfos, and the segment still has a valid empty points file triple.
//!
//! **Scope: single packed-value shape per field across all sources,
//! `num_index_dims` may be less than `num_dims`.**
//! [`lucene_codecs::points::write_packed`] supports fields whose `num_index_dims`
//! is less than `num_dims` (data-only, non-indexed trailing dimensions --
//! e.g. a `LatLonShape`-style bounding box), and this merge preserves that:
//! every point's full `num_dims`-wide packed value (index dims plus any
//! trailing data-only dims) is carried through unchanged. Because
//! field-number reconciliation only records the *first-seen* source's
//! `FieldInfo` as the merged one, every other live-doc-contributing source's
//! own BKD tree shape (`num_dims`/`bytes_per_dim`/`num_index_dims`) is
//! independently checked against the merged field's declared shape and
//! rejected with [`Error::PointsShapeDisagreement`] (for `num_dims`/
//! `bytes_per_dim`) or [`Error::PointsIndexDimsDisagreement`] (for
//! `num_index_dims`) on a mismatch -- otherwise a source using, say, 2
//! dimensions could have its points silently misinterpreted as
//! 1-dimensional (or vice versa), or a source with a different index/data-
//! dim boundary could have its data-only dims silently reinterpreted as
//! index dims, whenever an earlier, differently-shaped source happened to be
//! picked as canonical. Multi-dimension points (e.g. `LatLonPoint`-shaped 2D
//! fields) and multi-valued points (multiple points per doc for the same
//! field) are both supported -- this is exactly what
//! [`lucene_codecs::points::write_packed`] itself already handles, and the
//! concatenation this merge performs preserves both.
//!
//! Same caveat as doc values/norms/term vectors/postings: nothing in this
//! port's normal flush path produces a segment with points yet (`.kdm`/
//! `.kdi`/`.kdd` are written by [`lucene_codecs::points::write_packed`] as a
//! standalone function, not from a per-field indexing flush path), so this
//! merge logic is real and tested on its own (including a full round-trip
//! through the unmodified [`lucene_codecs::points::PointsReader`] and
//! `lucene_search` points range-query stack), but not yet reachable from a
//! real end-to-end "flush two segments, merge them" caller.
//!
//! # Vectors
//!
//! [`merge_vectors`] merges KNN vector fields (`.vec`/`.vemf` flat store plus
//! `.vem`/`.vex` HNSW graph) -- the port of `SegmentMerger.mergeVectorValues`
//! over `Lucene99HnswVectorsWriter.mergeOneField`. The flat store is merged
//! first and **defines the merged ordinal space** (vectors are copied, never
//! decoded, and a run of surviving consecutive ordinals from one source is
//! one `memcpy`); the graph is then merged against exactly the bytes just
//! written, reusing the largest usable source graph rather than rebuilding
//! (`IncrementalHnswGraphMerger`). There is deliberately no bulk-copy fast
//! path and no `MatchingReaders` consultation: the merged ordinal space is
//! new on every merge, and Lucene re-writes the data file too.
//!
//! See `docs/parity.md` for the exact scope line.

use std::collections::{HashMap, HashSet};

use crate::segment_info::{self, IndexSortField, LuceneVersion, SegmentInfo, SortKeyComparator};
use crate::segment_infos::SegmentCommitInfo;
use lucene_codecs::blocktree::{self, FieldTerms};
use lucene_codecs::doc_values::{
    self, BinaryEntry, NumericEntry, SortedEntry, SortedNumericEntry, SortedSetEntry, SortedSetKind,
};
use lucene_codecs::field_infos::{self, FieldInfo, IndexOptions, VectorEncoding};
use lucene_codecs::norms::{self, NormsEntry};
use lucene_codecs::per_field_doc_values;
use lucene_codecs::per_field_knn_vectors;
use lucene_codecs::per_field_postings;
use lucene_codecs::points;
use lucene_codecs::postings::DocInput;
use lucene_codecs::postings_writer::{self, FieldPostingsInput, TermSource};
use lucene_codecs::scalar_quantized_vectors;
use lucene_codecs::stored_fields;
use lucene_codecs::term_vectors::{self, TermVectorsDocument, TermVectorsReader};
use lucene_codecs::terms_dict;
use lucene_codecs::{hnsw, hnsw_vectors, vectors};
use lucene_store::codec_util::ID_LENGTH;
use lucene_store::data_output::DataOutput;
use lucene_store::directory::Directory;
use lucene_util::fixed_bit_set::FixedBitSet;
mod written_files;
use written_files::describe_written_files;

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
    /// Every vector-format error -- `vectors`, `hnsw` and `hnsw_vectors` all
    /// share one `Error` type, the way they share one on-disk format family.
    #[error(transparent)]
    Vectors(#[from] lucene_codecs::vectors::Error),
    /// The IVF index of a `IvfVectorsFormat` group.
    #[error(transparent)]
    Ivf(#[from] lucene_ivf::Error),
    /// The disk files of a `DiskHnswVectorsFormat` instance.
    #[error(transparent)]
    DiskHnsw(#[from] lucene_disk_hnsw::Error),
    #[error(transparent)]
    DocValuesWrite(#[from] lucene_codecs::doc_values::WriteError),
    #[error(transparent)]
    NormsWrite(#[from] lucene_codecs::norms::WriteError),
    #[error(transparent)]
    Blocktree(#[from] lucene_codecs::blocktree::Error),
    #[error(transparent)]
    PostingsWrite(#[from] lucene_codecs::postings_writer::Error),
    /// A `MergeSource`'s stored fields referenced a field number absent from
    /// that same source's own `field_infos` -- an inconsistent/malformed
    /// `MergeSource` (its `reader` and `field_infos` don't actually describe
    /// the same segment), not something a well-formed caller can trigger.
    #[error(
        "source segment's stored field number {field_number} has no entry in that source's own field_infos"
    )]
    UnknownSourceFieldNumber { field_number: i32 },
    /// A source's `.liv` bitset and its stored-fields `maxDoc` disagree about
    /// how many documents the segment has.
    ///
    /// The two come from different files, and `FixedBitSet::get` bounds
    /// nothing at runtime -- a short bitset would silently answer "live" or
    /// "deleted" for a ghost bit past its end, or panic outright once the doc
    /// id is a word past it. Reported rather than clamped: a merge that
    /// guessed here would write a segment containing documents that were
    /// deleted, or missing documents that were not.
    /// `OneMerge.reorder`'s doc map is not one the merge can apply.
    #[error("reorder: {0}")]
    InvalidReorder(String),
    /// A [`SortedNumericRemap`] met a value outside its map: Java's
    /// `ArrayIndexOutOfBoundsException` (or `Math.toIntExact`'s
    /// `ArithmeticException`) in `OrdinalMappingLeafReader`.
    #[error("sorted-numeric remap: {0}")]
    InvalidRemap(String),
    /// One merged term holds more than `u32::MAX` positions: past what a
    /// merge materializing a term's postings can index.
    #[error("merged term has more than {} positions", u32::MAX)]
    MergedTermTooLarge,
    #[error(
        "source {source_index}: live docs cover {live_docs_len} documents but the segment's maxDoc is {max_doc}"
    )]
    LiveDocsLengthMismatch {
        source_index: usize,
        max_doc: i32,
        live_docs_len: usize,
    },
    /// A sorted merge was asked for with no sort tiers. `numSortFields == 0`
    /// *is* "unsorted" on disk, so an empty sort has no encoding distinct
    /// from `None` -- the same rule `IndexWriter::set_index_sort` applies.
    #[error("a sorted merge needs at least one sort field (pass None for an unsorted merge)")]
    EmptySortFields,
    /// A [`MergeSortKeySpec`]'s key table is the wrong shape: it must have one
    /// entry per source, and each source's slice one entry per document of
    /// *that* source (`source.reader.max_doc()`), because it is indexed by the
    /// source's own pre-merge doc id.
    #[error(
        "sort field {field:?}: {} has {found} entries, expected {expected}",
        match source_index {
            Some(i) => format!("source {i}'s key slice"),
            None => "per_source_keys".to_string(),
        }
    )]
    SortKeysWrongLength {
        field: String,
        /// `None` when the outer list is the wrong length.
        source_index: Option<usize>,
        expected: usize,
        found: usize,
    },
    /// A field has norms in at least one source that contributes live docs,
    /// but not in every such source (or not for every one of that source's
    /// live docs) -- see this module's doc comment on the "sparse across
    /// sources" rule.
    #[error(
        "merged field number {merged_field_number} has norms in some sources but not in every source that contributes live docs (or not for every one of that source's live docs)"
    )]
    NormsFieldMissingInSource { merged_field_number: i32 },
    /// A source's SORTED column gives a document an ordinal outside that
    /// source's own term dictionary -- a corrupt source, since the merge
    /// resolves each ordinal to its bytes through that dictionary.
    #[error(
        "merged field number {merged_field_number}: a source's sorted doc values name an ordinal outside its own dictionary"
    )]
    SortedDocValuesFieldMissingInSource { merged_field_number: i32 },
    /// Same as [`Error::SortedDocValuesFieldMissingInSource`], for
    /// SORTED_SET.
    #[error(
        "merged field number {merged_field_number}: a source's sorted-set doc values name an ordinal outside its own dictionary"
    )]
    SortedSetDocValuesFieldMissingInSource { merged_field_number: i32 },
    /// A field's merged `index_options` indexes positions, but a source
    /// contributing live docs for it wasn't given an opened `.pos` reader
    /// (`SourcePostings::pos_in`) -- a caller-side wiring inconsistency,
    /// a caller wiring bug rather
    /// than a panic, since a real, well-formed segment for this field always
    /// has a `.pos` file to open.
    #[error(
        "merged field number {merged_field_number} indexes positions but source index {source_index} has no opened .pos reader"
    )]
    PostingsPositionsInputMissingInSource {
        merged_field_number: i32,
        source_index: usize,
    },
    /// This port's postings merge handles `IndexOptions::Docs`,
    /// `IndexOptions::DocsAndFreqs`, `IndexOptions::DocsAndCustomFreqs`
    /// (wire-identical to `DocsAndFreqs` -- see `lucene_codecs::postings`'s
    /// doc comment), `IndexOptions::DocsAndFreqsAndPositions`, and
    /// `IndexOptions::DocsAndFreqsAndPositionsAndOffsets` -- any other
    /// `IndexOptions` variant is rejected here.
    #[error(
        "merging postings for merged field number {merged_field_number} isn't supported: index_options {index_options:?} isn't one of IndexOptions::Docs/DocsAndFreqs/DocsAndCustomFreqs/DocsAndFreqsAndPositions/DocsAndFreqsAndPositionsAndOffsets"
    )]
    PostingsIndexOptionsNotSupported {
        merged_field_number: i32,
        index_options: IndexOptions,
    },
    /// One source's `field_infos` names the same field twice. Real Lucene's
    /// `FieldInfos(FieldInfo[])` constructor throws "duplicate field names"
    /// for exactly this, and so does this port's `field_infos::parse` -- but
    /// [`MergeSource::field_infos`] is a caller-supplied slice that may be
    /// hand-built, and a duplicate would otherwise silently lose one of the
    /// two field numbers when the map is inverted for the postings/points
    /// merge.
    #[error("source index {source_index} names the field {field_name:?} more than once")]
    DuplicateFieldNameInSource {
        field_name: String,
        source_index: usize,
    },
    /// Port of `FieldInfo.verifySameIndexOptions`, raised by
    /// [`reconcile_field_numbers`] when two sources name the same field with
    /// different `index_options`. Real Lucene's `FieldInfos.Builder.add`
    /// runs `verifySameSchema` for every field of every merged segment and
    /// throws `IllegalArgumentException` on a mismatch -- without it, the
    /// merged `.fnm` would silently record only the first-seen source's
    /// options while the merged postings carry another source's richer (or
    /// poorer) data.
    #[error(
        "merged field number {merged_field_number} has disagreeing index_options across sources: source claims {source_index_options:?} but the merged field is {merged_index_options:?}"
    )]
    PostingsIndexOptionsDisagreement {
        merged_field_number: i32,
        merged_index_options: IndexOptions,
        source_index_options: IndexOptions,
    },
    /// Port of the rest of `FieldInfo.verifySameSchema` -- everything
    /// [`Error::PostingsIndexOptionsDisagreement`] and
    /// [`Error::PointsShapeDisagreement`]/
    /// [`Error::PointsIndexDimsDisagreement`] don't already cover:
    /// `omit_norms` and `store_term_vectors` (both checked only when the
    /// field is indexed, matching Java's `if (this.indexOptions !=
    /// IndexOptions.NONE)` guard), `doc_values_type`,
    /// `doc_values_skip_index_type`, and the KNN vector options. `attribute`
    /// names the disagreeing `FieldInfo` field.
    ///
    /// Note `store_payloads` is deliberately *not* here: real Lucene ORs it
    /// (`FieldInfos.Builder.add`'s `if (fi.hasPayloads())
    /// curFi.setStorePayloads()`) rather than rejecting a mismatch, and so
    /// does [`reconcile_field_numbers`].
    #[error(
        "field {field_name:?} (merged number {merged_field_number}) has disagreeing {attribute} across sources: source claims {source_value} but the merged field is {merged_value}"
    )]
    FieldSchemaDisagreement {
        field_name: String,
        merged_field_number: i32,
        attribute: &'static str,
        merged_value: String,
        source_value: String,
    },
    #[error(transparent)]
    Points(#[from] lucene_codecs::points::Error),
    /// A live-doc-contributing source *declares* this points field in its own
    /// `FieldInfos` but its [`MergeSource::points`] carries no entry for it (or
    /// its opened reader has no such field) -- a caller wiring bug.
    /// Deliberately stricter than Java (real `PointsWriter.merge` tolerates
    /// `values == null` for a declared field): unlike an indexed field, a
    /// points field cannot be declared by a segment without a value, since
    /// every points instance carries one. A source whose `FieldInfos` never saw
    /// the field at all is fine and simply contributes no points, matching
    /// `PointsWriter.merge`.
    #[error(
        "merged field number {merged_field_number} is declared by a live-doc-contributing source whose MergeSource supplied no BKD points for it"
    )]
    PointsFieldMissingInSource { merged_field_number: i32 },
    /// Field-number reconciliation only records the *first-seen* source's
    /// `FieldInfo` as the merged one (see `reconcile_field_numbers`) and
    /// never checks that every other live-doc-contributing source's own BKD
    /// tree shape (dimension count / bytes per dimension) agrees. Without
    /// this check, a source whose points use a different shape than the
    /// merged field's declared shape would either panic deep inside
    /// [`lucene_codecs::points::write_packed`] (wrong packed-value length) or, if
    /// the lengths happened to coincidentally match, silently produce a
    /// merged tree with garbage values -- so this is checked explicitly and
    /// rejected loudly instead.
    #[error(
        "merged field number {merged_field_number} has disagreeing BKD points shape across sources: source has num_dims={source_num_dims}/bytes_per_dim={source_bytes_per_dim}, but the merged field is num_dims={merged_num_dims}/bytes_per_dim={merged_bytes_per_dim}"
    )]
    PointsShapeDisagreement {
        merged_field_number: i32,
        merged_num_dims: i32,
        merged_bytes_per_dim: i32,
        source_num_dims: i32,
        source_bytes_per_dim: i32,
    },
    /// A source's points reader handed the merge a packed value whose length
    /// is not the field's `num_dims * bytes_per_dim`, after the shapes were
    /// checked equal. The reader decodes by that shape, so this is a reader
    /// defect or corruption, never a shape the sources disagree about.
    #[error(
        "merged field number {merged_field_number}: a source point is {actual} bytes, the field's shape needs {expected}"
    )]
    PointValueLength {
        merged_field_number: i32,
        expected: usize,
        actual: usize,
    },
    /// [`lucene_codecs::points::write_packed`] supports `num_index_dims <=
    /// num_dims` (data-only, non-indexed trailing dimensions), but every
    /// contributing source must agree on `num_index_dims` for a given merged
    /// field -- same reasoning as [`Error::PointsShapeDisagreement`] for
    /// `num_dims`/`bytes_per_dim`: field-number reconciliation only records
    /// the first-seen source's `FieldInfo`, so a source with a different
    /// `num_index_dims` would otherwise have its data-only/index-dim split
    /// silently reinterpreted against the wrong boundary.
    #[error(
        "merged field number {merged_field_number} has disagreeing BKD points num_index_dims across sources: source has num_index_dims={source_num_index_dims} (num_dims={num_dims}), but the merged field is num_index_dims={merged_num_index_dims}"
    )]
    PointsIndexDimsDisagreement {
        merged_field_number: i32,
        num_dims: i32,
        merged_num_index_dims: i32,
        source_num_index_dims: i32,
    },
    /// [`check_format_coverage`]: a source segment's own `.si` lists files
    /// for a format whose readers the caller did not put on that source's
    /// [`MergeSource`]. The merge would then run to completion and write a
    /// well-formed, checksummed, `CheckIndex`-clean segment with that
    /// format's data simply gone. Refused instead.
    #[error(
        "source segment `{segment}` has {format} files ({files}) that this merge never opened; merging it would silently drop them"
    )]
    MergeFormatNotOpened {
        segment: String,
        format: &'static str,
        files: String,
    },
    /// [`check_format_coverage`]: a source segment's `.si` lists a file whose
    /// extension no [`SegmentFormat`] claims and which is not one of the
    /// named non-format extensions. This is the gate's anti-rot arm: a flush
    /// path that learns to write a *new* format lands here on the first merge
    /// rather than dropping it silently, and the fix is to add the
    /// [`SegmentFormat`] variant -- whose two exhaustive `match`es then force
    /// the caller to open it.
    #[error(
        "source segment `{segment}` has file `{file}`, whose extension no merge format claims; add a `merge::SegmentFormat` variant for it (and open it in `IndexWriter::execute_merge`) or record it as a non-format extension"
    )]
    UnknownSegmentFormat { segment: String, file: String },
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
/// [`SourceNumericDocValues`], for [`NormsEntry`]/`.nvd` instead. `data` is
/// the whole `.nvd` file, including its header and footer, so a contributing
/// source's integrity can be checked before any merged output is created.
pub struct SourceNorms<'a> {
    pub data: &'a [u8],
    pub entry: NormsEntry,
}

/// One source's BINARY doc-values data for a single field -- same shape as
/// [`SourceNumericDocValues`], for [`BinaryEntry`]/`.dvd` instead.
pub struct SourceBinaryDocValues<'a> {
    pub data: &'a [u8],
    pub entry: BinaryEntry,
}

/// One source's SORTED doc-values data for a single field -- same shape as
/// [`SourceNumericDocValues`], for [`SortedEntry`]/`.dvd` instead. Unlike
/// NUMERIC/BINARY, `entry` also carries that source's own term dictionary
/// (`entry.terms`) -- needed because each source's dictionary is built
/// independently, so ordinal `N` in one source generally isn't the same term
/// as ordinal `N` in another (see [`merge_sorted_doc_values`]).
pub struct SourceSortedDocValues<'a> {
    pub data: &'a [u8],
    pub entry: SortedEntry,
}

/// One source's SORTED_NUMERIC doc-values data for a single field -- same
/// shape as [`SourceNumericDocValues`], for [`SortedNumericEntry`]/`.dvd`
/// instead. Each live doc resolves to a `Vec<i64>` of zero-or-more values via
/// [`doc_values::sorted_numeric_values`] (no shared dictionary to worry
/// about, unlike SORTED/SORTED_SET -- see [`merge_sorted_numeric_doc_values`]).
pub struct SourceSortedNumericDocValues<'a> {
    pub data: &'a [u8],
    pub entry: SortedNumericEntry,
}

/// One source's SORTED_SET doc-values data for a single field -- same shape
/// as [`SourceNumericDocValues`], for [`SortedSetEntry`]/`.dvd` instead.
/// `entry.kind` may be [`SortedSetKind::Single`] (this source happened to
/// collapse to one value per doc) or [`SortedSetKind::Multi`] (true
/// multi-valued) -- [`merge_sorted_set_doc_values`] handles both uniformly,
/// same "resolve each of this doc's own source's ordinals to term bytes via
/// that source's own dictionary" approach as [`merge_sorted_doc_values`],
/// just per-value instead of per-doc.
pub struct SourceSortedSetDocValues<'a> {
    pub data: &'a [u8],
    pub entry: SortedSetEntry,
}

/// One source's postings (term dictionary + doc/freq data) for a single
/// field -- `field_number` is that source's own original field number
/// (pre-merge, same convention as [`SourceNumericDocValues::entry`]'s
/// `field_number`). `field_terms` is that field's already-decoded term
/// dictionary (via [`lucene_codecs::blocktree::open`] +
/// [`lucene_codecs::blocktree::BlockTreeFields::field`]); `doc_in` is that
/// source's already-opened `.doc` file reader ([`DocInput::open`]), needed
/// to resolve any term whose `docFreq > 1` (`docFreq == 1` singleton terms
/// need no `.doc` bytes at all -- `None` is fine if every field in this
/// source's segment happens to have no `docFreq > 1` terms, though in
/// practice almost every real segment needs one).
///
/// # Positions/offsets/payloads
///
/// `pos_in`/`pay_in` are this source's already-opened `.pos`/`.pay` file
/// readers (via [`lucene_codecs::postings::PosInput::open`]/
/// [`lucene_codecs::postings::PayInput::open`]), needed by [`merge_term_range`]
/// whenever the merged field's `index_options` indexes positions
/// ([`IndexOptions::DocsAndFreqsAndPositions`] or
/// [`IndexOptions::DocsAndFreqsAndPositionsAndOffsets`]) -- both are `None`
/// for a `Docs`/`DocsAndFreqs`/`DocsAndCustomFreqs` field, exactly like
/// `doc_in` is `None` when a source never opens a `.doc` file. `pay_in` may
/// still be `None` even for a field with offsets/payloads if no term's
/// `total_term_freq` spans a full 256-position block (see
/// [`lucene_codecs::postings::read_positions`]'s doc comment).
pub struct SourcePostings<'a> {
    pub field_number: i32,
    pub field_terms: &'a FieldTerms,
    pub doc_in: Option<&'a DocInput<'a>>,
    pub pos_in: Option<&'a lucene_codecs::postings::PosInput<'a>>,
    pub pay_in: Option<&'a lucene_codecs::postings::PayInput<'a>>,
    /// Documents whose postings *for this field* the merge keeps, when that is
    /// narrower than the documents it keeps at all -- OpenSearch's
    /// `PrunePostingsMergePolicy`, which drops a soft-deleted document's `_id`
    /// postings while `SoftDeletesRetentionMergePolicy` still carries the
    /// document itself over for its history. `None` keeps every merged
    /// document's postings. A term left with no document is dropped, and a
    /// field left with no term is not written, as `BlockTreeTermsWriter` does.
    pub live: Option<&'a FixedBitSet>,
}

/// One source's BKD points (`.kdm`/`.kdi`/`.kdd`) data for a single field --
/// `field_number` is that source's own original field number (pre-merge,
/// same convention as [`SourcePostings::field_number`]). `reader` is that
/// source's already-opened [`lucene_codecs::points::PointsReader`] (via
/// [`lucene_codecs::points::open`]) -- the exact same read path
/// `lucene_search`'s points range query already uses, reused verbatim here
/// rather than re-deriving points decoding.
///
/// # Scope: one packed value shape per merged field, `num_index_dims <= num_dims`
///
/// [`lucene_codecs::points::write_packed`] supports `num_index_dims` less than
/// `num_dims` (data-only, non-indexed trailing dimensions), and this reader
/// hands back every point's full `num_dims`-wide packed value regardless --
/// [`merge_points`] independently checks every contributing source's own
/// `num_dims`/`bytes_per_dim`/`num_index_dims` against the merged field's
/// declared shape and rejects a disagreement with
/// [`Error::PointsShapeDisagreement`]/[`Error::PointsIndexDimsDisagreement`]
/// rather than silently truncating or corrupting the merged tree.
pub struct SourcePoints<'a> {
    pub field_number: i32,
    pub reader: &'a lucene_codecs::points::PointsReader<'a>,
}

/// One source segment's already-decoded input to a merge: its field infos
/// (from `.fnm`, via [`lucene_codecs::field_infos::parse`]), a stored-fields
/// reader over its `.fdt`/`.fdx`/`.fdm` (via [`stored_fields::open`]), an
/// optional live-docs bitset (`None` means "no deletions -- every doc up to
/// `reader.max_doc()` is live", matching a segment whose `SegmentCommitInfo`
/// has `del_gen == -1`), and optional per-field doc-values/norms/
/// term-vectors data (all empty/`None` by default -- a source with none of
/// these contributes only stored fields, same as a plain stored-fields merge).
pub struct MergeSource<'a> {
    pub field_infos: &'a [FieldInfo],
    pub reader: &'a lucene_codecs::stored_fields::StoredFieldsReader<'a>,
    pub live_docs: Option<&'a FixedBitSet>,
    /// This source's numeric doc-values fields, if any (see this module's
    /// doc comment: at most one distinct field across *all* sources may
    /// have numeric doc-values data in one merge call).
    pub numeric_doc_values: &'a [SourceNumericDocValues<'a>],
    /// This source's BINARY doc-values fields, if any (same one-field-
    /// across-all-sources limit as `numeric_doc_values`).
    pub binary_doc_values: &'a [SourceBinaryDocValues<'a>],
    /// This source's SORTED doc-values fields, if any (same one-field-
    /// across-all-sources limit as `numeric_doc_values`).
    pub sorted_doc_values: &'a [SourceSortedDocValues<'a>],
    /// This source's SORTED_NUMERIC doc-values fields, if any (same
    /// one-field-across-all-sources limit as `numeric_doc_values`).
    pub sorted_numeric_doc_values: &'a [SourceSortedNumericDocValues<'a>],
    /// This source's SORTED_SET doc-values fields, if any (same
    /// one-field-across-all-sources limit as `numeric_doc_values`).
    pub sorted_set_doc_values: &'a [SourceSortedSetDocValues<'a>],
    /// This source's norms fields, if any (same one-field-across-all-sources
    /// limit as `numeric_doc_values`).
    pub norms: &'a [SourceNorms<'a>],
    /// This source's term-vectors reader, or `None` if this source has no
    /// term vectors at all (every doc then contributes an empty
    /// [`TermVectorsDocument`]).
    pub term_vectors: Option<&'a TermVectorsReader<'a>>,
    /// This source's postings (term dictionary + doc/freq data) fields, if
    /// any -- unlike doc-values/norms, [`postings_writer::write_fields`]
    /// already supports any number of fields per call, so there is no
    /// single-field-per-merge-call limit here (see [`SourcePostings`] for
    /// the Docs/DocsAndFreqs-only scope of what gets merged).
    pub postings: &'a [SourcePostings<'a>],
    /// This source's BKD points fields, if any -- like postings,
    /// [`lucene_codecs::points::write_packed`] already supports any number of
    /// fields per call, so there is no single-field-per-merge-call limit
    /// here (see [`SourcePoints`] for the exact scope of what gets merged).
    pub points: &'a [SourcePoints<'a>],
    /// This source's KNN vector readers, or `None` if this source has no
    /// vectors at all. One pair of readers covers every vector field of the
    /// segment, so unlike postings/points this is not a per-field list (see
    /// [`SourceVectors`]).
    pub vectors: Option<&'a SourceVectors<'a>>,
    /// This source segment's `SegmentInfo.minVersion` -- Java's
    /// `LeafMetaData.minVersion()`, which a `SegmentReader` reads straight
    /// off the source's own `.si`.
    ///
    /// `SegmentMerger`'s constructor folds these into the merged segment's:
    /// `minVersion = Version.LATEST; for each reader { if its minVersion is
    /// null, the merged one is null; else keep the smaller }`. It is the
    /// oldest version that ever wrote *any* of the data in the merged
    /// segment, so a caller cannot recover it from the merge's own writer
    /// version -- and getting it wrong makes a segment claim it was never
    /// touched by the older Lucene whose bytes it is still carrying, which
    /// is exactly the fact back-compat and upgrade tooling keys on.
    ///
    /// `None` is Java's `null`: a segment that records no minVersion at all
    /// (written before Lucene 7). One such source makes the merged segment's
    /// `minVersion` null too, by Java's rule -- and `check_index`'s
    /// `commit.segment_records_min_version` then reports it for an index
    /// created at major 7 or later, exactly as `SegmentInfos.applyMergeChanges`
    /// refuses it in Java.
    pub min_version: Option<LuceneVersion>,
    /// This source segment's `SegmentInfo.getHasBlocks()` -- Java's
    /// `LeafMetaData.hasBlocks()`, i.e. "this segment contains at least one
    /// run of documents added together by `addDocuments`/`updateDocuments`
    /// and guaranteed to occupy contiguous doc ids".
    ///
    /// The merged segment carries it iff **any** source does
    /// (`IndexWriter.mergeMiddle`'s loop over `merge.segments`, and
    /// `SlowCompositeCodecReaderWrapper`/`ParallelLeafReader`'s
    /// `hasBlocks |= readerMeta.hasBlocks()`). Concatenation preserves each
    /// source's own document order, so its blocks stay contiguous; a
    /// *sorted* merge would shred them, which is why a block plus an index
    /// sort is refused at flush (`IndexWriter::set_index_sort`) and so cannot reach here.
    ///
    /// Losing the flag is silent: the merged segment reads back perfectly and
    /// every parent/child join query against it is quietly invalid.
    pub has_blocks: bool,
    /// A stored field, and the NUMERIC doc-values field of the same name,
    /// this merge drops from some of this source's documents -- see
    /// [`PrunedField`]. `None` keeps every field of every merged document.
    pub pruned_field: Option<PrunedField<'a>>,
    /// NUMERIC doc-values fields this merge rewrites on some of this
    /// source's documents -- see [`SentinelValues`]. `None` copies every
    /// value as it is.
    pub numeric_floor: Option<SentinelValues<'a>>,
}

/// Documents whose NUMERIC doc values a merge rewrites to fixed values:
/// OpenSearch's `_seq_no` and `_primary_term` of every document soft-deletes
/// retention no longer needs, as `UNASSIGNED_SEQ_NO` and
/// `UNASSIGNED_PRIMARY_TERM` (Elasticsearch's
/// `index.disable_sequence_numbers`). A document without a value keeps none.
#[derive(Clone, Copy, Debug)]
pub struct SentinelValues<'a> {
    /// The documents rewritten (this source's doc ids).
    pub docs: &'a FixedBitSet,
    /// Each rewritten field's number in this source, and its new value.
    pub fields: &'a [(i32, i64)],
}

impl SentinelValues<'_> {
    /// `value` of the field numbered `field_number` (in this source) on
    /// `doc`, as the merge writes it.
    fn apply(&self, field_number: i32, doc: i32, value: i64) -> i64 {
        match self.fields.iter().find(|(n, _)| *n == field_number) {
            Some(&(_, replacement)) if self.docs.get_doc(doc) => replacement,
            _ => value,
        }
    }
}

/// OpenSearch's `RecoverySourcePruneMergePolicy`: `_recovery_source` -- a
/// stored copy of the original source, flagged by a NUMERIC doc-values field
/// of the same name -- is kept only for the documents soft-deletes retention
/// still needs, and every other merged document loses both.
#[derive(Clone, Copy)]
pub struct PrunedField<'a> {
    /// The field's number in this source (pre-merge).
    pub field_number: i32,
    /// The documents that keep it; `None` drops it from every document.
    pub keep: Option<&'a FixedBitSet>,
}

impl PrunedField<'_> {
    /// Whether `doc`'s value for the field numbered `field_number` (in this
    /// source) is dropped.
    fn drops(&self, field_number: i32, doc: i32) -> bool {
        field_number == self.field_number && !self.keep.is_some_and(|keep| keep.get_doc(doc))
    }
}

impl<'a> MergeSource<'a> {
    /// Convenience constructor for the common "stored fields only" case
    /// (matches this module's original, pre-doc-values/norms/term-vectors
    /// shape) -- avoids every existing caller having to spell out the
    /// empty/`None` per-format fields.
    ///
    /// `min_version` is **not** defaulted, unlike everything else here, and
    /// that is deliberate. It is this source's own
    /// `SegmentInfo.minVersion`, and `merge_segments` propagates a `None`
    /// straight into the merged `.si` (Java's rule -- see
    /// [`MergeSource::min_version`]). A merged segment with no `minVersion` is
    /// one **real Lucene refuses to open** for an index created at major 7 or
    /// later: `SegmentInfos.readCommit` throws `CorruptIndexException`, and
    /// both `SegmentInfos.write` and `applyMergeChanges` throw
    /// `IllegalStateException`. Defaulting it would make "the caller said
    /// nothing" produce an unopenable index by omission, so the caller has to
    /// say. Pass `Some(v)` for a segment written at version `v` -- which is
    /// what every segment this port flushes records -- and `None` only for a
    /// genuine pre-Lucene-7 segment.
    ///
    /// `has_blocks` *is* defaulted, to `false`, because that is both Java's
    /// default and the safe direction for it: the flag is viral and can only
    /// be set, and a merge that has not been told about blocks has no way to
    /// invent them.
    pub fn stored_only(
        field_infos: &'a [FieldInfo],
        reader: &'a lucene_codecs::stored_fields::StoredFieldsReader<'a>,
        live_docs: Option<&'a FixedBitSet>,
        min_version: Option<LuceneVersion>,
    ) -> Self {
        Self {
            field_infos,
            reader,
            live_docs,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &[],
            vectors: None,
            min_version,
            has_blocks: false,
            pruned_field: None,
            numeric_floor: None,
        }
    }
}

/// One per-segment Lucene format, as the pair of facts a merge needs about
/// it: **which file extensions it owns**, and **whether a
/// [`MergeSource`] carries its opened readers**.
///
/// # Why this type exists
///
/// Something must check mechanically that `execute_merge` opens every format
/// the flush can write, because the failure is silent by construction. A format
/// the merge does not open contributes nothing; [`describe_written_files`] then
/// clears the corresponding capability off the merged `.fnm` so the segment
/// stays
/// *openable*; and what is left is well-formed, checksummed and
/// `CheckIndex`-clean, with the data gone or (for norms) wrong. Nothing
/// downstream reports it.
///
/// [`check_format_coverage`] closes that loop mechanically. See its doc
/// comment for the two properties that make it hard to rot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SegmentFormat {
    /// `Lucene90CompressingStoredFieldsFormat`.
    StoredFields,
    /// `Lucene90CompressingTermVectorsFormat`.
    TermVectors,
    /// `Lucene104PostingsFormat` + `Lucene103BlockTreeTermsFormat`.
    Postings,
    /// `Lucene90DocValuesFormat`.
    DocValues,
    /// `Lucene90NormsFormat`.
    Norms,
    /// `Lucene90PointsFormat` (BKD).
    Points,
    /// `Lucene99HnswVectorsFormat` and its flat store.
    KnnVectors,
}

/// A file extension that is per-segment bookkeeping rather than a format's
/// data, with the reason it is legitimately absent from
/// [`check_format_coverage`]'s reckoning. Written as a table rather than a
/// `match` arm apiece so the reasons stay next to the names.
const NON_FORMAT_EXTENSIONS: &[(&str, &str)] = &[
    (
        "si",
        "the segment's own metadata; the merge writes a fresh one",
    ),
    (
        "fnm",
        "the field schema; the merge rebuilds it in `describe_written_files`",
    ),
    (
        "liv",
        "deletions; the merge resolves them by dropping the documents, so a \
         merged segment has none by design",
    ),
];

impl SegmentFormat {
    /// Every format, in the order [`merge_segments`] writes them. The array
    /// length is spelled out so adding a variant without extending this list
    /// is a compile error, not a silently shorter sweep.
    pub const ALL: [SegmentFormat; 7] = [
        SegmentFormat::StoredFields,
        SegmentFormat::TermVectors,
        SegmentFormat::Postings,
        SegmentFormat::DocValues,
        SegmentFormat::Norms,
        SegmentFormat::Points,
        SegmentFormat::KnnVectors,
    ];

    /// This format's name, for error messages.
    pub fn name(self) -> &'static str {
        match self {
            SegmentFormat::StoredFields => "stored fields",
            SegmentFormat::TermVectors => "term vectors",
            SegmentFormat::Postings => "postings",
            SegmentFormat::DocValues => "doc values",
            SegmentFormat::Norms => "norms",
            SegmentFormat::Points => "points",
            SegmentFormat::KnnVectors => "KNN vectors",
        }
    }

    /// Every file extension this format owns, including the optional ones
    /// (`.pos`/`.pay` exist only for a positional field; `.vem`/`.vex` only
    /// above `HNSW_GRAPH_THRESHOLD`). Extensions are matched on the segment
    /// file name's suffix after the last `.`, which is also how
    /// `IndexFileNames.getExtension` does it -- per-field suffixed names like
    /// `_0_Lucene104_0.doc` end in the same extension as an unsuffixed one.
    pub fn extensions(self) -> &'static [&'static str] {
        match self {
            SegmentFormat::StoredFields => &["fdt", "fdx", "fdm"],
            SegmentFormat::TermVectors => &["tvd", "tvx", "tvm"],
            // `.lkp`/`.cmp` are `Completion104`'s suggester files: its
            // `merge` rebuilds them from the merged postings' payloads, so
            // the postings being opened is what they need.
            // Another terms dictionary's files (`BlockTreeOrds`' `.tio`, ...)
            // are claimed through `lucene_codecs::terms_dictionary` in
            // `for_extension`: `per_field_postings::open_groups` reads them
            // with the rest.
            SegmentFormat::Postings => &[
                "tim", "tip", "tmd", "doc", "psm", "pos", "pay", "lkp", "cmp",
            ],
            SegmentFormat::DocValues => &["dvd", "dvm", "dvs", "dvp"],
            SegmentFormat::Norms => &["nvd", "nvm"],
            SegmentFormat::Points => &["kdd", "kdi", "kdm"],
            // `.vemq`/`.veq` and `.vemb`/`.veb`: the retired quantized formats' codes
            // (Lucene 9.9-10.3), which a merge reads past to the raw `.vec`;
            // `.ivm`/`.ivp`/`.ivv`: `IvfVectorsFormat`'s IVF index;
            // `.vqm`/`.vqd`: `HnswQuantizedVectorsFormat`'s codes;
            // `.dhm`/`.dhc`/`.dhn`: `DiskHnswVectorsFormat`'s disk layout.
            SegmentFormat::KnnVectors => &[
                "vec", "vemf", "vem", "vex", "vemq", "veq", "vemb", "veb", "ivm", "ivp", "ivv",
                "vqm", "vqd", "dhm", "dhc", "dhn",
            ],
        }
    }

    /// The format that owns `extension`, or `None` when no format claims it.
    pub fn for_extension(extension: &str) -> Option<SegmentFormat> {
        SegmentFormat::ALL
            .into_iter()
            .find(|f| f.extensions().contains(&extension))
            .or_else(|| {
                lucene_codecs::terms_dictionary::all()
                    .iter()
                    .any(|d| d.extensions().contains(&extension))
                    .then_some(SegmentFormat::Postings)
            })
    }

    /// Whether `source` carries this format's opened readers -- i.e. whether
    /// the caller opened it.
    ///
    /// `StoredFields` is unconditionally `true` because [`MergeSource`]'s
    /// `reader` is not optional: a source without stored fields cannot be
    /// constructed at all.
    pub fn is_opened(self, source: &MergeSource<'_>) -> bool {
        match self {
            SegmentFormat::StoredFields => true,
            SegmentFormat::TermVectors => source.term_vectors.is_some(),
            SegmentFormat::Postings => !source.postings.is_empty(),
            SegmentFormat::DocValues => {
                !source.numeric_doc_values.is_empty()
                    || !source.binary_doc_values.is_empty()
                    || !source.sorted_doc_values.is_empty()
                    || !source.sorted_numeric_doc_values.is_empty()
                    || !source.sorted_set_doc_values.is_empty()
            }
            SegmentFormat::Norms => !source.norms.is_empty(),
            SegmentFormat::Points => !source.points.is_empty(),
            SegmentFormat::KnnVectors => source.vectors.is_some(),
        }
    }
}

/// **The merge-completeness gate.** For every source, every format its own
/// `.si` lists files for must be a format the caller opened onto that
/// source's [`MergeSource`].
///
/// `source_files` is one entry per source, in the same order as `sources`,
/// holding that source's `SegmentInfo::files` -- the segment's *own* claim
/// about what it has, which is also what `IndexFileDeleter` reference-counts
/// and what `CheckIndex` walks, so there is no third place for a format to
/// hide.
///
/// `postings_opened` is, per source, whether the caller opened that source's
/// term dictionary at all. It is needed because a routed field outlives its
/// last term (see `merge_postings`), so an opened dictionary can hand the
/// merge no [`SourcePostings`] entry and still be complete.
///
/// # What makes it hard to rot
///
/// Two properties, and neither is a hand-maintained list of "the formats we
/// have today":
///
/// 1. **Unknown extensions are an error.** Any extension in a source's `.si`
///    that no [`SegmentFormat`] claims and that is not in
///    [`NON_FORMAT_EXTENSIONS`] fails with [`Error::UnknownSegmentFormat`].
///    So a flush path that learns to write a *new* format is caught by this
///    gate on the very first merge of a segment it wrote -- before anything
///    can be dropped -- and the only way to satisfy it is to add a
///    [`SegmentFormat`] variant. That variant then forces
///    [`SegmentFormat::extensions`] and [`SegmentFormat::is_opened`] (two
///    exhaustive `match`es) and [`SegmentFormat::ALL`] (a fixed-length
///    array), and `is_opened` cannot be written without a [`MergeSource`]
///    field to read -- which cannot be populated without the caller opening
///    the format.
/// 2. **It runs on every merge, not in one test.** It is called from
///    `IndexWriter::execute_merge` on the real path, so every existing and
///    future merge test exercises it over whatever formats that test's
///    segments happen to have.
///
/// # What it deliberately does not check, and cannot
///
/// - **That an opened format is merged *correctly*.** A format whose doc map
///   is wrong produces exactly the same well-formed segment; that is what
///   `assert_every_format_agrees` and `VerifySortedSegment` are for. This
///   gate is "opened at all".
/// - **A format the flush writes without listing it in the `.si`.** Such a
///   file is already invisible to `IndexFileDeleter` and to `CheckIndex`, so
///   it is a different (and louder) defect.
/// - **A format opened per source but dropped per *field*.** For example a
///   per-type drop inside an opened `.dvm`; the `debug_assert!` in
///   `execute_merge` covers the doc-values case specifically, and this gate
///   covers the coarser "not opened at all".
/// - **Compound segments.** `IndexWriter` always flushes loose files
///   (`use_compound_file: false`), so a `.cfs`/`.cfe` source never reaches
///   here; if one ever did it would be an unknown extension, which is the
///   right answer -- its formats are not visible in the `.si` at all.
pub fn check_format_coverage(
    source_names: &[&str],
    source_files: &[&[String]],
    sources: &[MergeSource<'_>],
    postings_opened: &[bool],
) -> Result<()> {
    debug_assert_eq!(source_names.len(), sources.len());
    debug_assert_eq!(source_files.len(), sources.len());
    debug_assert_eq!(postings_opened.len(), sources.len());
    for (((name, files), source), &postings_opened) in source_names
        .iter()
        .zip(source_files)
        .zip(sources)
        .zip(postings_opened)
    {
        // Which formats this source's own `.si` says it has, and one example
        // file per format for the message.
        let mut present: Vec<(SegmentFormat, Vec<&str>)> = Vec::new();
        for file in files.iter() {
            let Some((_, extension)) = file.rsplit_once('.') else {
                return Err(Error::UnknownSegmentFormat {
                    segment: (*name).to_string(),
                    file: file.clone(),
                });
            };
            if NON_FORMAT_EXTENSIONS.iter().any(|(e, _)| *e == extension) {
                continue;
            }
            let Some(format) = SegmentFormat::for_extension(extension) else {
                return Err(Error::UnknownSegmentFormat {
                    segment: (*name).to_string(),
                    file: file.clone(),
                });
            };
            match present.iter_mut().find(|(f, _)| *f == format) {
                Some((_, seen)) => seen.push(file),
                None => present.push((format, vec![file])),
            }
        }
        for (format, seen) in present {
            // A routed field survives a merge that removed its last term
            // (`merge_postings`' Java note), so a source can hold postings
            // files that its caller opened and that still name no term: an
            // empty `source.postings` alone does not mean "never opened".
            let opened_empty = format == SegmentFormat::Postings && postings_opened;
            if !opened_empty && !format.is_opened(source) {
                return Err(Error::MergeFormatNotOpened {
                    segment: (*name).to_string(),
                    format: format.name(),
                    files: seen.join(", "),
                });
            }
        }
    }
    Ok(())
}

/// [`reconcile_field_numbers`]' result: the merged field list, plus one
/// `original field number -> merged field number` map per source.
pub type ReconciledFieldNumbers = (Vec<FieldInfo>, Vec<HashMap<i32, i32>>);

/// Reconciles field numbering across `sources_fields` (one source's
/// [`FieldInfos`](field_infos::FieldInfos)-equivalent field list per entry):
/// assigns every distinct field *name* a single merged field number --
/// mirrors real Lucene's `FieldInfos.FieldNumbers.addOrGet`, which hands out
/// a writer-wide number per name and reuses it for every segment that has
/// that field.
///
/// A name keeps the number its first-seen source gave it, as long as no
/// earlier-seen name already holds that number. The sources of one index
/// share their writer's global numbering, so a segment holding only some of
/// the fields (`_0` with just field 3, as Lucene writes it) keeps its numbers
/// instead of being packed down to 0..n. That keeps the merged `.fnm` the
/// one Java writes, and it keeps every such source's map the identity, which
/// is what lets [`matching_readers`] pick the bulk stored-fields and
/// term-vectors copy. A name whose number is taken (two sources numbering
/// different fields alike) gets a fresh number past every kept one, in
/// first-seen order.
///
/// Returns the merged field list (one [`FieldInfo`] per distinct name, seeded
/// from the *first* source's metadata for that name, ordered by number as
/// `FieldInfos` iterates) and, per source, a map from that source's original
/// field number to the merged number.
///
/// # Cross-source schema reconciliation
///
/// This is also the port of `FieldInfos.Builder.add`'s second half, which
/// real Lucene runs for every `FieldInfo` of every segment being merged:
/// - **`verifySameSchema`** -- a second source naming an already-seen field
///   must agree on `index_options`
///   ([`Error::PostingsIndexOptionsDisagreement`]), on `omit_norms` and
///   `store_term_vectors` when the field is indexed, on `doc_values_type`
///   and `doc_values_skip_index_type`, and on the KNN vector options
///   ([`Error::FieldSchemaDisagreement`]); points dimensions get their own
///   [`Error::PointsShapeDisagreement`]/[`Error::PointsIndexDimsDisagreement`]
///   variants, shared with the BKD-shape check [`merge_points`] performs
///   against each source's actual on-disk tree.
/// - **`setStorePayloads`** -- `store_payloads` is one of the two attributes
///   Java *merges* rather than verifies (`if (fi.hasPayloads())
///   curFi.setStorePayloads()`), so the merged field stores payloads if any
///   source does. That is exactly right for the postings merge:
///   `Position::payload` comes back empty for a source that has none, and
///   the postings writer treats an empty payload as "no payload at this
///   occurrence". `FieldInfo.setStorePayloads`' own
///   `indexOptions.subsumes(DOCS_AND_FREQS_AND_POSITIONS)` guard is ported
///   too, so a source claiming payloads on a positionless field cannot push
///   the merged `FieldInfo` into a state `checkConsistency` rejects.
/// - **`putAttributes`** -- the other merged (not verified) attribute: a
///   later source's `attributes` are `Map.putAll`'d onto the merged field,
///   so its value wins for a shared key and its own keys are added.
///
/// Without this, a merged `.fnm` would record only the first-seen source's
/// view of a field while the merged data files carried another source's --
/// e.g. claiming a field has no term vectors while `write_merged_term_vectors`
/// happily wrote them, or dropping a later source's real payload bytes.
pub fn reconcile_field_numbers(sources_fields: &[&[FieldInfo]]) -> Result<ReconciledFieldNumbers> {
    let numbers = assign_merged_field_numbers(sources_fields);
    let mut merged_fields: Vec<FieldInfo> = Vec::new();
    // Name -> (merged number, position in `merged_fields`).
    let mut name_to_merged: HashMap<String, (i32, usize)> = HashMap::new();
    let mut per_source_maps: Vec<HashMap<i32, i32>> = Vec::with_capacity(sources_fields.len());

    for (source_index, fields) in sources_fields.iter().enumerate() {
        let mut map = HashMap::with_capacity(fields.len());
        // Java's `FieldInfos(FieldInfo[])` constructor rejects a segment that
        // names the same field twice; `field_infos::parse` does too, but
        // `MergeSource::field_infos` is a caller-supplied slice that may be
        // hand-built, and letting a duplicate through would silently drop one
        // of the two numbers when the map is inverted for the postings/points
        // merge.
        let mut seen_in_source: HashSet<&str> = HashSet::with_capacity(fields.len());
        for f in *fields {
            if !seen_in_source.insert(f.name.as_str()) {
                return Err(Error::DuplicateFieldNameInSource {
                    field_name: f.name.clone(),
                    source_index,
                });
            }
            match name_to_merged.get(&f.name) {
                Some(&(merged_number, position)) => {
                    let merged = &mut merged_fields[position];
                    verify_same_schema(merged, f)?;
                    // FieldInfos.Builder.add: payloads are ORed in, not
                    // verified -- but through `FieldInfo.setStorePayloads`,
                    // which is itself guarded by
                    // `indexOptions.subsumes(DOCS_AND_FREQS_AND_POSITIONS)`
                    // and would otherwise trip `checkConsistency`'s "indexed
                    // field cannot have payloads without positions". Without
                    // the guard a hand-built `MergeSource` claiming payloads
                    // on a `Docs` field would write a `.fnm` real Lucene's
                    // `FieldInfo` constructor rejects.
                    if f.store_payloads && subsumes_positions(merged.index_options) {
                        merged.store_payloads = true;
                    }
                    // FieldInfos.Builder.add: `curFi.putAttributes(fi.attributes())`
                    // -- a `Map.putAll`, so a later source's value wins for a
                    // key both declare, and keys only a later source has are
                    // added.
                    for (key, value) in &f.attributes {
                        match merged.attributes.iter_mut().find(|(k, _)| k == key) {
                            Some(existing) => existing.1 = value.clone(),
                            None => merged.attributes.push((key.clone(), value.clone())),
                        }
                    }
                    map.insert(f.number, merged_number);
                }
                None => {
                    let number = numbers.get(f.name.as_str()).copied().unwrap_or(f.number);
                    let mut renumbered = f.clone();
                    renumbered.number = number;
                    name_to_merged.insert(f.name.clone(), (number, merged_fields.len()));
                    merged_fields.push(renumbered);
                    map.insert(f.number, number);
                }
            }
        }
        per_source_maps.push(map);
    }

    merged_fields.sort_by_key(|f| f.number);
    Ok((merged_fields, per_source_maps))
}

/// [`reconcile_field_numbers`]' numbering: each name's first-seen number
/// when no earlier name holds it, otherwise a fresh one past every number
/// kept. A source naming one field twice is left to the caller's check.
fn assign_merged_field_numbers<'a>(sources_fields: &[&'a [FieldInfo]]) -> HashMap<&'a str, i32> {
    let mut numbers: HashMap<&str, i32> = HashMap::new();
    let mut taken: HashSet<i32> = HashSet::new();
    let mut displaced: Vec<&str> = Vec::new();
    for f in sources_fields.iter().flat_map(|fields| fields.iter()) {
        if numbers.contains_key(f.name.as_str()) || displaced.contains(&f.name.as_str()) {
            continue;
        }
        if f.number >= 0 && taken.insert(f.number) {
            numbers.insert(f.name.as_str(), f.number);
        } else {
            displaced.push(f.name.as_str());
        }
    }
    let mut next = taken.iter().max().map_or(0, |&max| max.saturating_add(1));
    for name in displaced {
        while taken.contains(&next) {
            next = next.saturating_add(1);
        }
        taken.insert(next);
        numbers.insert(name, next);
    }
    numbers
}

/// Port of `IndexOptions.subsumes(DOCS_AND_FREQS_AND_POSITIONS)`: does this
/// option index positions? (`DocsAndCustomFreqs` deliberately does not --
/// Java special-cases it to subsume as if it were `DocsAndFreqs`.)
/// `lucene_codecs`' own `IndexOptions::subsumes_positions` is `pub(crate)`
/// there, so this mirrors it rather than widening that crate's API.
fn subsumes_positions(index_options: IndexOptions) -> bool {
    matches!(
        index_options,
        IndexOptions::DocsAndFreqsAndPositions | IndexOptions::DocsAndFreqsAndPositionsAndOffsets
    )
}

/// Port of `FieldInfo.verifySameSchema` -- every attribute two segments
/// sharing a field *name* must agree on before they can be merged. Ordered
/// exactly as Java orders its `verifySame*` calls so the first disagreement
/// reported is the same one Java would report.
fn verify_same_schema(merged: &FieldInfo, source: &FieldInfo) -> Result<()> {
    if merged.index_options != source.index_options {
        return Err(Error::PostingsIndexOptionsDisagreement {
            merged_field_number: merged.number,
            merged_index_options: merged.index_options,
            source_index_options: source.index_options,
        });
    }
    let disagreement = |attribute: &'static str, merged_value: String, source_value: String| {
        Error::FieldSchemaDisagreement {
            field_name: merged.name.clone(),
            merged_field_number: merged.number,
            attribute,
            merged_value,
            source_value,
        }
    };
    if merged.index_options != IndexOptions::None {
        if merged.omit_norms != source.omit_norms {
            return Err(disagreement(
                "omit_norms",
                merged.omit_norms.to_string(),
                source.omit_norms.to_string(),
            ));
        }
        if merged.store_term_vectors != source.store_term_vectors {
            return Err(disagreement(
                "store_term_vectors",
                merged.store_term_vectors.to_string(),
                source.store_term_vectors.to_string(),
            ));
        }
    }
    if merged.doc_values_type != source.doc_values_type {
        return Err(disagreement(
            "doc_values_type",
            format!("{:?}", merged.doc_values_type),
            format!("{:?}", source.doc_values_type),
        ));
    }
    if merged.doc_values_skip_index_type != source.doc_values_skip_index_type {
        return Err(disagreement(
            "doc_values_skip_index_type",
            format!("{:?}", merged.doc_values_skip_index_type),
            format!("{:?}", source.doc_values_skip_index_type),
        ));
    }
    if merged.point_dimension_count != source.point_dimension_count
        || merged.point_num_bytes != source.point_num_bytes
    {
        return Err(Error::PointsShapeDisagreement {
            merged_field_number: merged.number,
            merged_num_dims: merged.point_dimension_count,
            merged_bytes_per_dim: merged.point_num_bytes,
            source_num_dims: source.point_dimension_count,
            source_bytes_per_dim: source.point_num_bytes,
        });
    }
    if merged.point_index_dimension_count != source.point_index_dimension_count {
        return Err(Error::PointsIndexDimsDisagreement {
            merged_field_number: merged.number,
            num_dims: source.point_dimension_count,
            merged_num_index_dims: merged.point_index_dimension_count,
            source_num_index_dims: source.point_index_dimension_count,
        });
    }
    if merged.vector_dimension != source.vector_dimension {
        return Err(disagreement(
            "vector_dimension",
            merged.vector_dimension.to_string(),
            source.vector_dimension.to_string(),
        ));
    }
    if merged.vector_encoding != source.vector_encoding {
        return Err(disagreement(
            "vector_encoding",
            format!("{:?}", merged.vector_encoding),
            format!("{:?}", source.vector_encoding),
        ));
    }
    if merged.vector_similarity_function != source.vector_similarity_function {
        return Err(disagreement(
            "vector_similarity_function",
            format!("{:?}", merged.vector_similarity_function),
            format!("{:?}", source.vector_similarity_function),
        ));
    }
    Ok(())
}

/// Builds the "concatenate sources in order" doc-visit order every
/// `merge_*_doc_values`/`merge_norms`/`write_merged_term_vectors` helper walks for
/// [`merge_stored_only_segments`]: source 0's live docs (in ascending
/// original-doc-id order), then source 1's, etc. -- exactly the order
/// `merged_docs` itself is built in above. Factored out so the same
/// `merge_*` helpers can also be driven by [`merge_sorted_stored_only_segments`]'s
/// k-way-merge order instead, without duplicating each helper's field-
/// resolution/candidate logic per call site.
/// `SortingCodecReader.wrap(mergedView, docMap)` compacted by the merge: the
/// merged view's documents in `new_to_old` order, each a `(source, doc)`,
/// the deleted ones left out. `new_to_old` must be a permutation of the
/// merged view's `0..sum(maxDoc)` -- `Sorter.DocMap`'s own invariant, checked
/// here because a hook supplied it.
fn reordered_doc_order(
    new_to_old: &[i32],
    per_source_max_doc: &[i32],
    sources: &[MergeSource<'_>],
) -> Result<Vec<(usize, i32)>> {
    // Each merged-view document's source and its id there.
    let mut owner: Vec<(usize, i32)> = Vec::new();
    for (src, &max_doc) in per_source_max_doc.iter().enumerate() {
        owner.extend((0..max_doc.max(0)).map(|d| (src, d)));
    }
    if new_to_old.len() != owner.len() {
        return Err(Error::InvalidReorder(format!(
            "the doc map covers {} documents but the merged view has {}",
            new_to_old.len(),
            owner.len()
        )));
    }
    let mut seen = vec![false; owner.len()];
    let mut order = Vec::with_capacity(owner.len());
    for &old in new_to_old {
        let Some(at) = usize::try_from(old).ok().filter(|&o| o < owner.len()) else {
            return Err(Error::InvalidReorder(format!(
                "old document {old} is outside the merged view"
            )));
        };
        if std::mem::replace(&mut seen[at], true) {
            return Err(Error::InvalidReorder(format!(
                "old document {old} appears twice"
            )));
        }
        let (src, doc) = owner[at];
        // FBS: `doc < max_doc`, and the caller checked every source's
        // `bits.len() == max_doc` (`LiveDocsLengthMismatch`); bounded again
        // so a caller that did not is a dead document, not a ghost bit.
        let live = sources[src].live_docs.is_none_or(|bits| {
            let d = doc as usize;
            d < bits.len() && bits.get(d)
        });
        if live {
            order.push((src, doc));
        }
    }
    Ok(order)
}

fn concat_doc_order(per_source_live_ids: &[Vec<i32>]) -> Vec<(usize, i32)> {
    let mut order = Vec::with_capacity(per_source_live_ids.iter().map(Vec::len).sum());
    for (src_idx, live_ids) in per_source_live_ids.iter().enumerate() {
        for &doc_id in live_ids {
            order.push((src_idx, doc_id));
        }
    }
    order
}

/// Port of `codecs/compressing/MatchingReaders`: which source segments have a
/// field name -> number mapping that survives the merge *unchanged*, so their
/// stored fields (and, in Java, term vectors) can be bulk merged.
///
/// Java asks, per source, whether every one of its `FieldInfo`s finds a
/// merged `FieldInfo` at the *same number* with the same *name*. Here
/// [`reconcile_field_numbers`] has already produced that mapping explicitly,
/// so the same question is "is this source's map the identity?" -- a merged
/// number equal to the original one can only have come from the same name,
/// because merged numbers are unique per name.
///
/// It matters because both fast merge paths copy bytes that *encode field
/// numbers*: a chunk's compressed payload is a sequence of
/// `(fieldNumber << 3) | type` vints. Copying those for a source whose
/// fields were renumbered would produce a segment that reads back plausible
/// but wrong documents -- values landing under the wrong field name -- which
/// is precisely the failure mode Java's "bulk merge is scary" comment is
/// about.
fn matching_readers(sources: &[MergeSource], per_source_maps: &[HashMap<i32, i32>]) -> Vec<bool> {
    sources
        .iter()
        .zip(per_source_maps)
        .map(|(source, map)| {
            source
                .field_infos
                .iter()
                .all(|f| map.get(&f.number) == Some(&f.number))
        })
        .collect()
}

/// Which of `Lucene90CompressingStoredFieldsWriter`'s three merge paths a
/// source qualifies for -- see [`stored_fields::StoredFieldsWriter`] for what
/// each one costs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StoredFieldsMergeStrategy {
    /// Copy whole compressed chunks.
    Bulk,
    /// Copy serialized documents, without parsing them into fields.
    Doc,
    /// Parse every field and recompress.
    Visitor,
}

/// Port of `Lucene90CompressingStoredFieldsWriter.getMergeStrategy`.
///
/// Java also checks the reader's concrete class and format version; here
/// `stored_fields::open` already refuses anything that is not this port's one
/// `VERSION_CURRENT` in this port's one chunk format, so a `MergeSource`
/// cannot be holding a reader those checks would reject.
///
/// A writer with the high-compression LZ4 match finder on copies only the
/// chunks of a source written the same way (`source_lz4_hc`): any other
/// source is recompressed once, so a merged segment's chunks are all of one
/// kind.
fn stored_fields_merge_strategy(
    writer: &stored_fields::StoredFieldsWriter,
    source: &MergeSource,
    matching: bool,
    source_lz4_hc: bool,
) -> StoredFieldsMergeStrategy {
    // A pruning source hands the writer a filtering reader in Java, which is
    // never the matching compressing reader either.
    if !matching || source.pruned_field.is_some() {
        return StoredFieldsMergeStrategy::Visitor;
    }
    // "its not worth fine-graining this if there are deletions" -- plus the
    // compressor/chunk-size/dirtiness trio, which `can_bulk_copy` owns.
    if source.live_docs.is_none()
        && writer.can_bulk_copy(source.reader)
        && (source_lz4_hc || !writer.lz4_high_compression())
    {
        StoredFieldsMergeStrategy::Bulk
    } else {
        StoredFieldsMergeStrategy::Doc
    }
}

/// Writes the merged segment's `.fdt`/`.fdx`/`.fdm` by walking `doc_order`
/// -- the (source index, source doc id) pairs, in merged doc id order, that
/// both merge entry points already compute -- and picking
/// `Lucene90CompressingStoredFieldsWriter.merge`'s strategy per source.
///
/// This is the port of that `merge` method's `DocIDMerger` loop: a run of
/// consecutive documents from a BULK source is handed to `copyChunks` in one
/// call (Java's `while ((sub = docIDMerger.next()) == current)` run
/// detection), a DOC source's document is copied without being parsed, and
/// only a VISITOR source's document is materialised and renumbered.
///
/// For the unsorted merge every BULK source contributes exactly one run,
/// `0..maxDoc`, so every chunk it owns is copied verbatim. For the sorted
/// merge the runs are whatever the k-way merge produces, and `copy_chunks`
/// falls back to per-document copying at each run's ragged ends.
#[allow(clippy::too_many_arguments)]
fn write_merged_stored_fields(
    mode: stored_fields::Mode,
    lz4_high_compression: bool,
    lz4_hc_sources: &[bool],
    zstd_dictionary_min_bytes: u64,
    sources: &[MergeSource],
    per_source_maps: &[HashMap<i32, i32>],
    doc_order: &[(usize, i32)],
    merged_segment_id: &[u8; ID_LENGTH],
    fdt_out: lucene_store::FsIndexOutput,
) -> Result<(lucene_store::FsIndexOutput, Vec<u8>, Vec<u8>)> {
    let mut writer =
        stored_fields::StoredFieldsWriter::new(mode, merged_segment_id, mode.segment_suffix());
    writer.set_lz4_high_compression(lz4_high_compression);
    writer.stream_to(fdt_out);
    // Java's per-reader cached `BlockState`: the DOC path reads a run of
    // documents out of the same chunk, and decompressing that chunk once per
    // document instead of once per chunk is the difference between a merge
    // that costs O(documents) inflations and one that costs O(chunks).
    let mut chunk_cursors: Vec<stored_fields::ChunkCursor> = (0..sources.len())
        .map(|_| stored_fields::ChunkCursor::new())
        .collect();
    let matching = matching_readers(sources, per_source_maps);
    // Java runs `reader.checkIntegrity(mergeState.oneMerge)` on every
    // source before it picks a strategy, and it matters most for
    // BULK, which copies a source's compressed bytes verbatim
    // and then writes a freshly computed, valid footer over them, so a bit
    // flip in the source would be laundered into a merged segment that
    // passes every checksum from then on. `stored_fields::open` only
    // validates the footer's shape (`retrieve_checksum`), which is the
    // right trade for a random-access reader and not enough here.
    //
    // Each check hashes a whole source file, so the sources are checked in
    // parallel, reporting the first corrupt one in source order.
    {
        use rayon::prelude::*;
        in_source_order(
            sources
                .par_iter()
                .map(|source| Ok(source.reader.check_integrity()?)),
        )?;
    }
    if let stored_fields::Mode::ZstdPaged(_) = mode {
        let bytes: u64 = sources
            .iter()
            .map(|s| s.reader.max_pointer().max(0) as u64)
            .sum();
        if bytes >= zstd_dictionary_min_bytes {
            let readers: Vec<&stored_fields::StoredFieldsReader<'_>> =
                sources.iter().map(|s| s.reader).collect();
            if let Some(dict) = stored_fields::train_zstd_paged_dictionary(&readers)? {
                writer.set_zstd_dictionary(&dict);
            }
        }
    }
    let strategies: Vec<StoredFieldsMergeStrategy> = sources
        .iter()
        .zip(&matching)
        .enumerate()
        .map(|(i, (source, &m))| {
            let hc = lz4_hc_sources.get(i).copied().unwrap_or(false);
            stored_fields_merge_strategy(&writer, source, m, hc)
        })
        .collect();

    if is_scattered(doc_order, sources.len()) {
        write_scattered_stored_fields(
            &mut writer,
            sources,
            per_source_maps,
            &strategies,
            &mut chunk_cursors,
            doc_order,
        )?;
        return finish_streamed_stored(writer);
    }

    let mut i = 0usize;
    while i < doc_order.len() {
        let (src_idx, doc_id) = doc_order[i];
        match strategies[src_idx] {
            StoredFieldsMergeStrategy::Bulk => {
                // ARITH: every `doc_order` entry is a `(source index, doc id)`
                // pair `merge_segments` built out of that source's
                // `0..max_doc`, so `0 <= doc_id < max_doc <= i32::MAX` and
                // `doc_id + 1` cannot overflow. `to_doc` steps only while
                // `doc_order[j]` *is* `(src_idx, to_doc)`, i.e. while `to_doc`
                // is itself one of those doc ids, so the same bound holds on
                // every step. `i` and `j` are indices into `doc_order`, a
                // `Vec` whose length is at most `isize::MAX`.
                #[allow(clippy::arithmetic_side_effects)]
                let (to_doc, j) = {
                    let mut to_doc = doc_id + 1;
                    let mut j = i + 1;
                    while j < doc_order.len() && doc_order[j] == (src_idx, to_doc) {
                        to_doc += 1;
                        j += 1;
                    }
                    (to_doc, j)
                };
                // The cursor is the caller-owned equivalent of Java's
                // per-reader cached `BlockState`: an index-sorted merge
                // interleaves the sources, so a BULK source's runs are a
                // document or two and every run's ragged ends read out of a
                // partially-copied chunk. Without the shared cursor each of
                // those decompresses the whole chunk again.
                writer.copy_chunks_with_cursor(
                    sources[src_idx].reader,
                    &mut chunk_cursors[src_idx],
                    doc_id,
                    to_doc,
                )?;
                i = j;
            }
            StoredFieldsMergeStrategy::Doc => {
                let (num_stored_fields, bytes) =
                    chunk_cursors[src_idx].document(sources[src_idx].reader, doc_id)?;
                writer.add_serialized_document(num_stored_fields, bytes);
                // ARITH: `i < doc_order.len()` is this `while`'s condition and
                // `doc_order` is a `Vec`, so `i + 1 <= isize::MAX`.
                #[allow(clippy::arithmetic_side_effects)]
                {
                    i += 1;
                }
            }
            StoredFieldsMergeStrategy::Visitor => {
                // Parsed out of the same cached chunk the DOC path reads --
                // the field renumbering is what rules out copying the bytes,
                // not anything about how they are decompressed.
                let (num_stored_fields, bytes) =
                    chunk_cursors[src_idx].document(sources[src_idx].reader, doc_id)?;
                add_renumbered_document(
                    &mut writer,
                    &sources[src_idx],
                    &per_source_maps[src_idx],
                    doc_id,
                    num_stored_fields,
                    bytes,
                )?;
                // ARITH: as above -- `i < doc_order.len()`.
                #[allow(clippy::arithmetic_side_effects)]
                {
                    i += 1;
                }
            }
        }
    }
    finish_streamed_stored(writer)
}

/// The streamed stored-fields writer's output and its `.fdx`/`.fdm`.
fn finish_streamed_stored(
    writer: stored_fields::StoredFieldsWriter,
) -> Result<(lucene_store::FsIndexOutput, Vec<u8>, Vec<u8>)> {
    let (out, fdx, fdm) = writer.finish_streamed();
    let out = out.expect("write_merged_stored_fields streams the .fdt");
    Ok((out, fdx, fdm))
}

/// A VISITOR source's document: parsed, its pruned field dropped, and its
/// fields renumbered into the merged segment's numbers.
fn add_renumbered_document(
    writer: &mut stored_fields::StoredFieldsWriter,
    source: &MergeSource,
    field_number_map: &HashMap<i32, i32>,
    doc_id: i32,
    num_stored_fields: i64,
    bytes: &[u8],
) -> Result<()> {
    let mut doc = stored_fields::parse_document(num_stored_fields, bytes)?;
    if let Some(pruned) = source.pruned_field {
        doc.fields
            .retain(|field| !pruned.drops(field.field_number, doc_id));
    }
    for field in &mut doc.fields {
        field.field_number =
            *field_number_map
                .get(&field.field_number)
                .ok_or(Error::UnknownSourceFieldNumber {
                    field_number: field.field_number,
                })?;
    }
    writer.add_document(&doc);
    Ok(())
}

/// Whether `doc_order` visits a source's documents out of their order often
/// enough -- a reordered merge does on nearly every document -- that reading
/// each document where it falls would decompress a chunk per document.
fn is_scattered(doc_order: &[(usize, i32)], num_sources: usize) -> bool {
    let mut last = vec![-1i32; num_sources];
    let mut backwards = 0usize;
    for &(src, doc) in doc_order {
        let Some(prev) = last.get_mut(src) else {
            continue;
        };
        if doc < *prev {
            backwards = backwards.saturating_add(1);
        }
        *prev = doc;
    }
    backwards.saturating_mul(64) > doc_order.len()
}

/// About how many uncompressed bytes of documents a window of
/// [`write_scattered_stored_fields`] holds.
const SCATTERED_WINDOW_BYTES: usize = 64 << 20;

/// [`write_merged_stored_fields`] for a scattered `doc_order`: the merged
/// documents a window at a time, each window's documents read in source
/// order -- every chunk decompressed once per window, not once per document
/// -- then written in merged order. The window is sized from the documents
/// seen so far to hold about [`SCATTERED_WINDOW_BYTES`].
fn write_scattered_stored_fields(
    writer: &mut stored_fields::StoredFieldsWriter,
    sources: &[MergeSource],
    per_source_maps: &[HashMap<i32, i32>],
    strategies: &[StoredFieldsMergeStrategy],
    chunk_cursors: &mut [stored_fields::ChunkCursor],
    doc_order: &[(usize, i32)],
) -> Result<()> {
    let mut window = 4096usize;
    let mut start = 0usize;
    let mut arena: Vec<u8> = Vec::new();
    let mut slots: Vec<(i64, usize, usize)> = Vec::new();
    let mut by_source: Vec<usize> = Vec::new();
    while start < doc_order.len() {
        let end = start.saturating_add(window).min(doc_order.len());
        let entries = &doc_order[start..end];
        by_source.clear();
        by_source.extend(0..entries.len());
        by_source.sort_unstable_by_key(|&k| entries[k]);
        arena.clear();
        slots.clear();
        slots.resize(entries.len(), (0, 0, 0));
        for &k in &by_source {
            let (src, doc) = entries[k];
            let (num_stored_fields, bytes) =
                chunk_cursors[src].document(sources[src].reader, doc)?;
            slots[k] = (num_stored_fields, arena.len(), bytes.len());
            arena.extend_from_slice(bytes);
        }
        for (&(src, doc), &(num_stored_fields, at, len)) in entries.iter().zip(&slots) {
            // ARITH: `at..at + len` is a range this loop's reads appended.
            #[allow(clippy::arithmetic_side_effects)]
            let bytes = &arena[at..at + len];
            match strategies[src] {
                StoredFieldsMergeStrategy::Bulk | StoredFieldsMergeStrategy::Doc => {
                    writer.add_serialized_document(num_stored_fields, bytes);
                }
                StoredFieldsMergeStrategy::Visitor => add_renumbered_document(
                    writer,
                    &sources[src],
                    &per_source_maps[src],
                    doc,
                    num_stored_fields,
                    bytes,
                )?,
            }
        }
        let per_doc = arena.len().checked_div(entries.len()).unwrap_or(1).max(1);
        window = SCATTERED_WINDOW_BYTES
            .checked_div(per_doc)
            .unwrap_or(1024)
            .clamp(1024, 1 << 16);
        start = end;
    }
    Ok(())
}

/// Java's `StoredFieldsWriter.MergeVisitor`: forward each decoded value
/// immediately, so an earlier writer failure precedes later source corruption.
struct MergedStoredFieldsVisitor<'a> {
    writer: &'a mut dyn lucene_codecs::stored_fields_writer::StoredFieldsWriteSink,
    field_map: &'a HashMap<i32, i32>,
    pruned: Option<PrunedField<'a>>,
    doc_id: i32,
    unknown_field: Option<i32>,
}

impl MergedStoredFieldsVisitor<'_> {
    fn write_field(
        &mut self,
        field_number: i32,
        value: stored_fields::FieldValue,
    ) -> stored_fields::Result<()> {
        if self
            .pruned
            .is_some_and(|pruned| pruned.drops(field_number, self.doc_id))
        {
            return Ok(());
        }
        let Some(&number) = self.field_map.get(&field_number) else {
            // The lower-level visitor returns only stored-fields errors.
            // Abort traversal immediately, then restore the typed merge error
            // at the call site before propagating any decoder/writer error.
            self.unknown_field = Some(field_number);
            return Err(lucene_store::Error::Io(std::io::Error::other(
                "stored-fields merge field mapping failed",
            ))
            .into());
        };
        self.writer.write_field(number, &value)
    }
}

impl stored_fields::StoredFieldVisitor for MergedStoredFieldsVisitor<'_> {
    fn needs_field(
        &mut self,
        _field_number: i32,
    ) -> stored_fields::Result<stored_fields::VisitStatus> {
        // Preserve validation of pruned values, then prune in source space
        // before looking up the merged field number.
        Ok(stored_fields::VisitStatus::Yes)
    }

    fn string_field(&mut self, number: i32, value: &str) -> stored_fields::Result<()> {
        self.write_field(number, stored_fields::FieldValue::String(value.to_owned()))
    }

    fn binary_field(&mut self, number: i32, value: &[u8]) -> stored_fields::Result<()> {
        self.write_field(number, stored_fields::FieldValue::Binary(value.to_vec()))
    }

    fn int_field(&mut self, number: i32, value: i32) -> stored_fields::Result<()> {
        self.write_field(number, stored_fields::FieldValue::Int(value))
    }

    fn long_field(&mut self, number: i32, value: i64) -> stored_fields::Result<()> {
        self.write_field(number, stored_fields::FieldValue::Long(value))
    }

    fn float_field(&mut self, number: i32, value: f32) -> stored_fields::Result<()> {
        self.write_field(number, stored_fields::FieldValue::Float(value))
    }

    fn double_field(&mut self, number: i32, value: f64) -> stored_fields::Result<()> {
        self.write_field(number, stored_fields::FieldValue::Double(value))
    }
}

/// Java's `StoredFieldsWriter.merge` visitor path re-enters the installed
/// writer for every live document, including when native fields could be
/// copied as compressed chunks. This is necessary for lifecycle assertions.
fn write_merged_stored_fields_with_factory(
    factory: &dyn lucene_codecs::stored_fields_writer::StoredFieldsWriterFactory,
    state: lucene_codecs::stored_fields_writer::StoredFieldsWriteState<'_>,
    sources: &[MergeSource],
    per_source_maps: &[HashMap<i32, i32>],
    doc_order: &[(usize, i32)],
) -> Result<()> {
    let mut cursors: Vec<stored_fields::ChunkCursor> = (0..sources.len())
        .map(|_| stored_fields::ChunkCursor::new())
        .collect();
    lucene_codecs::stored_fields_writer::with_writer(factory, state, |writer| -> Result<()> {
        // Java opens the writer before its merge checks source integrity.
        // Keep that error order and close the opened writer on corruption.
        {
            use rayon::prelude::*;
            in_source_order(
                sources
                    .par_iter()
                    .map(|source| Ok(source.reader.check_integrity()?)),
            )?;
        }
        for &(source_index, doc_id) in doc_order {
            let source = &sources[source_index];
            // Java starts the output document before visiting/decoding its
            // source, so a start failure precedes a corrupt-document error.
            writer.start_document()?;
            let (count, bytes) = cursors[source_index].document(source.reader, doc_id)?;
            let mut visitor = MergedStoredFieldsVisitor {
                writer,
                field_map: &per_source_maps[source_index],
                pruned: source.pruned_field,
                doc_id,
                unknown_field: None,
            };
            let visited = stored_fields::visit_document(count, bytes, &mut visitor);
            if let Some(field_number) = visitor.unknown_field {
                return Err(Error::UnknownSourceFieldNumber { field_number });
            }
            visited?;
            writer.finish_document()?;
        }
        // The same final document sequence supplies the segment's doc_count.
        writer.finish(doc_order.len() as i32)?;
        Ok(())
    })
}

/// Merges `sources` (already-opened, in source order) into one brand-new
/// segment named `merged_segment_name` inside `dir`, exactly as
/// [`crate::segment_writer::flush_stored_only_segment`] writes a freshly
/// flushed one -- deleted docs (per each source's `live_docs`) are dropped,
/// surviving docs are renumbered contiguously by concatenating sources in
/// order, and field numbers are reconciled by name (see
/// [`reconcile_field_numbers`]).
///
/// A source with `live_docs` fully cleared (every doc deleted) naturally
/// contributes zero docs to the merge -- this port merges it anyway rather
/// than requiring the caller to have already dropped it (real Lucene's
/// `IndexWriter` drops a 100%-deleted segment before a merge is even
/// scheduled, purely as a merge-policy optimization; skipping that
/// optimization here costs nothing but a no-op source pass).
///
/// This is [`merge_segments`] with no index sort and default
/// [`MergeOptions`]; the merged `.si` therefore records no index sort, which
/// is honest for what concatenation produces. To *preserve* an index sort,
/// use [`merge_sorted_stored_only_segments`] (or [`merge_segments`] directly).
/// `SegmentMerger`'s constructor: the merged segment's `minVersion` is the
/// **smallest** `minVersion` among its readers, or `null` if any reader has
/// none.
///
/// ```text
/// Version minVersion = Version.LATEST;
/// for (CodecReader reader : readers) {
///   Version leafMinVersion = reader.getMetaData().minVersion();
///   if (leafMinVersion == null) { minVersion = null; break; }
///   if (minVersion.onOrAfter(leafMinVersion)) { minVersion = leafMinVersion; }
/// }
/// ```
///
/// `latest` stands for Java's `Version.LATEST` seed -- the version this
/// writer is, which is also what an empty source list yields (Java would
/// keep the seed too).
///
/// Writing the writer's own version unconditionally is right only while every
/// source was written by this writer at this version: once a merge takes in a
/// segment another Lucene wrote, the merged `.si` would claim a `minVersion`
/// newer than the oldest bytes it carries, which is the one fact upgrade
/// tooling and `IndexFormatTooOldException` are decided from.
fn merged_min_version(sources: &[MergeSource], latest: LuceneVersion) -> Option<LuceneVersion> {
    fold_min_version(latest, sources.iter().map(|s| s.min_version))
}

/// [`merged_min_version`]'s loop over anything that yields one reader's
/// `minVersion` -- the fold itself, separated from where the versions come
/// from so a test can drive it directly instead of restating it.
fn fold_min_version(
    latest: LuceneVersion,
    mins: impl IntoIterator<Item = Option<LuceneVersion>>,
) -> Option<LuceneVersion> {
    let mut min = latest;
    for leaf in mins {
        // `if (leafMinVersion == null) { minVersion = null; break; }`
        let leaf = leaf?;
        // `if (minVersion.onOrAfter(leafMinVersion)) { minVersion = leafMinVersion; }`
        if version_key(min) >= version_key(leaf) {
            min = leaf;
        }
    }
    Some(min)
}

/// A [`LuceneVersion`] as one comparable triple, so `onOrAfter` is a tuple
/// compare rather than three nested branches. `LuceneVersion` derives only
/// `PartialEq`/`Eq` (ordering a *version* is not meaningful outside this
/// major-minor-bugfix rule, so it is spelled out where it is used rather
/// than derived onto the type).
///
/// Exactly Java's comparison: `Version.onOrAfter` compares
/// `encodedValue = major << 18 | minor << 10 | bugfix << 2 | prerelease`,
/// whose fields do not overlap, so it *is* lexicographic on the triple. The
/// `prerelease` component has no bearing here -- `Lucene99SegmentInfoFormat`
/// writes only the three ints, so anything read back off a `.si` has
/// `prerelease == 0`.
fn version_key(v: LuceneVersion) -> (i32, i32, i32) {
    (v.major, v.minor, v.bugfix)
}

pub fn merge_stored_only_segments(
    dir: &dyn Directory,
    sources: &[MergeSource],
    merged_segment_name: &str,
    merged_segment_id: [u8; ID_LENGTH],
    codec_name: &str,
    lucene_version: LuceneVersion,
) -> Result<SegmentCommitInfo> {
    merge_segments(
        dir,
        sources,
        None,
        &MergeOptions::default(),
        merged_segment_name,
        merged_segment_id,
        codec_name,
        lucene_version,
    )
}

/// A single field's priority tier for [`merge_sorted_stored_only_segments`]'s
/// k-way merge -- the cross-source analogue of
/// [`crate::segment_writer::SortKeySpec`]: `per_source_keys[i][d]` is source
/// `i`'s original (pre-merge) doc `d`'s value for this field, or `None` if
/// that doc has no value. `per_source_keys` must have exactly one entry per
/// source, and each source's slice must have exactly one entry per doc in
/// that source (`source.reader.max_doc()` entries).
pub struct MergeSortKeySpec<'a> {
    pub sort: &'a IndexSortField,
    pub per_source_keys: &'a [&'a [Option<i64>]],
}

/// Merges `sources` into one brand-new segment whose documents are produced
/// in **global** sort order across all sources -- a genuine k-way merge by
/// sort key (at each step, take whichever source's current head doc has the
/// smallest key, in `sort_fields` priority order), not a concatenation of
/// source 0's docs, then source 1's, etc. the way
/// [`merge_stored_only_segments`] works. This is the real behavior of merging
/// index-sorted segments in Lucene: since every source is already internally
/// sorted by the same key, the merged segment can be produced by a single
/// forward pass over all sources at once.
///
/// [`merge_segments`] with `Some(sort_fields)` and default [`MergeOptions`];
/// see there for the scope (every format the plain merge writes) and for the
/// precondition on `sort_fields`.
pub fn merge_sorted_stored_only_segments(
    dir: &dyn Directory,
    sources: &[MergeSource],
    sort_fields: &[MergeSortKeySpec<'_>],
    merged_segment_name: &str,
    merged_segment_id: [u8; ID_LENGTH],
    codec_name: &str,
    lucene_version: LuceneVersion,
) -> Result<SegmentCommitInfo> {
    merge_segments(
        dir,
        sources,
        Some(sort_fields),
        &MergeOptions::default(),
        merged_segment_name,
        merged_segment_id,
        codec_name,
        lucene_version,
    )
}

/// [`merge_segments`]'s result, with what `IndexWriter.commitMerge` needs to
/// carry deletes made **during** the merge onto the merged segment
/// (`commitMergedDeletesAndUpdates`): each source's doc-id map into the
/// merged segment (`-1` for a document the merge dropped), and the files the
/// merge wrote, for an abandoned merge to delete.
pub struct MergedSegment {
    pub info: SegmentCommitInfo,
    pub doc_id_maps: Vec<Vec<i32>>,
    pub files: Vec<String>,
}

/// The one merge: `SegmentMerger.merge()`.
///
/// Every format any source supplies is merged and written -- stored fields,
/// postings, term vectors, norms, doc values, BKD points and KNN vectors --
/// and the *only* thing `sort_fields` changes is the order documents come out
/// in, i.e. `MergeState.docMaps`. That is deliberate and is the point of this
/// function existing: while the sorted merge was a separate entry point that
/// wrote a subset of the formats, routing an index-sorted index through it
/// would have traded a lost sort for lost data, and routing it through the
/// plain merge lost the sort silently (a merged segment that is valid,
/// `CheckIndex`-clean, and no longer in the order its own `.si` claims). One
/// function cannot drift from itself.
///
/// - `sort_fields: None` -- documents are concatenated in source order
///   (`MergeState.buildDeletionDocMaps`), and the merged `.si` records no
///   index sort.
/// - `sort_fields: Some(..)` -- documents are produced by a k-way merge on
///   the key (`MultiSorter.sort`), and the merged `.si` records that sort.
///
/// # Precondition on `sort_fields` (caller-guaranteed, not re-checked here)
///
/// Real Lucene requires every segment being merged to share the exact same
/// index sort -- merging segments with different (or absent-vs-present)
/// index sorts is a hard error in `IndexWriter.validateIndexSort`, not
/// something this function tries to detect or repair. `sort_fields` is the
/// *one* shared sort every source is already ordered by: each source's own
/// doc 0, 1, 2, ... must already be non-decreasing by this exact key -- true
/// for any segment written by
/// [`crate::segment_writer::flush_sorted_stored_only_segment`], by
/// `IndexWriter::flush` with an index sort configured, or by a previous call
/// to this function with the same key. It is the caller's job to have
/// verified this by comparing each source's own `SegmentInfo.index_sort`
/// against `sort_fields` (which is what `IndexWriter::execute_merge` does);
/// this function does not re-verify it, keeping it usable from a plain
/// in-memory source list. Passing sources that are not actually sorted by
/// `sort_fields` silently produces a merged segment that is not sorted
/// either -- garbage in, garbage out, exactly like the analogous
/// precondition on `flush_sorted_stored_only_segment`'s caller-supplied
/// `SortKeySpec::keys`.
///
/// # Cost of the sort
///
/// The k-way merge itself is O(docs x sources) comparisons with a linear head
/// scan. What it really costs is the **byte-copy fast paths**: stored fields
/// and term vectors can copy whole compressed chunks only for a run of
/// consecutive documents from one source, and an index sort interleaves the
/// sources, so those runs collapse. `write_merged_stored_fields` and
/// `write_merged_term_vectors` detect runs from `doc_order` itself, so the
/// fast path is taken exactly when it is legal and never when it is not.
#[allow(clippy::too_many_arguments)]
pub fn merge_segments(
    dir: &dyn Directory,
    sources: &[MergeSource],
    sort_fields: Option<&[MergeSortKeySpec<'_>]>,
    options: &MergeOptions,
    merged_segment_name: &str,
    merged_segment_id: [u8; ID_LENGTH],
    codec_name: &str,
    lucene_version: LuceneVersion,
) -> Result<SegmentCommitInfo> {
    merge_segments_mapped(
        dir,
        sources,
        sort_fields,
        options,
        merged_segment_name,
        merged_segment_id,
        codec_name,
        lucene_version,
    )
    .map(|merged| merged.info)
}

/// [`merge_segments`], also returning the doc-id maps and files -- see
/// [`MergedSegment`].
#[allow(clippy::too_many_arguments)]
pub fn merge_segments_mapped(
    dir: &dyn Directory,
    sources: &[MergeSource],
    sort_fields: Option<&[MergeSortKeySpec<'_>]>,
    options: &MergeOptions,
    merged_segment_name: &str,
    merged_segment_id: [u8; ID_LENGTH],
    codec_name: &str,
    lucene_version: LuceneVersion,
) -> Result<MergedSegment> {
    // Java: IndexWriter.mergeMiddle tracks every successfully created output.
    // Track the whole merge, including files completed before a later source
    // decoder or codec fails. Per-format consumers only clean their own writes.
    let outputs = crate::created_outputs::CreatedOutputs::new(dir);
    let result = merge_segments_mapped_inner(
        &outputs,
        sources,
        sort_fields,
        options,
        merged_segment_name,
        merged_segment_id,
        codec_name,
        lucene_version,
    );
    if result.is_err() {
        // Successful creations establish ownership; a failed create must never
        // make a colliding, preexisting file eligible for deletion. Preserve
        // the merge error if a best-effort cleanup deletion also fails.
        for name in outputs.files().iter() {
            let _ = dir.delete_file(name);
        }
    }
    result
}

/// `MultiSorter.sort`: the `(source index, source doc id)` pairs of every
/// live document, in the order the shared index sort puts them in.
///
/// A linear scan over the (few) source heads per step rather than Java's
/// `PriorityQueue`: this port merges `max_merge_at_once` segments at a time,
/// which defaults to 10, so a heap's log factor buys nothing against its
/// constant.
///
/// Java walks *every* document through the queue and only increments the
/// mapped id for live ones; this walks only the live ones, which produces the
/// same relative order (a deleted document can never sit between two live
/// ones in the output) and skips the comparisons for documents that cannot
/// appear.
fn sorted_doc_order(
    sort_fields: &[MergeSortKeySpec<'_>],
    per_source_live_ids: &[Vec<i32>],
) -> Vec<(usize, i32)> {
    let cmps = merge_comparators(sort_fields);
    let mut cursors = vec![0usize; per_source_live_ids.len()];
    let mut doc_order: Vec<(usize, i32)> =
        Vec::with_capacity(per_source_live_ids.iter().map(|ids| ids.len()).sum());
    loop {
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
                        &cmps,
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
        // ARITH: `src_idx` was only chosen for a source whose cursor is still
        // `< live_ids.len()` (the `continue` above), so this increment lands
        // at most at that `Vec`'s length.
        #[allow(clippy::arithmetic_side_effects)]
        {
            cursors[src_idx] += 1;
        }
        doc_order.push((src_idx, doc_id));
    }
    doc_order
}

/// The merged position of each partition's first document: where the
/// first sort tier's key changes along `doc_order`.
fn partition_starts(doc_order: &[(usize, i32)], tier: &MergeSortKeySpec<'_>) -> Vec<i32> {
    let key = |&(source, doc): &(usize, i32)| {
        tier.per_source_keys
            .get(source)
            .and_then(|keys| keys.get(usize::try_from(doc).ok()?))
            .copied()
            .flatten()
    };
    let mut starts = Vec::new();
    let mut previous = None;
    for (new, entry) in doc_order.iter().enumerate() {
        let k = key(entry);
        if new == 0 || k != previous {
            starts.push(i32::try_from(new).unwrap_or(i32::MAX));
        }
        previous = k;
    }
    starts
}

/// Multi-tier comparator for the k-way merge: folds `sort_fields` in
/// priority order using [`SortKeyComparator`] (the exact same per-tier
/// comparator [`crate::segment_writer::flush_sorted_stored_only_segment`]
/// uses within one batch -- reused, not reimplemented), then breaks any
/// remaining tie first by source index and finally by original doc id,
/// giving a fully deterministic total order.
///
/// `cmps` is resolved once by the caller, outside the O(n log k) comparison
/// loop; see [`merge_comparators`] for why a sort with no comparator cannot
/// reach here.
fn compare_heads(
    sort_fields: &[MergeSortKeySpec<'_>],
    cmps: &[SortKeyComparator],
    src_a: usize,
    doc_a: i32,
    src_b: usize,
    doc_b: i32,
) -> std::cmp::Ordering {
    sort_fields
        .iter()
        .zip(cmps)
        .fold(std::cmp::Ordering::Equal, |acc, (spec, cmp)| {
            acc.then_with(|| {
                let key_a = spec.per_source_keys[src_a][doc_a as usize];
                let key_b = spec.per_source_keys[src_b][doc_b as usize];
                cmp.compare(key_a, key_b)
            })
        })
        .then_with(|| src_a.cmp(&src_b))
        .then_with(|| doc_a.cmp(&doc_b))
}

/// The comparators `sort_fields` induces, in priority order -- the merge's
/// twin of `segment_writer`'s own.
fn merge_comparators(sort_fields: &[MergeSortKeySpec<'_>]) -> Vec<SortKeyComparator> {
    sort_fields
        .iter()
        .map(|spec| SortKeyComparator::new(spec.sort))
        .collect()
}

fn write_file(dir: &dyn Directory, name: &str, bytes: &[u8]) -> Result<()> {
    let mut out = dir.create_output(name)?;
    out.write_bytes(bytes);
    out.close()?;
    Ok(())
}

/// One merged doc-values field, resolved into exactly the shape
/// [`lucene_codecs::doc_values::write_dense_fields`] takes -- the merged
/// segment's whole `.dvm`/`.dvd`/`.dvs` is one call over a slice of these,
/// which is what a real multi-field `Lucene90DocValuesFormat` segment is.
enum MergedDocValuesField {
    /// Every merged document has a value.
    Numeric(i32, Vec<i64>),
    /// `(merged doc id, value)` for the documents that have one --
    /// `SortField.setMissingValue`'s normal case, and the shape a
    /// multi-tier index sort's tiers routinely take.
    SparseNumeric(i32, Vec<(i32, i64)>),
    /// BINARY and SORTED_NUMERIC, dense or sparse, streamed in merged-doc
    /// order into one flat buffer each.
    Binary(i32, doc_values::BinaryColumn),
    SortedNumeric(i32, doc_values::SortedNumericColumn),
    /// SORTED and SORTED_SET, dense or sparse, with the merged dictionary
    /// already built by [`OrdinalMap`] and every document's ordinals
    /// remapped into it.
    SortedOrds(i32, doc_values::SortedOrdsColumn),
    SortedSetOrds(i32, doc_values::SortedSetOrdsColumn),
}

impl MergedDocValuesField {
    fn as_dense_field(&self) -> doc_values::DenseField<'_> {
        match self {
            MergedDocValuesField::Numeric(n, v) => doc_values::DenseField::Numeric(*n, v),
            MergedDocValuesField::SparseNumeric(n, v) => {
                doc_values::DenseField::SparseNumeric(*n, v)
            }
            MergedDocValuesField::Binary(n, c) => doc_values::DenseField::BinaryColumn(*n, c),
            MergedDocValuesField::SortedNumeric(n, c) => {
                doc_values::DenseField::SortedNumericColumn(*n, c)
            }
            MergedDocValuesField::SortedOrds(n, c) => doc_values::DenseField::SortedOrds(*n, c),
            MergedDocValuesField::SortedSetOrds(n, c) => {
                doc_values::DenseField::SortedSetOrds(*n, c)
            }
        }
    }
}

/// Every merged field number some source has data of this kind for, ascending
/// -- the candidate list every `merge_*_doc_values` function starts from.
///
/// A source with no live documents is skipped: it contributes nothing to the
/// merged output, so whatever doc-values fields it happens to carry cannot
/// affect it (the same exemption is applied again when checking whether a
/// source is missing a field the merge needs).
fn doc_values_candidates<'a, T: 'a>(
    sources: &'a [MergeSource<'a>],
    per_source_maps: &[HashMap<i32, i32>],
    per_source_live_ids: &[Vec<i32>],
    per_source_fields: impl Fn(&'a MergeSource<'a>) -> &'a [T],
    field_number: impl Fn(&T) -> i32,
) -> Vec<i32> {
    let mut candidates: Vec<i32> = Vec::new();
    for ((source, map), live_ids) in sources.iter().zip(per_source_maps).zip(per_source_live_ids) {
        if live_ids.is_empty() {
            continue;
        }
        for field in per_source_fields(source) {
            if let Some(&merged_number) = map.get(&field_number(field)) {
                candidates.push(merged_number);
            }
        }
    }
    candidates.sort_unstable();
    candidates.dedup();
    candidates
}

/// Each source's `fields`, keyed by the source's own field number (the first
/// entry wins, as the linear `find` this replaces did), and empty for a
/// source with no live documents -- built once per merge stage so resolving
/// a candidate field in every source is a hash lookup rather than a scan of
/// the source's whole field list per (field, source) pair.
fn index_source_fields<'a, T>(
    sources: &'a [MergeSource<'a>],
    per_source_live_ids: &[Vec<i32>],
    fields: impl Fn(&'a MergeSource<'a>) -> &'a [T],
    field_number: impl Fn(&T) -> i32,
) -> Vec<HashMap<i32, &'a T>> {
    sources
        .iter()
        .zip(per_source_live_ids)
        .map(|(source, live_ids)| {
            let mut by_number = HashMap::new();
            if !live_ids.is_empty() {
                for field in fields(source) {
                    by_number.entry(field_number(field)).or_insert(field);
                }
            }
            by_number
        })
        .collect()
}

/// Merges numeric doc-values data across `sources`, one
/// [`MergedDocValuesField`] per merged field number any source has a NUMERIC
/// column for, with each field's values in the same doc order `doc_order`
/// gives -- so an index-sorted merge's columns are ordered by the merged
/// segment's physical order, not by source.
///
/// # Multiple fields, and per-document sparsity
///
/// There is no single-field-per-merge limit: every candidate is merged, and
/// they all land in the one `.dvm`/`.dvd` the caller writes. That is what a
/// **multi-tier index sort** needs (a second tier is a second NUMERIC
/// column), and it is what a real segment looks like.
///
/// A live document with **no** value for the field is not an error either: it
/// is `SortField.setMissingValue`'s normal case, and the merged column is then
/// written sparsely through the same `IndexedDISI` + values body
/// `Lucene90DocValuesConsumer.writeValues` uses; so is a source with no column
/// for the field at all (see the comment in the body). A field for which **no**
/// merged document has a value is still written -- as
/// `Lucene90DocValuesConsumer.writeValues`' all-missing column
/// (`docsWithFieldOffset = -2`), not omitted -- because omitting it would zero
/// the field's `DocValuesType` in the merged `.fnm` and, for an index-sort
/// tier, leave a segment whose `.si` claims an order nothing can re-derive.
fn merge_numeric_doc_values(
    sources: &[MergeSource],
    per_source_maps: &[HashMap<i32, i32>],
    reverse_maps: &[HashMap<i32, i32>],
    per_source_live_ids: &[Vec<i32>],
    doc_order: &[(usize, i32)],
    only: Option<i32>,
) -> Result<Vec<MergedDocValuesField>> {
    let candidates = doc_values_candidates(
        sources,
        per_source_maps,
        per_source_live_ids,
        |s| s.numeric_doc_values,
        |f| f.entry.field_number,
    );
    let candidates: Vec<i32> = candidates
        .into_iter()
        .filter(|n| only.is_none_or(|o| o == *n))
        .collect();
    let by_number = index_source_fields(
        sources,
        per_source_live_ids,
        |s| s.numeric_doc_values,
        |f: &SourceNumericDocValues| f.entry.field_number,
    );
    let mut merged = Vec::with_capacity(candidates.len());
    for merged_field_number in candidates {
        // Resolve each contributing source's entry once, up front --
        // `doc_order` may interleave sources in any order (a k-way sorted
        // merge does), so the per-source resolution can no longer be folded
        // into a single linear pass over `per_source_live_ids` the way
        // concatenation could.
        // `DocValuesConsumer.getMergedNumericDocValues`: a source with no
        // NUMERIC column for this field is simply **not a sub** --
        //
        // ```java
        // FieldInfo readerFieldInfo = mergeState.fieldInfos[i].fieldInfo(mergeFieldInfo.name);
        // if (readerFieldInfo != null
        //     && readerFieldInfo.getDocValuesType() == DocValuesType.NUMERIC) {
        //   values = docValuesProducer.getNumeric(readerFieldInfo);
        // }
        // if (values != null) { subs.add(...); }
        // ```
        //
        // -- so every one of its documents comes out of the merge with no
        // value, which is the sparse column `SortField.setMissingValue`
        // already models and `write_dense_fields` already writes. Two ordinary
        // cases depend on it: a field added to the schema after some segments
        // were flushed, and a doc-values **update** against a field whose base
        // flush wrote no column.
        //
        // It is safe not to treat this as a caller error because a better
        // detector for the same class runs one level up:
        // [`check_format_coverage`] refuses a merge whose caller never opened
        // a format a source's own `.si` lists, and the `debug_assert!` in
        // `IndexWriter::execute_merge` pins that every entry an opened `.dvm`
        // declares reaches the merge. A silently-dropped column can no longer
        // hide behind this branch, so the branch can have Java's meaning.
        let per_source_entry: Vec<Option<&SourceNumericDocValues>> =
            per_source_entries(&by_number, reverse_maps, merged_field_number);

        // One `NumericReader` per source, not a `numeric_value` call per
        // document: that free function allocates a fresh `DisiCursor` and
        // re-walks the docs-with-field region from its start every time, and
        // its own doc comment names a *sort* as the caller that must not do
        // that. `doc_order` visits each source's documents in ascending order
        // (`build_doc_id_maps` is monotone within a source), which is exactly
        // the forward-only access the cursor wants.
        let mut per_source_reader: Vec<Option<doc_values::NumericReader>> = per_source_entry
            .iter()
            .map(|entry| entry.map(|e| doc_values::NumericReader::new(e.data, &e.entry)))
            .collect();
        // Dense until the first document without a value; that document
        // converts what is there once (every earlier document has a value,
        // so `dense[d]` is merged doc `d`'s) and the rest goes sparse only.
        let mut dense: Vec<i64> = Vec::with_capacity(doc_order.len());
        let mut sparse: Option<Vec<(i32, i64)>> = None;
        for (merged_doc_id, &(src_idx, doc_id)) in doc_order.iter().enumerate() {
            // No reader for this source means the source has no column for
            // this field at all: every one of its documents is missing a
            // value, exactly like a document the column itself skips.
            let value = match per_source_reader[src_idx].as_mut() {
                Some(reader) => reader.value(doc_id)?,
                None => None,
            };
            let pruned = sources[src_idx]
                .pruned_field
                .zip(per_source_entry[src_idx])
                .is_some_and(|(pruned, entry)| pruned.drops(entry.entry.field_number, doc_id));
            let value = value.filter(|_| !pruned).map(|v| {
                match (sources[src_idx].numeric_floor, per_source_entry[src_idx]) {
                    (Some(floor), Some(entry)) => floor.apply(entry.entry.field_number, doc_id, v),
                    _ => v,
                }
            });
            match (value, sparse.as_mut()) {
                (Some(value), None) => dense.push(value),
                (Some(value), Some(sparse)) => sparse.push((merged_doc_id as i32, value)),
                (None, None) => {
                    sparse = Some(
                        std::mem::take(&mut dense)
                            .into_iter()
                            .enumerate()
                            .map(|(doc, value)| (doc as i32, value))
                            .collect(),
                    );
                }
                (None, Some(_)) => {}
            }
        }
        if let Some(sparse) = sparse {
            // Including when `sparse` is empty. `Lucene90DocValuesConsumer`
            // is called for every field the merged `FieldInfos` gives a
            // `DocValuesType`, and `writeValues` records an all-missing
            // column as `docsWithFieldOffset = -2` -- it does not omit the
            // field. Omitting it here would zero the field's
            // `DocValuesType` in the merged `.fnm`
            // (`describe_written_files`), which for an **index-sort tier** is
            // a segment whose `.si` claims an order no reader can re-derive:
            // real Lucene's `DocValues.getNumeric` throws for a field whose
            // `FieldInfo` exists but declares no doc values, so
            // `CheckIndex.testSort` fails rather than degrading, and this
            // port's own `execute_merge` could never merge the segment again
            // (`Error::MergeSortColumnMissing`, propagated out of `commit`).
            merged.push(MergedDocValuesField::SparseNumeric(
                merged_field_number,
                sparse,
            ));
        } else {
            merged.push(MergedDocValuesField::Numeric(merged_field_number, dense));
        }
    }
    Ok(merged)
}

/// How many doc-values columns a merge builds at once.
const DOC_VALUES_MERGE_WINDOW: usize = 4;

/// Every merged field number some source has doc-values data of any kind for,
/// ascending: what the five `merge_*_doc_values` stages would produce between
/// them.
fn merged_doc_values_field_numbers(
    sources: &[MergeSource],
    per_source_maps: &[HashMap<i32, i32>],
    per_source_live_ids: &[Vec<i32>],
) -> Vec<i32> {
    let mut numbers = doc_values_candidates(
        sources,
        per_source_maps,
        per_source_live_ids,
        |s| s.numeric_doc_values,
        |f| f.entry.field_number,
    );
    numbers.extend(doc_values_candidates(
        sources,
        per_source_maps,
        per_source_live_ids,
        |s| s.binary_doc_values,
        |f| f.entry.field_number,
    ));
    numbers.extend(doc_values_candidates(
        sources,
        per_source_maps,
        per_source_live_ids,
        |s| s.sorted_doc_values,
        |f| f.entry.field_number,
    ));
    numbers.extend(doc_values_candidates(
        sources,
        per_source_maps,
        per_source_live_ids,
        |s| s.sorted_numeric_doc_values,
        |f| f.entry.field_number,
    ));
    numbers.extend(doc_values_candidates(
        sources,
        per_source_maps,
        per_source_live_ids,
        |s| s.sorted_set_doc_values,
        |f| f.entry.field_number,
    ));
    numbers.sort_unstable();
    numbers
}

/// Each contributing source's entry for `merged_field_number`, or `None` for
/// a source with no live documents or no such column --
/// `DocValuesConsumer.getMerged*DocValues`' rule that a source whose
/// `FieldInfo` lacks the column is simply **not a sub**, so every one of its
/// documents comes out of the merge with no value (see
/// [`merge_numeric_doc_values`] for why that is safe to rely on here).
///
/// `by_number` is [`index_source_fields`]' per-source index and
/// `reverse_maps` [`invert_field_number_maps`]' merged -> original maps, so
/// this is two hash lookups per source.
fn per_source_entries<'a, T>(
    by_number: &[HashMap<i32, &'a T>],
    reverse_maps: &[HashMap<i32, i32>],
    merged_field_number: i32,
) -> Vec<Option<&'a T>> {
    by_number
        .iter()
        .zip(reverse_maps)
        .map(|(fields, reverse)| {
            let original = reverse.get(&merged_field_number)?;
            fields.get(original).copied()
        })
        .collect()
}

/// Merges BINARY doc-values data across `sources`, one
/// [`MergedDocValuesField`] per merged field number any source has a BINARY
/// column for, in `doc_order` -- `DocValuesConsumer.mergeBinaryField`. A
/// document with no value (its source has no column, or the column skips
/// it) is missing from the merged column, which is then written sparsely.
fn merge_binary_doc_values(
    sources: &[MergeSource],
    per_source_maps: &[HashMap<i32, i32>],
    reverse_maps: &[HashMap<i32, i32>],
    per_source_live_ids: &[Vec<i32>],
    doc_order: &[(usize, i32)],
    only: Option<i32>,
) -> Result<Vec<MergedDocValuesField>> {
    let candidates = doc_values_candidates(
        sources,
        per_source_maps,
        per_source_live_ids,
        |s| s.binary_doc_values,
        |f| f.entry.field_number,
    );
    let candidates: Vec<i32> = candidates
        .into_iter()
        .filter(|n| only.is_none_or(|o| o == *n))
        .collect();
    let by_number = index_source_fields(
        sources,
        per_source_live_ids,
        |s| s.binary_doc_values,
        |f: &SourceBinaryDocValues| f.entry.field_number,
    );
    let mut merged = Vec::with_capacity(candidates.len());
    for merged_field_number in candidates {
        let per_source_entry = per_source_entries(&by_number, reverse_maps, merged_field_number);
        // One forward cursor per source (`doc_order` visits each source's
        // documents in ascending order), as `getMergedBinaryDocValues`
        // iterates each sub.
        let mut readers: Vec<Option<doc_values::BinaryReader>> = per_source_entry
            .iter()
            .map(|e| e.map(|e| doc_values::BinaryReader::new(e.data, &e.entry)))
            .collect();
        let mut column = doc_values::BinaryColumn::default();
        for (merged_doc_id, &(src_idx, doc_id)) in doc_order.iter().enumerate() {
            let Some(reader) = readers[src_idx].as_mut() else {
                continue;
            };
            if let Some(value) = reader.value_ref(doc_id)? {
                column.docs.push(merged_doc_id as i32);
                column.lengths.push(value.len() as u32);
                column.bytes.extend_from_slice(value);
            }
        }
        merged.push(MergedDocValuesField::Binary(merged_field_number, column));
    }
    Ok(merged)
}

/// `OrdinalMap`: the merged dictionary of several sources' sorted, unique
/// dictionaries, and each source's ordinal -> merged-ordinal map.
///
/// Java's walks the sources' `TermsEnum`s through a priority queue ordered by
/// current term, handing out the next global ordinal each time the smallest
/// term changes; this is that walk, over dictionaries already decoded. A
/// source with no column (`None`) contributes nothing and gets an empty map.
struct OrdinalMap {
    dict: Vec<Vec<u8>>,
    /// `segment_to_global[source][ord]`; `None` for a term no live document
    /// of the source uses.
    segment_to_global: Vec<Vec<Option<i64>>>,
}

impl OrdinalMap {
    /// `OrdinalMap.build` over `source_dicts`, each restricted to its
    /// `live_ords` when it has one -- `mergeSortedField`/`mergeSortedSetField`
    /// hand a source with deletions over as a `BitsFilteredTermsEnum` of the
    /// ordinals its live documents use, so a term only deleted documents
    /// carried leaves the merged dictionary (Java's `CheckIndex` refuses a
    /// dictionary with a term no document uses).
    fn build(source_dicts: Vec<Option<Vec<Vec<u8>>>>, live_ords: Vec<Option<FixedBitSet>>) -> Self {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;
        let mut segment_to_global: Vec<Vec<Option<i64>>> = source_dicts
            .iter()
            .map(|d| Vec::with_capacity(d.as_ref().map_or(0, Vec::len)))
            .collect();
        let dicts: Vec<Vec<Vec<u8>>> = source_dicts
            .into_iter()
            .map(Option::unwrap_or_default)
            .collect();
        // (current term, source, ord within source), smallest term first.
        let mut queue: BinaryHeap<Reverse<(&[u8], usize, usize)>> = dicts
            .iter()
            .enumerate()
            .filter(|(_, d)| !d.is_empty())
            .map(|(src, d)| Reverse((d[0].as_slice(), src, 0)))
            .collect();
        let mut dict: Vec<Vec<u8>> = Vec::new();
        while let Some(Reverse((term, src, ord))) = queue.pop() {
            let live = live_ords
                .get(src)
                .and_then(Option::as_ref)
                .is_none_or(|live| live.get_doc(i32::try_from(ord).unwrap_or(i32::MAX)));
            if !live {
                segment_to_global[src].push(None);
            } else {
                if dict.last().map(Vec::as_slice) != Some(term) {
                    dict.push(term.to_vec());
                }
                // ARITH: `dict` is non-empty after the push above.
                #[allow(clippy::arithmetic_side_effects)]
                segment_to_global[src].push(Some((dict.len() - 1) as i64));
            }
            // ARITH: `ord` indexes `dicts[src]`, so `ord + 1 <= len`.
            #[allow(clippy::arithmetic_side_effects)]
            let next = ord + 1;
            if let Some(t) = dicts[src].get(next) {
                queue.push(Reverse((t.as_slice(), src, next)));
            }
        }
        Self {
            dict,
            segment_to_global,
        }
    }

    /// `getGlobalOrds(source).get(ord)`; an ordinal outside the source's
    /// dictionary is a corrupt source.
    fn global(&self, source: usize, ord: i64) -> Option<i64> {
        usize::try_from(ord)
            .ok()
            .and_then(|o| self.segment_to_global[source].get(o).copied())
            .flatten()
    }
}

/// The ordinals `source`'s live documents use, for [`OrdinalMap::build`] --
/// `None` when the source has no deletions, as Java hands such a source's
/// terms over unfiltered. `ords` writes one document's ordinals.
fn live_ords(
    source: &MergeSource,
    live_ids: &[i32],
    value_count: usize,
    mut ords: impl FnMut(i32, &mut Vec<i64>) -> Result<()>,
) -> Result<Option<FixedBitSet>> {
    if source.live_docs.is_none() {
        return Ok(None);
    }
    let mut used = FixedBitSet::new(value_count);
    let mut scratch = Vec::new();
    for &doc in live_ids {
        scratch.clear();
        ords(doc, &mut scratch)?;
        for &ord in &scratch {
            // An ordinal outside the dictionary stays unset here and is
            // reported by the merge's own lookup.
            if let Ok(ord) = usize::try_from(ord) {
                if ord < value_count {
                    // FBS: `ord < value_count`, the bitset's length.
                    used.set(ord);
                }
            }
        }
    }
    Ok(Some(used))
}

/// Merges SORTED doc-values data across `sources`, one
/// [`MergedDocValuesField`] per merged field number any source has a SORTED
/// column for, in `doc_order` -- `DocValuesConsumer.mergeSortedField`: an
/// [`OrdinalMap`] over the sources' dictionaries, then each document's
/// ordinal remapped through it. No term bytes are copied per document and no
/// dictionary is rebuilt by sorting. Dense or sparse as
/// [`merge_binary_doc_values`].
fn merge_sorted_doc_values(
    sources: &[MergeSource],
    per_source_maps: &[HashMap<i32, i32>],
    reverse_maps: &[HashMap<i32, i32>],
    per_source_live_ids: &[Vec<i32>],
    doc_order: &[(usize, i32)],
    only: Option<i32>,
) -> Result<Vec<MergedDocValuesField>> {
    let candidates = doc_values_candidates(
        sources,
        per_source_maps,
        per_source_live_ids,
        |s| s.sorted_doc_values,
        |f| f.entry.field_number,
    );
    let candidates: Vec<i32> = candidates
        .into_iter()
        .filter(|n| only.is_none_or(|o| o == *n))
        .collect();
    let by_number = index_source_fields(
        sources,
        per_source_live_ids,
        |s| s.sorted_doc_values,
        |f: &SourceSortedDocValues| f.entry.field_number,
    );
    let mut merged = Vec::with_capacity(candidates.len());
    for merged_field_number in candidates {
        let per_source_entry = per_source_entries(&by_number, reverse_maps, merged_field_number);
        let dicts: Vec<Option<Vec<Vec<u8>>>> = per_source_entry
            .iter()
            .map(|e| {
                e.map(|e| terms_dict::decode_all_terms(e.data, &e.entry.terms))
                    .transpose()
            })
            .collect::<std::result::Result<_, _>>()?;
        let mut live = Vec::with_capacity(sources.len());
        for (idx, entry) in per_source_entry.iter().enumerate() {
            live.push(match (entry, &dicts[idx]) {
                (Some(e), Some(dict)) => {
                    let mut reader = doc_values::NumericReader::new(e.data, &e.entry.ords);
                    live_ords(
                        &sources[idx],
                        &per_source_live_ids[idx],
                        dict.len(),
                        |doc, out| {
                            out.extend(reader.value(doc)?);
                            Ok(())
                        },
                    )?
                }
                _ => None,
            });
        }
        let map = OrdinalMap::build(dicts, live);
        let mut readers: Vec<Option<doc_values::NumericReader>> = per_source_entry
            .iter()
            .map(|e| e.map(|e| doc_values::NumericReader::new(e.data, &e.entry.ords)))
            .collect();
        let mut column = doc_values::SortedOrdsColumn::default();
        column.docs.reserve_exact(doc_order.len());
        column.ords.reserve_exact(doc_order.len());
        for (merged_doc_id, &(src_idx, doc_id)) in doc_order.iter().enumerate() {
            let Some(reader) = readers[src_idx].as_mut() else {
                continue;
            };
            let Some(ord) = reader.value(doc_id)? else {
                continue;
            };
            let global =
                map.global(src_idx, ord)
                    .ok_or(Error::SortedDocValuesFieldMissingInSource {
                        merged_field_number,
                    })?;
            column.docs.push(merged_doc_id as i32);
            column.ords.push(global);
        }
        column.dict = map.dict;
        merged.push(MergedDocValuesField::SortedOrds(
            merged_field_number,
            column,
        ));
    }
    Ok(merged)
}

/// Merges SORTED_NUMERIC doc-values data across `sources`, one
/// [`MergedDocValuesField`] per merged field number any source has a
/// SORTED_NUMERIC column for, in `doc_order` --
/// `DocValuesConsumer.mergeSortedNumericField`. A document is missing when
/// its source has no column or the column gives it no values.
///
/// A field `remap` names has each document's values rewritten through it.
#[allow(clippy::too_many_arguments)]
fn merge_sorted_numeric_doc_values(
    sources: &[MergeSource],
    per_source_maps: &[HashMap<i32, i32>],
    reverse_maps: &[HashMap<i32, i32>],
    per_source_live_ids: &[Vec<i32>],
    doc_order: &[(usize, i32)],
    merged_fields: &[FieldInfo],
    remap: Option<&SortedNumericRemap>,
    only: Option<i32>,
) -> Result<Vec<MergedDocValuesField>> {
    let candidates = doc_values_candidates(
        sources,
        per_source_maps,
        per_source_live_ids,
        |s| s.sorted_numeric_doc_values,
        |f| f.entry.field_number,
    );
    let candidates: Vec<i32> = candidates
        .into_iter()
        .filter(|n| only.is_none_or(|o| o == *n))
        .collect();
    let by_number = index_source_fields(
        sources,
        per_source_live_ids,
        |s| s.sorted_numeric_doc_values,
        |f: &SourceSortedNumericDocValues| f.entry.field_number,
    );
    let mut merged = Vec::with_capacity(candidates.len());
    for merged_field_number in candidates {
        let per_source_entry = per_source_entries(&by_number, reverse_maps, merged_field_number);
        let mut readers: Vec<Option<doc_values::SortedNumericReader>> = per_source_entry
            .iter()
            .map(|e| e.map(|e| doc_values::SortedNumericReader::new(e.data, &e.entry)))
            .collect();
        let field_remap = remap.filter(|r| {
            merged_fields
                .iter()
                .find(|f| f.number == merged_field_number)
                .is_some_and(|f| r.fields.contains(&f.name))
        });
        let mut scratch: Vec<i64> = Vec::new();
        let mut column = doc_values::SortedNumericColumn::default();
        column.docs.reserve_exact(doc_order.len());
        column.counts.reserve_exact(doc_order.len());
        column.values.reserve_exact(doc_order.len());
        for (merged_doc_id, &(src_idx, doc_id)) in doc_order.iter().enumerate() {
            let Some(reader) = readers[src_idx].as_mut() else {
                continue;
            };
            reader.values(doc_id, &mut scratch)?;
            if let Some(r) = field_remap {
                r.apply(&mut scratch)?;
            }
            if !scratch.is_empty() {
                column.docs.push(merged_doc_id as i32);
                column.counts.push(scratch.len() as u32);
                column.values.extend_from_slice(&scratch);
            }
        }
        merged.push(MergedDocValuesField::SortedNumeric(
            merged_field_number,
            column,
        ));
    }
    Ok(merged)
}

/// Resolves one live doc's SORTED_SET ordinals for `doc_id`, regardless of
/// whether `entry.kind` collapsed to [`SortedSetKind::Single`] (one ordinal
/// or none) or stayed [`SortedSetKind::Multi`] (zero or more via the same
/// [`SortedNumericEntry`] layout [`doc_values::sorted_numeric_values`]
/// already decodes). Kept for tests and single lookups; a merge holds a
/// [`SortedSetOrdsReader`].
#[cfg(test)]
fn sorted_set_doc_ordinals(data: &[u8], entry: &SortedSetEntry, doc_id: i32) -> Result<Vec<i64>> {
    match &entry.kind {
        SortedSetKind::Single(sorted) => Ok(doc_values::sorted_ord(data, sorted, doc_id)?
            .into_iter()
            .collect()),
        SortedSetKind::Multi { ords, .. } => {
            Ok(doc_values::sorted_numeric_values(data, ords, doc_id)?)
        }
    }
}

/// A forward cursor over one source's SORTED_SET ordinals, whichever of
/// [`SortedSetKind`]'s two shapes it was written in.
enum SortedSetOrdsReader<'a> {
    Single(Box<doc_values::NumericReader<'a>>),
    Multi(Box<doc_values::SortedNumericReader<'a>>),
}

impl<'a> SortedSetOrdsReader<'a> {
    fn new(data: &'a [u8], entry: &'a SortedSetEntry) -> Self {
        match &entry.kind {
            SortedSetKind::Single(sorted) => {
                Self::Single(Box::new(doc_values::NumericReader::new(data, &sorted.ords)))
            }
            SortedSetKind::Multi { ords, .. } => {
                Self::Multi(Box::new(doc_values::SortedNumericReader::new(data, ords)))
            }
        }
    }

    fn ords(&mut self, doc: i32, out: &mut Vec<i64>) -> Result<()> {
        match self {
            Self::Single(reader) => {
                out.clear();
                out.extend(reader.value(doc)?);
            }
            Self::Multi(reader) => reader.values(doc, out)?,
        }
        Ok(())
    }
}

/// Decodes one source's whole SORTED_SET term dictionary, in ordinal order --
/// the input [`OrdinalMap`] merges.
fn sorted_set_source_dict(data: &[u8], entry: &SortedSetEntry) -> Result<Vec<Vec<u8>>> {
    match &entry.kind {
        SortedSetKind::Single(sorted) => Ok(terms_dict::decode_all_terms(data, &sorted.terms)?),
        SortedSetKind::Multi { terms, .. } => Ok(terms_dict::decode_all_terms(data, terms)?),
    }
}

/// Merges SORTED_SET doc-values data across `sources`, one
/// [`MergedDocValuesField`] per merged field number any source has a
/// SORTED_SET column for, in `doc_order` --
/// `DocValuesConsumer.mergeSortedSetField`: an [`OrdinalMap`] over the
/// sources' dictionaries, then each document's ordinals remapped through it.
/// A source's per-document ordinals are ascending and unique, and the map is
/// monotonic, so the remapped ones are too. Dense or sparse as
/// [`merge_binary_doc_values`].
fn merge_sorted_set_doc_values(
    sources: &[MergeSource],
    per_source_maps: &[HashMap<i32, i32>],
    reverse_maps: &[HashMap<i32, i32>],
    per_source_live_ids: &[Vec<i32>],
    doc_order: &[(usize, i32)],
    only: Option<i32>,
) -> Result<Vec<MergedDocValuesField>> {
    let candidates = doc_values_candidates(
        sources,
        per_source_maps,
        per_source_live_ids,
        |s| s.sorted_set_doc_values,
        |f| f.entry.field_number,
    );
    let candidates: Vec<i32> = candidates
        .into_iter()
        .filter(|n| only.is_none_or(|o| o == *n))
        .collect();
    let by_number = index_source_fields(
        sources,
        per_source_live_ids,
        |s| s.sorted_set_doc_values,
        |f: &SourceSortedSetDocValues| f.entry.field_number,
    );
    let mut merged = Vec::with_capacity(candidates.len());
    for merged_field_number in candidates {
        let per_source_entry = per_source_entries(&by_number, reverse_maps, merged_field_number);
        let dicts: Vec<Option<Vec<Vec<u8>>>> = per_source_entry
            .iter()
            .map(|e| {
                e.map(|e| sorted_set_source_dict(e.data, &e.entry))
                    .transpose()
            })
            .collect::<Result<_>>()?;
        let mut live = Vec::with_capacity(sources.len());
        for (idx, entry) in per_source_entry.iter().enumerate() {
            live.push(match (entry, &dicts[idx]) {
                (Some(e), Some(dict)) => {
                    let mut reader = SortedSetOrdsReader::new(e.data, &e.entry);
                    live_ords(
                        &sources[idx],
                        &per_source_live_ids[idx],
                        dict.len(),
                        |doc, out| reader.ords(doc, out),
                    )?
                }
                _ => None,
            });
        }
        let map = OrdinalMap::build(dicts, live);
        let mut readers: Vec<Option<SortedSetOrdsReader>> = per_source_entry
            .iter()
            .map(|e| e.map(|e| SortedSetOrdsReader::new(e.data, &e.entry)))
            .collect();
        let mut scratch: Vec<i64> = Vec::new();
        let mut column = doc_values::SortedSetOrdsColumn::default();
        column.docs.reserve_exact(doc_order.len());
        column.counts.reserve_exact(doc_order.len());
        column.ords.reserve_exact(doc_order.len());
        for (merged_doc_id, &(src_idx, doc_id)) in doc_order.iter().enumerate() {
            let Some(reader) = readers[src_idx].as_mut() else {
                continue;
            };
            reader.ords(doc_id, &mut scratch)?;
            if scratch.is_empty() {
                continue;
            }
            for &ord in &scratch {
                column.ords.push(map.global(src_idx, ord).ok_or(
                    Error::SortedSetDocValuesFieldMissingInSource {
                        merged_field_number,
                    },
                )?);
            }
            column.docs.push(merged_doc_id as i32);
            column.counts.push(scratch.len() as u32);
        }
        column.dict = map.dict;
        merged.push(MergedDocValuesField::SortedSetOrds(
            merged_field_number,
            column,
        ));
    }
    Ok(merged)
}

/// Same shape and same rules as [`merge_numeric_doc_values`], for norms --
/// **every** norms field the sources declare, not one.
///
/// Returns one `(merged field number, one value per merged document)` pair
/// per field, in ascending merged-field-number order so a merged `.nvm`'s
/// entry order is a function of the schema rather than of which source
/// happened to be first.
///
/// Every field goes into one `.nvm`/`.nvd` pair through `norms::write_fields`,
/// as `Lucene90NormsConsumer` gets one `addNormsField` call per field.
///
/// Which fields get a column, and what a document with no norm becomes, are
/// both Java's:
///
/// ```java
/// for (FieldInfo mergeFieldInfo : mergeState.mergeFieldInfos) {
///   if (mergeFieldInfo.hasNorms()) {
///     mergeNormsField(mergeFieldInfo, mergeState);
///   }
/// }
/// ```
///
/// -- the candidate list comes from the **merged** field infos
/// (`hasNorms() == !omitNorms && indexOptions != NONE`), not from which
/// sources happen to carry norms files, and `mergeNormsField` then merges
/// only the sources whose own `FieldInfo` says `hasNorms()`:
/// a source that never declared the field, and a document inside a source
/// whose norm column simply has no entry for it, both contribute *nothing*
/// and appear as a gap in the merged column, which is written sparse.
///
/// A segment whose documents do not all carry the field has a sparse column,
/// and merging two segments that disagree about a field is the ordinary case
/// for a schema that grew, so neither is an error.
fn merge_norms(
    sources: &[MergeSource],
    merged_fields: &[FieldInfo],
    reverse_maps: &[HashMap<i32, i32>],
    per_source_live_ids: &[Vec<i32>],
    doc_order: &[(usize, i32)],
) -> Result<Vec<MergedNorms>> {
    let candidates: Vec<i32> = merged_fields
        .iter()
        .filter(|f| !f.omit_norms && f.index_options != IndexOptions::None)
        .map(|f| f.number)
        .collect();
    // Same "fully-deleted source can't affect the merged output" exemption
    // as merge_numeric_doc_values: such a source indexes no fields.
    let by_number = index_source_fields(
        sources,
        per_source_live_ids,
        |s| s.norms,
        |f: &SourceNorms| f.entry.field_number,
    );

    let mut merged: Vec<MergedNorms> = Vec::with_capacity(candidates.len());
    for merged_field_number in candidates {
        // Java: NormsConsumer.mergeNormsField retains one NumericDocValues
        // iterator per source. Reuse sparse cursors instead of rescanning each
        // source's document prefix for every merged document. A source whose
        // reverse map has no entry is `readerFieldInfo == null` in
        // `mergeNormsField`: it never declared the field, so it adds no sub.
        let mut per_source_readers: Vec<Option<norms::NormsReader<'_>>> =
            per_source_entries(&by_number, reverse_maps, merged_field_number)
                .into_iter()
                .map(|entry| entry.map(|entry| norms::NormsReader::new(entry.data, &entry.entry)))
                .collect();

        // Dense (missing = 1) for every field, since the postings impacts
        // read it; the sparse wire column only once a document turns out to
        // have no norm, seeded from the all-present prefix.
        let mut values: Vec<i64> = Vec::with_capacity(doc_order.len());
        let mut sparse: Option<Vec<(i32, i64)>> = None;
        for (merged_doc_id, &(src_idx, doc_id)) in doc_order.iter().enumerate() {
            let value = match per_source_readers[src_idx].as_mut() {
                Some(reader) => reader.value(doc_id)?,
                None => None,
            };
            match (value, sparse.as_mut()) {
                (Some(value), None) => values.push(value),
                (Some(value), Some(sparse)) => {
                    values.push(value);
                    sparse.push((merged_doc_id as i32, value));
                }
                (None, sparse_so_far) => {
                    if sparse_so_far.is_none() {
                        sparse = Some(
                            values
                                .iter()
                                .enumerate()
                                .map(|(doc, &value)| (doc as i32, value))
                                .collect(),
                        );
                    }
                    values.push(1);
                }
            }
        }
        merged.push(MergedNorms {
            field_number: merged_field_number,
            values,
            sparse,
        });
    }
    Ok(merged)
}

/// One merged norms field, as [`merge_norms`] returns it.
#[derive(Debug, PartialEq)]
struct MergedNorms {
    field_number: i32,
    /// One value per merged document, a missing norm as `1` -- what the
    /// postings writer's impacts read, and the `.nvd` column itself when
    /// `sparse` is `None`. A present zero stays zero.
    values: Vec<i64>,
    /// `(merged doc id, norm)` for the documents that have one, or `None`
    /// when every document does.
    sparse: Option<Vec<(i32, i64)>>,
}

impl MergedNorms {
    fn as_norms_field(&self) -> norms::NormsField<'_> {
        match &self.sparse {
            None => norms::NormsField::Dense(self.field_number, &self.values),
            Some(sparse) => norms::NormsField::Sparse(self.field_number, sparse),
        }
    }

    /// Back to one `Option` per merged document.
    #[cfg(test)]
    fn options(&self) -> Vec<Option<i64>> {
        match &self.sparse {
            None => self.values.iter().copied().map(Some).collect(),
            Some(sparse) => {
                let mut out = vec![None; self.values.len()];
                for &(doc, value) in sparse {
                    out[doc as usize] = Some(value);
                }
                out
            }
        }
    }
}

/// One written term-vectors segment: `.tvd`, `.tvx`, `.tvm`.
type TermVectorsFiles = (Vec<u8>, Vec<u8>, Vec<u8>);

/// Which of `Lucene90CompressingTermVectorsWriter.merge`'s two paths a
/// source qualifies for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TermVectorsMergeStrategy {
    /// Copy whole compressed chunks verbatim (`copyChunks`).
    Bulk,
    /// Decode each document and re-encode it (`addAllDocVectors`).
    PerDoc,
}

/// Port of `Lucene90CompressingTermVectorsWriter.canPerformBulkMerge`, minus
/// the reader-class/version half [`term_vectors::open`] already enforces (this
/// port has exactly one `VERSION_CURRENT`, one `packedIntsVersion` and one
/// compression mode, and refuses to open anything else).
fn term_vectors_merge_strategy(
    writer: &term_vectors::TermVectorsWriter,
    source: &MergeSource,
    matching: bool,
) -> TermVectorsMergeStrategy {
    match source.term_vectors {
        // "its not worth fine-graining this if there are deletions", plus
        // the chunk-size/dirtiness pair `can_bulk_copy` owns, plus
        // `MatchingReaders` -- a copied chunk encodes its own field numbers.
        Some(reader) if matching && source.live_docs.is_none() && writer.can_bulk_copy(reader) => {
            TermVectorsMergeStrategy::Bulk
        }
        _ => TermVectorsMergeStrategy::PerDoc,
    }
}

/// Java: `TermVectorsWriter.merge`, over mapped owned document values.
fn write_installed_term_vectors(
    factory: &dyn lucene_codecs::term_vectors_writer::TermVectorsWriterFactory,
    state: lucene_codecs::term_vectors_writer::TermVectorsWriteState<'_>,
    sources: &[MergeSource],
    per_source_maps: &[HashMap<i32, i32>],
    doc_order: &[(usize, i32)],
    infos: &[FieldInfo],
) -> Result<Vec<String>> {
    // Java checks every supplied reader, including a fully deleted source.
    {
        use rayon::prelude::*;
        in_source_order(sources.par_iter().map(|source| match source.term_vectors {
            Some(reader) => Ok(reader.check_integrity()?),
            None => Ok(()),
        }))?;
    }
    let mut cursors: Vec<_> = sources
        .iter()
        .map(|_| term_vectors::ChunkCursor::new())
        .collect();
    crate::term_vectors_writer::write_installed(factory, state, |writer| -> Result<()> {
        for &(source, doc_id) in doc_order {
            let mut document = match sources[source].term_vectors {
                Some(reader) => cursors[source]
                    .document(reader, doc_id)?
                    .unwrap_or_default(),
                None => TermVectorsDocument::default(),
            };
            for field in &mut document.fields {
                field.field_number = *per_source_maps[source].get(&field.field_number).ok_or(
                    Error::UnknownSourceFieldNumber {
                        field_number: field.field_number,
                    },
                )?;
            }
            crate::term_vectors_writer::write_document(writer, &document, infos)?;
        }
        writer.finish(state.max_doc)?;
        Ok(())
    })
}

/// Writes the merged segment's `.tvd`/`.tvx`/`.tvm` by walking `doc_order`
/// -- the (source index, source doc id) pairs, in merged doc id order, that
/// both merge entry points already compute -- or `Ok(None)` if no source has
/// a term-vectors reader at all.
///
/// This is the port of `Lucene90CompressingTermVectorsWriter.merge`'s
/// `DocIDMerger` loop, and it has the same two-way shape as its
/// stored-fields twin [`write_merged_stored_fields`]: a run of consecutive
/// documents from a bulk-eligible source is handed to `copyChunks` in one
/// call (Java's `while ((sub = docIDMerger.next()) == current)` run
/// detection), and every other document is decoded and re-encoded with its
/// field numbers remapped.
///
/// Unlike postings, term-vector data is entirely doc-local: one source's
/// [`TermVectorsReader::document`] call already returns a fully
/// self-contained [`TermVectorsDocument`], so there is no cross-source
/// term-level combination to do -- only renumbering, and on the bulk path
/// not even that (`MatchingReaders` guarantees the numbers survive).
///
/// A source with no term-vectors reader contributes an empty document per
/// doc, exactly as `TermVectorsWriter.merge` does when
/// `mergeState.termVectorsReaders[i]` is null.
fn write_merged_term_vectors(
    format: Option<&'static lucene_codecs::compression::CompressingFormat>,
    sources: &[MergeSource],
    per_source_maps: &[HashMap<i32, i32>],
    doc_order: &[(usize, i32)],
    merged_segment_id: &[u8; ID_LENGTH],
) -> Result<Option<TermVectorsFiles>> {
    if sources.iter().all(|s| s.term_vectors.is_none()) {
        return Ok(None);
    }

    let mut writer = match format {
        None => term_vectors::TermVectorsWriter::new(merged_segment_id, ""),
        Some(format) => term_vectors::TermVectorsWriter::with_format(merged_segment_id, format),
    };
    let matching = matching_readers(sources, per_source_maps);
    // Java's `reader.checkIntegrity(mergeState.oneMerge)`, run on every
    // source *before* a strategy is picked. It matters most for the bulk
    // path: that path copies a source's compressed bytes verbatim and
    // then writes a freshly computed, valid footer over them, so a bit
    // flip in the source would be laundered into a merged segment that
    // passes every checksum from then on. `term_vectors::open` only
    // validates the footer's shape (`retrieve_checksum`). In parallel, the
    // first corrupt source in source order reported.
    {
        use rayon::prelude::*;
        in_source_order(sources.par_iter().map(|source| match source.term_vectors {
            Some(reader) => Ok(reader.check_integrity()?),
            None => Ok(()),
        }))?;
    }
    let strategies: Vec<TermVectorsMergeStrategy> = sources
        .iter()
        .zip(&matching)
        .map(|(source, &m)| term_vectors_merge_strategy(&writer, source, m))
        .collect();
    // Java re-reads a chunk's metadata arrays on every `get(doc)`; this port
    // keeps the decoded chunk, so a run of documents from one source costs
    // one decode per chunk instead of one per document.
    let mut cursors: Vec<term_vectors::ChunkCursor> = (0..sources.len())
        .map(|_| term_vectors::ChunkCursor::new())
        .collect();

    let mut i = 0usize;
    while i < doc_order.len() {
        let (src_idx, doc_id) = doc_order[i];
        match strategies[src_idx] {
            TermVectorsMergeStrategy::Bulk => {
                let reader = sources[src_idx]
                    .term_vectors
                    .expect("the bulk strategy is only chosen for a source with a reader");
                // ARITH: identical to `write_merged_stored_fields`' run
                // detection -- `doc_order`'s doc ids are drawn from each
                // source's own `0..max_doc`, so `doc_id + 1 <= i32::MAX`, and
                // `i`/`j` index a `Vec`.
                #[allow(clippy::arithmetic_side_effects)]
                let (to_doc, j) = {
                    let mut to_doc = doc_id + 1;
                    let mut j = i + 1;
                    while j < doc_order.len() && doc_order[j] == (src_idx, to_doc) {
                        to_doc += 1;
                        j += 1;
                    }
                    (to_doc, j)
                };
                // Shared with the per-document path below, for the same
                // reason as `write_merged_stored_fields`: an index-sorted
                // merge's runs are short, and every run's ragged ends read
                // out of a chunk the next run will read again.
                writer.copy_chunks_with_cursor(reader, &mut cursors[src_idx], doc_id, to_doc)?;
                i = j;
            }
            TermVectorsMergeStrategy::PerDoc => {
                let mut doc = match sources[src_idx].term_vectors {
                    Some(reader) => cursors[src_idx]
                        .document(reader, doc_id)?
                        .unwrap_or_default(),
                    None => TermVectorsDocument::default(),
                };
                let map = &per_source_maps[src_idx];
                for field in &mut doc.fields {
                    field.field_number =
                        *map.get(&field.field_number)
                            .ok_or(Error::UnknownSourceFieldNumber {
                                field_number: field.field_number,
                            })?;
                }
                writer.add_document(&doc);
                // ARITH: as above -- `i < doc_order.len()`.
                #[allow(clippy::arithmetic_side_effects)]
                {
                    i += 1;
                }
            }
        }
    }
    Ok(Some(writer.finish()))
}

/// One source segment's vector data: its already-opened flat store
/// (`.vec`/`.vemf`) and, when the segment has one, its HNSW graph
/// (`.vem`/`.vex`). A segment may legitimately have the flat pair and no
/// graph (`numLevels = 0`, i.e. it was below
/// [`lucene_codecs::hnsw::HNSW_GRAPH_THRESHOLD`]); a source with a graph and
/// no flat store cannot exist, so the flat reader is not optional.
///
/// One reader pair covers every vector field of the segment, so this is
/// per-source rather than per-field -- unlike [`SourcePostings`]/
/// [`SourcePoints`], which name one field each because their readers are
/// opened per field.
pub struct SourceVectors<'a> {
    /// One per `PerFieldKnnVectorsFormat` group the source holds (a segment
    /// an older Lucene wrote with per-field quantized formats has several,
    /// each under its own suffix); a field is served by the group whose flat
    /// reader has it.
    pub groups: Vec<VectorGroup<'a>>,
}

/// One per-field vector group of a source: its raw vectors and, when it has
/// one, its `Lucene99HnswVectorsFormat` graph.
#[derive(Clone, Copy)]
pub struct VectorGroup<'a> {
    pub flat: &'a vectors::FlatVectorsReader<'a>,
    pub graph: Option<&'a hnsw_vectors::HnswVectorsReader<'a>>,
    /// The quantized codes of a `Lucene104ScalarQuantizedVectorsFormat`
    /// group -- the one reader `Lucene104ScalarQuantizedVectorsWriter
    /// .getCentroid` recognises, so the only centroids a merge reuses.
    pub flat_quantized: Option<&'a scalar_quantized_vectors::ScalarQuantizedVectorsReader<'a>>,
    /// The IVF files of a `IvfVectorsFormat` group, whose
    /// clusters a merge into that format reuses.
    pub ivf: Option<&'a crate::ivf_vectors::IvfInputs>,
}

impl<'a> SourceVectors<'a> {
    /// A source with one group.
    pub fn single(
        flat: &'a vectors::FlatVectorsReader<'a>,
        graph: Option<&'a hnsw_vectors::HnswVectorsReader<'a>>,
    ) -> Self {
        SourceVectors {
            groups: vec![VectorGroup {
                flat,
                graph,
                flat_quantized: None,
                ivf: None,
            }],
        }
    }

    /// The group that holds `field_number` (the first, for a field no group
    /// has -- its lookups then fail as they always did).
    fn group_of(&self, field_number: i32) -> VectorGroup<'a> {
        self.groups
            .iter()
            .find(|g| g.flat.field(field_number).is_some())
            .or(self.groups.first())
            .copied()
            .expect("a SourceVectors has at least one group")
    }
}

/// Per-merge settings that are not derivable from the sources.
///
/// Only the HNSW graph parameters live here so far, and they matter only for
/// the from-scratch rebuild inside
/// [`lucene_codecs::hnsw_vectors::merge_one_field`]: when a source graph is
/// reused as the base, the merged graph inherits **its** `maxConn`, exactly
/// as `IncrementalHnswGraphMerger` does.
///
/// `postings_formats` is `PerFieldMergeState`'s view of the *merged*
/// segment's codec: the fields routed to a non-default postings format, as
/// [`crate::index_writer::IndexWriter::set_postings_format_for_field`] set
/// them. The sources' own routing is irrelevant -- Java re-groups on write.
#[derive(Debug, Clone)]
pub struct MergeOptions {
    pub hnsw_m: i32,
    pub hnsw_beam_width: i32,
    pub postings_formats: Vec<(String, per_field_postings::Lucene104PostingsFormat)>,
    /// The fields routed to a non-default doc-values format
    /// ([`crate::index_writer::IndexWriter::set_doc_values_format_for_field`]);
    /// regrouped on write like `postings_formats`.
    pub doc_values_formats: Vec<(String, per_field_doc_values::Lucene90DocValuesFormat)>,
    /// What replaces the default `Lucene90DocValuesFormat()` for a field
    /// nothing else routes
    /// ([`crate::index_writer::IndexWriter::set_default_doc_values_format`]).
    pub default_doc_values_format: per_field_doc_values::Lucene90DocValuesFormat,
    /// The Lucene release whose index formats the merged segment is written in.
    pub index_format: lucene_codecs::index_format::IndexFormat,
    /// What replaces the default `Lucene104PostingsFormat()` for a field
    /// nothing else routes
    /// ([`crate::index_writer::IndexWriter::set_default_postings_format`]).
    pub default_postings_format: per_field_postings::Lucene104PostingsFormat,
    /// Whether `BEST_SPEED` stored fields are written with the
    /// high-compression LZ4 match finder
    /// ([`crate::index_writer::IndexWriter::set_merge_lz4_high_compression`]).
    pub stored_fields_lz4_high_compression: bool,
    /// The fewest vectors IVF fields of a sorted merge cluster a partition
    /// of the first sort field alone with, `None` for no per-partition
    /// clusters ([`crate::index_writer::IndexWriter::set_ivf_partition_clusters`]).
    pub ivf_partition_clusters: Option<usize>,
    /// Per source, whether its stored fields were already written that way
    /// (its `.si` records [`crate::segment_writer::LZ4_HC_ATTRIBUTE`]); a
    /// missing entry is `false`. With the option on, only these sources'
    /// chunks are copied verbatim.
    pub lz4_high_compression_sources: Vec<bool>,
    /// The least source `.fdt` bytes for which a
    /// [`stored_fields::Mode::ZstdPaged`] merge trains a dictionary
    /// ([`crate::index_writer::IndexWriter::set_zstd_dictionary_min_bytes`]).
    pub zstd_dictionary_min_bytes: u64,
    /// The fields routed to a KNN vectors format other than the default
    /// `Lucene99HnswVectorsFormat(hnsw_m, hnsw_beam_width)`
    /// ([`crate::index_writer::IndexWriter::set_knn_vectors_format_for_field`]).
    pub knn_vectors_formats: Vec<(String, per_field_knn_vectors::KnnVectorsFormat)>,
    /// The writer's codec ([`crate::index_writer::IndexWriter::set_codec`]),
    /// consulted for a field the lists above do not route and for its optional
    /// stored-fields writer factory.
    pub codec: Option<std::sync::Arc<dyn lucene_codecs::codec::Lucene104Codec>>,
    /// `OneMerge.reorder`'s `Sorter.DocMap` as `newToOld` over every
    /// source's documents concatenated (deleted ones included): the merged
    /// segment holds the live ones in that order -- `SortingCodecReader`
    /// over the merged view. Refused together with an index sort, and when
    /// it is not a permutation of the merged view's documents.
    pub reorder: Option<Vec<i32>>,
    /// Values of the named SORTED_NUMERIC fields rewritten on the way
    /// through -- see [`SortedNumericRemap`].
    pub sorted_numeric_remap: Option<std::sync::Arc<SortedNumericRemap>>,
    /// What the merged `.si`'s diagnostics say produced it.
    pub kind: MergeKind,
    /// Diagnostics the merge's hooks add to the merged `.si`
    /// (`OneMerge.setMergeInfo`), after the writer's own.
    pub extra_diagnostics: Vec<(String, String)>,
    /// `.si` attributes the merged segment records after the writer's own,
    /// a later value for a key replacing an earlier one.
    pub extra_attributes: Vec<(String, String)>,
}

/// The diagnostics key OpenSearch's `ShuffleForcedMergePolicy` marks an
/// interleaved merged segment with (`isInterleavedSegment`).
pub const SHUFFLE_MERGE_KEY: &str = "opensearch.shuffle_merge";

/// The writer operation a merge carries out, as the merged segment's
/// diagnostics record it (`IndexWriter.setDiagnostics`' source and details).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeKind {
    /// A `OneMerge` (`_mergeInit`): source `merge`, with `mergeFactor` (the
    /// number of sources) and `mergeMaxNumSegments` (`OneMerge.maxNumSegments`:
    /// `forceMerge`'s argument for a forced merge, `-1` otherwise).
    ///
    /// `shuffled`: OpenSearch's `ShuffleForcedMergePolicy` interleaved the
    /// sources, which the diagnostics record as `opensearch.shuffle_merge`.
    Merge {
        max_num_segments: i32,
        shuffled: bool,
    },
    /// `addIndexes(CodecReader...)`'s merge: source
    /// `addIndexes(CodecReader...)`, no details.
    AddIndexes,
}

impl Default for MergeKind {
    fn default() -> Self {
        MergeKind::Merge {
            max_num_segments: -1,
            shuffled: false,
        }
    }
}

impl MergeKind {
    /// [`segment_info::writer_diagnostics`] for a merge of `merge_factor`
    /// sources.
    pub fn diagnostics(self, merge_factor: usize, version: LuceneVersion) -> Vec<(String, String)> {
        match self {
            MergeKind::Merge {
                max_num_segments,
                shuffled,
            } => {
                let mut diagnostics = segment_info::writer_diagnostics(
                    segment_info::SOURCE_MERGE,
                    version,
                    &[
                        ("mergeMaxNumSegments", max_num_segments.to_string()),
                        ("mergeFactor", merge_factor.to_string()),
                    ],
                );
                if shuffled {
                    diagnostics.push((SHUFFLE_MERGE_KEY.to_string(), String::new()));
                }
                diagnostics
            }
            MergeKind::AddIndexes => segment_info::writer_diagnostics(
                segment_info::SOURCE_ADDINDEXES_READERS,
                version,
                &[],
            ),
        }
    }
}

/// `OrdinalMappingLeafReader`'s view of a merge source (lucene-facet's
/// `TaxonomyMergeUtils`): in each of `fields`, every SORTED_NUMERIC value
/// `v` becomes `map[v]`, and each document's values are sorted again.
/// Java wraps the source readers; this port's merge reads segments, not
/// readers, so the rewrite rides along as a merge option.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SortedNumericRemap {
    /// The fields whose values are rewritten.
    pub fields: Vec<String>,
    /// `map[old] = new`.
    pub map: Vec<i32>,
}

impl SortedNumericRemap {
    /// Rewrites one document's values in place and sorts them.
    fn apply(&self, values: &mut [i64]) -> Result<()> {
        for v in values.iter_mut() {
            let mapped = usize::try_from(*v)
                .ok()
                .and_then(|i| self.map.get(i))
                .ok_or_else(|| {
                    Error::InvalidRemap(format!(
                        "value {v} is outside the map of {} ordinals",
                        self.map.len()
                    ))
                })?;
            *v = i64::from(*mapped);
        }
        values.sort_unstable();
        Ok(())
    }
}

impl MergeOptions {
    /// Whether the codec's `knnVectorsFormat()` is a format itself rather
    /// than per-field routing.
    fn knn_vectors_direct(&self) -> bool {
        self.codec
            .as_ref()
            .is_some_and(|c| c.knn_vectors_format_is_direct())
    }

    /// The KNN vectors format `field` is routed to.
    fn knn_vectors_format_for(&self, field: &str) -> per_field_knn_vectors::KnnVectorsFormat {
        self.knn_vectors_formats
            .iter()
            .find(|(name, _)| name == field)
            .map(|(_, format)| *format)
            .or_else(|| {
                self.codec
                    .as_ref()
                    .map(|c| c.knn_vectors_format_for_field(field))
            })
            .unwrap_or(per_field_knn_vectors::KnnVectorsFormat::Hnsw {
                max_conn: self.hnsw_m,
                beam_width: self.hnsw_beam_width,
            })
    }

    /// The doc-values format `field` is routed to.
    fn doc_values_format_for(&self, field: &str) -> per_field_doc_values::Lucene90DocValuesFormat {
        let disi_file = self.index_format.doc_values_disi_file();
        // A field routed explicitly keeps its format, even the default one.
        if let Some((_, format)) = self
            .doc_values_formats
            .iter()
            .find(|(name, _)| name == field)
        {
            return format.with_disi_file(disi_file);
        }
        let routed = self
            .codec
            .as_ref()
            .map(|c| c.doc_values_format_for_field(field))
            .unwrap_or_default();
        if routed == per_field_doc_values::Lucene90DocValuesFormat::default() {
            self.default_doc_values_format
        } else {
            routed
        }
        .with_disi_file(disi_file)
    }

    /// The postings format `field` is routed to.
    fn postings_format_for(&self, field: &str) -> per_field_postings::Lucene104PostingsFormat {
        route_postings_format(
            field,
            &self.postings_formats,
            self.codec.as_deref(),
            self.default_postings_format,
        )
    }
}

/// `field`'s postings format: its explicit routing, else the codec's, with
/// the default `Lucene104PostingsFormat()` replaced by `default` -- shared by
/// flush ([`crate::index_writer::IndexingConfig::postings_format_for`]) and
/// merge ([`MergeOptions`]).
pub(crate) fn route_postings_format(
    field: &str,
    explicit: &[(String, per_field_postings::Lucene104PostingsFormat)],
    codec: Option<&dyn lucene_codecs::codec::Lucene104Codec>,
    default: per_field_postings::Lucene104PostingsFormat,
) -> per_field_postings::Lucene104PostingsFormat {
    if let Some((_, format)) = explicit.iter().find(|(name, _)| name == field) {
        return *format;
    }
    let routed = codec
        .map(|c| c.postings_format_for_field(field))
        .unwrap_or_default();
    if routed == per_field_postings::Lucene104PostingsFormat::default() {
        default
    } else {
        routed
    }
}

impl Default for MergeOptions {
    fn default() -> Self {
        MergeOptions {
            hnsw_m: hnsw::DEFAULT_MAX_CONN,
            hnsw_beam_width: hnsw::DEFAULT_BEAM_WIDTH,
            postings_formats: Vec::new(),
            doc_values_formats: Vec::new(),
            default_doc_values_format: Default::default(),
            index_format: Default::default(),
            default_postings_format: Default::default(),
            stored_fields_lz4_high_compression: false,
            ivf_partition_clusters: None,
            lz4_high_compression_sources: Vec::new(),
            zstd_dictionary_min_bytes: stored_fields::ZSTD_PAGED_DICTIONARY_MIN_BYTES,
            knn_vectors_formats: Vec::new(),
            codec: None,
            reorder: None,
            sorted_numeric_remap: None,
            kind: MergeKind::default(),
            extra_diagnostics: Vec::new(),
            extra_attributes: Vec::new(),
        }
    }
}

/// The four files a vector merge writes, plus the merged field numbers that
/// actually ended up with vectors (which is what the merged `.fnm` may claim
/// -- see [`describe_written_files`]).
struct MergedVectorFiles {
    /// One per `PerFieldKnnVectorsFormat` instance: its codec suffix and its
    /// files by extension.
    groups: Vec<(String, GroupFiles)>,
    /// Every field written: `(number, (format name, suffix number))`, no
    /// pair for a direct `knnVectorsFormat()` (no per-field attributes).
    written: Vec<(i32, Option<(&'static str, u32)>)>,
    /// Files the merge already wrote to the directory (each group's `.vec`,
    /// written before its graph is built so it is read from the file rather
    /// than held in memory).
    written_early: Vec<String>,
}

/// Merges KNN vectors across `sources` -- the port of `SegmentMerger`'s
/// `mergeVectorValues` step, i.e. `Lucene99HnswVectorsWriter.mergeOneField`
/// over `Lucene99FlatVectorsWriter.mergeOneFlatVectorField`.
///
/// Returns `Ok(None)` when no merged field ends up with a single vector.
///
/// The two halves, in the order they have to happen:
///
/// 1. **The flat store defines the merged ordinal space.**
///    [`lucene_codecs::vectors::FlatVectorsWriter::merge_one_flat_vector_field`]
///    assigns merged ordinals by ascending merged doc id and copies the
///    surviving vectors' bytes straight across, so nothing is decoded and,
///    with no deletions and no index sort, each source costs one `memcpy`.
/// 2. **The graphs are merged against that space**, by reopening exactly the
///    `.vec`/`.vemf` just written -- the same "build the graph over the bytes
///    that were written, not over the ones we meant to write" rule
///    `IndexWriter::build_vectors_output` follows at flush time, and the same
///    thing `mergeOneField` does. `merge_one_field` reuses the largest usable
///    source graph rather than rebuilding, which is the whole point of the
///    incremental merger.
///
/// A source whose own `FieldInfo` gives the field `vector_dimension == 0`
/// contributes nothing and is skipped entirely, for both halves and in the
/// same order -- Java's
/// `hasVectorValues(mergeState.fieldInfos[i], fieldInfo.name)` filter in
/// `buildAndWriteGraph`. The count matters: "every source contributed a
/// usable graph" is what decides whether any merged ordinal still has to be
/// inserted from scratch.
///
/// There is deliberately **no bulk-copy fast path** and no `MatchingReaders`
/// consultation: the merged ordinal space is new, so Lucene re-writes the
/// data file on every merge too.
#[allow(clippy::too_many_arguments)]
fn merge_vectors(
    dir: &dyn Directory,
    merged_segment_name: &str,
    sources: &[MergeSource],
    per_source_maps: &[HashMap<i32, i32>],
    doc_id_maps: &[Vec<i32>],
    merged_fields: &[FieldInfo],
    merged_max_doc: i32,
    options: &MergeOptions,
    merged_segment_id: &[u8; ID_LENGTH],
    partitions: Option<crate::ivf_vectors::Partitions<'_>>,
) -> Result<Option<MergedVectorFiles>> {
    if sources.iter().all(|s| s.vectors.is_none()) {
        return Ok(None);
    }
    let reverse_maps = invert_field_number_maps(per_source_maps);

    let mut plan: Vec<(&FieldInfo, Vec<FieldSource>)> = Vec::new();
    for merged_field in merged_fields {
        if merged_field.vector_dimension <= 0 {
            continue;
        }
        let mut field_sources = Vec::new();
        for (src_idx, (source, reverse)) in sources.iter().zip(&reverse_maps).enumerate() {
            let Some(source_vectors) = source.vectors else {
                continue;
            };
            let Some(&original_number) = reverse.get(&merged_field.number) else {
                continue;
            };
            // Java's `hasVectorValues(mergeState.fieldInfos[i], fieldInfo.name)`:
            // a source that never indexed the field contributes nothing, and
            // must not be counted as a graph-less source either.
            let has_vectors = source
                .field_infos
                .iter()
                .any(|f| f.number == original_number && f.vector_dimension > 0);
            if !has_vectors {
                continue;
            }
            let ord_to_doc = source_ord_to_doc(
                source_vectors.group_of(original_number).flat,
                original_number,
                merged_field.vector_encoding,
            )?;
            field_sources.push(FieldSource {
                source_index: src_idx,
                original_field_number: original_number,
                ord_to_doc,
            });
        }
        // A field every one of whose vectors was deleted is still written,
        // as an empty one: `KnnVectorsWriter.merge` merges every field of
        // the merged `FieldInfos` that `hasVectorValues()`, so the merged
        // segment keeps the field (`getFloatVectorValues` of size 0), as
        // `testDeleteAllVectorDocs` reads it.
        plan.push((merged_field, field_sources));
    }
    if plan.is_empty() {
        return Ok(None);
    }

    // `PerFieldKnnVectorsFormat.FieldsWriter.mergeOneField`: every merged
    // field, in field-number order, to the instance of the format it is
    // routed to now (the sources' routing is not consulted).
    let names: Vec<(i32, &str)> = plan
        .iter()
        .map(|(f, _)| (f.number, f.name.as_str()))
        .collect();
    let plan_by_number: HashMap<i32, &(&FieldInfo, Vec<FieldSource>)> =
        plan.iter().map(|entry| (entry.0.number, entry)).collect();
    let format_for = |name: &str| options.knn_vectors_format_for(name);
    let groups = match names.first() {
        Some(&(_, first)) if options.knn_vectors_direct() => {
            per_field_knn_vectors::direct_group(&names, format_for(first))
        }
        _ => per_field_knn_vectors::group_fields(&names, &format_for),
    };
    let mut out = MergedVectorFiles {
        groups: Vec::with_capacity(groups.len()),
        written: Vec::new(),
        written_early: Vec::new(),
    };
    for group in &groups {
        let suffix = group.codec_suffix();
        let group_plan: Vec<&(&FieldInfo, Vec<FieldSource>)> = group
            .field_numbers
            .iter()
            .filter_map(|n| plan_by_number.get(n).copied())
            .collect();
        let files = merge_vector_group(
            dir,
            merged_segment_name,
            sources,
            doc_id_maps,
            &group_plan,
            group.format,
            merged_max_doc,
            merged_segment_id,
            &suffix,
            partitions,
        )?;
        for n in &group.field_numbers {
            out.written.push((
                *n,
                (!group.direct).then_some((group.format.name(), group.suffix)),
            ));
        }
        let (files, early) = files;
        out.written_early.extend(early);
        out.groups.push((suffix, files));
    }
    Ok(Some(out))
}

/// One KNN format instance's files, by extension.
type GroupFiles = Vec<(&'static str, Vec<u8>)>;

/// One source's centroid (when visible), every vector it holds for a field,
/// and whether it has deletions -- a [`scalar_quantized_vectors::QuantizedMergeSource`]'s
/// owned parts.
type OwnedQuantizedSource<'a> = (
    Option<&'a [f32]>,
    std::borrow::Cow<'a, [f32]>,
    bool,
    Option<Vec<bool>>,
);

/// One source's contribution to one merged vector field, resolved once and
/// reused by both halves of [`merge_vector_group`] so the flat merge and the
/// graph merge cannot disagree about which sources took part or in what
/// order. `ord_to_doc` is that source's own ordinal -> its own doc id,
/// materialised because both halves need it (the flat merge to place each
/// vector, the graph merge to build `IncrementalHnswGraphMerger`'s ordinal
/// maps).
struct FieldSource {
    source_index: usize,
    original_field_number: i32,
    ord_to_doc: Vec<i32>,
}

/// One `PerFieldKnnVectorsFormat` instance's merge
/// (`Lucene99HnswVectorsWriter.mergeOneField` or a flat format's
/// `mergeOneFlatVectorField`): the raw vectors, the quantized codes when the
/// format quantizes (`Lucene104ScalarQuantizedVectorsWriter
/// .mergeOneFlatVectorField`, centroids from `mergeAndRecalculateCentroids`),
/// then -- for a format with a graph -- the graphs, over the vectors just
/// written, scored with the quantized scorer for a quantized `FLOAT32` field
/// (`getRandomVectorScorerSupplierForMerge`) and the raw one otherwise.
/// Returns the instance's files by extension.
#[allow(clippy::too_many_arguments)]
/// Writes `vectors` (`dimension` floats each) to `name`, each vector
/// unit-normalized as [`lucene_ivf::prepare_vectors`] does for a cosine
/// field, a few thousand at a time.
fn write_normalized_vectors(
    dir: &dyn Directory,
    name: &str,
    dimension: usize,
    vectors: &[f32],
) -> Result<()> {
    const ROWS: usize = 4096;
    let mut out = dir.create_output(name)?;
    let mut rows: Vec<f32> = Vec::new();
    let mut bytes: Vec<u8> = Vec::new();
    for chunk in vectors.chunks(dimension.max(1).saturating_mul(ROWS)) {
        rows.clear();
        rows.extend_from_slice(chunk);
        lucene_ivf::prepare_vectors(lucene_ivf::Similarity::Cosine, dimension, &mut rows);
        bytes.clear();
        for x in &rows {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        out.write_bytes(&bytes);
    }
    out.close()?;
    Ok(())
}

fn merge_vector_group(
    dir: &dyn Directory,
    merged_segment_name: &str,
    sources: &[MergeSource],
    doc_id_maps: &[Vec<i32>],
    plan: &[&(&FieldInfo, Vec<FieldSource>)],
    format: per_field_knn_vectors::KnnVectorsFormat,
    merged_max_doc: i32,
    merged_segment_id: &[u8; ID_LENGTH],
    suffix: &str,
    partitions: Option<crate::ivf_vectors::Partitions<'_>>,
) -> Result<(GroupFiles, Vec<String>)> {
    let mut flat_writer =
        vectors::FlatVectorsWriter::new(merged_max_doc, merged_segment_id, suffix);
    // The merged vectors go to their file as they are copied: the merge
    // never holds the file whole.
    let vec_name = crate::index_writer::segment_file_name(merged_segment_name, suffix, "vec");
    flat_writer.stream_to(dir.create_output(&vec_name)?);
    for (merged_field, field_sources) in plan.iter().map(|p| (&p.0, &p.1)) {
        let encoding = merged_field.vector_encoding;
        let mut flat_sources: Vec<vectors::FlatVectorMergeSource<'_>> =
            Vec::with_capacity(field_sources.len());
        for fs in field_sources {
            let source_vectors = sources[fs.source_index]
                .vectors
                .expect("only sources with a flat reader are planned");
            let group = source_vectors.group_of(fs.original_field_number);
            let values = match encoding {
                VectorEncoding::Float32 => vectors::MergeSourceValues::Float32(
                    group.flat.float_vector_values(fs.original_field_number)?,
                ),
                VectorEncoding::Byte => vectors::MergeSourceValues::Byte(
                    group.flat.byte_vector_values(fs.original_field_number)?,
                ),
            };
            flat_sources.push(vectors::FlatVectorMergeSource {
                values,
                doc_map: &doc_id_maps[fs.source_index],
            });
        }
        flat_writer.merge_one_flat_vector_field(&vectors::MergedFlatVectorField {
            field_number: merged_field.number,
            encoding,
            similarity: merged_field.vector_similarity_function,
            dimension: merged_field.vector_dimension,
            sources: &flat_sources,
        })?;
    }
    let (vec_out, vemf_bytes) = flat_writer.finish_streamed();
    // Everything built over the vectors reads their file: the merge holds
    // no in-memory copy of every vector.
    if let Some(out) = vec_out {
        out.close()?;
    }
    let vec_input = dir.open(&vec_name)?;
    let vec_bytes: &[u8] = &vec_input;

    // Reopen exactly the bytes just written and build everything else over
    // *those*, so the graph's ordinals, the codes' ordinals and the flat
    // store's ordinals are the same fact rather than three derivations of it.
    let mut flat =
        vectors::FlatVectorsReader::open(&vemf_bytes, vec_bytes, merged_segment_id, suffix)?;
    flat.resolve_similarities(|number| {
        plan.iter()
            .find(|p| p.0.number == number)
            .map(|p| p.0.vector_similarity_function)
    })?;
    let mut files: Vec<(&'static str, Vec<u8>)> = Vec::new();
    let mut ivv_name: Option<String> = None;

    if let Some(encoding) = format.quantization() {
        let mut writer = scalar_quantized_vectors::ScalarQuantizedVectorsWriter::new(
            encoding,
            merged_max_doc,
            merged_segment_id,
            suffix,
        );
        for (merged_field, field_sources) in plan.iter().map(|p| (&p.0, &p.1)) {
            if merged_field.vector_encoding != VectorEncoding::Float32 {
                continue;
            }
            // Every source's vectors for the field, deleted ones included,
            // in its ordinal order -- what `calculateCentroid` iterates.
            let mut owned: Vec<OwnedQuantizedSource<'_>> = Vec::new();
            for fs in field_sources {
                let source = &sources[fs.source_index];
                let group = source
                    .vectors
                    .expect("only sources with a flat reader are planned")
                    .group_of(fs.original_field_number);
                let all = group
                    .flat
                    .float_vector_values(fs.original_field_number)?
                    .all_vectors()?;
                let live = match source.live_docs {
                    Some(live_docs) => {
                        let values = group.flat.float_vector_values(fs.original_field_number)?;
                        let mut flags =
                            Vec::with_capacity(usize::try_from(values.size()).unwrap_or(0));
                        for ord in 0..values.size() {
                            flags.push(live_docs.get_doc(values.ord_to_doc(ord)?));
                        }
                        Some(flags)
                    }
                    None => None,
                };
                owned.push((
                    group
                        .flat_quantized
                        .and_then(|q| q.centroid(fs.original_field_number)),
                    all,
                    source.live_docs.is_some(),
                    live,
                ));
            }
            let merge_sources: Vec<scalar_quantized_vectors::QuantizedMergeSource<'_>> = owned
                .iter()
                .map(|(centroid, vectors, has_deletions, live)| {
                    scalar_quantized_vectors::QuantizedMergeSource {
                        centroid: *centroid,
                        vectors,
                        has_deletions: *has_deletions,
                        live: live.as_deref(),
                    }
                })
                .collect();
            let values = flat.float_vector_values(merged_field.number)?;
            let mut docs = Vec::with_capacity(usize::try_from(values.size()).unwrap_or(0));
            for ord in 0..values.size() {
                docs.push(values.ord_to_doc(ord)?);
            }
            let merged = values.all_vectors()?;
            writer.merge_field(
                merged_field.number,
                merged_field.vector_similarity_function,
                merged_field.vector_dimension,
                &merge_sources,
                &docs,
                &merged,
            )?;
        }
        let (veq, vemq) = writer.finish();
        files.push(("veq", veq));
        files.push(("vemq", vemq));
    }

    if let Some(mut ivf) =
        crate::ivf_vectors::IvfGroupWriter::new(format, merged_segment_id, suffix)?
    {
        // The `.ivv` goes to its file as it is built rather than being held
        // whole beside the vectors it is built from.
        if plan
            .iter()
            .any(|p| p.0.vector_encoding == VectorEncoding::Float32)
        {
            let name = crate::index_writer::segment_file_name(
                merged_segment_name,
                suffix,
                crate::ivf_vectors::EXTENSIONS[2],
            );
            ivf.stream_vectors_to(dir.create_output(&name)?);
            ivv_name = Some(name);
        }
        // Each source group's IVF index, opened once for every field it
        // serves.
        let mut readers: Vec<(*const crate::ivf_vectors::IvfInputs, lucene_ivf::IvfReader)> =
            Vec::new();
        for (merged_field, field_sources) in plan.iter().map(|p| (&p.0, &p.1)) {
            if merged_field.vector_encoding != VectorEncoding::Float32 {
                continue;
            }
            // The IVF reader keys fields by the source's own numbers; a
            // source that numbered the field differently lends nothing.
            let mut lent: Vec<(usize, *const crate::ivf_vectors::IvfInputs)> = Vec::new();
            for fs in field_sources {
                if fs.original_field_number != merged_field.number {
                    continue;
                }
                let group = sources[fs.source_index]
                    .vectors
                    .expect("only sources with a flat reader are planned")
                    .group_of(fs.original_field_number);
                if let Some(inputs) = group.ivf {
                    let key: *const crate::ivf_vectors::IvfInputs = inputs;
                    if !readers.iter().any(|(k, _)| *k == key) {
                        readers.push((key, inputs.open()?));
                    }
                    lent.push((fs.source_index, key));
                }
            }
            let ivf_sources: Vec<crate::ivf_vectors::IvfSource<'_>> = lent
                .iter()
                .filter_map(|&(source, key)| {
                    let reader = &readers.iter().find(|(k, _)| *k == key)?.1;
                    Some(crate::ivf_vectors::IvfSource {
                        reader,
                        doc_map: &doc_id_maps[source],
                    })
                })
                .collect();
            let values = flat.float_vector_values(merged_field.number)?;
            let mut docs = Vec::with_capacity(usize::try_from(values.size()).unwrap_or(0));
            for ord in 0..values.size() {
                docs.push(values.ord_to_doc(ord)?);
            }
            let merged = values.all_vectors()?;
            let dimension = usize::try_from(merged_field.vector_dimension).unwrap_or(0);
            let similarity = merged_field.vector_similarity_function;
            if similarity != field_infos::VectorSimilarityFunction::Cosine {
                ivf.add(
                    &crate::ivf_vectors::IvfField {
                        number: merged_field.number,
                        similarity,
                        dimension,
                        docs: &docs,
                        vectors: &merged,
                        partitions,
                    },
                    &ivf_sources,
                )?;
            } else {
                // A cosine field is written unit-normalized: the normalized
                // vectors go to a scratch file the writer reads mapped, so the
                // merge holds neither them nor the source's vectors on the heap.
                let fits_half = ivf.fits_half(&merged);
                let scratch =
                    crate::index_writer::segment_file_name(merged_segment_name, suffix, "ivn");
                write_normalized_vectors(dir, &scratch, dimension, &merged)?;
                drop(merged);
                let input = dir.open(&scratch)?;
                match lucene_util::vector_util::le_f32s(&input) {
                    Some(normalized) => ivf.add_prepared(
                        &crate::ivf_vectors::IvfField {
                            number: merged_field.number,
                            similarity,
                            dimension,
                            docs: &docs,
                            vectors: normalized,
                            partitions,
                        },
                        &ivf_sources,
                        fits_half,
                    )?,
                    None => {
                        return Err(lucene_store::Error::Io(std::io::Error::other(
                            "the normalized vectors are not aligned floats",
                        ))
                        .into())
                    }
                }
                drop(input);
                dir.delete_file(&scratch)?;
            }
        }
        match ivf.finish_streamed() {
            Some((written, out)) => {
                files.extend(written);
                match out {
                    Some(out) => {
                        out.close()?;
                    }
                    None => ivv_name = None,
                }
            }
            None => {
                if let Some(name) = ivv_name.take() {
                    dir.delete_file(&name)?;
                }
            }
        }
    }

    if format.hnsw_quantizer().is_some() {
        // The merged vectors are encoded afresh, against their own mean.
        let mut merged_fields = Vec::new();
        for merged_field in plan.iter().map(|p| &p.0) {
            if merged_field.vector_encoding != VectorEncoding::Float32 {
                continue;
            }
            let values = flat.float_vector_values(merged_field.number)?;
            merged_fields.push((merged_field, values.all_vectors()?));
        }
        let fields: Vec<crate::hnsw_quant_vectors::CodesField<'_>> = merged_fields
            .iter()
            .map(|(field, vectors)| crate::hnsw_quant_vectors::CodesField {
                number: field.number,
                similarity: field.vector_similarity_function,
                dimension: usize::try_from(field.vector_dimension).unwrap_or(0),
                vectors,
            })
            .collect();
        if let Some(codes) =
            crate::hnsw_quant_vectors::write(format, &fields, merged_segment_id, suffix)?
        {
            files.extend(codes);
        }
    }

    if let Some((m, beam_width)) = format.graph() {
        let quantized = match files.iter().find(|(e, _)| *e == "vemq") {
            Some((_, vemq)) => {
                let veq = &files
                    .iter()
                    .find(|(e, _)| *e == "veq")
                    .expect("written with the .vemq")
                    .1;
                Some(
                    scalar_quantized_vectors::ScalarQuantizedVectorsReader::open(
                        vemq,
                        veq,
                        merged_segment_id,
                        suffix,
                    )?,
                )
            }
            None => None,
        };
        let mut hnsw_fields: Vec<hnsw_vectors::HnswVectorsField<'_>> = Vec::new();
        let mut graphs: Vec<Option<hnsw::OnHeapHnswGraph>> = Vec::with_capacity(plan.len());
        let mut counts: Vec<i32> = Vec::with_capacity(plan.len());
        // Each field's graph is merged on its own (a fixed seed, no shared
        // state), so the fields run in parallel; collected in field order,
        // the first failing field is the one reported, as the serial loop did.
        let merge_field_graph = |merged_field: &&FieldInfo,
                                 field_sources: &Vec<FieldSource>|
         -> Result<(i32, Option<hnsw::OnHeapHnswGraph>)> {
            // Every source's own graph for this field, in the same order the
            // flat merge consumed them. A source with the flat pair and no
            // graph still contributes vectors -- they are simply inserted.
            let mut source_graphs: Vec<Option<hnsw_vectors::OffHeapHnswGraph<'_>>> =
                Vec::with_capacity(field_sources.len());
            for fs in field_sources {
                let source_vectors = sources[fs.source_index]
                    .vectors
                    .expect("only sources with a flat reader are planned");
                let graph = match source_vectors.group_of(fs.original_field_number).graph {
                    Some(reader) if reader.field(fs.original_field_number).is_some() => {
                        reader.graph(fs.original_field_number)?
                    }
                    _ => None,
                };
                source_graphs.push(graph);
            }
            let graph_sources: Vec<
                hnsw_vectors::GraphMergeSource<'_, hnsw_vectors::OffHeapHnswGraph<'_>>,
            > = field_sources
                .iter()
                .zip(&source_graphs)
                .map(|(fs, graph)| hnsw_vectors::GraphMergeSource {
                    graph: graph.as_ref(),
                    ord_to_doc: &fs.ord_to_doc,
                    doc_map: &doc_id_maps[fs.source_index],
                })
                .collect();

            let (count, graph) = match merged_field.vector_encoding {
                VectorEncoding::Float32 => {
                    let values = flat.float_vector_values(merged_field.number)?;
                    let merged_ord_to_doc: Vec<i32> = (0..values.size())
                        .map(|ord| values.ord_to_doc(ord))
                        .collect::<std::result::Result<Vec<i32>, _>>()?;
                    let graph = match &quantized {
                        Some(reader) if values.size() > 0 => {
                            let qv = reader.quantized_vector_values(merged_field.number)?;
                            let similarity = merged_field.vector_similarity_function;
                            let query_side = if qv.encoding().is_asymmetric() {
                                let raw = values.all_vectors()?;
                                Some(scalar_quantized_vectors::binarized_query_data(
                                    &qv, similarity, &raw,
                                )?)
                            } else {
                                None
                            };
                            let scorer = scalar_quantized_vectors::merge_scorer_supplier(
                                &qv,
                                similarity,
                                query_side.as_deref(),
                            )?;
                            hnsw_vectors::merge_one_field(
                                scorer,
                                m,
                                beam_width,
                                hnsw::DEFAULT_RAND_SEED,
                                &merged_ord_to_doc,
                                &graph_sources,
                            )?
                        }
                        _ => hnsw_vectors::merge_one_field(
                            values.ord_scorer(),
                            m,
                            beam_width,
                            hnsw::DEFAULT_RAND_SEED,
                            &merged_ord_to_doc,
                            &graph_sources,
                        )?,
                    };
                    (values.size(), graph)
                }
                VectorEncoding::Byte => {
                    let values = flat.byte_vector_values(merged_field.number)?;
                    let merged_ord_to_doc: Vec<i32> = (0..values.size())
                        .map(|ord| values.ord_to_doc(ord))
                        .collect::<std::result::Result<Vec<i32>, _>>()?;
                    let graph = hnsw_vectors::merge_one_field(
                        values.ord_scorer(),
                        m,
                        beam_width,
                        hnsw::DEFAULT_RAND_SEED,
                        &merged_ord_to_doc,
                        &graph_sources,
                    )?;
                    (values.size(), graph)
                }
            };
            Ok((count, graph))
        };
        let merged_graphs: Vec<Result<(i32, Option<hnsw::OnHeapHnswGraph>)>> = {
            use rayon::prelude::*;
            plan.par_iter()
                .map(|p| merge_field_graph(&p.0, &p.1))
                .collect()
        };
        for merged in merged_graphs {
            let (count, graph) = merged?;
            counts.push(count);
            graphs.push(graph);
        }
        for ((merged_field, _), (graph, count)) in plan
            .iter()
            .map(|p| (&p.0, &p.1))
            .zip(graphs.iter().zip(&counts))
        {
            hnsw_fields.push(hnsw_vectors::HnswVectorsField {
                field_number: merged_field.number,
                encoding: merged_field.vector_encoding,
                similarity: merged_field.vector_similarity_function,
                dimension: merged_field.vector_dimension,
                count: *count,
                graph: graph.as_ref(),
                m,
            });
        }
        let (vex_bytes, vem_bytes) =
            hnsw_vectors::write_hnsw_vectors(&hnsw_fields, merged_segment_id, suffix)?;
        files.push(("vex", vex_bytes));
        files.push(("vem", vem_bytes));
        if format.hnsw_disk_quantizer().is_some() {
            // The merged graph and vectors are copied into node records
            // afresh.
            let mut merged_fields = Vec::new();
            for (merged_field, graph) in plan.iter().map(|p| &p.0).zip(&graphs) {
                let Some(graph) = graph else { continue };
                if merged_field.vector_encoding != VectorEncoding::Float32 {
                    continue;
                }
                let values = flat.float_vector_values(merged_field.number)?;
                let docs = (0..values.size())
                    .map(|ord| values.ord_to_doc(ord))
                    .collect::<std::result::Result<Vec<i32>, _>>()?;
                merged_fields.push((merged_field, docs, values.all_vectors()?, graph));
            }
            let disk: Vec<crate::disk_hnsw_vectors::DiskField<'_>> = merged_fields
                .iter()
                .map(
                    |(field, docs, vectors, graph)| crate::disk_hnsw_vectors::DiskField {
                        number: field.number,
                        similarity: field.vector_similarity_function,
                        dimension: usize::try_from(field.vector_dimension).unwrap_or(0),
                        docs,
                        vectors,
                        graph,
                    },
                )
                .collect();
            if let Some(written) =
                crate::disk_hnsw_vectors::write_merged(format, &disk, merged_segment_id, suffix)?
            {
                files.extend(written);
            }
        }
    }
    files.insert(0, ("vemf", vemf_bytes));
    let mut early = vec![vec_name];
    early.extend(ivv_name);
    Ok((files, early))
}

/// One source field's ordinal -> its own doc id, materialised from the flat
/// store (`KnnVectorValues.ordToDoc` in a `Vec`). Both halves of
/// [`merge_vectors`] need it, and the two encodings differ only in which
/// values view answers.
fn source_ord_to_doc(
    flat: &vectors::FlatVectorsReader<'_>,
    field_number: i32,
    encoding: VectorEncoding,
) -> Result<Vec<i32>> {
    match encoding {
        VectorEncoding::Float32 => {
            let values = flat.float_vector_values(field_number)?;
            Ok((0..values.size())
                .map(|ord| values.ord_to_doc(ord))
                .collect::<std::result::Result<Vec<i32>, _>>()?)
        }
        VectorEncoding::Byte => {
            let values = flat.byte_vector_values(field_number)?;
            Ok((0..values.size())
                .map(|ord| values.ord_to_doc(ord))
                .collect::<std::result::Result<Vec<i32>, _>>()?)
        }
    }
}

/// Builds, per source, `MergeState.docMaps[i]`: a map from that source's own
/// (pre-merge) doc ids to the merged doc id space, `-1` for a document the
/// merge drops.
///
/// It is derived from `doc_order` -- the `(source index, source doc id)` pairs
/// in merged doc id order that every entry point computes -- rather than from
/// the concatenation rule, so it is correct for the sorted merge (where the
/// sources interleave) and for the plain one (where they do not) with no
/// second implementation to drift. `doc_order[m] == (s, d)` says exactly
/// "merged doc `m` is source `s`'s doc `d`", which inverted per source *is*
/// the doc map.
///
/// Within one source the map is still monotonically increasing in both
/// entry points -- concatenation visits a source's live docs in ascending
/// order, and the k-way sorted merge preserves each source's internal order
/// because every source is already sorted by the same key. That is what lets
/// [`merge_term_range`] treat one source's contribution to a term as already
/// ascending and only interleave across sources.
///
/// # Representation
///
/// Sized by each source's own `max_doc`, not by its last live doc: a vector
/// merge looks documents up past that point
/// (`FlatVectorsWriter::merge_one_flat_vector_field` treats a doc map that is
/// *short* as corruption, where `mapped_doc_id` treats it as "deleted"), so
/// the map covers the whole source and only the `-1`s distinguish the two.
///
/// Real Lucene's `MergeState.DocMap` is a dense, array-backed
/// old-doc-id -> new-doc-id lookup (`-1` for a deleted doc) precisely
/// because it sits in the innermost merge loop -- one lookup per posting,
/// per point, per vector, per doc-values entry. This mirrors that: a
/// `Vec<i32>` indexed by the source's own doc id, `-1` where the doc was
/// deleted, sized to the source's highest live doc id + 1. That turns each
/// lookup into a bounds check plus an index (and a `>= 0` test) instead of
/// hashing an `i32` and chasing a `HashMap` bucket, and it drops the
/// per-entry hash-table overhead from the allocation too.
fn build_doc_id_maps(per_source_max_doc: &[i32], doc_order: &[(usize, i32)]) -> Vec<Vec<i32>> {
    let mut maps: Vec<Vec<i32>> = per_source_max_doc
        .iter()
        // `max_doc` is a **file-derived value sizing an allocation**:
        // `stored_fields::open` checks only that its `.fdm` `maxDoc` is
        // non-negative, so `i32::MAX` here would reserve 8.6 GB -- a SIGABRT
        // (`index_writer`'s `a_segment_whose_fdm_disagrees_with_its_si_about_max_doc_is_not_merged`
        // covers it). The bound is at the caller, where the `.si`'s document count
        // (the figure Java's `SegmentMerger` works from) is in scope:
        // `IndexWriter::execute_merge` reports `SegmentDocCountMismatch`
        // rather than reserving for a number one file made up. The
        // `.max(0)` stays as the second line of defence for a `pub`
        // `merge_segments` caller that assembled its own `MergeSource`.
        .map(|&max_doc| vec![-1i32; max_doc.max(0) as usize])
        .collect();
    for (merged_doc_id, &(src_idx, doc_id)) in doc_order.iter().enumerate() {
        maps[src_idx][doc_id as usize] = merged_doc_id as i32;
    }
    maps
}

/// Whether a source's live docs keep their relative order in the merged
/// segment (deleted docs, `-1`, are skipped).
fn doc_map_is_monotonic(map: &[i32]) -> bool {
    let mut previous = -1;
    map.iter().filter(|&&merged| merged >= 0).all(|&merged| {
        let ascending = merged > previous;
        previous = merged;
        ascending
    })
}

/// One source's `DocMap.get`: the merged doc id for `doc_id`, or `None` if
/// that doc was deleted (or is past the source's last live doc).
#[inline]
fn mapped_doc_id(doc_id_map: &[i32], doc_id: i32) -> Option<i32> {
    if doc_id < 0 {
        return None;
    }
    match doc_id_map.get(doc_id as usize) {
        Some(&merged) if merged >= 0 => Some(merged),
        _ => None,
    }
}

/// Collects per-source results computed in parallel, in source order, and
/// returns the first error in that order -- the error a serial loop over the
/// sources would have stopped at.
fn in_source_order(
    results: impl rayon::iter::IndexedParallelIterator<Item = Result<()>>,
) -> Result<()> {
    let results: Vec<Result<()>> = results.collect();
    results.into_iter().collect()
}

/// Inverts the `original field number -> merged field number` maps
/// [`reconcile_field_numbers`] returns, once per merge, so
/// [`plan_postings`]/[`merge_points`] can answer "what did *this* source
/// call the field the merged segment numbers `n`?" with a single hash lookup
/// instead of a linear scan of the whole forward map per (candidate field,
/// source) pair -- which is `O(fields^2 * sources)` on a wide schema.
fn invert_field_number_maps(per_source_maps: &[HashMap<i32, i32>]) -> Vec<HashMap<i32, i32>> {
    per_source_maps
        .iter()
        .map(|map| map.iter().map(|(&orig, &merged)| (merged, orig)).collect())
        .collect()
}

/// One source's forward cursor over a merged field's term dictionary -- the
/// port of one sub-`TermsEnum` inside Java's `MultiTerms`/`MappedMultiFields`.
///
/// `TermsEnum::try_next` hands back a borrowed term that cannot outlive the
/// next call, so the current term is copied into an owned buffer the merge can
/// compare across cursors. That is one `Vec<u8>` per cursor, reused for every
/// term, against the old `BTreeSet<Vec<u8>>`'s one heap allocation and one
/// tree insert per (term, source) for the whole field at once.
struct TermCursor<'a> {
    terms: blocktree::TermsEnum<'a>,
    term: Vec<u8>,
    done: bool,
}

impl<'a> TermCursor<'a> {
    /// A cursor positioned on `pf`'s first term, or `None` if that source's
    /// dictionary for the field is empty.
    fn open(pf: &'a SourcePostings<'a>) -> Result<Option<Self>> {
        let mut cursor = TermCursor {
            terms: pf.field_terms.iter(),
            term: Vec::new(),
            done: false,
        };
        cursor.advance()?;
        Ok((!cursor.done).then_some(cursor))
    }

    /// [`Self::open`], positioned on the first term `>= lower` instead.
    fn open_at(pf: &'a SourcePostings<'a>, lower: Option<&[u8]>) -> Result<Option<Self>> {
        let Some(lower) = lower else {
            return Self::open(pf);
        };
        let mut terms = pf.field_terms.iter();
        if terms.try_seek_ceil(lower)? == blocktree::SeekStatus::End {
            return Ok(None);
        }
        let term = match terms.try_current()? {
            Some((term, _)) => term.to_vec(),
            None => return Ok(None),
        };
        Ok(Some(TermCursor {
            terms,
            term,
            done: false,
        }))
    }

    fn advance(&mut self) -> Result<()> {
        match self.terms.try_next()? {
            Some((term, _stats)) => {
                self.term.clear();
                self.term.extend_from_slice(term);
            }
            None => self.done = true,
        }
        Ok(())
    }

    fn exhausted(&self) -> bool {
        self.done
    }
}

/// One term range's merged postings, every term's streams concatenated into
/// shared buffers: a term costs one [`RunTerm`] (four ends) instead of a
/// struct of seven `Vec`s and their heap blocks. Holding a field's whole
/// merged postings until the writer has seen every term is what bounds the
/// merge's peak memory, so the per-term overhead is paid on every term of
/// the segment at once. Java: `MappedMultiPostingsEnum`, materialized for the
/// native writer.
#[derive(Debug, Default)]
struct MergedRun {
    has_positions: bool,
    has_offsets: bool,
    has_payloads: bool,
    term_bytes: Vec<u8>,
    terms: Vec<RunTerm>,
    /// `(doc, freq)` of every term's documents, term after term.
    docs: Vec<(i32, i32)>,
    /// Parallel to `docs` when the field indexes positions: where each
    /// document's positions (and offsets) end, counted from its term's first
    /// position -- half the size of an absolute `usize`, and this buffer is
    /// one entry per merged posting.
    pos_ends: Vec<u32>,
    /// A term's positions passed `u32::MAX`: [`Self::finish_term`] fails.
    too_large: bool,
    positions: Vec<i32>,
    offsets: Vec<(i32, i32)>,
    payload_lengths: Vec<u32>,
    payload_bytes: Vec<u8>,
}

/// Where one term of a [`MergedRun`] ends in each of its buffers; it starts
/// where the previous term ends.
#[derive(Debug, Clone, Copy, Default)]
struct RunTerm {
    term: usize,
    docs: usize,
    positions: usize,
    payload_lengths: usize,
    payload_bytes: usize,
}

impl MergedRun {
    fn new(has_positions: bool, has_offsets: bool, has_payloads: bool) -> Self {
        Self {
            has_positions,
            has_offsets,
            has_payloads,
            ..Self::default()
        }
    }

    /// Room for a `1 / parts` share of the sources' postings, from their
    /// dictionaries' totals, so the buffers do not grow by doubling: every
    /// doubling copies what was merged so far into freshly faulted pages.
    /// Deleted documents make it an overestimate; the untouched tail is
    /// address space only.
    fn reserve_share(&mut self, sources: &[Option<&SourcePostings<'_>>], parts: usize) {
        let parts = parts.max(1);
        let total = |f: fn(&FieldTerms) -> i64| -> usize {
            sources
                .iter()
                .flatten()
                .map(|s| usize::try_from(f(s.field_terms)).unwrap_or(0))
                .fold(0usize, usize::saturating_add)
                .checked_div(parts)
                .unwrap_or(0)
        };
        let terms = total(|t| t.num_terms);
        let docs = total(|t| t.sum_doc_freq);
        self.terms.reserve(terms);
        self.docs.reserve(docs);
        if self.has_positions {
            // a field without frequencies reports -1, which counts as none
            let positions = total(|t| t.sum_total_term_freq);
            self.pos_ends.reserve(docs);
            self.positions.reserve(positions);
            if self.has_offsets {
                self.offsets.reserve(positions);
            }
        }
    }

    /// The ends of the last finished term: where the open one starts.
    fn open_start(&self) -> RunTerm {
        self.terms.last().copied().unwrap_or_default()
    }

    /// Opens term `term`; [`Self::finish_term`] closes it.
    fn start_term(&mut self, term: &[u8]) {
        self.term_bytes.extend_from_slice(term);
    }

    /// The open term's bytes.
    fn open_term(&self) -> &[u8] {
        &self.term_bytes[self.open_start().term..]
    }

    /// The source reader has validated every occurrence before live filtering.
    /// Append only the retained document's borrowed slices, including one zero
    /// payload length per occurrence from a payload-free contributing source.
    fn push_document(
        &mut self,
        doc: i32,
        freq: i32,
        occurrences: Option<lucene_codecs::postings::DocOccurrences<'_>>,
    ) {
        self.docs.push((doc, freq));
        if let Some(occurrences) = occurrences {
            self.positions.extend_from_slice(occurrences.positions);
            let end = self
                .positions
                .len()
                .saturating_sub(self.open_start().positions);
            self.too_large |= end > u32::MAX as usize;
            self.pos_ends.push(end as u32);
            if self.has_offsets {
                self.offsets.extend(
                    occurrences
                        .start_offsets
                        .iter()
                        .copied()
                        .zip(occurrences.end_offsets.iter().copied()),
                );
            }
            if self.has_payloads {
                if occurrences.payload_lengths.is_empty() {
                    self.payload_lengths
                        .extend(std::iter::repeat_n(0, occurrences.positions.len()));
                } else {
                    self.payload_lengths
                        .extend_from_slice(occurrences.payload_lengths);
                    self.payload_bytes
                        .extend_from_slice(occurrences.payload_bytes);
                }
            }
        }
    }

    /// Closes the open term: kept, its documents in `DocIDMerger` order, when
    /// it reached a live document; dropped otherwise.
    fn finish_term(&mut self) -> Result<()> {
        if self.too_large {
            return Err(Error::MergedTermTooLarge);
        }
        let start = self.open_start();
        if self.docs.len() == start.docs {
            self.term_bytes.truncate(start.term);
            return Ok(());
        }
        self.sort_open_documents(start);
        self.terms.push(RunTerm {
            term: self.term_bytes.len(),
            docs: self.docs.len(),
            positions: self.positions.len(),
            payload_lengths: self.payload_lengths.len(),
            payload_bytes: self.payload_bytes.len(),
        });
        Ok(())
    }

    /// Gives back what the buffers grew past their contents: a finished
    /// range waits, whole, for the writer, while the others still grow.
    fn shrink_to_fit(&mut self) {
        self.term_bytes.shrink_to_fit();
        self.terms.shrink_to_fit();
        self.docs.shrink_to_fit();
        self.pos_ends.shrink_to_fit();
        self.positions.shrink_to_fit();
        self.offsets.shrink_to_fit();
        self.payload_lengths.shrink_to_fit();
        self.payload_bytes.shrink_to_fit();
    }

    /// `DocIDMerger` order: plain merges already ascend, while sorted merges
    /// interleave sources. Every occurrence stream of the open term follows
    /// the same permutation; earlier terms are not touched.
    fn sort_open_documents(&mut self, start: RunTerm) {
        let docs = &self.docs[start.docs..];
        if docs.windows(2).all(|w| w[0].0 < w[1].0) {
            return;
        }
        let mut permutation: Vec<usize> = (0..docs.len()).collect();
        permutation.sort_unstable_by_key(|&i| docs[i].0);
        let sorted_docs: Vec<(i32, i32)> = permutation.iter().map(|&i| docs[i]).collect();
        self.docs.truncate(start.docs);
        self.docs.extend_from_slice(&sorted_docs);
        if !self.has_positions {
            return;
        }
        // The open term's positions start where the previous term's end;
        // its ends count from there.
        let pos_start = start.positions;
        let ends = &self.pos_ends[start.docs..];
        let span = |i: usize| {
            let from = i.checked_sub(1).map_or(0, |prev| ends[prev] as usize);
            pos_start.saturating_add(from)..pos_start.saturating_add(ends[i] as usize)
        };
        // Counts come from the same spans that built the payload run, before
        // changing any document order. Each span is one validated i32 freq, so
        // its length fits u32; no count read from disk sizes an allocation here.
        let counts: Vec<u32> = if self.payload_lengths.len() > start.payload_lengths {
            (0..ends.len()).map(|i| span(i).len() as u32).collect()
        } else {
            Vec::new()
        };
        // ALLOC: capacities are lengths of already-owned buffers; each original
        // document span is appended once by a permutation of 0..docs.len().
        let mut positions = Vec::with_capacity(self.positions.len().saturating_sub(pos_start));
        let mut offsets = Vec::with_capacity(if self.has_offsets {
            self.offsets.len().saturating_sub(pos_start)
        } else {
            0
        });
        let mut pos_ends = Vec::with_capacity(ends.len());
        for &doc in &permutation {
            let range = span(doc);
            positions.extend_from_slice(&self.positions[range.clone()]);
            if self.has_offsets {
                offsets.extend_from_slice(&self.offsets[range]);
            }
            // At most the unsorted term's own last end, which fit.
            pos_ends.push(positions.len() as u32);
        }
        self.positions.truncate(pos_start);
        self.positions.extend_from_slice(&positions);
        if self.has_offsets {
            self.offsets.truncate(pos_start);
            self.offsets.extend_from_slice(&offsets);
        }
        self.pos_ends.truncate(start.docs);
        self.pos_ends.extend_from_slice(&pos_ends);
        if !counts.is_empty() {
            let (bytes, lengths) = postings_writer::permute_payload_run(
                &self.payload_bytes[start.payload_bytes..],
                &self.payload_lengths[start.payload_lengths..],
                &counts,
                &permutation,
            );
            self.payload_bytes.truncate(start.payload_bytes);
            self.payload_bytes.extend_from_slice(&bytes);
            self.payload_lengths.truncate(start.payload_lengths);
            self.payload_lengths.extend_from_slice(&lengths);
        }
    }

    /// One [`MergedTermView`] per finished term, in order.
    fn views(&self) -> impl Iterator<Item = MergedTermView<'_>> + '_ {
        (0..self.terms.len()).map(move |term| MergedTermView { run: self, term })
    }

    /// The ends of the term before `term`: where it starts.
    fn start_of(&self, term: usize) -> RunTerm {
        term.checked_sub(1)
            .map_or_else(RunTerm::default, |prev| self.terms[prev])
    }
}

/// One finished term of a [`MergedRun`], read in place by the writer.
#[derive(Debug, Clone, Copy)]
struct MergedTermView<'a> {
    run: &'a MergedRun,
    term: usize,
}

impl MergedTermView<'_> {
    /// Document `doc`'s span in the run's positions (and offsets).
    fn document_range(&self, doc: usize) -> Option<std::ops::Range<usize>> {
        let run = self.run;
        let (start, end) = (run.start_of(self.term).docs, run.terms[self.term].docs);
        let at = start.checked_add(doc).filter(|&at| at < end)?;
        let base = run.start_of(self.term).positions;
        let from = match doc.checked_sub(1) {
            None => 0,
            Some(_) => *run.pos_ends.get(at.checked_sub(1)?)? as usize,
        };
        let to = *run.pos_ends.get(at)? as usize;
        Some(base.saturating_add(from)..base.saturating_add(to))
    }
}

impl TermSource for MergedTermView<'_> {
    fn term(&self) -> &[u8] {
        let run = self.run;
        &run.term_bytes[run.start_of(self.term).term..run.terms[self.term].term]
    }
    fn docs(&self) -> &[(i32, i32)] {
        let run = self.run;
        &run.docs[run.start_of(self.term).docs..run.terms[self.term].docs]
    }
    fn position_lists(&self) -> usize {
        if self.run.has_positions {
            self.docs().len()
        } else {
            0
        }
    }
    fn positions(&self, doc: usize) -> &[i32] {
        if !self.run.has_positions {
            return &[];
        }
        self.document_range(doc)
            .and_then(|range| self.run.positions.get(range))
            .unwrap_or_default()
    }
    fn offset_lists(&self) -> usize {
        if self.run.has_offsets {
            self.docs().len()
        } else {
            0
        }
    }
    fn offsets(&self, doc: usize) -> &[(i32, i32)] {
        if !self.run.has_offsets {
            return &[];
        }
        self.document_range(doc)
            .and_then(|range| self.run.offsets.get(range))
            .unwrap_or_default()
    }
    fn payload_bytes(&self) -> &[u8] {
        let run = self.run;
        &run.payload_bytes
            [run.start_of(self.term).payload_bytes..run.terms[self.term].payload_bytes]
    }
    fn payload_lengths(&self) -> &[u32] {
        let run = self.run;
        &run.payload_lengths
            [run.start_of(self.term).payload_lengths..run.terms[self.term].payload_lengths]
    }
}

/// One merged field's postings, ready to hand to
/// [`lucene_codecs::postings_writer::write_fields`] (via a borrowed
/// [`FieldPostingsInput`] built from `terms`).
struct MergedPostingsField {
    field_number: i32,
    index_options: IndexOptions,
    doc_count: i32,
    has_payloads: bool,
    /// The field's term ranges, in term order.
    runs: Vec<MergedRun>,
}

/// Merges postings (term dictionaries + doc/freq data) across `sources` for
/// every field any source declares postings for, returning one
/// [`MergedPostingsField`] per distinct merged field number that has
/// postings data in at least one source -- or an empty `Vec` if no source
/// supplied any postings data at all.
///
/// Each source's term dictionary is independent (the same reason
/// [`merge_sorted_doc_values`] can't just concatenate ordinals): this
/// resolves each contributing source's own term dictionary directly to
/// term *bytes* (no cross-source ordinal-remapping table), unions those
/// bytes across sources into one sorted term set, and for each term walks
/// the contributing sources **in source order**, concatenating each
/// source's `(mergedDocId, freq)` pairs for that term (dropping non-live
/// docs via [`build_doc_id_maps`]) -- ascending overall because merged doc
/// ids are assigned in increasing, source-disjoint ranges (see
/// [`build_doc_id_maps`]'s doc comment).
///
/// Unlike doc-values/norms, [`postings_writer::write_fields`] already
/// supports any number of fields per call (`numFields` in `.tmd` is simply
/// `inputs.len()`), so there is no single-field-per-merge-call limit here
/// the way `TooManyNumericDocValuesFields` etc. enforce for doc-values.
///
/// # The "sparse across sources" rule, postings edition
///
/// A term's postings are naturally sparse per-doc (most docs don't contain
/// most terms) -- that sparsity is exactly what a term dictionary already
/// models, and is not an error here. Nor is a source that declares the
/// field but has no terms for it at all -- every document of that segment
/// had an empty value -- which contributes nothing, as in `MultiFields`.
///
/// # Positions/offsets/payloads
///
/// See [`SourcePostings`]'s doc comment: a candidate field whose merged
/// `index_options` indexes positions has every contributing source's
/// per-term positions (and offsets/payloads, when applicable) read back and
/// concatenated in the same source order as docs/freqs.
///
/// This gathers every field whole, for the formats that need it (see
/// [`per_field_postings::write_streaming`]); the block-tree format streams
/// [`merge_field_runs`] straight into its writer instead.
fn materialize_postings(
    plans: &[PostingsFieldPlan<'_, '_>],
    doc_id_maps: &[Vec<i32>],
) -> Result<Vec<MergedPostingsField>> {
    let mut result = Vec::with_capacity(plans.len());
    for plan in plans {
        let mut runs = Vec::new();
        let doc_count = merge_field_runs(plan, doc_id_maps, &mut |run| {
            runs.push(run);
            Ok(())
        })?;
        result.push(MergedPostingsField {
            field_number: plan.field_number,
            index_options: plan.index_options,
            doc_count,
            has_payloads: plan.shape.has_payloads,
            runs,
        });
    }
    Ok(result)
}

/// One merged postings field as [`plan_postings`] resolves it: what
/// [`merge_field_runs`] needs to merge its terms.
struct PostingsFieldPlan<'a, 'b> {
    field_number: i32,
    index_options: IndexOptions,
    /// Per source, in source order, the field's postings there, or `None`
    /// when that source contributes nothing to it.
    per_source_field: Vec<Option<&'a SourcePostings<'b>>>,
    shape: FieldShape,
}

/// The fields the merge writes postings for, in field-number order, each with its
/// contributing sources resolved: every check that needs no term decoded.
fn plan_postings<'a, 'b>(
    sources: &'a [MergeSource<'b>],
    per_source_maps: &[HashMap<i32, i32>],
    per_source_live_ids: &[Vec<i32>],
    doc_order: &[(usize, i32)],
    merged_fields: &[FieldInfo],
) -> Result<Vec<PostingsFieldPlan<'a, 'b>>> {
    let mut candidates: Vec<i32> = Vec::new();
    for ((source, map), live_ids) in sources.iter().zip(per_source_maps).zip(per_source_live_ids) {
        if live_ids.is_empty() {
            // Same "fully-deleted source can't affect the merged output"
            // exemption as merge_numeric_doc_values.
            continue;
        }
        // Java: PerFieldPostingsFormat.FieldsReader.iterator retains routed
        // fields even when an earlier merge removed their last term. A later
        // merge must still open their group and preserve routing attributes.
        let routed = source.field_infos.iter().filter(|field| {
            field.index_options != IndexOptions::None
                && field
                    .attributes
                    .iter()
                    .any(|(key, _)| key == per_field_postings::PER_FIELD_FORMAT_KEY)
        });
        for number in source
            .postings
            .iter()
            .map(|field| field.field_number)
            .chain(routed.map(|field| field.number))
        {
            if let Some(&merged_number) = map.get(&number) {
                if !candidates.contains(&merged_number) {
                    candidates.push(merged_number);
                }
            }
        }
    }
    candidates.sort_unstable();
    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    let reverse_maps = invert_field_number_maps(per_source_maps);
    let merged_doc_count: usize = doc_order.len();
    let mut result = Vec::with_capacity(candidates.len());

    for merged_field_number in candidates {
        let merged_field = merged_fields
            .iter()
            .find(|f| f.number == merged_field_number)
            .expect("merged_field_number came from reconcile_field_numbers over these same sources, so it must have an entry in merged_fields");
        let index_options = merged_field.index_options;
        if !matches!(
            index_options,
            IndexOptions::Docs
                | IndexOptions::DocsAndFreqs
                | IndexOptions::DocsAndCustomFreqs
                | IndexOptions::DocsAndFreqsAndPositions
                | IndexOptions::DocsAndFreqsAndPositionsAndOffsets
        ) {
            return Err(Error::PostingsIndexOptionsNotSupported {
                merged_field_number,
                index_options,
            });
        }
        let has_positions = matches!(
            index_options,
            IndexOptions::DocsAndFreqsAndPositions
                | IndexOptions::DocsAndFreqsAndPositionsAndOffsets
        );
        let has_offsets = matches!(
            index_options,
            IndexOptions::DocsAndFreqsAndPositionsAndOffsets
        );
        let has_payloads = merged_field.store_payloads;

        // Per-source (in source order) this field's `SourcePostings`, or
        // `None` when this source contributes nothing to the field: a
        // fully-deleted source, or a source whose own `FieldInfos` never saw
        // the field at all. The latter mirrors `FieldsConsumer.merge`'s
        // `Terms terms = fields.terms(field); if (terms == null) continue;`
        // -- adding a field to an index that already has segments is normal,
        // and those older segments simply contribute no terms. (Declaring
        // the field with *different* `index_options` is a different matter
        // and is already rejected up front by `reconcile_field_numbers`'
        // `verifySameSchema` port.)
        let mut per_source_field: Vec<Option<&SourcePostings<'_>>> =
            Vec::with_capacity(sources.len());
        for ((source, reverse), live_ids) in
            sources.iter().zip(&reverse_maps).zip(per_source_live_ids)
        {
            if live_ids.is_empty() {
                per_source_field.push(None);
                continue;
            }
            let Some(&original_number) = reverse.get(&merged_field_number) else {
                per_source_field.push(None);
                continue;
            };
            // A source that declares the field but has no terms for it --
            // every one of its documents had an empty value, so its term
            // dictionary never got the field -- contributes nothing, as in
            // `MultiFields`. It cannot be a source whose `.tim` went
            // unopened: `check_format_coverage` refuses that merge first.
            let Some(pf) = source
                .postings
                .iter()
                .find(|pf| pf.field_number == original_number)
            else {
                per_source_field.push(None);
                continue;
            };
            if has_positions && pf.pos_in.is_none() {
                return Err(Error::PostingsPositionsInputMissingInSource {
                    merged_field_number,
                    source_index: per_source_field.len(),
                });
            }
            per_source_field.push(Some(pf));
        }

        result.push(PostingsFieldPlan {
            field_number: merged_field_number,
            index_options,
            per_source_field,
            shape: FieldShape {
                has_positions,
                has_offsets,
                has_payloads,
                merged_doc_count,
            },
        });
    }

    Ok(result)
}

/// Merges `plan`'s terms, handing each term range's [`MergedRun`] to `sink`
/// in term order, and returns the field's `docsSeen` count.
///
/// Ranges are merged in parallel a window of one per thread at a time, and
/// each window's runs are handed over (and so can be written and dropped)
/// before the next is decoded: a streaming writer holds at most one window
/// of the field's decoded postings, where Java's `FieldsConsumer.merge`
/// holds one term's.
fn merge_field_runs(
    plan: &PostingsFieldPlan<'_, '_>,
    doc_id_maps: &[Vec<i32>],
    sink: &mut dyn FnMut(MergedRun) -> Result<()>,
) -> Result<i32> {
    let per_source_field = &plan.per_source_field;
    let shape = plan.shape;
    let ranges = term_ranges(per_source_field)?;
    let mut docs_seen = FixedBitSet::new(shape.merged_doc_count.max(1));
    if ranges.len() == 1 {
        let (run, seen) = merge_term_range(per_source_field, doc_id_maps, shape, None, None, 1)?;
        docs_seen.or(&seen);
        sink(run)?;
    } else {
        use rayon::prelude::*;
        let window = rayon::current_num_threads().max(1);
        let mut first = 0usize;
        while first < ranges.len() {
            let last = first.saturating_add(window).min(ranges.len());
            let parts: Vec<(MergedRun, FixedBitSet)> = (first..last)
                .into_par_iter()
                .map(|i| {
                    let lower = (i > 0).then(|| ranges[i].as_slice());
                    let upper = ranges.get(i.saturating_add(1)).map(Vec::as_slice);
                    merge_term_range(
                        per_source_field,
                        doc_id_maps,
                        shape,
                        lower,
                        upper,
                        ranges.len(),
                    )
                })
                .collect::<Result<_>>()?;
            for (run, seen) in parts {
                docs_seen.or(&seen);
                sink(run)?;
            }
            first = last;
        }
    }
    // Candidates come from live source dictionaries or routed FieldInfos,
    // including a prior merge's termless field with no SourcePostings.
    // Keep its group even when every source Terms is null or every posting
    // was pruned. The encoder omits only the empty field's `.tmd` record.
    Ok(docs_seen.cardinality() as i32)
}

/// What [`merge_term_range`] needs to know about the merged field.
#[derive(Clone, Copy)]
struct FieldShape {
    has_positions: bool,
    has_offsets: bool,
    has_payloads: bool,
    merged_doc_count: usize,
}

/// Fewer terms than this across a field's sources merge on one thread: the
/// split walk and the per-range cursors would cost more than they save.
const PARALLEL_MERGE_MIN_TERMS: i64 = 8192;

/// Roughly how many terms (summed over the sources) one merge range of a
/// large field covers.
const TERMS_PER_MERGE_RANGE: i64 = 16384;

/// Roughly how many postings (documents plus positions, summed over the
/// sources) one merge range of a large field holds: a few MB of decoded run
/// per range, a window of one range per thread at a time.
const POSTINGS_PER_MERGE_RANGE: i64 = 1 << 20;

/// Where to split a field's term space for a parallel merge: `ranges[0]` is
/// an unused placeholder for "from the first term", and range `i` covers the
/// terms in `ranges[i]..ranges[i + 1]` (the last one to the end).
///
/// The split terms are taken in one walk of the dictionary of the source
/// with the most postings, evenly spaced in its postings. Ranges are disjoint and
/// cover every term of every source, so merging them independently and
/// concatenating the results in order is the single merge.
fn term_ranges(per_source_field: &[Option<&SourcePostings<'_>>]) -> Result<Vec<Vec<u8>>> {
    let total: i64 = per_source_field
        .iter()
        .flatten()
        .map(|pf| pf.field_terms.num_terms.max(0))
        .sum();
    let threads = rayon::current_num_threads();
    // Documents plus positions: what a range's run holds.
    let postings = |pf: &SourcePostings<'_>| {
        pf.field_terms
            .sum_doc_freq
            .max(0)
            .saturating_add(pf.field_terms.sum_total_term_freq.max(0))
    };
    let largest = per_source_field
        .iter()
        .flatten()
        .max_by_key(|pf| (postings(pf), pf.field_terms.num_terms));
    let (Some(largest), true) = (largest, threads > 1 && total >= PARALLEL_MERGE_MIN_TERMS) else {
        return Ok(vec![Vec::new()]);
    };
    // Four ranges per thread: ranges differ in how many sources and
    // postings they hold, so finer ranges balance better. A large field gets
    // more, so that the window `merge_field_runs` decodes at a time (one
    // range per thread) stays a small part of it, by terms and by postings.
    let total_postings: i64 = per_source_field
        .iter()
        .flatten()
        .map(|pf| postings(pf))
        .fold(0, i64::saturating_add);
    let ranges = (threads.saturating_mul(4) as i64)
        .max(total / TERMS_PER_MERGE_RANGE)
        .max(total_postings / POSTINGS_PER_MERGE_RANGE);
    // Split points evenly spaced in the largest source's postings, not its
    // terms: a few frequent terms hold most of a field's postings (Zipf), and
    // equal term counts would give one range most of them.
    let step = postings(largest).checked_div(ranges).unwrap_or(0).max(1);
    let mut out = vec![Vec::new()];
    let mut terms = largest.field_terms.iter();
    let mut seen = 0i64;
    let mut next_split = step;
    while let Some((term, stats)) = terms.try_next()? {
        if seen >= next_split && (out.len() as i64) < ranges {
            out.push(term.to_vec());
            while next_split <= seen {
                next_split = next_split.saturating_add(step);
            }
        }
        seen = seen
            .saturating_add(i64::from(stats.doc_freq.max(0)))
            .saturating_add(stats.total_term_freq.max(0));
    }
    Ok(out)
}

/// Merges the terms in `lower..upper` (`None`: unbounded) of one field
/// across its sources, returning them in order with the merged documents
/// they reached.
fn merge_term_range(
    per_source_field: &[Option<&SourcePostings<'_>>],
    doc_id_maps: &[Vec<i32>],
    shape: FieldShape,
    lower: Option<&[u8]>,
    upper: Option<&[u8]>,
    parts: usize,
) -> Result<(MergedRun, FixedBitSet)> {
    let FieldShape {
        has_positions,
        has_offsets,
        has_payloads,
        merged_doc_count,
    } = shape;
    // A streaming k-way merge of the contributing sources' term
    // dictionaries -- the port of `FieldsConsumer.merge`'s
    // `MultiTerms`/`MappedMultiPostingsEnum` pipeline. One forward cursor
    // per source, advanced together; at each step the smallest current
    // term is the next merged term, and only the sources actually
    // standing on it contribute postings, decoded straight off their own
    // cursor position (no dictionary seek at all -- see
    // `TermsEnum::try_current_postings`).
    //
    // What this replaces: a `BTreeSet<Vec<u8>>` holding every distinct
    // term of the merged field in memory at once, built by one full
    // traversal per source, followed by *another* seek per (term, source)
    // -- including for the sources that did not have the term -- and, for
    // a positional field, a second seek and a second docs/freqs decode on
    // top of that.
    let mut cursors: Vec<Option<TermCursor<'_>>> = Vec::with_capacity(per_source_field.len());
    for pf in per_source_field {
        cursors.push(match pf {
            Some(pf) => TermCursor::open_at(pf, lower)?,
            None => None,
        });
    }

    let mut run = MergedRun::new(has_positions, has_offsets, has_payloads);
    run.reserve_share(per_source_field, parts);
    // One source term's decoded postings at a time, their buffers reused
    // from term to term.
    let mut source_postings = lucene_codecs::postings::Postings::default();
    let mut occurrences_buf = lucene_codecs::postings::FlatOccurrences::default();
    let mut docs_seen = FixedBitSet::new(merged_doc_count.max(1));
    loop {
        // Smallest term across the live cursors.
        let mut smallest: Option<&[u8]> = None;
        for cursor in cursors.iter().flatten() {
            if smallest.is_none_or(|current| cursor.term.as_slice() < current) {
                smallest = Some(&cursor.term);
            }
        }
        let Some(smallest) = smallest else { break };
        if upper.is_some_and(|upper| smallest >= upper) {
            break;
        }
        run.start_term(smallest);
        for src_idx in 0..cursors.len() {
            let Some(cursor) = cursors[src_idx].as_mut() else {
                continue;
            };
            if cursor.term.as_slice() != run.open_term() {
                continue;
            }
            let pf = per_source_field[src_idx]
                .expect("a cursor only exists for a source that contributes this field");
            let on_term = if has_positions {
                cursor
                    .terms
                    .try_current_postings_and_occurrences_flat_into(
                        pf.doc_in,
                        pf.pos_in.expect(
                            "checked as Error::PostingsPositionsInputMissingInSource above",
                        ),
                        pf.pay_in,
                        &mut source_postings,
                        &mut occurrences_buf,
                    )?
            } else {
                cursor
                    .terms
                    .try_current_postings_into(pf.doc_in, &mut source_postings)?
            };
            assert!(on_term, "the cursor is standing on this term");
            let source_positions = has_positions.then_some(&occurrences_buf);

            let doc_id_map = &doc_id_maps[src_idx];
            for (doc_idx, (&doc_id, &freq)) in source_postings
                .docs
                .iter()
                .zip(source_postings.freqs.iter())
                .enumerate()
            {
                if pf.live.is_some_and(|live| !live.get_doc(doc_id)) {
                    continue;
                }
                if let Some(merged_doc_id) = mapped_doc_id(doc_id_map, doc_id) {
                    // Java's `docsSeen` bitset, not a `HashSet<i32>`: the
                    // merged doc id space is dense and known up front.
                    //
                    // FBS: `docs_seen` is
                    // `FixedBitSet::new(merged_doc_count.max(1))` where
                    // `merged_doc_count == doc_order.len()`, and
                    // `build_doc_id_maps` writes only `doc_order`'s own
                    // enumeration indices into every map -- so every
                    // `mapped_doc_id` result is in `0..doc_order.len()`,
                    // which is this bitset's `len()`.
                    docs_seen.set(merged_doc_id as usize);
                    let occurrences = source_positions.as_ref().map(|positions| {
                        positions
                            .document(doc_idx)
                            .expect("occurrence boundaries came from these same decoded freqs")
                    });
                    run.push_document(merged_doc_id, freq, occurrences);
                }
            }
            cursor.advance()?;
            if cursor.exhausted() {
                cursors[src_idx] = None;
            }
        }
        run.finish_term()?;
    }
    run.shrink_to_fit();
    Ok((run, docs_seen))
}

/// One source's (or the merged) points for one field, laid out flat: point
/// `i` is `docs[i]` with packed value `values[i * stride .. (i + 1) *
/// stride]` -- what [`points::PackedPointsField`] takes, with no allocation
/// per point.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct PointStream {
    docs: Vec<i32>,
    values: Vec<u8>,
}

/// Collects every point of a field, in leaf order (which, for one index
/// dimension, is value order), remapped into the merged doc-id space and with
/// deleted documents dropped. Every cell "crosses", so every leaf is visited
/// point by point -- `BKDReader`'s `MergeReader` iteration, through the
/// reader's own allocation-free leaf decoder.
struct CollectPoints<'m> {
    doc_id_map: &'m [i32],
    stride: usize,
    stream: PointStream,
    /// The first packed value whose length disagreed with the field's
    /// declared shape -- reported after the walk, since a visitor cannot fail.
    bad_length: Option<usize>,
}

impl points::IntersectVisitor for CollectPoints<'_> {
    fn compare(&mut self, _min: &[u8], _max: &[u8]) -> points::Relation {
        points::Relation::CellCrossesQuery
    }

    fn visit(&mut self, _doc_id: i32) {
        // Only reached for a cell reported *inside* the query, which
        // `compare` above never does.
    }

    fn visit_with_value(&mut self, doc_id: i32, packed_value: &[u8]) {
        if packed_value.len() != self.stride {
            self.bad_length.get_or_insert(packed_value.len());
            return;
        }
        if let Some(merged) = mapped_doc_id(self.doc_id_map, doc_id) {
            self.stream.docs.push(merged);
            self.stream.values.extend_from_slice(packed_value);
        }
    }
}

/// Merges BKD points (`.kdm`/`.kdi`/`.kdd`) data across `sources` for every
/// field any source declares points for, returning one [`points::MergePointsField`]
/// per distinct merged field number declaring points, even when all of its
/// source values are null or deleted. No declared points returns an empty Vec.
///
/// Unlike SORTED doc values or postings, a points field has no shared
/// dictionary to resolve ordinals against -- it's fundamentally a per-doc set
/// of fixed-width packed values (like NUMERIC/SORTED_NUMERIC doc values, but
/// with a merged tree rebuilt from scratch rather than a single scalar per
/// doc). So this simply reads back every live doc's points via each source's
/// own already-opened [`lucene_codecs::points::PointsReader`]
/// ([`SourcePoints::reader`], the same reader `lucene_search`'s points range
/// query uses, walked by [`CollectPoints`] into a flat [`PointStream`]),
/// drops non-live docs and remaps surviving doc ids to the
/// merged id space via [`build_doc_id_maps`] (same mechanism
/// [`merge_term_range`] uses), and combines the sources' streams
/// ([`points::write_merged`]). The BKD writer rebuilds the merged BKD
/// tree (leaf plan, packed index, bounding boxes) from this flat list
/// itself, so there is no tree-merging logic to get wrong here, and -- like
/// postings, unlike doc-values/norms -- `write` already supports any number
/// of fields per call, so there is no single-field-per-merge-call limit.
///
/// # The "sparse across sources" rule, points edition
///
/// A field has no per-doc sparsity of its own to model here (a live doc
/// contributes whatever packed values the source's
/// [`lucene_codecs::points::PointsReader::intersect`] walk visits for it, or
/// none) --
/// this merge does not require every live doc to have a point (multi-valued
/// points and docs with zero points for a field are both realistic and
/// simply mean fewer points end up in the merged tree for that doc). What
/// *is* an error, matching the same philosophy as doc-values/norms/postings:
/// a source declaring point dimensions must supply its opened reader via
/// SourcePoints, otherwise this returns [`Error::PointsFieldMissingInSource`].
/// A supplied reader without a BKD entry is legitimate: an earlier merge may
/// have deleted every point while preserving the field's declared shape.
///
/// # Cross-source shape validation
///
/// Because field-number reconciliation only records the first-seen source's
/// `FieldInfo` (see `reconcile_field_numbers`), every contributing source's
/// own BKD tree shape (`num_dims`/`bytes_per_dim`, from that source's own
/// [`lucene_codecs::points::PointsField`]) is checked against the merged
/// field's declared shape (`FieldInfo::point_dimension_count`/
/// `point_num_bytes`) and rejected with [`Error::PointsShapeDisagreement`] on
/// a mismatch, and any source field whose `num_index_dims` disagrees with
/// the merged field's declared `point_index_dimension_count` is rejected
/// with [`Error::PointsIndexDimsDisagreement`].
pub(crate) fn merge_points(
    sources: &[MergeSource],
    per_source_maps: &[HashMap<i32, i32>],
    _per_source_live_ids: &[Vec<i32>],
    doc_id_maps: &[Vec<i32>],
    merged_fields: &[FieldInfo],
    only: Option<i32>,
) -> Result<Vec<points::MergePointsField>> {
    let mut candidates = points_candidates(merged_fields);
    candidates.retain(|n| only.is_none_or(|o| o == *n));
    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    let reverse_maps = invert_field_number_maps(per_source_maps);
    let mut result = Vec::with_capacity(candidates.len());

    for merged_field_number in candidates {
        let merged_field = merged_fields
            .iter()
            .find(|f| f.number == merged_field_number)
            .expect("merged_field_number came from reconcile_field_numbers over these same sources, so it must have an entry in merged_fields");
        let merged_num_dims = merged_field.point_dimension_count;
        let merged_num_index_dims = merged_field.point_index_dimension_count;
        let merged_bytes_per_dim = merged_field.point_num_bytes;
        // Every contributing source's shape is checked equal to this below;
        // with no contributing source it is never used.
        let stride = usize::try_from(merged_num_dims)
            .unwrap_or(0)
            .saturating_mul(usize::try_from(merged_bytes_per_dim).unwrap_or(0));

        // One stream per contributing source, kept separate so a
        // single-index-dimension field can be k-way merged below instead of
        // concatenated and re-sorted -- `BKDWriter.merge` versus
        // `PointsWriter.mergeOneField`.
        let mut per_source_points: Vec<points::PointsMergeSource> = Vec::new();
        // `totMaxSize`: every contributing source's `size()`, deleted
        // documents' points included -- `BKDWriter` sizes its heap buffer
        // (or decides to spill) from it.
        let mut max_point_count = 0u64;
        for (src_idx, (source, reverse)) in sources.iter().zip(&reverse_maps).enumerate() {
            let Some(&original_number) = reverse.get(&merged_field_number) else {
                // This source's own `FieldInfos` never saw the field --
                // `PointsWriter.merge`'s `if (readerFieldInfo == null)
                // continue;`. Adding a points field to an index that already
                // has segments is normal; those segments contribute no
                // points.
                continue;
            };
            let Some(sp) = source
                .points
                .iter()
                .find(|sp| sp.field_number == original_number)
            else {
                return Err(Error::PointsFieldMissingInSource {
                    merged_field_number,
                });
            };
            let Some(field_meta) = sp.reader.field(original_number) else {
                // Lucene90PointsReader.getValues may return null with a
                // declared shape: an earlier merge deleted every point.
                continue;
            };
            if field_meta.num_dims != merged_num_dims
                || field_meta.bytes_per_dim != merged_bytes_per_dim
            {
                return Err(Error::PointsShapeDisagreement {
                    merged_field_number,
                    merged_num_dims,
                    merged_bytes_per_dim,
                    source_num_dims: field_meta.num_dims,
                    source_bytes_per_dim: field_meta.bytes_per_dim,
                });
            }
            if field_meta.num_index_dims != merged_num_index_dims {
                return Err(Error::PointsIndexDimsDisagreement {
                    merged_field_number,
                    num_dims: field_meta.num_dims,
                    merged_num_index_dims,
                    source_num_index_dims: field_meta.num_index_dims,
                });
            }

            let mut collect = CollectPoints {
                doc_id_map: &doc_id_maps[src_idx],
                stride,
                stream: PointStream::default(),
                bad_length: None,
            };
            max_point_count =
                max_point_count.saturating_add(u64::try_from(field_meta.point_count).unwrap_or(0));
            // Sized up front from the source's own point count (deleted
            // documents' points included, so at least what survives). The
            // count is file-derived, so a reservation the allocator refuses
            // is skipped rather than aborting: the vectors then simply grow.
            let expected = usize::try_from(field_meta.point_count).unwrap_or(0);
            let _ = collect.stream.docs.try_reserve_exact(expected);
            let _ = collect
                .stream
                .values
                .try_reserve_exact(expected.saturating_mul(stride));
            sp.reader.intersect(original_number, &mut collect)?;
            if let Some(actual) = collect.bad_length {
                return Err(Error::PointValueLength {
                    merged_field_number,
                    expected: stride,
                    actual,
                });
            }
            per_source_points.push(points::PointsMergeSource {
                docs: collect.stream.docs,
                values: collect.stream.values,
            });
        }

        result.push(points::MergePointsField {
            field_number: merged_field_number,
            field_name: merged_field.name.clone(),
            num_dims: merged_num_dims,
            num_index_dims: merged_num_index_dims,
            bytes_per_dim: merged_bytes_per_dim,
            sources: per_source_points,
            max_point_count,
        });
    }

    Ok(result)
}

/// Java iterates merged FieldInfos, including fields whose reader has no
/// BKD entry because an earlier merge deleted every point.
fn points_candidates(merged_fields: &[FieldInfo]) -> Vec<i32> {
    let mut candidates: Vec<i32> = merged_fields
        .iter()
        .filter(|field| field.point_dimension_count > 0)
        .map(|field| field.number)
        .collect();
    candidates.sort_unstable();
    candidates
}

mod write_merged;
use write_merged::merge_segments_mapped_inner;

#[cfg(test)]
mod tests;
