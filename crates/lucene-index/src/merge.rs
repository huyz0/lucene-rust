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
//! # Doc values / norms / term vectors: mergeable, and now real flush callers exist
//!
//! [`segment_writer::flush_stored_only_segment`] itself still only ever
//! writes stored-fields-only segments. But `IndexWriter::commit`
//! (`index_writer.rs`) is a real caller for postings, term vectors, doc
//! values, and norms: postings and term vectors support multiple fields per
//! commit (`add_postings_field`/`add_term_vector_field`), while doc values
//! and norms are still single-field-per-commit
//! (`set_doc_values_field`/`set_norms_field`). Each produces a real segment
//! carrying that format, decodable by this module's merge sources. This
//! module still cannot be exercised end-to-end for the *general*
//! multi-doc-values/norms-field-per-source case ("flush two real segments
//! each with several doc-values/norms fields, merge them") from a single
//! commit, since those two formats are only wired one field at a time today.
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
//! **Doc values have no per-merge field limit.** Every merged field of every
//! type is written into one `.dvm`/`.dvd`/`.dvs` triple through
//! [`lucene_codecs::doc_values::write_dense_fields`], which is the real
//! on-disk shape of a `Lucene90DocValuesFormat` segment (`numFields`
//! interleaved meta entries over shared data). A **multi-tier index sort**
//! needs exactly that: a second sort tier is a second NUMERIC column, and
//! before this the merge could carry only one field per type and only one
//! type per call.
//!
//! **Norms have no per-merge field limit either** (c26): every merged norms
//! field goes into one `.nvm`/`.nvd` pair through
//! [`lucene_codecs::norms::write_fields`], which is the real on-disk shape of
//! a `Lucene90NormsFormat` segment (`Lucene90NormsConsumer` gets one
//! `addNormsField` call per field into one pair). Until c26 this was capped at
//! one field per merge, because [`lucene_codecs::norms::write_single_dense_field`]
//! writes a whole pair and two of them in one merge overwrote each other.
//! Term vectors and postings have never had a limit.
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
//! - **NUMERIC** doc values may be sparse. A merged column only some
//!   documents have a value for is written through the same `IndexedDISI` +
//!   values body `Lucene90DocValuesConsumer.writeValues` uses, and a column
//!   **no** merged document has a value for is still written, as that
//!   method's all-missing form (`docsWithFieldOffset = -2`).
//! - **BINARY/SORTED/SORTED_NUMERIC/SORTED_SET and norms** must still be
//!   dense over the merged segment, because
//!   [`lucene_codecs::doc_values::DenseField`]'s only sparse variant is the
//!   numeric one.
//!
//! In every case a live-doc-contributing source that never declared the
//! field at all is a **schema** mismatch, not sparsity, and stays a hard
//! error ([`Error::DocValuesFieldMissingInSource`] /
//! [`Error::BinaryDocValuesFieldMissingInSource`] /
//! [`Error::SortedDocValuesFieldMissingInSource`] /
//! [`Error::NormsFieldMissingInSource`]) rather than a silently dropped
//! field.
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
//! [`merge_postings`] merges each source's term dictionary + doc/freq data
//! (`.tim`/`.tip`/`.tmd`/`.doc`) for every field any source declares
//! postings for ([`SourcePostings`], attached per source via
//! [`MergeSource::postings`]), re-encoding the result with
//! [`lucene_codecs::postings_writer::write_fields`]. Because each source's
//! term dictionary is independent (same reason SORTED doc values need
//! special handling above), this resolves each source's own terms straight
//! to bytes via that source's already-opened
//! [`lucene_codecs::blocktree::FieldTerms`], unions those bytes across
//! sources into one sorted term set, and for each term concatenates every
//! contributing source's `(mergedDocId, freq)` pairs in source order --
//! ascending overall for free, since merged doc ids occupy disjoint,
//! increasing per-source ranges (see [`build_doc_id_maps`]'s doc comment).
//! `write_fields` already accepts any number of fields in one call, so
//! unlike doc-values/norms there is no single-field-per-merge-call limit
//! for postings. Unlike doc values/norms there is no "sparse across
//! sources" restriction here: a source whose own `FieldInfos` never saw the
//! field simply contributes no terms, matching `FieldsConsumer.merge`'s
//! `Terms terms = fields.terms(field); if (terms == null) continue;` (this
//! is the normal state of a segment written before the field existed). A
//! source that *does* declare the field but whose caller supplied no
//! [`SourcePostings`] for it is still a hard error
//! ([`Error::PostingsFieldMissingInSource`]) -- that is a caller wiring
//! bug, not index evolution. Ordinary per-doc/per-term sparsity (most docs
//! don't contain most terms) is of course not an error either.
//!
//! **Scope: `Docs`/`DocsAndFreqs`/`DocsAndCustomFreqs`, plus positions/
//! offsets/payloads.** A field whose merged `index_options` indexes
//! positions ([`IndexOptions::DocsAndFreqsAndPositions`] or
//! [`IndexOptions::DocsAndFreqsAndPositionsAndOffsets`]) has every
//! contributing source's positions read back via
//! [`lucene_codecs::blocktree::FieldTerms::positions`] (the same read path
//! `lucene_search`'s phrase matching uses) and concatenated in the same
//! source order as docs/freqs; offsets are carried along automatically
//! since `Position` bundles them, and payloads whenever the merged field's
//! `FieldInfo::store_payloads` is set. Any other `index_options` this port
//! doesn't model (there are none left in the [`IndexOptions`] enum) would
//! still be rejected with [`Error::PostingsIndexOptionsNotSupported`].
//! Because field-number reconciliation
//! records the *first-seen* source's `FieldInfo` as the merged one, every
//! other source sharing that field name must agree on `index_options`
//! ([`Error::PostingsIndexOptionsDisagreement`], raised by
//! [`reconcile_field_numbers`]) -- otherwise a source with positions could
//! have them silently dropped whenever an earlier, positions-free source
//! happened to be picked as canonical. `store_payloads` is the one
//! attribute that is *merged* (ORed) rather than verified, matching
//! `FieldInfos.Builder.add`: a source without payloads contributes empty
//! `Position::payload`s, which the postings writer encodes as "no payload
//! at this occurrence".
//!
//! Same caveat as doc values/norms/term vectors: nothing in this port's
//! normal flush path produces a segment with postings yet (`.tim`/`.tip`/
//! `.tmd`/`.doc` are written by
//! [`lucene_codecs::postings_writer::write_fields`] as a standalone
//! function, not from a per-field indexing flush path), so this merge
//! logic is real and tested on its own, but not yet reachable from a real
//! end-to-end "flush two segments, merge them" caller.
//!
//! # Points
//!
//! [`merge_points`] merges each source's BKD points data (`.kdm`/`.kdi`/
//! `.kdd`) for every field any source declares points for ([`SourcePoints`],
//! attached per source via [`MergeSource::points`]), re-encoding the result
//! with [`lucene_codecs::points::write`]. Unlike SORTED doc values or
//! postings, a points field has no shared term dictionary to resolve
//! ordinals against -- it's a flat, per-doc set of fixed-width packed values
//! (closer in spirit to SORTED_NUMERIC doc values than to postings), so this
//! reads back every live doc's points via that source's own already-opened
//! [`lucene_codecs::points::PointsReader`] (the *exact same* reader
//! `lucene_search`'s points range query already uses -- no new BKD decode
//! logic was written for this merge), drops non-live docs and remaps
//! surviving doc ids to the merged id space (reusing
//! [`build_doc_id_maps`], the same mechanism [`merge_postings`] uses), and
//! concatenates the results across sources in source order.
//! [`lucene_codecs::points::write`] already accepts any number of fields per
//! call, so, like postings and unlike doc-values/norms, there is no
//! single-field-per-merge-call limit for points.
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
//! up with zero points after the merge is simply omitted from the merged
//! segment (matching real Lucene's `finish()` returning `null`/omitting the
//! field, and [`lucene_codecs::points::write`]'s own documented
//! `EmptyField` restriction).
//!
//! **Scope: single packed-value shape per field across all sources,
//! `num_index_dims` may be less than `num_dims`.**
//! [`lucene_codecs::points::write`] supports fields whose `num_index_dims`
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
//! [`lucene_codecs::points::write`] itself already handles, and the
//! concatenation this merge performs preserves both.
//!
//! Same caveat as doc values/norms/term vectors/postings: nothing in this
//! port's normal flush path produces a segment with points yet (`.kdm`/
//! `.kdi`/`.kdd` are written by [`lucene_codecs::points::write`] as a
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
//! See `docs/parity.md` and `PLAN.md`'s Phase 5 section for the exact,
//! currently-true scope line.

use std::collections::{HashMap, HashSet};

use crate::index_writer::{
    per_field_codec_suffix, per_field_segment, DOC_VALUES_FORMAT_NAME, PER_FIELD_SUFFIX,
    POSTINGS_FORMAT_NAME,
};
use crate::segment_info::{self, IndexSortField, LuceneVersion, SegmentInfo, SortMissingValue};
use crate::segment_infos::SegmentCommitInfo;
use lucene_codecs::blocktree::{self, FieldTerms};
use lucene_codecs::doc_values::{
    self, BinaryEntry, NumericEntry, SortedEntry, SortedNumericEntry, SortedSetEntry, SortedSetKind,
};
use lucene_codecs::field_infos::{self, FieldInfo, IndexOptions, VectorEncoding};
use lucene_codecs::norms::{self, NormsEntry};
use lucene_codecs::points::{self, WritePointsField};
use lucene_codecs::postings::DocInput;
use lucene_codecs::postings_writer::{self, FieldPostingsInput, TermPostings};
use lucene_codecs::stored_fields;
use lucene_codecs::term_vectors::{self, TermVectorsDocument, TermVectorsReader};
use lucene_codecs::terms_dict;
use lucene_codecs::{hnsw, hnsw_vectors, vectors};
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
    /// Every vector-format error -- `vectors`, `hnsw` and `hnsw_vectors` all
    /// share one `Error` type, the way they share one on-disk format family.
    #[error(transparent)]
    Vectors(#[from] lucene_codecs::vectors::Error),
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
    /// Same as [`Error::DocValuesFieldMissingInSource`], for BINARY doc
    /// values.
    #[error(
        "merged field number {merged_field_number} has binary doc values in some sources but not in every source that contributes live docs (or not for every one of that source's live docs)"
    )]
    BinaryDocValuesFieldMissingInSource { merged_field_number: i32 },
    /// Same as [`Error::DocValuesFieldMissingInSource`], for SORTED doc
    /// values.
    #[error(
        "merged field number {merged_field_number} has sorted doc values in some sources but not in every source that contributes live docs (or not for every one of that source's live docs)"
    )]
    SortedDocValuesFieldMissingInSource { merged_field_number: i32 },
    /// Same as [`Error::DocValuesFieldMissingInSource`], for SORTED_NUMERIC
    /// doc values -- also raised for a live doc whose resolved value list
    /// came back empty, since
    /// [`lucene_codecs::doc_values::write_single_dense_sorted_numeric_field`]
    /// requires every doc to have at least one value.
    #[error(
        "merged field number {merged_field_number} has sorted-numeric doc values in some sources but not in every source that contributes live docs (or not for every one of that source's live docs)"
    )]
    SortedNumericDocValuesFieldMissingInSource { merged_field_number: i32 },
    /// Same as [`Error::DocValuesFieldMissingInSource`], for SORTED_SET doc
    /// values -- also raised for a live doc whose resolved value set came
    /// back empty, since
    /// [`lucene_codecs::doc_values::write_single_dense_sorted_set_field`]
    /// requires every doc to have at least one value.
    #[error(
        "merged field number {merged_field_number} has sorted-set doc values in some sources but not in every source that contributes live docs (or not for every one of that source's live docs)"
    )]
    SortedSetDocValuesFieldMissingInSource { merged_field_number: i32 },
    /// A live-doc-contributing source *declares* this field in its own
    /// `FieldInfos` but its [`MergeSource::postings`] carries no entry for
    /// it -- a caller wiring bug (the segment has postings on disk but the
    /// caller never opened them), not index evolution. A source whose
    /// `FieldInfos` never saw the field at all is fine and simply
    /// contributes no terms, matching `FieldsConsumer.merge`.
    ///
    /// **Deliberately stricter than Java here**: real Lucene's `MultiFields`
    /// also tolerates a declared field whose reader returns no `Terms` (a
    /// segment where every doc happened to have no value for the field). This
    /// port cannot tell that apart from "the caller forgot to open the
    /// `.tim`", because [`SourcePostings`] is caller-supplied rather than
    /// pulled from a reader, and silently merging away a whole source's
    /// postings is the worse failure. Revisit if `MergeSource` ever takes a
    /// reader instead of pre-opened per-field handles.
    #[error(
        "merged field number {merged_field_number} is declared by a live-doc-contributing source whose MergeSource supplied no postings for it"
    )]
    PostingsFieldMissingInSource { merged_field_number: i32 },
    /// A field's merged `index_options` indexes positions, but a source
    /// contributing live docs for it wasn't given an opened `.pos` reader
    /// (`SourcePostings::pos_in`) -- a caller-side wiring inconsistency,
    /// same class of error as [`Error::PostingsFieldMissingInSource`] rather
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
    /// A live-doc-contributing source *declares* this points field in its
    /// own `FieldInfos` but its [`MergeSource::points`] carries no entry for
    /// it (or its opened reader has no such field) -- a caller wiring bug,
    /// same distinction [`Error::PostingsFieldMissingInSource`] draws
    /// (including its "deliberately stricter than Java" note: real
    /// `PointsWriter.merge` also tolerates `values == null` for a declared
    /// field). A source whose `FieldInfos` never saw the field at all is fine
    /// and simply contributes no points, matching `PointsWriter.merge`.
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
    /// [`lucene_codecs::points::write`] (wrong packed-value length) or, if
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
    /// [`lucene_codecs::points::write`] supports `num_index_dims <=
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
    /// format's data simply gone -- the exact shape of c22's findings 14, 22,
    /// 23 and 24. Refused instead.
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
/// [`SourceNumericDocValues`], for [`NormsEntry`]/`.nvd` instead.
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
/// [`lucene_codecs::postings::PayInput::open`]), needed by [`merge_postings`]
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
/// [`lucene_codecs::points::write`] supports `num_index_dims` less than
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
    /// [`lucene_codecs::points::write`] already supports any number of
    /// fields per call, so there is no single-field-per-merge-call limit
    /// here (see [`SourcePoints`] for the exact scope of what gets merged).
    pub points: &'a [SourcePoints<'a>],
    /// This source's KNN vector readers, or `None` if this source has no
    /// vectors at all. One pair of readers covers every vector field of the
    /// segment, so unlike postings/points this is not a per-field list (see
    /// [`SourceVectors`]).
    pub vectors: Option<&'a SourceVectors<'a>>,
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
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &[],
            vectors: None,
        }
    }
}

/// One per-segment Lucene format, as the pair of facts a merge needs about
/// it: **which file extensions it owns**, and **whether a
/// [`MergeSource`] carries its opened readers**.
///
/// # Why this type exists
///
/// c22 fixed eleven correctness defects in the merge path; its Tier-2 review
/// traced three of them (findings 22, 23 and 24 -- doc-values types other
/// than NUMERIC, positional postings, and `has_blocks`) plus one older one
/// (finding 14, norms, standing since c4 and making **every merged BM25 score
/// wrong**) to a single structural gap, in its own words:
///
/// > nothing mechanically checks that `execute_merge` opens every format the
/// > flush can write
///
/// The failure is silent by construction. A format the merge does not open
/// contributes nothing; [`describe_written_files`] then clears the
/// corresponding capability off the merged `.fnm` so the segment stays
/// *openable*; and what is left is well-formed, checksummed and
/// `CheckIndex`-clean, with the data gone or (for norms) wrong. Nothing
/// downstream reports it, which is why all four were found by reading.
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
            SegmentFormat::Postings => &["tim", "tip", "tmd", "doc", "psm", "pos", "pay"],
            SegmentFormat::DocValues => &["dvd", "dvm", "dvs"],
            SegmentFormat::Norms => &["nvd", "nvm"],
            SegmentFormat::Points => &["kdd", "kdi", "kdm"],
            SegmentFormat::KnnVectors => &["vec", "vemf", "vem", "vex"],
        }
    }

    /// The format that owns `extension`, or `None` when no format claims it.
    pub fn for_extension(extension: &str) -> Option<SegmentFormat> {
        SegmentFormat::ALL
            .into_iter()
            .find(|f| f.extensions().contains(&extension))
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
///    the format. That is the closed loop c22 asked for.
/// 2. **It runs on every merge, not in one test.** It is called from
///    `IndexWriter::execute_merge` on the real path, so every existing and
///    future merge test exercises it over whatever formats that test's
///    segments happen to have.
///
/// # What it deliberately does not check, and cannot
///
/// - **That an opened format is merged *correctly*.** A format whose doc map
///   is wrong produces exactly the same well-formed segment; that is what
///   c22's `assert_every_format_agrees` and c17's `VerifySortedSegment` are
///   for. This gate is "opened at all", which is the failure c22 actually hit
///   four times.
/// - **A format the flush writes without listing it in the `.si`.** Such a
///   file is already invisible to `IndexFileDeleter` and to `CheckIndex`, so
///   it is a different (and louder) defect.
/// - **A format opened per source but dropped per *field*.** c22 finding 22
///   was a per-type drop inside an opened `.dvm`; the `debug_assert!` in
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
) -> Result<()> {
    debug_assert_eq!(source_names.len(), sources.len());
    debug_assert_eq!(source_files.len(), sources.len());
    for ((name, files), source) in source_names.iter().zip(source_files).zip(sources) {
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
            if !format.is_opened(source) {
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
/// assigns every distinct field *name* a single, contiguous merged field
/// number, in first-seen order across sources (source 0's fields first, then
/// any new names introduced by source 1, etc.) -- mirrors real Lucene's
/// `FieldInfos.FieldNumbers.addOrGet`, which hands out a process-wide number
/// per name and reuses it for every segment that has that field, regardless
/// of what number that segment originally used.
///
/// Returns the merged field list (one [`FieldInfo`] per distinct name, seeded
/// from the *first* source's metadata for that name) and, per source, a map
/// from that source's original field number to the merged number.
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
    let mut merged_fields: Vec<FieldInfo> = Vec::new();
    let mut name_to_merged_number: HashMap<String, i32> = HashMap::new();
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
            match name_to_merged_number.get(&f.name) {
                Some(&merged_number) => {
                    let merged = &mut merged_fields[merged_number as usize];
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
                    let number = merged_fields.len() as i32;
                    let mut renumbered = f.clone();
                    renumbered.number = number;
                    merged_fields.push(renumbered);
                    name_to_merged_number.insert(f.name.clone(), number);
                    map.insert(f.number, number);
                }
            }
        }
        per_source_maps.push(map);
    }

    Ok((merged_fields, per_source_maps))
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
fn concat_doc_order(per_source_live_ids: &[Vec<i32>]) -> Vec<(usize, i32)> {
    let mut order = Vec::new();
    for (src_idx, live_ids) in per_source_live_ids.iter().enumerate() {
        for &doc_id in live_ids {
            order.push((src_idx, doc_id));
        }
    }
    order
}

/// Rewrites the merged field list so the merged `.fnm` describes the files
/// the merge **actually wrote**, the way `IndexWriter`'s
/// `fields_with_per_field_attributes` already does at flush time.
///
/// A merged `FieldInfo` is seeded from a source segment's own `.fnm` (see
/// [`reconcile_field_numbers`]), which describes *that* segment's files. Five
/// things in it can be wrong for the merged segment, and every one of them is
/// silent or fatal in real Lucene rather than merely cosmetic:
///
/// - **`PerFieldPostingsFormat.format`/`.suffix`, the `PerFieldDocValues`
///   pair, and the `PerFieldKnnVectors` pair.** Real Lucene routes a field to
///   a postings/doc-values/vectors format purely through these attributes.
///   Without them a merged segment's `.doc`/`.tim`/`.tip`/`.tmd` are never
///   opened at all: `MultiTerms.getTerms(reader, field)` returns `null` and
///   the field silently reads back as having no terms, with no error anywhere
///   -- the exact silent-failure shape `VerifyFullSegment` was written to
///   catch at flush time. Conversely, an attribute inherited from a source for
///   data this merge did *not* write sends a reader looking for a file that
///   does not exist.
/// - **`doc_values_type`.** A `.fnm` claiming a type the merged `.dvm` has no
///   entry for makes `PerFieldDocValuesFormat.FieldsReader` register no
///   producer, so `getNumeric` returns `null` and real
///   `CheckIndex.testDocValues` -- which iterates every field whose `.fnm`
///   claims doc values -- dereferences it.
/// - **`vector_dimension`.** Same shape: a positive dimension makes
///   `FieldInfo.hasVectorValues()` true, which `IncrementalHnswGraphMerger`
///   and `CheckIndex` key off, while `PerFieldKnnVectorsFormat` registers no
///   reader without the attribute.
/// - **`omit_norms`.** `DirectoryReader.open` throws on a missing `.nvm`
///   rather than degrading, so an indexed field whose merged `.fnm` still
///   claims norms the merge did not write makes the whole index unopenable.
/// - **`store_term_vectors`.** Same shape, for a segment with no `.tvd` at
///   all.
///
/// Nothing here invents data. But every one of these rewrites **tolerates a
/// loss** rather than refusing it, and that is a deliberate, uncomfortable
/// choice worth stating plainly: this function's job is to describe files, and
/// a merge that produced fewer of them than its sources had is already a fact
/// by the time it runs. Clearing the claim keeps the merged segment openable
/// and honest; it does not put the data back.
///
/// The guard against that ever mattering is **on the caller side**, and it is
/// the one thing to preserve when adding a format here:
/// [`crate::index_writer::IndexWriter::execute_merge`] opens every format its
/// own flush can write (stored fields, postings, term vectors, norms, all five
/// doc-values types and vectors), and
/// [`crate::index_writer::IndexWriter::segment_stats`] withholds from the
/// merge policy the one case it cannot round-trip (a doc-values *generation*).
/// A format opened by neither would be dropped here silently -- which is what
/// happened to norms until `c22-sorted-merge`, for the whole life of
/// `execute_merge`.
///
fn describe_written_files(
    merged_fields: &mut [FieldInfo],
    postings_field_numbers: &[i32],
    doc_values_field_numbers: &[i32],
    norms_field_numbers: &[i32],
    wrote_term_vectors: bool,
    vector_field_numbers: &[i32],
) {
    for f in merged_fields.iter_mut() {
        f.attributes.retain(|(key, _)| {
            key != "PerFieldPostingsFormat.format"
                && key != "PerFieldPostingsFormat.suffix"
                && key != "PerFieldDocValuesFormat.format"
                && key != "PerFieldDocValuesFormat.suffix"
                && key != "PerFieldKnnVectorsFormat.format"
                && key != "PerFieldKnnVectorsFormat.suffix"
        });
        if postings_field_numbers.contains(&f.number) {
            f.attributes.push((
                "PerFieldPostingsFormat.format".to_string(),
                POSTINGS_FORMAT_NAME.to_string(),
            ));
            f.attributes.push((
                "PerFieldPostingsFormat.suffix".to_string(),
                PER_FIELD_SUFFIX.to_string(),
            ));
        }
        if doc_values_field_numbers.contains(&f.number) {
            f.attributes.push((
                "PerFieldDocValuesFormat.format".to_string(),
                DOC_VALUES_FORMAT_NAME.to_string(),
            ));
            f.attributes.push((
                "PerFieldDocValuesFormat.suffix".to_string(),
                PER_FIELD_SUFFIX.to_string(),
            ));
        } else {
            // A `.fnm` claiming a `DocValuesType` the merged `.dvm` has no
            // entry for makes `PerFieldDocValuesFormat.FieldsReader` register
            // no producer, so `getNumeric` returns `null` and real
            // `CheckIndex.testDocValues` -- which iterates every field whose
            // `.fnm` claims doc values -- dereferences it. Same rule
            // `IndexWriter::fields_with_per_field_attributes` applies at
            // flush time.
            f.doc_values_type = lucene_codecs::field_infos::DocValuesType::None;
            f.doc_values_skip_index_type = lucene_codecs::field_infos::DocValuesSkipIndexType::None;
        }
        if f.index_options != IndexOptions::None && !norms_field_numbers.contains(&f.number) {
            f.omit_norms = true;
        }
        if !wrote_term_vectors {
            f.store_term_vectors = false;
        }
        if vector_field_numbers.contains(&f.number) {
            f.attributes.push((
                "PerFieldKnnVectorsFormat.format".to_string(),
                crate::index_writer::KNN_VECTORS_FORMAT_NAME.to_string(),
            ));
            f.attributes.push((
                "PerFieldKnnVectorsFormat.suffix".to_string(),
                PER_FIELD_SUFFIX.to_string(),
            ));
        } else {
            // The vectors twin of the doc-values rule above, and it tolerates
            // a loss on the same terms as `omit_norms`: a caller that supplies
            // no `MergeSource::vectors` for sources that have them gets a
            // merged segment with no vectors and an honest `.fnm`, rather than
            // one that claims them. `IndexWriter::execute_merge` always opens
            // them, so the loss is not reachable from this port's own writer.
            // A positive `vector_dimension` makes `FieldInfo.hasVectorValues()` true,
            // which `IncrementalHnswGraphMerger` and `CheckIndex` key off,
            // while `PerFieldKnnVectorsFormat` registers no reader without
            // the format attribute -- so the field reads back as
            // vector-capable and yields nothing, with no error anywhere.
            f.vector_dimension = 0;
        }
    }
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
fn stored_fields_merge_strategy(
    writer: &stored_fields::StoredFieldsWriter,
    source: &MergeSource,
    matching: bool,
) -> StoredFieldsMergeStrategy {
    if !matching {
        return StoredFieldsMergeStrategy::Visitor;
    }
    // "its not worth fine-graining this if there are deletions" -- plus the
    // compressor/chunk-size/dirtiness trio, which `can_bulk_copy` owns.
    if source.live_docs.is_none() && writer.can_bulk_copy(source.reader) {
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
fn write_merged_stored_fields(
    sources: &[MergeSource],
    per_source_maps: &[HashMap<i32, i32>],
    doc_order: &[(usize, i32)],
    merged_segment_id: &[u8; ID_LENGTH],
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let mut writer = stored_fields::StoredFieldsWriter::new(
        stored_fields::Mode::BestSpeed,
        merged_segment_id,
        "",
    );
    // Java's per-reader cached `BlockState`: the DOC path reads a run of
    // documents out of the same chunk, and decompressing that chunk once per
    // document instead of once per chunk is the difference between a merge
    // that costs O(documents) inflations and one that costs O(chunks).
    let mut chunk_cursors: Vec<stored_fields::ChunkCursor> = (0..sources.len())
        .map(|_| stored_fields::ChunkCursor::new())
        .collect();
    let matching = matching_readers(sources, per_source_maps);
    let mut strategies: Vec<StoredFieldsMergeStrategy> = Vec::with_capacity(sources.len());
    for (source, &m) in sources.iter().zip(&matching) {
        // Java runs `reader.checkIntegrity(mergeState.oneMerge)` on every
        // source before it picks a strategy, and it matters most for the one
        // this batch added: BULK copies a source's compressed bytes verbatim
        // and then writes a freshly computed, valid footer over them, so a bit
        // flip in the source would be laundered into a merged segment that
        // passes every checksum from then on. `stored_fields::open` only
        // validates the footer's shape (`retrieve_checksum`), which is the
        // right trade for a random-access reader and not enough here.
        source.reader.check_integrity()?;
        strategies.push(stored_fields_merge_strategy(&writer, source, m));
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
                let mut doc = stored_fields::parse_document(num_stored_fields, bytes)?;
                let field_number_map = &per_source_maps[src_idx];
                for field in &mut doc.fields {
                    field.field_number = *field_number_map.get(&field.field_number).ok_or(
                        Error::UnknownSourceFieldNumber {
                            field_number: field.field_number,
                        },
                    )?;
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
    Ok(writer.finish())
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
    pub field: &'a str,
    pub reverse: bool,
    pub missing: SortMissingValue,
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
/// fast path is taken exactly when it is legal and never when it is not --
/// see `docs/sweep/m2/c22-sorted-merge.md` for the measurement.
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
    // Reported, not asserted: this is a `pub` entry point, and a caller that
    // hands it a mis-shaped key table gets a named error rather than a panic
    // (the same call this module's `read`-side errors already make).
    if let Some(sort_fields) = sort_fields {
        if sort_fields.is_empty() {
            return Err(Error::EmptySortFields);
        }
        for spec in sort_fields {
            if spec.per_source_keys.len() != sources.len() {
                return Err(Error::SortKeysWrongLength {
                    field: spec.field.to_string(),
                    source_index: None,
                    expected: sources.len(),
                    found: spec.per_source_keys.len(),
                });
            }
            for (source_index, (source, keys)) in
                sources.iter().zip(spec.per_source_keys.iter()).enumerate()
            {
                if keys.len() != source.reader.max_doc().max(0) as usize {
                    return Err(Error::SortKeysWrongLength {
                        field: spec.field.to_string(),
                        source_index: Some(source_index),
                        expected: source.reader.max_doc().max(0) as usize,
                        found: keys.len(),
                    });
                }
            }
        }
    }

    let sources_fields: Vec<&[FieldInfo]> = sources.iter().map(|s| s.field_infos).collect();
    let (merged_fields, per_source_maps) = reconcile_field_numbers(&sources_fields)?;

    // Per-source live (pre-merge) doc ids, ascending.
    let mut per_source_live_ids: Vec<Vec<i32>> = Vec::with_capacity(sources.len());
    for (source_index, source) in sources.iter().enumerate() {
        let max_doc = source.reader.max_doc();
        // `docs/arithmetic-gate.md`'s crate rule: *never index a
        // `FixedBitSet` with an index bounded against anything other than
        // that bitset's own `len()`.* The doc id below is bounded by
        // `max_doc`, which comes off the source's `.fdm`, and the bitset
        // comes off its `.liv` -- two independent files. `FixedBitSet::get`
        // does `words[index >> 6]` behind a bare `debug_assert`, so a `.liv`
        // one word short of `max_doc` reads a **ghost bit** in release (a
        // silently wrong live/dead answer, i.e. a document merged or dropped
        // that should not have been), and one 64 or more bits short is an
        // index panic in both profiles.
        //
        // Checked once per source rather than per document: it is both the
        // bound the loop needs and cheaper than the per-document `min` that
        // would otherwise hide the disagreement.
        if let Some(bits) = source.live_docs {
            if bits.len() != usize::try_from(max_doc).unwrap_or(0) {
                return Err(Error::LiveDocsLengthMismatch {
                    source_index,
                    max_doc,
                    live_docs_len: bits.len(),
                });
            }
        }
        let mut live_ids = Vec::new();
        for doc_id in 0..max_doc {
            // ARITH-adjacent: `doc_id < max_doc == bits.len()` by the check
            // above, so this index is bounded by the bitset's own length.
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

    let doc_order = match sort_fields {
        None => concat_doc_order(&per_source_live_ids),
        Some(sort_fields) => sorted_doc_order(sort_fields, &per_source_live_ids),
    };
    let doc_count = doc_order.len() as i32;
    let per_source_max_doc: Vec<i32> = sources.iter().map(|s| s.reader.max_doc()).collect();
    let doc_id_maps = build_doc_id_maps(&per_source_max_doc, &doc_order);

    // Doc values, norms and term vectors, all resolved through the one
    // `doc_order` -- the merged segment's physical order, whichever way it
    // was produced.
    let mut merged_doc_values =
        merge_numeric_doc_values(sources, &per_source_maps, &per_source_live_ids, &doc_order)?;
    merged_doc_values.extend(merge_binary_doc_values(
        sources,
        &per_source_maps,
        &per_source_live_ids,
        &doc_order,
    )?);
    merged_doc_values.extend(merge_sorted_doc_values(
        sources,
        &per_source_maps,
        &per_source_live_ids,
        &doc_order,
    )?);
    merged_doc_values.extend(merge_sorted_numeric_doc_values(
        sources,
        &per_source_maps,
        &per_source_live_ids,
        &doc_order,
    )?);
    merged_doc_values.extend(merge_sorted_set_doc_values(
        sources,
        &per_source_maps,
        &per_source_live_ids,
        &doc_order,
    )?);
    merged_doc_values.sort_by_key(|f| f.field_number());

    let merged_norms = merge_norms(sources, &per_source_maps, &per_source_live_ids, &doc_order)?;
    let tv_files =
        write_merged_term_vectors(sources, &per_source_maps, &doc_order, &merged_segment_id)?;
    let merged_postings_fields = merge_postings(
        sources,
        &per_source_maps,
        &per_source_live_ids,
        &doc_order,
        &doc_id_maps,
        &merged_fields,
    )?;
    let merged_points_fields = merge_points(
        sources,
        &per_source_maps,
        &per_source_live_ids,
        &doc_id_maps,
        &merged_fields,
    )?;
    let merged_vectors = merge_vectors(
        sources,
        &per_source_maps,
        &doc_id_maps,
        &merged_fields,
        doc_count,
        options,
        &merged_segment_id,
    )?;

    // What the merged `.fnm` is allowed to claim, captured before the values
    // below are consumed by their writers (see `describe_written_files`).
    let postings_field_numbers: Vec<i32> = merged_postings_fields
        .iter()
        .map(|f| f.field_number)
        .collect();
    let doc_values_field_numbers: Vec<i32> =
        merged_doc_values.iter().map(|f| f.field_number()).collect();
    let vector_field_numbers: Vec<i32> = merged_vectors
        .as_ref()
        .map(|v| v.field_numbers.clone())
        .unwrap_or_default();
    let norms_field_numbers: Vec<i32> = merged_norms.iter().map(|(n, _)| *n).collect();
    let wrote_term_vectors = tv_files.is_some();

    let mut files: Vec<String> = Vec::new();

    let (fdt, fdx, fdm) =
        write_merged_stored_fields(sources, &per_source_maps, &doc_order, &merged_segment_id)?;
    let fdt_name = format!("{merged_segment_name}.fdt");
    let fdx_name = format!("{merged_segment_name}.fdx");
    let fdm_name = format!("{merged_segment_name}.fdm");
    for (name, bytes) in [(&fdt_name, &fdt), (&fdx_name, &fdx), (&fdm_name, &fdm)] {
        write_file(dir, name, bytes)?;
        files.push(name.clone());
    }

    let fnm_name = format!("{merged_segment_name}.fnm");
    let mut merged_fields = merged_fields;
    describe_written_files(
        &mut merged_fields,
        &postings_field_numbers,
        &doc_values_field_numbers,
        &norms_field_numbers,
        wrote_term_vectors,
        &vector_field_numbers,
    );
    let fnm = field_infos::write(&merged_fields, &merged_segment_id, "");
    write_file(dir, &fnm_name, &fnm)?;
    files.push(fnm_name);

    // Every doc-values field of the merged segment shares one
    // `.dvm`/`.dvd`/`.dvs` triple, which is what a real multi-field
    // `Lucene90DocValuesFormat` segment looks like.
    if !merged_doc_values.is_empty() {
        let dense_fields: Vec<doc_values::DenseField<'_>> = merged_doc_values
            .iter()
            .map(|f| f.as_dense_field())
            .collect();
        let (dvm, dvd, dvs) = doc_values::write_dense_fields(
            &dense_fields,
            doc_count,
            &merged_segment_id,
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        )?;
        for (ext, bytes) in [("dvm", &dvm), ("dvd", &dvd), ("dvs", &dvs)] {
            let name = format!(
                "{}.{ext}",
                per_field_segment(merged_segment_name, DOC_VALUES_FORMAT_NAME)
            );
            write_file(dir, &name, bytes)?;
            files.push(name);
        }
    }

    // Every norms field of the merged segment shares one `.nvm`/`.nvd`
    // pair, which is what a real `Lucene90NormsFormat` segment is
    // (`Lucene90NormsConsumer` gets one `addNormsField` call per field into
    // one pair) -- and what `write_dense_fields` already did for doc values.
    // Until c26 this was one `write_single_dense_field` call, so a merge of
    // segments with norms on two fields was `Error::TooManyNormsFields`.
    if !merged_norms.is_empty() {
        let norms_fields: Vec<norms::NormsField<'_>> = merged_norms
            .iter()
            .map(|(number, values)| norms::NormsField::Dense(*number, values))
            .collect();
        let (nvm, nvd) = norms::write_fields(&norms_fields, doc_count, &merged_segment_id, "")?;
        for (ext, bytes) in [("nvm", &nvm), ("nvd", &nvd)] {
            let name = format!("{merged_segment_name}.{ext}");
            write_file(dir, &name, bytes)?;
            files.push(name);
        }
    }

    if let Some((tvd, tvx, tvm)) = tv_files {
        for (ext, bytes) in [("tvd", &tvd), ("tvx", &tvx), ("tvm", &tvm)] {
            let name = format!("{merged_segment_name}.{ext}");
            write_file(dir, &name, bytes)?;
            files.push(name);
        }
    }

    if !merged_postings_fields.is_empty() {
        let inputs: Vec<FieldPostingsInput<'_>> = merged_postings_fields
            .iter()
            .map(|f| FieldPostingsInput {
                field_number: f.field_number,
                index_options: f.index_options,
                doc_count: f.doc_count,
                has_payloads: f.has_payloads,
                terms: &f.terms,
            })
            .collect();
        let output = postings_writer::write_fields(
            &inputs,
            &merged_segment_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )?;
        let mut exts: Vec<(&str, &[u8])> = vec![
            ("doc", &output.doc),
            ("psm", &output.psm),
            ("tim", &output.tim),
            ("tip", &output.tip),
            ("tmd", &output.tmd),
        ];
        if !output.pos.is_empty() {
            exts.push(("pos", &output.pos));
        }
        if !output.pay.is_empty() {
            exts.push(("pay", &output.pay));
        }
        for (ext, bytes) in exts {
            let name = format!(
                "{}.{ext}",
                per_field_segment(merged_segment_name, POSTINGS_FORMAT_NAME)
            );
            write_file(dir, &name, bytes)?;
            files.push(name);
        }
    }

    // A merged field with zero surviving points (every contributing live
    // doc happened to have none) is simply omitted -- `points::write`
    // doesn't support empty fields (see its own doc comment), and this
    // matches real Lucene's `finish()` returning `null`/omitting the field
    // entirely in that case.
    //
    // `points` is *moved* into the `WritePointsField`, not cloned: the merged
    // point list is one `Vec<u8>` allocation per point, and copying the whole
    // thing again on the way to the writer was pure waste.
    let inputs: Vec<WritePointsField> = merged_points_fields
        .into_iter()
        .filter(|f| !f.points.is_empty())
        .map(|f| WritePointsField {
            field_number: f.field_number,
            num_dims: f.num_dims,
            num_index_dims: f.num_index_dims,
            bytes_per_dim: f.bytes_per_dim,
            points: f.points,
        })
        .collect();
    if !inputs.is_empty() {
        let (kdm, kdi, kdd) = points::write(
            &inputs,
            points::DEFAULT_MAX_POINTS_IN_LEAF_NODE,
            &merged_segment_id,
            "",
        )?;
        for (ext, bytes) in [("kdm", &kdm), ("kdi", &kdi), ("kdd", &kdd)] {
            let name = format!("{merged_segment_name}.{ext}");
            write_file(dir, &name, bytes)?;
            files.push(name);
        }
    }

    if let Some(vector_files) = merged_vectors {
        for (ext, bytes) in [
            ("vec", &vector_files.vec),
            ("vemf", &vector_files.vemf),
            ("vem", &vector_files.vem),
            ("vex", &vector_files.vex),
        ] {
            let name = format!(
                "{}.{ext}",
                per_field_segment(
                    merged_segment_name,
                    crate::index_writer::KNN_VECTORS_FORMAT_NAME
                )
            );
            write_file(dir, &name, bytes)?;
            files.push(name);
        }
    }

    // `Lucene99SegmentInfoFormat.write` does `si.addFile(fileName)` *before*
    // writing, so a real Lucene `.si` lists itself -- and `IndexFileDeleter`
    // reference-counts from exactly this set, so a merged `.si` absent from
    // its own file list is a file nothing holds a reference to (and a
    // `CheckIndex` failure: `si.files_lists_itself`). The flush path
    // (`segment_writer`) already did this; the merge did not.
    let si_name = format!("{merged_segment_name}.si");
    files.push(si_name.clone());

    let si = SegmentInfo {
        id: merged_segment_id,
        version: lucene_version,
        min_version: Some(lucene_version),
        doc_count,
        is_compound_file: false,
        // `IndexWriter.mergeMiddle`: the merged segment holds blocks iff any
        // source did. Concatenation preserves each source's own document
        // order, so its blocks stay contiguous; a *sorted* merge would shred
        // them, which is why a block plus an index sort is refused at flush
        // (`IndexWriter::set_index_sort`, c17 finding 8) and so cannot reach
        // here.
        has_blocks: options.has_blocks,
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
        // The merged segment claims the sort exactly when this merge produced
        // it. A concatenating merge must not claim one (it would describe an
        // order the bytes do not have); the k-way merge genuinely preserves
        // the shared sort of its inputs, so it must.
        index_sort: sort_fields.map(|sort_fields| {
            sort_fields
                .iter()
                .map(|spec| IndexSortField {
                    field: spec.field.to_string(),
                    reverse: spec.reverse,
                    missing: spec.missing,
                })
                .collect()
        }),
    };
    let si_bytes = segment_info::write(&si, "");
    write_file(dir, &si_name, &si_bytes)?;

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
        // `IndexWriter.mergeMiddle` line 5086: a merged segment gets its own
        // `StringHelper.randomId()` too, same as a flushed one.
        sci_id: Some(crate::segment_writer::derive_sci_id(&merged_segment_id)),
        field_infos_files: vec![],
        dv_update_files: vec![],
        ..Default::default()
    })
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
    Binary(i32, Vec<Vec<u8>>),
    Sorted(i32, Vec<Vec<u8>>),
    SortedNumeric(i32, Vec<Vec<i64>>),
    SortedSet(i32, Vec<Vec<Vec<u8>>>),
}

impl MergedDocValuesField {
    fn field_number(&self) -> i32 {
        match self {
            MergedDocValuesField::Numeric(n, _)
            | MergedDocValuesField::SparseNumeric(n, _)
            | MergedDocValuesField::Binary(n, _)
            | MergedDocValuesField::Sorted(n, _)
            | MergedDocValuesField::SortedNumeric(n, _)
            | MergedDocValuesField::SortedSet(n, _) => *n,
        }
    }

    fn as_dense_field(&self) -> doc_values::DenseField<'_> {
        match self {
            MergedDocValuesField::Numeric(n, v) => doc_values::DenseField::Numeric(*n, v),
            MergedDocValuesField::SparseNumeric(n, v) => {
                doc_values::DenseField::SparseNumeric(*n, v)
            }
            MergedDocValuesField::Binary(n, v) => doc_values::DenseField::Binary(*n, v),
            MergedDocValuesField::Sorted(n, v) => doc_values::DenseField::Sorted(*n, v),
            MergedDocValuesField::SortedNumeric(n, v) => {
                doc_values::DenseField::SortedNumeric(*n, v)
            }
            MergedDocValuesField::SortedSet(n, v) => doc_values::DenseField::SortedSet(*n, v),
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
                if !candidates.contains(&merged_number) {
                    candidates.push(merged_number);
                }
            }
        }
    }
    candidates.sort_unstable();
    candidates
}

/// The field number `source` used for the field the merge numbers
/// `merged_field_number`, or `None` if that source never saw the field --
/// the reverse of `reconcile_field_numbers`' per-source forward map.
fn original_field_number(map: &HashMap<i32, i32>, merged_field_number: i32) -> Option<i32> {
    map.iter()
        .find(|&(_, &merged)| merged == merged_field_number)
        .map(|(&orig, _)| orig)
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
/// A live document with **no** value for the field is not an error either:
/// it is `SortField.setMissingValue`'s normal case, and the merged column is
/// then written sparsely through the same `IndexedDISI` + values body
/// `Lucene90DocValuesConsumer.writeValues` uses. What *is* still an error is
/// a live-doc-contributing source that never declared the field at all
/// ([`Error::DocValuesFieldMissingInSource`]) -- a schema mismatch, not
/// sparsity. A field for which **no** merged document has a value is still
/// written -- as `Lucene90DocValuesConsumer.writeValues`' all-missing column
/// (`docsWithFieldOffset = -2`), not omitted -- because omitting it would
/// zero the field's `DocValuesType` in the merged `.fnm` and, for an
/// index-sort tier, leave a segment whose `.si` claims an order nothing can
/// re-derive.
fn merge_numeric_doc_values(
    sources: &[MergeSource],
    per_source_maps: &[HashMap<i32, i32>],
    per_source_live_ids: &[Vec<i32>],
    doc_order: &[(usize, i32)],
) -> Result<Vec<MergedDocValuesField>> {
    let candidates = doc_values_candidates(
        sources,
        per_source_maps,
        per_source_live_ids,
        |s| s.numeric_doc_values,
        |f| f.entry.field_number,
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
        // already models and `write_dense_fields` already writes. This port
        // used to raise `DocValuesFieldMissingInSource` instead, which made
        // two ordinary cases unmergeable: a field added to the schema after
        // some segments were flushed, and a doc-values **update** against a
        // field whose base flush wrote no column -- the update's generation
        // is then the field's only column and the other sources have none at
        // all, which is what kept generational segments out of the merge
        // policy until c26.
        //
        // What makes it safe to stop treating this as a caller error is that
        // c26 added a better detector for the same class one level up:
        // [`check_format_coverage`] refuses a merge whose caller never opened
        // a format a source's own `.si` lists, and the `debug_assert!` in
        // `IndexWriter::execute_merge` pins that every entry an opened `.dvm`
        // declares reaches the merge. A silently-dropped column can no longer
        // hide behind this branch, so the branch can have Java's meaning.
        let mut per_source_entry: Vec<Option<&SourceNumericDocValues>> = vec![None; sources.len()];
        for (idx, ((source, map), live_ids)) in sources
            .iter()
            .zip(per_source_maps)
            .zip(per_source_live_ids)
            .enumerate()
        {
            if live_ids.is_empty() {
                continue;
            }
            per_source_entry[idx] =
                original_field_number(map, merged_field_number).and_then(|original_number| {
                    source
                        .numeric_doc_values
                        .iter()
                        .find(|nf| nf.entry.field_number == original_number)
                });
        }

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
        let mut dense: Vec<i64> = Vec::with_capacity(doc_order.len());
        let mut sparse: Vec<(i32, i64)> = Vec::new();
        let mut every_doc_has_one = true;
        for (merged_doc_id, &(src_idx, doc_id)) in doc_order.iter().enumerate() {
            // No reader for this source means the source has no column for
            // this field at all: every one of its documents is missing a
            // value, exactly like a document the column itself skips.
            let value = match per_source_reader[src_idx].as_mut() {
                Some(reader) => reader.value(doc_id)?,
                None => None,
            };
            match value {
                Some(value) => {
                    dense.push(value);
                    sparse.push((merged_doc_id as i32, value));
                }
                None => every_doc_has_one = false,
            }
        }
        if every_doc_has_one {
            merged.push(MergedDocValuesField::Numeric(merged_field_number, dense));
        } else {
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
        }
    }
    Ok(merged)
}

/// Merges BINARY doc-values data across `sources`, one
/// [`MergedDocValuesField`] per merged field number any source has a BINARY
/// column for, in `doc_order`.
///
/// Multi-field like [`merge_numeric_doc_values`], but **not** sparse:
/// [`lucene_codecs::doc_values::DenseField`]'s only sparse variant is the
/// numeric one, so a live document with no value is still
/// [`Error::BinaryDocValuesFieldMissingInSource`] here.
fn merge_binary_doc_values(
    sources: &[MergeSource],
    per_source_maps: &[HashMap<i32, i32>],
    per_source_live_ids: &[Vec<i32>],
    doc_order: &[(usize, i32)],
) -> Result<Vec<MergedDocValuesField>> {
    let candidates = doc_values_candidates(
        sources,
        per_source_maps,
        per_source_live_ids,
        |s| s.binary_doc_values,
        |f| f.entry.field_number,
    );
    let mut merged = Vec::with_capacity(candidates.len());
    for merged_field_number in candidates {
        let mut per_source_entry: Vec<Option<&SourceBinaryDocValues>> = vec![None; sources.len()];
        for (idx, ((source, map), live_ids)) in sources
            .iter()
            .zip(per_source_maps)
            .zip(per_source_live_ids)
            .enumerate()
        {
            if live_ids.is_empty() {
                continue;
            }
            let Some(original_number) = original_field_number(map, merged_field_number) else {
                return Err(Error::BinaryDocValuesFieldMissingInSource {
                    merged_field_number,
                });
            };
            let Some(entry) = source
                .binary_doc_values
                .iter()
                .find(|bf| bf.entry.field_number == original_number)
            else {
                return Err(Error::BinaryDocValuesFieldMissingInSource {
                    merged_field_number,
                });
            };
            per_source_entry[idx] = Some(entry);
        }

        let mut values: Vec<Vec<u8>> = Vec::with_capacity(doc_order.len());
        for &(src_idx, doc_id) in doc_order {
            let entry =
                per_source_entry[src_idx].ok_or(Error::BinaryDocValuesFieldMissingInSource {
                    merged_field_number,
                })?;
            let value = doc_values::binary_value(entry.data, &entry.entry, doc_id)?.ok_or(
                Error::BinaryDocValuesFieldMissingInSource {
                    merged_field_number,
                },
            )?;
            values.push(value.to_vec());
        }
        merged.push(MergedDocValuesField::Binary(merged_field_number, values));
    }
    Ok(merged)
}

/// Merges SORTED doc-values data across `sources`, one
/// [`MergedDocValuesField`] per merged field number any source has a SORTED
/// column for, in `doc_order`. Multi-field, dense-only, same as
/// [`merge_binary_doc_values`].
///
/// Unlike NUMERIC/BINARY, a SORTED field can't just be concatenated: each
/// source's term dictionary is built independently, so ordinal `N` in
/// source A's dictionary is generally a *different term* than ordinal `N`
/// in source B's dictionary (real Lucene's `OrdinalMap` exists to solve
/// exactly this). This port sidesteps building an explicit ordinal-
/// remapping table: for each live doc, it resolves that doc's *own source's*
/// ordinal straight to term bytes (via that source's own
/// [`terms_dict::decode_all_terms`]) and pushes the raw bytes, not an
/// ordinal, into the merged per-doc value list --
/// [`lucene_codecs::doc_values::write_dense_fields`] takes raw per-doc term
/// bytes and rebuilds the merged, deduplicated, sorted dictionary (and this
/// merge's ordinals) itself, so there's no separate remapping step to get
/// wrong: two sources' docs that happen to share a term end up pointing at
/// the exact same merged dictionary entry purely because the dictionary
/// building sorts and dedups by term *bytes*, not by ordinal.
fn merge_sorted_doc_values(
    sources: &[MergeSource],
    per_source_maps: &[HashMap<i32, i32>],
    per_source_live_ids: &[Vec<i32>],
    doc_order: &[(usize, i32)],
) -> Result<Vec<MergedDocValuesField>> {
    let candidates = doc_values_candidates(
        sources,
        per_source_maps,
        per_source_live_ids,
        |s| s.sorted_doc_values,
        |f| f.entry.field_number,
    );
    let mut merged = Vec::with_capacity(candidates.len());
    for merged_field_number in candidates {
        // This source's own dictionary, in ordinal order -- resolves this
        // source's ordinals to term bytes without needing any other source's
        // dictionary. Resolved once per source up front (see
        // `merge_numeric_doc_values`'s `per_source_entry` for why `doc_order`
        // rules out a single linear pass here).
        type SortedDvResolved<'a> = Option<(&'a SourceSortedDocValues<'a>, Vec<Vec<u8>>)>;
        let mut per_source_resolved: Vec<SortedDvResolved> = Vec::with_capacity(sources.len());
        for ((source, map), live_ids) in
            sources.iter().zip(per_source_maps).zip(per_source_live_ids)
        {
            if live_ids.is_empty() {
                per_source_resolved.push(None);
                continue;
            }
            let Some(original_number) = original_field_number(map, merged_field_number) else {
                return Err(Error::SortedDocValuesFieldMissingInSource {
                    merged_field_number,
                });
            };
            let Some(sf) = source
                .sorted_doc_values
                .iter()
                .find(|sf| sf.entry.field_number == original_number)
            else {
                return Err(Error::SortedDocValuesFieldMissingInSource {
                    merged_field_number,
                });
            };
            let source_dict = terms_dict::decode_all_terms(sf.data, &sf.entry.terms)?;
            per_source_resolved.push(Some((sf, source_dict)));
        }

        let mut values: Vec<Vec<u8>> = Vec::with_capacity(doc_order.len());
        for &(src_idx, doc_id) in doc_order {
            let (sf, source_dict) = per_source_resolved[src_idx].as_ref().ok_or(
                Error::SortedDocValuesFieldMissingInSource {
                    merged_field_number,
                },
            )?;
            let ord = doc_values::sorted_ord(sf.data, &sf.entry, doc_id)?.ok_or(
                Error::SortedDocValuesFieldMissingInSource {
                    merged_field_number,
                },
            )?;
            let term = source_dict.get(ord as usize).ok_or(
                Error::SortedDocValuesFieldMissingInSource {
                    merged_field_number,
                },
            )?;
            values.push(term.clone());
        }
        merged.push(MergedDocValuesField::Sorted(merged_field_number, values));
    }
    Ok(merged)
}

/// Merges SORTED_NUMERIC doc-values data across `sources`, one
/// [`MergedDocValuesField`] per merged field number any source has a
/// SORTED_NUMERIC column for, in `doc_order`. Multi-field, dense-only.
///
/// Unlike SORTED, SORTED_NUMERIC has no shared dictionary to reconcile: each
/// live doc simply contributes its own `Vec<i64>` of values (in whatever
/// order/count the source has), so merging is concatenation, exactly like
/// [`merge_numeric_doc_values`] generalized from one value per doc to a list
/// per doc. The writer requires every doc to have at least one value, so a
/// live doc whose resolved list comes back empty is treated the same as a
/// field missing from its source entirely.
fn merge_sorted_numeric_doc_values(
    sources: &[MergeSource],
    per_source_maps: &[HashMap<i32, i32>],
    per_source_live_ids: &[Vec<i32>],
    doc_order: &[(usize, i32)],
) -> Result<Vec<MergedDocValuesField>> {
    let candidates = doc_values_candidates(
        sources,
        per_source_maps,
        per_source_live_ids,
        |s| s.sorted_numeric_doc_values,
        |f| f.entry.field_number,
    );
    let mut merged = Vec::with_capacity(candidates.len());
    for merged_field_number in candidates {
        let mut per_source_entry: Vec<Option<&SourceSortedNumericDocValues>> =
            vec![None; sources.len()];
        for (idx, ((source, map), live_ids)) in sources
            .iter()
            .zip(per_source_maps)
            .zip(per_source_live_ids)
            .enumerate()
        {
            if live_ids.is_empty() {
                continue;
            }
            let Some(original_number) = original_field_number(map, merged_field_number) else {
                return Err(Error::SortedNumericDocValuesFieldMissingInSource {
                    merged_field_number,
                });
            };
            let Some(entry) = source
                .sorted_numeric_doc_values
                .iter()
                .find(|snf| snf.entry.field_number == original_number)
            else {
                return Err(Error::SortedNumericDocValuesFieldMissingInSource {
                    merged_field_number,
                });
            };
            per_source_entry[idx] = Some(entry);
        }

        let mut values: Vec<Vec<i64>> = Vec::with_capacity(doc_order.len());
        for &(src_idx, doc_id) in doc_order {
            let entry = per_source_entry[src_idx].ok_or(
                Error::SortedNumericDocValuesFieldMissingInSource {
                    merged_field_number,
                },
            )?;
            let doc_values = doc_values::sorted_numeric_values(entry.data, &entry.entry, doc_id)?;
            if doc_values.is_empty() {
                return Err(Error::SortedNumericDocValuesFieldMissingInSource {
                    merged_field_number,
                });
            }
            values.push(doc_values);
        }
        merged.push(MergedDocValuesField::SortedNumeric(
            merged_field_number,
            values,
        ));
    }
    Ok(merged)
}

/// Resolves one live doc's SORTED_SET ordinals for `doc_id`, regardless of
/// whether `entry.kind` collapsed to [`SortedSetKind::Single`] (one ordinal
/// or none) or stayed [`SortedSetKind::Multi`] (zero or more via the same
/// [`SortedNumericEntry`] layout [`doc_values::sorted_numeric_values`]
/// already decodes) -- mirrors the test-only `resolved_sorted_set_values`
/// helper in `lucene_codecs::doc_values`'s own test module, but per-doc
/// rather than for every doc in the field at once.
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

/// Decodes one source's whole SORTED_SET term dictionary, in ordinal order --
/// same "this source's own dictionary, used only to resolve this source's
/// own ordinals" role [`merge_sorted_doc_values`]'s `source_dict` plays,
/// generalized to either half of [`SortedSetKind`].
fn sorted_set_source_dict(data: &[u8], entry: &SortedSetEntry) -> Result<Vec<Vec<u8>>> {
    match &entry.kind {
        SortedSetKind::Single(sorted) => Ok(terms_dict::decode_all_terms(data, &sorted.terms)?),
        SortedSetKind::Multi { terms, .. } => Ok(terms_dict::decode_all_terms(data, terms)?),
    }
}

/// Merges SORTED_SET doc-values data across `sources`, one
/// [`MergedDocValuesField`] per merged field number any source has a
/// SORTED_SET column for, in `doc_order`. Multi-field, dense-only.
///
/// Exactly [`merge_sorted_doc_values`]'s "resolve to bytes, let the writer
/// dedupe" approach, applied per-*value* instead of per-doc: each live doc's
/// own source's ordinals ([`sorted_set_doc_ordinals`]) are resolved to term
/// bytes via that source's own dictionary ([`sorted_set_source_dict`]),
/// producing a `Vec<Vec<u8>>` per doc, which
/// [`lucene_codecs::doc_values::write_dense_fields`] then deduplicates (both
/// within a doc and across docs/sources) into the merged dictionary itself --
/// so, same as SORTED, there is no separate ordinal-remapping table to get
/// wrong. The writer requires every doc to have at least one value, so a live
/// doc whose resolved value set comes back empty is treated the same as a
/// field missing from its source entirely.
fn merge_sorted_set_doc_values(
    sources: &[MergeSource],
    per_source_maps: &[HashMap<i32, i32>],
    per_source_live_ids: &[Vec<i32>],
    doc_order: &[(usize, i32)],
) -> Result<Vec<MergedDocValuesField>> {
    let candidates = doc_values_candidates(
        sources,
        per_source_maps,
        per_source_live_ids,
        |s| s.sorted_set_doc_values,
        |f| f.entry.field_number,
    );
    let mut merged = Vec::with_capacity(candidates.len());
    for merged_field_number in candidates {
        // This source's own dictionary, in ordinal order -- resolves this
        // source's ordinals to term bytes without needing any other source's
        // dictionary. Resolved once per source up front, same reason as
        // `merge_sorted_doc_values`.
        type SortedSetDvResolved<'a> = Option<(&'a SourceSortedSetDocValues<'a>, Vec<Vec<u8>>)>;
        let mut per_source_resolved: Vec<SortedSetDvResolved> = Vec::with_capacity(sources.len());
        for ((source, map), live_ids) in
            sources.iter().zip(per_source_maps).zip(per_source_live_ids)
        {
            if live_ids.is_empty() {
                per_source_resolved.push(None);
                continue;
            }
            let Some(original_number) = original_field_number(map, merged_field_number) else {
                return Err(Error::SortedSetDocValuesFieldMissingInSource {
                    merged_field_number,
                });
            };
            let Some(ssf) = source
                .sorted_set_doc_values
                .iter()
                .find(|ssf| ssf.entry.field_number == original_number)
            else {
                return Err(Error::SortedSetDocValuesFieldMissingInSource {
                    merged_field_number,
                });
            };
            let source_dict = sorted_set_source_dict(ssf.data, &ssf.entry)?;
            per_source_resolved.push(Some((ssf, source_dict)));
        }

        let mut values: Vec<Vec<Vec<u8>>> = Vec::with_capacity(doc_order.len());
        for &(src_idx, doc_id) in doc_order {
            let (ssf, source_dict) = per_source_resolved[src_idx].as_ref().ok_or(
                Error::SortedSetDocValuesFieldMissingInSource {
                    merged_field_number,
                },
            )?;
            let ords = sorted_set_doc_ordinals(ssf.data, &ssf.entry, doc_id)?;
            if ords.is_empty() {
                return Err(Error::SortedSetDocValuesFieldMissingInSource {
                    merged_field_number,
                });
            }
            let mut doc_values: Vec<Vec<u8>> = Vec::with_capacity(ords.len());
            for ord in ords {
                let term = source_dict.get(ord as usize).ok_or(
                    Error::SortedSetDocValuesFieldMissingInSource {
                        merged_field_number,
                    },
                )?;
                doc_values.push(term.clone());
            }
            values.push(doc_values);
        }
        merged.push(MergedDocValuesField::SortedSet(merged_field_number, values));
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
/// Until c26 this took at most one field per merge
/// (`Error::TooManyNormsFields`), because `norms::write_single_dense_field`
/// writes a whole `.nvm`/`.nvd` pair and two of them in one merge overwrote
/// each other. `norms::write_fields` -- the norms analogue of
/// `doc_values::write_dense_fields` -- removed that limitation, and this is
/// the merge side taking it up. `Lucene90NormsConsumer` has never had the
/// limit: it gets one `addNormsField` call per field into one pair.
///
/// Norms are dense by construction here: a norm exists for every document of
/// every source that declares the field, so a source that declares it and a
/// document that has no value for it is [`Error::NormsFieldMissingInSource`]
/// rather than a sparse column. That is Java's invariant too --
/// `NormsConsumer.mergeNormsField` reads `getNormValues(field)` for every
/// live document and `FieldInfo.omitNorms` is what turns the field off, per
/// field, for the whole segment.
fn merge_norms(
    sources: &[MergeSource],
    per_source_maps: &[HashMap<i32, i32>],
    per_source_live_ids: &[Vec<i32>],
    doc_order: &[(usize, i32)],
) -> Result<Vec<(i32, Vec<i64>)>> {
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
    candidates.sort_unstable();

    let mut merged: Vec<(i32, Vec<i64>)> = Vec::with_capacity(candidates.len());
    for merged_field_number in candidates {
        let mut per_source_entry: Vec<Option<&SourceNorms>> = vec![None; sources.len()];
        for (idx, ((source, map), live_ids)) in sources
            .iter()
            .zip(per_source_maps)
            .zip(per_source_live_ids)
            .enumerate()
        {
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
            per_source_entry[idx] = Some(entry);
        }

        let mut values: Vec<i64> = Vec::with_capacity(doc_order.len());
        for &(src_idx, doc_id) in doc_order {
            let entry = per_source_entry[src_idx].ok_or(Error::NormsFieldMissingInSource {
                merged_field_number,
            })?;
            let value = norms::norm_value(entry.data, &entry.entry, doc_id)?.ok_or(
                Error::NormsFieldMissingInSource {
                    merged_field_number,
                },
            )?;
            values.push(value);
        }
        merged.push((merged_field_number, values));
    }
    Ok(merged)
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
    sources: &[MergeSource],
    per_source_maps: &[HashMap<i32, i32>],
    doc_order: &[(usize, i32)],
    merged_segment_id: &[u8; ID_LENGTH],
) -> Result<Option<TermVectorsFiles>> {
    if sources.iter().all(|s| s.term_vectors.is_none()) {
        return Ok(None);
    }

    let mut writer = term_vectors::TermVectorsWriter::new(merged_segment_id, "");
    let matching = matching_readers(sources, per_source_maps);
    let mut strategies: Vec<TermVectorsMergeStrategy> = Vec::with_capacity(sources.len());
    for (source, &m) in sources.iter().zip(&matching) {
        // Java's `reader.checkIntegrity(mergeState.oneMerge)`, run on every
        // source *before* a strategy is picked. It matters most for the bulk
        // path: that path copies a source's compressed bytes verbatim and
        // then writes a freshly computed, valid footer over them, so a bit
        // flip in the source would be laundered into a merged segment that
        // passes every checksum from then on. `term_vectors::open` only
        // validates the footer's shape (`retrieve_checksum`).
        if let Some(reader) = source.term_vectors {
            reader.check_integrity()?;
        }
        strategies.push(term_vectors_merge_strategy(&writer, source, m));
    }
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
    pub flat: &'a vectors::FlatVectorsReader<'a>,
    pub graph: Option<&'a hnsw_vectors::HnswVectorsReader<'a>>,
}

/// Per-merge settings that are not derivable from the sources.
///
/// Only the HNSW graph parameters live here so far, and they matter only for
/// the from-scratch rebuild inside
/// [`lucene_codecs::hnsw_vectors::merge_one_field`]: when a source graph is
/// reused as the base, the merged graph inherits **its** `maxConn`, exactly
/// as `IncrementalHnswGraphMerger` does.
#[derive(Debug, Clone, Copy)]
pub struct MergeOptions {
    pub hnsw_m: i32,
    pub hnsw_beam_width: i32,
    /// `IndexWriter.mergeMiddle`'s `hasBlocks`: true when **any** source
    /// segment's `.si` says it holds document blocks, which is the flag the
    /// merged `.si` must carry.
    ///
    /// It is an option rather than a per-[`MergeSource`] field because it is
    /// read off each source's `SegmentInfo`, which this module deliberately
    /// never opens (a `MergeSource` is a caller-supplied set of already-opened
    /// readers). Java reads it the same way -- from `SegmentCommitInfo`, in
    /// the writer, not in `SegmentMerger`.
    ///
    /// Defaulting to `false` is what every caller before this had, and it is
    /// the *unsafe* default: a merged segment reporting `hasBlocks = false`
    /// while holding blocks reads back perfectly and silently invalidates
    /// every parent/child join query against it. `IndexWriter::execute_merge`
    /// sets it.
    pub has_blocks: bool,
}

impl Default for MergeOptions {
    fn default() -> Self {
        MergeOptions {
            hnsw_m: hnsw::DEFAULT_MAX_CONN,
            hnsw_beam_width: hnsw::DEFAULT_BEAM_WIDTH,
            has_blocks: false,
        }
    }
}

/// The four files a vector merge writes, plus the merged field numbers that
/// actually ended up with vectors (which is what the merged `.fnm` may claim
/// -- see [`describe_written_files`]).
struct MergedVectorFiles {
    vec: Vec<u8>,
    vemf: Vec<u8>,
    vem: Vec<u8>,
    vex: Vec<u8>,
    field_numbers: Vec<i32>,
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
fn merge_vectors(
    sources: &[MergeSource],
    per_source_maps: &[HashMap<i32, i32>],
    doc_id_maps: &[Vec<i32>],
    merged_fields: &[FieldInfo],
    merged_max_doc: i32,
    options: &MergeOptions,
    merged_segment_id: &[u8; ID_LENGTH],
) -> Result<Option<MergedVectorFiles>> {
    if sources.iter().all(|s| s.vectors.is_none()) {
        return Ok(None);
    }
    let suffix = per_field_codec_suffix(crate::index_writer::KNN_VECTORS_FORMAT_NAME);
    let reverse_maps = invert_field_number_maps(per_source_maps);

    /// One source's contribution to one field, resolved once and reused by
    /// both halves so the flat merge and the graph merge cannot disagree
    /// about which sources took part or in what order. `ord_to_doc` is that
    /// source's own ordinal -> its own doc id, materialised here because both
    /// halves need it (the flat merge to place each vector, the graph merge
    /// to build `IncrementalHnswGraphMerger`'s ordinal maps).
    struct FieldSource {
        source_index: usize,
        original_field_number: i32,
        ord_to_doc: Vec<i32>,
    }

    let mut plan: Vec<(&FieldInfo, Vec<FieldSource>)> = Vec::new();
    for merged_field in merged_fields {
        if merged_field.vector_dimension <= 0 {
            continue;
        }
        let mut field_sources = Vec::new();
        let mut surviving = 0usize;
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
                source_vectors.flat,
                original_number,
                merged_field.vector_encoding,
            )?;
            let doc_map = &doc_id_maps[src_idx];
            // ARITH: a count of elements of an in-memory slice added to a
            // running total of the same, so the sum is bounded by the number
            // of `i32`s this process holds and cannot reach `usize::MAX`.
            #[allow(clippy::arithmetic_side_effects)]
            {
                surviving += ord_to_doc
                    .iter()
                    .filter(|&&doc| mapped_doc_id(doc_map, doc).is_some())
                    .count();
            }
            field_sources.push(FieldSource {
                source_index: src_idx,
                original_field_number: original_number,
                ord_to_doc,
            });
        }
        // A field every one of whose vectors was deleted is omitted rather
        // than written as an empty one, exactly as `build_vectors_output`
        // omits it at flush time -- and the merged `.fnm` then zeroes its
        // `vector_dimension` instead of claiming vectors that are not there.
        if surviving > 0 {
            plan.push((merged_field, field_sources));
        }
    }
    if plan.is_empty() {
        return Ok(None);
    }

    let mut flat_writer =
        vectors::FlatVectorsWriter::new(merged_max_doc, merged_segment_id, &suffix);
    let mut written: Vec<&FieldInfo> = Vec::new();
    for (merged_field, field_sources) in &plan {
        let encoding = merged_field.vector_encoding;
        let mut flat_sources: Vec<vectors::FlatVectorMergeSource<'_>> =
            Vec::with_capacity(field_sources.len());
        for fs in field_sources {
            let source_vectors = sources[fs.source_index]
                .vectors
                .expect("only sources with a flat reader are planned");
            let values = match encoding {
                VectorEncoding::Float32 => vectors::MergeSourceValues::Float32(
                    source_vectors
                        .flat
                        .float_vector_values(fs.original_field_number)?,
                ),
                VectorEncoding::Byte => vectors::MergeSourceValues::Byte(
                    source_vectors
                        .flat
                        .byte_vector_values(fs.original_field_number)?,
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
        written.push(merged_field);
    }
    let (vec_bytes, vemf_bytes) = flat_writer.finish();

    // Reopen exactly the bytes just written and build every graph over
    // *those*, so the graph's ordinals and the flat store's ordinals are the
    // same fact rather than two derivations of it.
    let flat =
        vectors::FlatVectorsReader::open(&vemf_bytes, &vec_bytes, merged_segment_id, &suffix)?;
    let mut hnsw_fields: Vec<hnsw_vectors::HnswVectorsField<'_>> = Vec::new();
    let mut graphs: Vec<Option<hnsw::OnHeapHnswGraph>> = Vec::with_capacity(written.len());
    let mut counts: Vec<i32> = Vec::with_capacity(written.len());
    for (merged_field, field_sources) in &plan {
        // Every source's own graph for this field, in the same order the flat
        // merge consumed them. A source with the flat pair and no graph
        // (`numLevels = 0`, i.e. below `HNSW_GRAPH_THRESHOLD` when it was
        // written) still contributes vectors -- they are simply inserted.
        let mut source_graphs: Vec<Option<hnsw_vectors::OffHeapHnswGraph<'_>>> =
            Vec::with_capacity(field_sources.len());
        for fs in field_sources {
            let source_vectors = sources[fs.source_index]
                .vectors
                .expect("only sources with a flat reader are planned");
            let graph = match source_vectors.graph {
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

        let (count, merged_ord_to_doc, graph) = match merged_field.vector_encoding {
            VectorEncoding::Float32 => {
                let values = flat.float_vector_values(merged_field.number)?;
                let merged_ord_to_doc: Vec<i32> = (0..values.size())
                    .map(|ord| values.ord_to_doc(ord))
                    .collect::<std::result::Result<Vec<i32>, _>>()?;
                let graph = hnsw_vectors::merge_one_field(
                    values.ord_scorer(),
                    options.hnsw_m,
                    options.hnsw_beam_width,
                    hnsw::DEFAULT_RAND_SEED,
                    &merged_ord_to_doc,
                    &graph_sources,
                )?;
                (values.size(), merged_ord_to_doc, graph)
            }
            VectorEncoding::Byte => {
                let values = flat.byte_vector_values(merged_field.number)?;
                let merged_ord_to_doc: Vec<i32> = (0..values.size())
                    .map(|ord| values.ord_to_doc(ord))
                    .collect::<std::result::Result<Vec<i32>, _>>()?;
                let graph = hnsw_vectors::merge_one_field(
                    values.ord_scorer(),
                    options.hnsw_m,
                    options.hnsw_beam_width,
                    hnsw::DEFAULT_RAND_SEED,
                    &merged_ord_to_doc,
                    &graph_sources,
                )?;
                (values.size(), merged_ord_to_doc, graph)
            }
        };
        debug_assert_eq!(merged_ord_to_doc.len() as i32, count);
        counts.push(count);
        graphs.push(graph);
    }

    for ((merged_field, _), (graph, count)) in plan.iter().zip(graphs.iter().zip(&counts)) {
        hnsw_fields.push(hnsw_vectors::HnswVectorsField {
            field_number: merged_field.number,
            encoding: merged_field.vector_encoding,
            similarity: merged_field.vector_similarity_function,
            dimension: merged_field.vector_dimension,
            count: *count,
            graph: graph.as_ref(),
            m: options.hnsw_m,
        });
    }
    let (vex_bytes, vem_bytes) =
        hnsw_vectors::write_hnsw_vectors(&hnsw_fields, merged_segment_id, &suffix)?;

    Ok(Some(MergedVectorFiles {
        vec: vec_bytes,
        vemf: vemf_bytes,
        vem: vem_bytes,
        vex: vex_bytes,
        field_numbers: written.iter().map(|f| f.number).collect(),
    }))
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

/// Reorders `values` by `permutation` (`permutation[n]` is the index of the
/// element that becomes position `n`), moving each element rather than
/// cloning it -- the per-document position/offset/payload lists a positional
/// postings merge builds are one `Vec` each, and cloning them to reorder a
/// term would double the merge's peak footprint for nothing.
fn take_permuted<T: Default>(values: &mut [T], permutation: &[usize]) -> Vec<T> {
    debug_assert_eq!(values.len(), permutation.len());
    permutation
        .iter()
        .map(|&i| std::mem::take(&mut values[i]))
        .collect()
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
/// [`merge_postings`] treat one source's contribution to a term as already
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
        // `max_doc` is a **file-derived value sizing an allocation**, and it
        // was not as validated as this comment used to claim:
        // `stored_fields::open` checks only that its `.fdm` `maxDoc` is
        // non-negative, so `i32::MAX` here reserves 8.6 GB -- a SIGABRT,
        // reproduced under `ulimit -v` by
        // `index_writer`'s `a_segment_whose_fdm_disagrees_with_its_si_about_max_doc_is_not_merged`.
        // The bound is now at the caller, where the `.si`'s document count
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

/// Inverts the `original field number -> merged field number` maps
/// [`reconcile_field_numbers`] returns, once per merge, so
/// [`merge_postings`]/[`merge_points`] can answer "what did *this* source
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

/// One merged field's postings, ready to hand to
/// [`lucene_codecs::postings_writer::write_fields`] (via a borrowed
/// [`FieldPostingsInput`] built from `terms`).
struct MergedPostingsField {
    field_number: i32,
    index_options: IndexOptions,
    doc_count: i32,
    has_payloads: bool,
    terms: Vec<TermPostings>,
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
/// models, and is not an error here. What *is* an error, matching the same
/// philosophy as doc-values/norms: if a merged field has postings data in
/// at least one source that contributes live docs, but another live-doc-
/// contributing source has no postings *field* at all for it (schema
/// mismatch across sources), this returns
/// [`Error::PostingsFieldMissingInSource`] rather than silently treating
/// that source's docs as having no terms for the field.
///
/// # Positions/offsets/payloads
///
/// See [`SourcePostings`]'s doc comment: a candidate field whose merged
/// `index_options` indexes positions has every contributing source's
/// per-term positions (and offsets/payloads, when applicable) read back and
/// concatenated in the same source order as docs/freqs.
fn merge_postings(
    sources: &[MergeSource],
    per_source_maps: &[HashMap<i32, i32>],
    per_source_live_ids: &[Vec<i32>],
    doc_order: &[(usize, i32)],
    doc_id_maps: &[Vec<i32>],
    merged_fields: &[FieldInfo],
) -> Result<Vec<MergedPostingsField>> {
    let mut candidates: Vec<i32> = Vec::new();
    for ((source, map), live_ids) in sources.iter().zip(per_source_maps).zip(per_source_live_ids) {
        if live_ids.is_empty() {
            // Same "fully-deleted source can't affect the merged output"
            // exemption as merge_numeric_doc_values.
            continue;
        }
        for pf in source.postings {
            if let Some(&merged_number) = map.get(&pf.field_number) {
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
            let Some(pf) = source
                .postings
                .iter()
                .find(|pf| pf.field_number == original_number)
            else {
                return Err(Error::PostingsFieldMissingInSource {
                    merged_field_number,
                });
            };
            if has_positions && pf.pos_in.is_none() {
                return Err(Error::PostingsPositionsInputMissingInSource {
                    merged_field_number,
                    source_index: per_source_field.len(),
                });
            }
            per_source_field.push(Some(pf));
        }

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
        for pf in &per_source_field {
            cursors.push(match pf {
                Some(pf) => TermCursor::open(pf)?,
                None => None,
            });
        }

        let mut terms_out: Vec<TermPostings> = Vec::new();
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
            let term = smallest.to_vec();

            let mut docs: Vec<(i32, i32)> = Vec::new();
            let mut positions: Vec<Vec<i32>> = Vec::new();
            let mut offsets: Vec<Vec<(i32, i32)>> = Vec::new();
            let mut payloads: Vec<Vec<Vec<u8>>> = Vec::new();
            for src_idx in 0..cursors.len() {
                let Some(cursor) = cursors[src_idx].as_mut() else {
                    continue;
                };
                if cursor.term != term {
                    continue;
                }
                let pf = per_source_field[src_idx]
                    .expect("a cursor only exists for a source that contributes this field");
                let (source_postings, source_positions) = if has_positions {
                    let (docs, positions) = cursor
                        .terms
                        .try_current_postings_and_positions(
                            pf.doc_in,
                            pf.pos_in.expect(
                                "checked as Error::PostingsPositionsInputMissingInSource above",
                            ),
                            pf.pay_in,
                        )?
                        .expect("the cursor is standing on this term");
                    (docs, Some(positions))
                } else {
                    (
                        cursor
                            .terms
                            .try_current_postings(pf.doc_in)?
                            .expect("the cursor is standing on this term"),
                        None,
                    )
                };

                let doc_id_map = &doc_id_maps[src_idx];
                for (doc_idx, (&doc_id, &freq)) in source_postings
                    .docs
                    .iter()
                    .zip(source_postings.freqs.iter())
                    .enumerate()
                {
                    if let Some(merged_doc_id) = mapped_doc_id(doc_id_map, doc_id) {
                        docs.push((merged_doc_id, freq));
                        // Java's `docsSeen` bitset, not a `HashSet<i32>`: the
                        // merged doc id space is dense and known up front.
                        docs_seen.set(merged_doc_id as usize);
                        if let Some(source_positions) = &source_positions {
                            let doc_positions = &source_positions[doc_idx];
                            positions.push(doc_positions.iter().map(|p| p.position).collect());
                            if has_offsets {
                                offsets.push(
                                    doc_positions
                                        .iter()
                                        .map(|p| (p.start_offset, p.end_offset))
                                        .collect(),
                                );
                            }
                            if has_payloads {
                                payloads.push(
                                    doc_positions.iter().map(|p| p.payload.clone()).collect(),
                                );
                            }
                        }
                    }
                }
                cursor.advance()?;
                if cursor.exhausted() {
                    cursors[src_idx] = None;
                }
            }
            if !docs.is_empty() {
                // `DocIDMerger.of(subs, mergeState.needsIndexSort)`: each
                // source contributed its own postings for this term already
                // ascending in the merged space (`build_doc_id_maps` is
                // monotone within a source), but a sorted merge interleaves
                // the sources, so the concatenation above is only globally
                // ascending when the merged ranges are source-disjoint --
                // i.e. for the plain merge. Where it is not, the four
                // parallel lists are re-ordered together; the check costs one
                // linear scan and moves nothing in the plain case.
                //
                // This is not cosmetic. `postings_writer` encodes doc ids as
                // deltas, so a descending step writes a negative delta that
                // decodes to a doc id no reader can resolve -- and a *pair* of
                // interleaved sources whose deltas happen to stay positive
                // would encode postings against the wrong documents.
                if !docs.windows(2).all(|w| w[0].0 < w[1].0) {
                    let mut permutation: Vec<usize> = (0..docs.len()).collect();
                    permutation.sort_unstable_by_key(|&i| docs[i].0);
                    docs = permutation.iter().map(|&i| docs[i]).collect();
                    if !positions.is_empty() {
                        positions = take_permuted(&mut positions, &permutation);
                    }
                    if !offsets.is_empty() {
                        offsets = take_permuted(&mut offsets, &permutation);
                    }
                    if !payloads.is_empty() {
                        payloads = take_permuted(&mut payloads, &permutation);
                    }
                }
                terms_out.push(TermPostings {
                    term,
                    docs,
                    positions,
                    offsets,
                    payloads,
                });
            }
        }

        let doc_count = docs_seen.cardinality() as i32;

        result.push(MergedPostingsField {
            field_number: merged_field_number,
            index_options,
            doc_count,
            has_payloads,
            terms: terms_out,
        });
    }

    Ok(result)
}

/// Port of `BKDWriter.merge`'s priority-queue loop: combines each source's
/// already-sorted point stream into one globally sorted stream, so
/// [`points::write`] never has to sort at all (see its `presorted_leaf_plan`).
///
/// Java only does this for `numDims == 1`, because that is the only case where
/// a segment's points come off disk in a single, well-defined sort order --
/// with more index dimensions the BKD tree's leaf order is not a total order
/// on values, so there is nothing to merge and `mergeOneField` re-indexes
/// instead. Same rule here, keyed on `num_index_dims` (the trailing data-only
/// dimensions never participate in a split, so they cannot make the order
/// ambiguous).
///
/// Java's `mergeComparator` orders by the packed value's bytes and then by
/// document id, "sorting smaller docIDs earlier"; this reproduces both. The
/// source index acts as an implicit final tiebreak -- the scan below only lets
/// a *strictly* smaller head displace the incumbent, so on a full tie the
/// lower-numbered source wins -- which makes the order total without a third
/// comparison key. (That tie is unreachable anyway: merged doc ids are
/// globally unique, whichever order the merge produced them in.)
///
/// Falls back to plain concatenation whenever the one-pass conditions do not
/// hold -- more than one index dimension, or a source whose stream is not
/// actually sorted (a hand-built `MergeSource`, or a segment some other writer
/// produced). `points::write` sorts in that case, exactly as before, so this
/// is a cost choice, never a correctness one.
fn merge_point_streams(
    mut per_source: Vec<Vec<(i32, Vec<u8>)>>,
    num_index_dims: usize,
    bytes_per_dim: usize,
) -> Vec<(i32, Vec<u8>)> {
    let total: usize = per_source.iter().map(|s| s.len()).sum();
    // `get(..)`, not `[..]`: `bytes_per_dim` comes off the `.kdm` and the
    // packed value off the `.kdd`, so nothing on the wire *relates* them
    // beyond `merge_points`' shape check. A short value is treated as "not
    // sorted", which falls through to plain concatenation -- `points::write`
    // sorts in that case, so this is a cost choice, never a correctness one,
    // and never a panic in the middle of a merge.
    fn key(p: &(i32, Vec<u8>), bytes_per_dim: usize) -> Option<&[u8]> {
        p.1.get(..bytes_per_dim)
    }
    let mergeable = num_index_dims == 1
        && per_source.iter().all(|s| {
            s.windows(2).all(
                |w| match (key(&w[0], bytes_per_dim), key(&w[1], bytes_per_dim)) {
                    (Some(a), Some(b)) => a <= b,
                    _ => false,
                },
            )
        });
    if !mergeable {
        let mut out = Vec::with_capacity(total);
        for stream in per_source {
            out.extend(stream);
        }
        return out;
    }

    // A linear scan over the (few) source heads per step rather than a
    // priority queue: this port merges `max_merge_at_once` segments at a time,
    // which defaults to 10 -- the same reasoning `sorted_doc_order`
    // records for its own k-way merge.
    let mut cursors = vec![0usize; per_source.len()];
    let mut out = Vec::with_capacity(total);
    loop {
        let mut best: Option<usize> = None;
        for (i, stream) in per_source.iter().enumerate() {
            let Some(head) = stream.get(cursors[i]) else {
                continue;
            };
            best = Some(match best {
                None => i,
                Some(b) => {
                    let current = &per_source[b][cursors[b]];
                    // `Option`'s own ordering carries a short value (`None`)
                    // to the front rather than panicking. `mergeable` above
                    // has already established every *adjacent pair* is
                    // ordered, so the only `None` that can reach here is a
                    // single-element stream's, where the ordering it picks
                    // cannot change the output's validity.
                    if (key(head, bytes_per_dim), head.0) < (key(current, bytes_per_dim), current.0)
                    {
                        i
                    } else {
                        b
                    }
                }
            });
        }
        let Some(i) = best else { break };
        let point = std::mem::take(&mut per_source[i][cursors[i]]);
        // ARITH: `i` was only chosen for a stream whose cursor still resolved
        // through `stream.get(cursors[i])`, so this increment lands at most at
        // that `Vec`'s length.
        #[allow(clippy::arithmetic_side_effects)]
        {
            cursors[i] += 1;
        }
        out.push(point);
    }
    out
}

/// One merged field's BKD points, ready to hand to
/// [`lucene_codecs::points::write`] (via a [`WritePointsField`] built from
/// `points`).
struct MergedPointsField {
    field_number: i32,
    num_dims: i32,
    num_index_dims: i32,
    bytes_per_dim: i32,
    points: Vec<(i32, Vec<u8>)>,
}

/// Merges BKD points (`.kdm`/`.kdi`/`.kdd`) data across `sources` for every
/// field any source declares points for, returning one [`MergedPointsField`]
/// per distinct merged field number that has points data in at least one
/// source -- or an empty `Vec` if no source supplied any points data at all.
///
/// Unlike SORTED doc values or postings, a points field has no shared
/// dictionary to resolve ordinals against -- it's fundamentally a per-doc set
/// of fixed-width packed values (like NUMERIC/SORTED_NUMERIC doc values, but
/// with a merged tree rebuilt from scratch rather than a single scalar per
/// doc). So this simply reads back every live doc's points via each source's
/// own already-opened [`lucene_codecs::points::PointsReader`]
/// ([`SourcePoints::reader`], the same reader `lucene_search`'s points range
/// query uses), drops non-live docs and remaps surviving doc ids to the
/// merged id space via [`build_doc_id_maps`] (same mechanism
/// [`merge_postings`] uses), and concatenates the results across sources in
/// source order. [`lucene_codecs::points::write`] rebuilds the merged BKD
/// tree (leaf plan, packed index, bounding boxes) from this flat list
/// itself, so there is no tree-merging logic to get wrong here, and -- like
/// postings, unlike doc-values/norms -- `write` already supports any number
/// of fields per call, so there is no single-field-per-merge-call limit.
///
/// # The "sparse across sources" rule, points edition
///
/// A field has no per-doc sparsity of its own to model here (a live doc
/// either contributes exactly one packed value for the field, from
/// [`lucene_codecs::points::PointsReader::decode_all_points`], or none) --
/// this merge does not require every live doc to have a point (multi-valued
/// points and docs with zero points for a field are both realistic and
/// simply mean fewer points end up in the merged tree for that doc). What
/// *is* an error, matching the same philosophy as doc-values/norms/postings:
/// if a merged field has points data in at least one source that contributes
/// live docs, but another live-doc-contributing source has no points *field*
/// at all for it (schema mismatch across sources), this returns
/// [`Error::PointsFieldMissingInSource`].
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
fn merge_points(
    sources: &[MergeSource],
    per_source_maps: &[HashMap<i32, i32>],
    per_source_live_ids: &[Vec<i32>],
    doc_id_maps: &[Vec<i32>],
    merged_fields: &[FieldInfo],
) -> Result<Vec<MergedPointsField>> {
    let mut candidates: Vec<i32> = Vec::new();
    for ((source, map), live_ids) in sources.iter().zip(per_source_maps).zip(per_source_live_ids) {
        if live_ids.is_empty() {
            // Same "fully-deleted source can't affect the merged output"
            // exemption as merge_numeric_doc_values.
            continue;
        }
        for sp in source.points {
            if let Some(&merged_number) = map.get(&sp.field_number) {
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
    let mut result = Vec::with_capacity(candidates.len());

    for merged_field_number in candidates {
        let merged_field = merged_fields
            .iter()
            .find(|f| f.number == merged_field_number)
            .expect("merged_field_number came from reconcile_field_numbers over these same sources, so it must have an entry in merged_fields");
        let merged_num_dims = merged_field.point_dimension_count;
        let merged_num_index_dims = merged_field.point_index_dimension_count;
        let merged_bytes_per_dim = merged_field.point_num_bytes;

        // One stream per contributing source, kept separate so a
        // single-index-dimension field can be k-way merged below instead of
        // concatenated and re-sorted -- `BKDWriter.merge` versus
        // `PointsWriter.mergeOneField`.
        let mut per_source_points: Vec<Vec<(i32, Vec<u8>)>> = Vec::new();
        for (src_idx, ((source, reverse), live_ids)) in sources
            .iter()
            .zip(&reverse_maps)
            .zip(per_source_live_ids)
            .enumerate()
        {
            if live_ids.is_empty() {
                continue;
            }
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
                return Err(Error::PointsFieldMissingInSource {
                    merged_field_number,
                });
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

            let doc_id_map = &doc_id_maps[src_idx];
            let mut stream: Vec<(i32, Vec<u8>)> = Vec::new();
            for point in sp.reader.decode_all_points(original_number)? {
                if let Some(merged_doc_id) = mapped_doc_id(doc_id_map, point.doc_id) {
                    stream.push((merged_doc_id, point.packed_value));
                }
            }
            per_source_points.push(stream);
        }

        let points = merge_point_streams(
            per_source_points,
            merged_num_index_dims as usize,
            merged_bytes_per_dim as usize,
        );

        result.push(MergedPointsField {
            field_number: merged_field_number,
            num_dims: merged_num_dims,
            num_index_dims: merged_num_index_dims,
            bytes_per_dim: merged_bytes_per_dim,
            points,
        });
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    // The arithmetic gate is about values read off disk; a fixture builder's
    // own index arithmetic is not one (see `docs/arithmetic-gate.md`).
    #![allow(clippy::arithmetic_side_effects)]

    use super::*;
    use crate::segment_writer;
    use lucene_codecs::field_infos::{
        DocValuesSkipIndexType, DocValuesType, IndexOptions, VectorEncoding,
        VectorSimilarityFunction,
    };
    use lucene_codecs::stored_fields::{self, Document, FieldValue, StoredField};
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
        let (merged, maps) = reconcile_field_numbers(&sources).unwrap();
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
        let (merged, maps) = reconcile_field_numbers(&sources).unwrap();

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
        let (merged, maps) = reconcile_field_numbers(&sources).unwrap();

        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].name, "id");
        assert_eq!(merged[1].name, "extra");
        assert_eq!(maps[0].len(), 1);
        assert_eq!(maps[1][&1], 1);
    }

    #[test]
    fn empty_sources_produce_empty_merged_fields() {
        let sources: Vec<&[FieldInfo]> = vec![];
        let (merged, maps) = reconcile_field_numbers(&sources).unwrap();
        assert!(merged.is_empty());
        assert!(maps.is_empty());
    }

    #[test]
    fn doc_id_maps_are_dense_arrays_with_minus_one_for_deleted_docs() {
        // Source 0 keeps docs 0 and 2 (1 deleted), source 1 keeps doc 1
        // only. Merged ids are assigned by concatenation: 0,1 for source 0
        // then 2 for source 1 -- Lucene's MergeState.DocMap contract.
        let live = vec![vec![0, 2], vec![1]];
        let maps = build_doc_id_maps(&[3, 2], &concat_doc_order(&live));
        assert_eq!(maps[0], vec![0, -1, 1]);
        assert_eq!(maps[1], vec![-1, 2]);

        assert_eq!(mapped_doc_id(&maps[0], 0), Some(0));
        assert_eq!(mapped_doc_id(&maps[0], 1), None, "deleted doc");
        assert_eq!(mapped_doc_id(&maps[0], 2), Some(1));
        // Past the source's own maxDoc.
        assert_eq!(mapped_doc_id(&maps[0], 3), None);
        assert_eq!(mapped_doc_id(&maps[0], -1), None);

        // A fully-deleted source gets an empty map, and every lookup misses.
        let maps = build_doc_id_maps(&[0], &[]);
        assert!(maps[0].is_empty());
        assert_eq!(mapped_doc_id(&maps[0], 0), None);
    }

    #[test]
    fn concat_doc_order_walks_sources_in_order() {
        let live = vec![vec![0, 2], vec![], vec![1]];
        assert_eq!(
            concat_doc_order(&live),
            vec![(0usize, 0i32), (0, 2), (2, 1)]
        );
    }

    #[test]
    fn a_source_naming_the_same_field_twice_is_rejected() {
        // Java's FieldInfos(FieldInfo[]) constructor: "duplicate field names".
        let fields = vec![field("body", 0), field("body", 1)];
        let sources: Vec<&[FieldInfo]> = vec![&fields];
        assert!(matches!(
            reconcile_field_numbers(&sources),
            Err(Error::DuplicateFieldNameInSource {
                source_index: 0,
                ..
            })
        ));

        // The same name in *different* sources is the normal case.
        let fields0 = vec![field("body", 0)];
        let fields1 = vec![field("body", 7)];
        let sources: Vec<&[FieldInfo]> = vec![&fields0, &fields1];
        let (merged, maps) = reconcile_field_numbers(&sources).unwrap();
        assert_eq!(merged.len(), 1);
        assert_eq!(maps[1][&7], 0);
    }

    #[test]
    fn store_payloads_is_ored_across_sources_not_taken_from_the_first() {
        // FieldInfos.Builder.add: `if (fi.hasPayloads()) curFi.setStorePayloads()`.
        let mut fields0 = vec![field("body", 0)];
        fields0[0].index_options = IndexOptions::DocsAndFreqsAndPositions;
        let mut fields1 = vec![field("body", 0)];
        fields1[0].index_options = IndexOptions::DocsAndFreqsAndPositions;
        fields1[0].store_payloads = true;
        let sources: Vec<&[FieldInfo]> = vec![&fields0, &fields1];
        let (merged, _) = reconcile_field_numbers(&sources).unwrap();
        assert!(merged[0].store_payloads);

        // ...and the OR is order-independent.
        let sources: Vec<&[FieldInfo]> = vec![&fields1, &fields0];
        let (merged, _) = reconcile_field_numbers(&sources).unwrap();
        assert!(merged[0].store_payloads);
    }

    #[test]
    fn store_payloads_or_respects_set_store_payloads_positions_guard() {
        // `FieldInfo.setStorePayloads` only sets the flag when
        // `indexOptions.subsumes(DOCS_AND_FREQS_AND_POSITIONS)`; otherwise
        // `checkConsistency` would throw "indexed field cannot have payloads
        // without positions". A source claiming payloads on a positionless
        // field must therefore NOT push the merged field into that state.
        for options in [
            IndexOptions::Docs,
            IndexOptions::DocsAndFreqs,
            // Java special-cases DocsAndCustomFreqs to subsume as if it were
            // DocsAndFreqs, so it does not subsume positions either.
            IndexOptions::DocsAndCustomFreqs,
        ] {
            let mut fields0 = vec![field("body", 0)];
            fields0[0].index_options = options;
            let mut fields1 = fields0.clone();
            fields1[0].store_payloads = true;
            let sources: Vec<&[FieldInfo]> = vec![&fields0, &fields1];
            let (merged, _) = reconcile_field_numbers(&sources).unwrap();
            assert!(
                !merged[0].store_payloads,
                "{options:?} does not subsume positions, so the OR must not apply"
            );
        }
    }

    #[test]
    fn attributes_are_put_all_ed_across_sources_not_taken_from_the_first() {
        // FieldInfos.Builder.add: `curFi.putAttributes(fi.attributes())` is a
        // Map.putAll -- a later source's value wins for a shared key, and its
        // own keys are added.
        let mut fields0 = vec![field("body", 0)];
        fields0[0].attributes = vec![
            ("shared".to_string(), "from-0".to_string()),
            ("only-0".to_string(), "a".to_string()),
        ];
        let mut fields1 = vec![field("body", 0)];
        fields1[0].attributes = vec![
            ("shared".to_string(), "from-1".to_string()),
            ("only-1".to_string(), "b".to_string()),
        ];
        let sources: Vec<&[FieldInfo]> = vec![&fields0, &fields1];
        let (merged, _) = reconcile_field_numbers(&sources).unwrap();
        let mut attrs = merged[0].attributes.clone();
        attrs.sort();
        assert_eq!(
            attrs,
            vec![
                ("only-0".to_string(), "a".to_string()),
                ("only-1".to_string(), "b".to_string()),
                ("shared".to_string(), "from-1".to_string()),
            ]
        );
    }

    #[test]
    fn disagreeing_index_options_across_sources_is_rejected_without_any_postings_data() {
        // FieldInfo.verifySameIndexOptions -- checked for every field of
        // every merged segment, not just fields whose postings are supplied.
        let mut fields0 = vec![field("body", 0)];
        fields0[0].index_options = IndexOptions::DocsAndFreqs;
        let mut fields1 = vec![field("body", 0)];
        fields1[0].index_options = IndexOptions::DocsAndFreqsAndPositions;
        let sources: Vec<&[FieldInfo]> = vec![&fields0, &fields1];
        assert!(matches!(
            reconcile_field_numbers(&sources),
            Err(Error::PostingsIndexOptionsDisagreement { .. })
        ));
    }

    #[test]
    fn disagreeing_store_term_vectors_across_sources_is_rejected() {
        // FieldInfo.verifySameStoreTermVectors. Without this the merged
        // .fnm would claim no term vectors while write_merged_term_vectors happily
        // wrote source 1's.
        let mut fields0 = vec![field("body", 0)];
        fields0[0].index_options = IndexOptions::DocsAndFreqs;
        let mut fields1 = fields0.clone();
        fields1[0].store_term_vectors = true;
        let sources: Vec<&[FieldInfo]> = vec![&fields0, &fields1];
        assert!(matches!(
            reconcile_field_numbers(&sources),
            Err(Error::FieldSchemaDisagreement {
                attribute: "store_term_vectors",
                ..
            })
        ));
    }

    #[test]
    fn disagreeing_omit_norms_across_sources_is_rejected_only_when_indexed() {
        // FieldInfo.verifySameOmitNorms, guarded by Java's
        // `if (this.indexOptions != IndexOptions.NONE)`.
        let mut fields0 = vec![field("body", 0)];
        fields0[0].index_options = IndexOptions::DocsAndFreqs;
        let mut fields1 = fields0.clone();
        fields1[0].omit_norms = true;
        let sources: Vec<&[FieldInfo]> = vec![&fields0, &fields1];
        assert!(matches!(
            reconcile_field_numbers(&sources),
            Err(Error::FieldSchemaDisagreement {
                attribute: "omit_norms",
                ..
            })
        ));

        // A non-indexed field's omitNorms/storeTermVector are not compared
        // (they are meaningless there), exactly as in Java.
        let fields0 = vec![field("body", 0)];
        let mut fields1 = fields0.clone();
        fields1[0].omit_norms = true;
        fields1[0].store_term_vectors = true;
        let sources: Vec<&[FieldInfo]> = vec![&fields0, &fields1];
        assert!(reconcile_field_numbers(&sources).is_ok());
    }

    #[test]
    fn disagreeing_doc_values_type_across_sources_is_rejected() {
        // FieldInfo.verifySameDocValuesType.
        let fields0 = vec![field("dv", 0)];
        let mut fields1 = fields0.clone();
        fields1[0].doc_values_type = DocValuesType::Numeric;
        let sources: Vec<&[FieldInfo]> = vec![&fields0, &fields1];
        assert!(matches!(
            reconcile_field_numbers(&sources),
            Err(Error::FieldSchemaDisagreement {
                attribute: "doc_values_type",
                ..
            })
        ));
    }

    #[test]
    fn disagreeing_points_shape_in_field_infos_is_rejected_without_any_points_data() {
        // FieldInfo.verifySamePointsOptions -- the FieldInfos-level half of
        // the check merge_points also performs against each source's actual
        // on-disk BKD tree.
        let mut fields0 = vec![field("pt", 0)];
        fields0[0].point_dimension_count = 1;
        fields0[0].point_index_dimension_count = 1;
        fields0[0].point_num_bytes = 4;
        let mut fields1 = fields0.clone();
        fields1[0].point_dimension_count = 2;
        fields1[0].point_index_dimension_count = 2;
        let sources: Vec<&[FieldInfo]> = vec![&fields0, &fields1];
        assert!(matches!(
            reconcile_field_numbers(&sources),
            Err(Error::PointsShapeDisagreement { .. })
        ));

        // num_dims/bytes_per_dim agree but the index/data dim split differs.
        let mut fields1 = fields0.clone();
        fields1[0].point_dimension_count = 1;
        fields1[0].point_index_dimension_count = 0;
        let sources: Vec<&[FieldInfo]> = vec![&fields0, &fields1];
        assert!(matches!(
            reconcile_field_numbers(&sources),
            Err(Error::PointsIndexDimsDisagreement { .. })
        ));
    }

    #[test]
    fn disagreeing_vector_options_across_sources_are_rejected() {
        // FieldInfo.verifySameVectorOptions.
        let mut fields0 = vec![field("vec", 0)];
        fields0[0].vector_dimension = 8;
        let mut fields1 = fields0.clone();
        fields1[0].vector_dimension = 16;
        let sources: Vec<&[FieldInfo]> = vec![&fields0, &fields1];
        assert!(matches!(
            reconcile_field_numbers(&sources),
            Err(Error::FieldSchemaDisagreement {
                attribute: "vector_dimension",
                ..
            })
        ));

        let mut fields1 = fields0.clone();
        fields1[0].vector_similarity_function = VectorSimilarityFunction::DotProduct;
        let sources: Vec<&[FieldInfo]> = vec![&fields0, &fields1];
        assert!(matches!(
            reconcile_field_numbers(&sources),
            Err(Error::FieldSchemaDisagreement {
                attribute: "vector_similarity_function",
                ..
            })
        ));
    }

    #[test]
    fn merged_field_keeps_first_sources_metadata() {
        let mut fields0 = vec![field("id", 0)];
        fields0[0].doc_values_gen = 99;
        let fields1 = vec![field("id", 5)];
        let sources: Vec<&[FieldInfo]> = vec![&fields0, &fields1];
        let (merged, _maps) = reconcile_field_numbers(&sources).unwrap();
        assert_eq!(merged[0].doc_values_gen, 99);
    }

    // --- merge_stored_only_segments (full round-trip via real Directory I/O) ---

    use lucene_util::test_support::TempDir;

    /// A scratch directory that removes itself when the test ends -- unless the
    /// test is panicking, in which case its bytes stay for inspection.
    fn tempdir() -> TempDir {
        TempDir::new("merge")
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
        tmp: &std::path::Path,
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
            fdt: std::fs::read(tmp.join(format!("{name}.fdt"))).unwrap(),
            fdx: std::fs::read(tmp.join(format!("{name}.fdx"))).unwrap(),
            fdm: std::fs::read(tmp.join(format!("{name}.fdm"))).unwrap(),
            fields: fields.to_vec(),
            segment_id,
        }
    }

    fn open_reader(seg: &FlushedSegment) -> stored_fields::StoredFieldsReader<'_> {
        stored_fields::open(&seg.fdt, &seg.fdx, &seg.fdm, &seg.segment_id, "").unwrap()
    }

    // --- the merge-completeness gate (`check_format_coverage`) ---

    /// The table `check_format_coverage` reads must partition extensions:
    /// two formats claiming one extension would attribute a file to the
    /// wrong `is_opened` question, and a format claiming a non-format
    /// extension (`si`/`fnm`/`liv`) would make the gate demand readers for
    /// the segment's own bookkeeping.
    #[test]
    fn no_two_segment_formats_claim_the_same_extension() {
        let mut seen: Vec<(&str, SegmentFormat)> = Vec::new();
        for format in SegmentFormat::ALL {
            assert!(
                !format.extensions().is_empty(),
                "{format:?} claims no extension, so nothing can ever require it to be opened"
            );
            for ext in format.extensions() {
                if let Some((_, other)) = seen.iter().find(|(e, _)| e == ext) {
                    panic!("extension `{ext}` is claimed by both {other:?} and {format:?}");
                }
                assert!(
                    !NON_FORMAT_EXTENSIONS.iter().any(|(e, _)| e == ext),
                    "extension `{ext}` is claimed by {format:?} and also listed as a non-format"
                );
                seen.push((ext, format));
            }
        }
        // Every non-format extension carries a stated reason, so the escape
        // hatch cannot grow silently.
        for (ext, reason) in NON_FORMAT_EXTENSIONS {
            assert!(
                !reason.is_empty(),
                "non-format extension `{ext}` has no reason"
            );
        }
    }

    /// `SegmentFormat::ALL` must actually be all of them. The fixed-length
    /// array makes a *short* list a compile error, but not a list with the
    /// same variant twice.
    #[test]
    fn segment_format_all_lists_each_variant_once() {
        let mut seen: Vec<SegmentFormat> = Vec::new();
        for format in SegmentFormat::ALL {
            assert!(!seen.contains(&format), "{format:?} appears twice in ALL");
            seen.push(format);
        }
        // Round-trips through `for_extension`, which is how the gate finds
        // the format for a file it sees.
        for format in SegmentFormat::ALL {
            for ext in format.extensions() {
                assert_eq!(SegmentFormat::for_extension(ext), Some(format), "{ext}");
            }
        }
    }

    /// **The gate's negative control, in c22's shape.** A source segment
    /// whose own `.si` lists `.nvm`/`.nvd` merged through a `MergeSource`
    /// with no norms is exactly finding 14 -- `execute_merge` passing
    /// `norms: &[]` while every merged BM25 score silently went wrong. The
    /// merge itself would succeed and produce a `CheckIndex`-clean segment,
    /// so nothing but this check reports it.
    #[test]
    fn a_format_the_si_lists_and_the_merge_never_opened_is_refused() {
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let fields = vec![field("id", 0)];
        let seg = flush(
            &dir,
            &tmp,
            "_0",
            [1u8; ID_LENGTH],
            &fields,
            &[doc_with(0, "a")],
        );
        let reader = open_reader(&seg);
        let source = MergeSource::stored_only(&seg.fields, &reader, None);

        // Exactly what a norms-bearing flush puts in its `.si`.
        let files = vec![
            "_0.fdt".to_string(),
            "_0.fdx".to_string(),
            "_0.fdm".to_string(),
            "_0.fnm".to_string(),
            "_0.si".to_string(),
            "_0.nvm".to_string(),
            "_0.nvd".to_string(),
        ];
        let err = check_format_coverage(&["_0"], &[&files], std::slice::from_ref(&source))
            .expect_err("a `.nvd` the merge never opened must be refused");
        match &err {
            Error::MergeFormatNotOpened {
                segment,
                format,
                files,
            } => {
                assert_eq!(segment, "_0");
                assert_eq!(*format, "norms");
                // Both files are named, not just the first one found.
                assert!(
                    files.contains("_0.nvm") && files.contains("_0.nvd"),
                    "{files}"
                );
            }
            other => panic!("expected MergeFormatNotOpened, got {other:?}"),
        }

        // The same `.si` minus the norms files is fine: the gate is about a
        // format the source *has*, not about every format existing.
        let without_norms: Vec<String> = files
            .iter()
            .filter(|f| !f.ends_with(".nvm") && !f.ends_with(".nvd"))
            .cloned()
            .collect();
        check_format_coverage(&["_0"], &[&without_norms], std::slice::from_ref(&source)).unwrap();
    }

    /// One case per format, so no `is_opened` arm can be wrong in the
    /// permissive direction without a test noticing. Every format except
    /// stored fields (whose reader is not optional on `MergeSource`) must be
    /// refused when the `.si` has it and the `MergeSource` does not.
    #[test]
    fn every_optional_format_is_refused_when_the_si_has_it_and_the_source_does_not() {
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let fields = vec![field("id", 0)];
        let seg = flush(
            &dir,
            &tmp,
            "_0",
            [1u8; ID_LENGTH],
            &fields,
            &[doc_with(0, "a")],
        );
        let reader = open_reader(&seg);
        let source = MergeSource::stored_only(&seg.fields, &reader, None);
        let base = ["_0.fdt", "_0.fdx", "_0.fdm", "_0.fnm", "_0.si"];

        for format in SegmentFormat::ALL {
            if format == SegmentFormat::StoredFields {
                assert!(
                    format.is_opened(&source),
                    "stored fields are not optional on a MergeSource"
                );
                continue;
            }
            let mut files: Vec<String> = base.iter().map(|f| (*f).to_string()).collect();
            for ext in format.extensions() {
                files.push(format!("_0.{ext}"));
            }
            let err = check_format_coverage(&["_0"], &[&files], std::slice::from_ref(&source))
                .unwrap_err();
            match err {
                Error::MergeFormatNotOpened { format: named, .. } => {
                    assert_eq!(named, format.name(), "{format:?}");
                }
                other => panic!("expected MergeFormatNotOpened for {format:?}, got {other:?}"),
            }
        }
    }

    /// **The gate's anti-rot arm.** A flush path that learns to write a new
    /// format lands here on the first merge of a segment it wrote, rather
    /// than having that format silently dropped -- and the only way to
    /// satisfy the error is to add a `SegmentFormat` variant, whose two
    /// exhaustive `match`es then force the caller to open it.
    #[test]
    fn an_extension_no_format_claims_is_refused_rather_than_ignored() {
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let fields = vec![field("id", 0)];
        let seg = flush(
            &dir,
            &tmp,
            "_0",
            [1u8; ID_LENGTH],
            &fields,
            &[doc_with(0, "a")],
        );
        let reader = open_reader(&seg);
        let source = MergeSource::stored_only(&seg.fields, &reader, None);

        for bad in ["_0.brandnew", "_0.cfs", "_0_no_extension_at_all"] {
            let files = vec![
                "_0.fdt".to_string(),
                "_0.fdx".to_string(),
                "_0.fdm".to_string(),
                "_0.si".to_string(),
                bad.to_string(),
            ];
            let err = check_format_coverage(&["_0"], &[&files], std::slice::from_ref(&source))
                .unwrap_err();
            match err {
                Error::UnknownSegmentFormat { file, .. } => assert_eq!(file, bad),
                other => panic!("expected UnknownSegmentFormat for {bad}, got {other:?}"),
            }
        }
    }

    /// Per source, not "any source": a two-source merge where only the
    /// second has norms and only the second's readers were opened must
    /// still be refused for the first, and vice versa. The bug shape this
    /// pins is a caller that opens a format for source 0 and forgets the
    /// loop for the rest.
    #[test]
    fn the_gate_asks_the_question_per_source() {
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
            &[doc_with(0, "b")],
        );
        let reader0 = open_reader(&seg0);
        let reader1 = open_reader(&seg1);
        let sources = vec![
            MergeSource::stored_only(&seg0.fields, &reader0, None),
            MergeSource::stored_only(&seg1.fields, &reader1, None),
        ];
        let plain: Vec<String> = ["fdt", "fdx", "fdm", "si"]
            .iter()
            .map(|e| format!("_0.{e}"))
            .collect();
        let with_vectors: Vec<String> = ["fdt", "fdx", "fdm", "si", "vec", "vemf"]
            .iter()
            .map(|e| format!("_1.{e}"))
            .collect();
        let err =
            check_format_coverage(&["_0", "_1"], &[&plain, &with_vectors], &sources).unwrap_err();
        match err {
            Error::MergeFormatNotOpened {
                segment, format, ..
            } => {
                assert_eq!(
                    segment, "_1",
                    "the second source is the one with the `.vec`"
                );
                assert_eq!(format, "KNN vectors");
            }
            other => panic!("expected MergeFormatNotOpened, got {other:?}"),
        }
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

        let merged_fdt = std::fs::read(tmp.join("_merged.fdt")).unwrap();
        let merged_fdx = std::fs::read(tmp.join("_merged.fdx")).unwrap();
        let merged_fdm = std::fs::read(tmp.join("_merged.fdm")).unwrap();
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

        let merged_fdt = std::fs::read(tmp.join("_merged2.fdt")).unwrap();
        let merged_fdx = std::fs::read(tmp.join("_merged2.fdx")).unwrap();
        let merged_fdm = std::fs::read(tmp.join("_merged2.fdm")).unwrap();
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

        let merged_fdt = std::fs::read(tmp.join("_merged3.fdt")).unwrap();
        let merged_fdx = std::fs::read(tmp.join("_merged3.fdx")).unwrap();
        let merged_fdm = std::fs::read(tmp.join("_merged3.fdm")).unwrap();
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

        let merged_fdt = std::fs::read(tmp.join("_merged4.fdt")).unwrap();
        let merged_fdx = std::fs::read(tmp.join("_merged4.fdx")).unwrap();
        let merged_fdm = std::fs::read(tmp.join("_merged4.fdm")).unwrap();
        let merged_fnm = std::fs::read(tmp.join("_merged4.fnm")).unwrap();
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
        let fdt = std::fs::read(tmp.join("_empty.fdt")).unwrap();
        let fdx = std::fs::read(tmp.join("_empty.fdx")).unwrap();
        let fdm = std::fs::read(tmp.join("_empty.fdm")).unwrap();
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
        //
        // The source declares "id" as field number 1, so the merge renumbers
        // it to 0 and the source is *not* a matching reader -- which is what
        // puts it on the VISITOR path, the only one of
        // `Lucene90CompressingStoredFieldsWriter.merge`'s three that parses
        // field numbers at all. A matching reader's chunk bytes are copied
        // verbatim, in this port exactly as in Java, so a segment whose
        // `.fdt` disagrees with its own `.fnm` is a corruption `CheckIndex`
        // reports rather than something the merge re-derives -- see
        // `a_matching_deletion_free_source_is_bulk_copied_verbatim`.
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let fields = vec![field("id", 1)];
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

        let merged_fdt = std::fs::read(tmp.join("_merged5.fdt")).unwrap();
        let merged_fdx = std::fs::read(tmp.join("_merged5.fdx")).unwrap();
        let merged_fdm = std::fs::read(tmp.join("_merged5.fdm")).unwrap();
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
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        )
        .unwrap();
        let field_infos = field_infos::FieldInfos {
            fields: vec![numeric_field("x", field_number)],
        };
        let (_version, parsed) = doc_values::parse_meta(
            &meta,
            &segment_id,
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            &field_infos,
        )
        .unwrap();
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

    fn binary_field(name: &str, number: i32) -> FieldInfo {
        let mut f = field(name, number);
        f.doc_values_type = DocValuesType::Binary;
        f
    }

    /// Same idea as [`FlushedNumericDv`], for BINARY doc values.
    struct FlushedBinaryDv {
        data: Vec<u8>,
        entry: BinaryEntry,
    }

    fn flush_binary_dv(
        field_number: i32,
        values: &[Vec<u8>],
        segment_id: [u8; ID_LENGTH],
    ) -> FlushedBinaryDv {
        let max_doc = values.len() as i32;
        let (meta, data, _skip) = doc_values::write_single_dense_binary_field(
            field_number,
            values,
            max_doc,
            &segment_id,
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        )
        .unwrap();
        let field_infos = field_infos::FieldInfos {
            fields: vec![binary_field("x", field_number)],
        };
        let (_version, parsed) = doc_values::parse_meta(
            &meta,
            &segment_id,
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            &field_infos,
        )
        .unwrap();
        let entry = parsed.binary_entry(field_number).unwrap().clone();
        FlushedBinaryDv { data, entry }
    }

    impl FlushedBinaryDv {
        fn source(&self) -> SourceBinaryDocValues<'_> {
            SourceBinaryDocValues {
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
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &[],
            vectors: None,
        };
        let source1 = MergeSource {
            field_infos: &stored1.fields,
            reader: &reader1,
            live_docs: None,
            numeric_doc_values: &dv1_source,
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &[],
            vectors: None,
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

        let dvd = std::fs::read(tmp.join(format!(
            "{}.dvd",
            per_field_segment("_merged_dv", DOC_VALUES_FORMAT_NAME)
        )))
        .unwrap();
        let dvm = std::fs::read(tmp.join(format!(
            "{}.dvm",
            per_field_segment("_merged_dv", DOC_VALUES_FORMAT_NAME)
        )))
        .unwrap();
        let merged_field_infos = field_infos::FieldInfos {
            fields: vec![numeric_field("num", 0)],
        };
        let (_v, meta) = doc_values::parse_meta(
            &dvm,
            &[9u8; ID_LENGTH],
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            &merged_field_infos,
        )
        .unwrap();
        let entry = meta.numeric_entry(0).unwrap();
        let values: Vec<i64> = (0..2)
            .map(|d| doc_values::numeric_value(&dvd, entry, d).unwrap().unwrap())
            .collect();
        assert_eq!(values, vec![10, 30]);
    }

    /// Task #54's numeric doc-values update overlay
    /// (`lucene_codecs::doc_values_updates`) composed with this module's
    /// numeric doc-values merge -- proving, end to end through both
    /// already-shipped features' real public APIs, that a doc's
    /// overlay-updated value does **not** survive a merge today.
    ///
    /// This is not a bug in either feature: [`doc_values_updates`] is
    /// documented (see that module's top doc comment) as a standalone
    /// single-generation overlay primitive with no `SegmentCommitInfo`/`.si`
    /// `docValuesGen` wiring, and nothing in `IndexWriter` ever applies an
    /// overlay to a committed segment (there is no `update_numeric_doc_value`-
    /// style method on `IndexWriter`; the overlay's only production consumer
    /// is `lucene_search::soft_deletes`'s query-time live-docs check, which
    /// never touches `merge.rs`). [`MergeSource`]/[`SourceNumericDocValues`]
    /// accordingly carry no overlay field at all -- there is no parameter a
    /// caller could even use to plug an overlay into a merge call. This test
    /// documents the current, honest shape of that gap: build a base numeric
    /// field, write a real overlay marking doc 0 with a new value via
    /// [`lucene_codecs::doc_values_updates::write_numeric_updates`], confirm
    /// the overlay-aware read
    /// ([`lucene_codecs::doc_values_updates::numeric_value_with_updates`])
    /// sees the new value, then run an actual `merge_stored_only_segments`
    /// call using only the base `data`/`entry` (since that's all
    /// `SourceNumericDocValues` can express) and confirm the merged segment
    /// still has the **stale** pre-update value -- i.e. if a caller ever did
    /// wire an overlay-writing update path into `IndexWriter` without also
    /// teaching the merge call site to resolve "effective" (overlay-folded)
    /// values before constructing `MergeSource`, updates would be silently
    /// lost across a merge. See `docs/parity.md` and `PLAN.md` for the full
    /// scope writeup.
    ///
    /// **Update (c14 sweep):** `IndexWriter` *does* now write doc-values
    /// updates against a committed segment -- as real Lucene generations, not
    /// as these overlays (`crate::field_updates`). The half of this gap that
    /// was reachable in production is closed from the other side:
    /// `IndexWriter::segment_stats` excludes any segment with
    /// `doc_values_gen != -1` from merge consideration, so a segment carrying
    /// updates is never fed to `merge_stored_only_segments` at all (see
    /// `docs/sweep/m2/c14-dv-updates-format.md` F-8). What this test still
    /// documents is the *shape* of the remaining gap: `MergeSource` reads a
    /// field's base column and nothing else, so whenever a doc-values-aware
    /// merge is built it must resolve each source's newest generation before
    /// constructing the `MergeSource`, exactly as this test's overlay stands
    /// in for.
    #[test]
    fn numeric_doc_values_update_overlay_does_not_survive_a_merge_today() {
        use lucene_codecs::doc_values_updates;

        let seg0_id = [1u8; ID_LENGTH];
        // Base segment: 2 docs, values [10, 20].
        let dv0 = flush_numeric_dv(0, &[10, 20], seg0_id);

        // Apply a real overlay update: doc 0's value becomes 999.
        let overlay_bytes =
            doc_values_updates::write_numeric_updates(&[(0, Some(999))], &seg0_id, "");
        let overlay = doc_values_updates::read_numeric_updates(&overlay_bytes, &seg0_id, "")
            .expect("overlay round-trips");

        // Sanity: the overlay-aware read genuinely sees the new value for
        // doc 0 and correctly falls back to the base value for doc 1 --
        // proving the overlay mechanism itself works in isolation.
        assert_eq!(
            doc_values_updates::numeric_value_with_updates(&dv0.entry, &dv0.data, &overlay, 0)
                .unwrap(),
            Some(999)
        );
        assert_eq!(
            doc_values_updates::numeric_value_with_updates(&dv0.entry, &dv0.data, &overlay, 1)
                .unwrap(),
            Some(20)
        );

        // Now actually merge this single source through the real merge
        // entry point. `SourceNumericDocValues` has no overlay field, so the
        // only thing that can be handed to the merge is the base data/entry
        // -- exactly what a hypothetical "apply overlay before merge" caller
        // would need to have already flattened, and exactly what today's
        // codebase has no code path that does.
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
        let reader0 = open_reader(&stored0);
        let dv0_source = [dv0.source()];
        let source0 = MergeSource {
            field_infos: &stored0.fields,
            reader: &reader0,
            live_docs: None,
            numeric_doc_values: &dv0_source,
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &[],
            vectors: None,
        };

        let sci = merge_stored_only_segments(
            &dir,
            &[source0],
            "_merged_dv_overlay",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();
        assert_eq!(sci.segment_name, "_merged_dv_overlay");

        let dvd = std::fs::read(tmp.join(format!(
            "{}.dvd",
            per_field_segment("_merged_dv_overlay", DOC_VALUES_FORMAT_NAME)
        )))
        .unwrap();
        let dvm = std::fs::read(tmp.join(format!(
            "{}.dvm",
            per_field_segment("_merged_dv_overlay", DOC_VALUES_FORMAT_NAME)
        )))
        .unwrap();
        let merged_field_infos = field_infos::FieldInfos {
            fields: vec![numeric_field("num", 0)],
        };
        let (_v, meta) = doc_values::parse_meta(
            &dvm,
            &[9u8; ID_LENGTH],
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            &merged_field_infos,
        )
        .unwrap();
        let entry = meta.numeric_entry(0).unwrap();
        let merged_values: Vec<i64> = (0..2)
            .map(|d| doc_values::numeric_value(&dvd, entry, d).unwrap().unwrap())
            .collect();

        // The merged segment has the STALE base value (10) for doc 0, not
        // the overlay-updated value (999) -- proving the update was silently
        // lost across this merge, since nothing resolved it before the
        // merge call.
        assert_eq!(merged_values, vec![10, 20]);
        assert_ne!(merged_values[0], 999);
    }

    /// `DocValuesConsumer.getMergedNumericDocValues` skips a reader with no
    /// NUMERIC column for the field, so every one of that reader's documents
    /// lands in the merged column as **missing** -- it does not fail the
    /// merge. This port refused it until c26, which made a field added to
    /// the schema after some segments were flushed, and a doc-values update
    /// against a field with no base column, both unmergeable.
    ///
    /// The reason it is safe to be permissive here is one level up:
    /// [`check_format_coverage`] refuses a merge whose caller never opened a
    /// format a source's `.si` lists, so a *dropped* column cannot hide
    /// behind this branch any more.
    #[test]
    fn numeric_doc_values_absent_from_one_source_come_back_missing_not_as_an_error() {
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
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &[],
            vectors: None,
        };
        // Source 1 has live docs but no numeric doc-values entry at all for
        // field "num": Java's `values == null` sub, i.e. all-missing.
        let source1 = MergeSource::stored_only(&stored1.fields, &reader1, None);

        merge_stored_only_segments(
            &dir,
            &[source0, source1],
            "_merged_dv_sparse",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();

        let seg = per_field_segment("_merged_dv_sparse", DOC_VALUES_FORMAT_NAME);
        let dvm = std::fs::read(tmp.join(format!("{seg}.dvm"))).unwrap();
        let dvd = std::fs::read(tmp.join(format!("{seg}.dvd"))).unwrap();
        let (_v, meta) = doc_values::parse_meta(
            &dvm,
            &[9u8; ID_LENGTH],
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            &field_infos::FieldInfos {
                fields: vec![numeric_field("num", 0)],
            },
        )
        .unwrap();
        let entry = meta
            .numeric_entry(0)
            .expect("the column source 0 has must still be written");
        assert_eq!(
            (0..2)
                .map(|d| doc_values::numeric_value(&dvd, entry, d).unwrap())
                .collect::<Vec<_>>(),
            vec![Some(10), None],
            "source 1's document has no value, rather than the merge failing"
        );
    }

    #[test]
    fn two_numeric_doc_values_fields_land_in_one_merged_dvm() {
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
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &[],
            vectors: None,
        };

        merge_stored_only_segments(
            &dir,
            &[source0],
            "_merged_dv_two",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();

        // Both columns are in the one `.dvm`/`.dvd`, each still attached to
        // its own field number -- what a real multi-field
        // `Lucene90DocValuesFormat` segment looks like, and what a
        // multi-tier index sort needs (a second tier is a second column).
        let (dvd, meta) = open_merged_doc_values(
            tmp.path(),
            "_merged_dv_two",
            [9u8; ID_LENGTH],
            vec![numeric_field("a", 0), numeric_field("b", 1)],
        );
        assert_eq!(
            doc_values::numeric_value(&dvd, meta.numeric_entry(0).unwrap(), 0).unwrap(),
            Some(1)
        );
        assert_eq!(
            doc_values::numeric_value(&dvd, meta.numeric_entry(1).unwrap(), 0).unwrap(),
            Some(2)
        );
    }

    /// Reads a merged segment's `.dvm`/`.dvd` back, parsed against a
    /// caller-supplied merged field list -- the multi-field analogue of the
    /// single-field `read_back_*` helpers, used by the tests that check
    /// several doc-values fields landed in the *same* file triple.
    fn open_merged_doc_values(
        tmp: &std::path::Path,
        segment_name: &str,
        merged_id: [u8; ID_LENGTH],
        fields: Vec<FieldInfo>,
    ) -> (Vec<u8>, doc_values::DocValuesMeta) {
        let dvd = std::fs::read(tmp.join(format!(
            "{}.dvd",
            per_field_segment(segment_name, DOC_VALUES_FORMAT_NAME)
        )))
        .unwrap();
        let dvm = std::fs::read(tmp.join(format!(
            "{}.dvm",
            per_field_segment(segment_name, DOC_VALUES_FORMAT_NAME)
        )))
        .unwrap();
        let (_v, meta) = doc_values::parse_meta(
            &dvm,
            &merged_id,
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            &field_infos::FieldInfos { fields },
        )
        .unwrap();
        (dvd, meta)
    }

    // --- binary doc-values merging (mirrors the numeric tests above) ---

    #[test]
    fn binary_doc_values_merge_across_two_sources_with_deletions() {
        let seg0_id = [1u8; ID_LENGTH];
        let seg1_id = [2u8; ID_LENGTH];
        // Source 0: 2 docs, doc 1 deleted -> only "aa" survives.
        let dv0 = flush_binary_dv(0, &[b"aa".to_vec(), b"bb".to_vec()], seg0_id);
        // Source 1: 1 doc, no deletions -> "cc" survives.
        let dv1 = flush_binary_dv(0, &[b"cc".to_vec()], seg1_id);

        let fields = vec![binary_field("bin", 0)];
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
        live0.set(0); // keep doc 0 ("a"/"aa"), drop doc 1 ("b"/"bb")

        let dv0_source = [dv0.source()];
        let dv1_source = [dv1.source()];
        let source0 = MergeSource {
            field_infos: &stored0.fields,
            reader: &reader0,
            live_docs: Some(&live0),
            numeric_doc_values: &[],
            binary_doc_values: &dv0_source,
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &[],
            vectors: None,
        };
        let source1 = MergeSource {
            field_infos: &stored1.fields,
            reader: &reader1,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &dv1_source,
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &[],
            vectors: None,
        };

        let sci = merge_stored_only_segments(
            &dir,
            &[source0, source1],
            "_merged_bdv",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();
        assert_eq!(sci.segment_name, "_merged_bdv");

        let dvd = std::fs::read(tmp.join(format!(
            "{}.dvd",
            per_field_segment("_merged_bdv", DOC_VALUES_FORMAT_NAME)
        )))
        .unwrap();
        let dvm = std::fs::read(tmp.join(format!(
            "{}.dvm",
            per_field_segment("_merged_bdv", DOC_VALUES_FORMAT_NAME)
        )))
        .unwrap();
        let merged_field_infos = field_infos::FieldInfos {
            fields: vec![binary_field("bin", 0)],
        };
        let (_v, meta) = doc_values::parse_meta(
            &dvm,
            &[9u8; ID_LENGTH],
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            &merged_field_infos,
        )
        .unwrap();
        let entry = meta.binary_entry(0).unwrap();
        let values: Vec<Vec<u8>> = (0..2)
            .map(|d| {
                doc_values::binary_value(&dvd, entry, d)
                    .unwrap()
                    .unwrap()
                    .to_vec()
            })
            .collect();
        assert_eq!(values, vec![b"aa".to_vec(), b"cc".to_vec()]);
    }

    #[test]
    fn binary_doc_values_missing_in_a_live_contributing_source_is_an_error() {
        let seg0_id = [1u8; ID_LENGTH];
        let dv0 = flush_binary_dv(0, &[b"aa".to_vec()], seg0_id);
        let fields = vec![binary_field("bin", 0)];
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
            numeric_doc_values: &[],
            binary_doc_values: &dv0_source,
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &[],
            vectors: None,
        };
        // Source 1 has live docs but no binary doc-values entry at all for
        // field "bin" -- the sparse-across-sources case this port refuses to
        // silently drop.
        let source1 = MergeSource::stored_only(&stored1.fields, &reader1, None);

        let result = merge_stored_only_segments(
            &dir,
            &[source0, source1],
            "_merged_bdv_err",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        );
        assert!(matches!(
            result,
            Err(Error::BinaryDocValuesFieldMissingInSource {
                merged_field_number: 0
            })
        ));
    }

    #[test]
    fn two_binary_doc_values_fields_land_in_one_merged_dvm() {
        let seg0_id = [1u8; ID_LENGTH];
        let dv_a = flush_binary_dv(0, &[b"x".to_vec()], seg0_id);
        let dv_b = flush_binary_dv(1, &[b"y".to_vec()], seg0_id);
        let fields = vec![binary_field("a", 0), binary_field("b", 1)];
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
        let binary = vec![
            SourceBinaryDocValues {
                data: sources_a.data,
                entry: sources_a.entry.clone(),
            },
            SourceBinaryDocValues {
                data: sources_b.data,
                entry: sources_b.entry.clone(),
            },
        ];
        let source0 = MergeSource {
            field_infos: &stored0.fields,
            reader: &reader0,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &binary,
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &[],
            vectors: None,
        };

        merge_stored_only_segments(
            &dir,
            &[source0],
            "_merged_bdv_two",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();

        let (dvd, meta) = open_merged_doc_values(
            tmp.path(),
            "_merged_bdv_two",
            [9u8; ID_LENGTH],
            vec![binary_field("a", 0), binary_field("b", 1)],
        );
        assert_eq!(
            doc_values::binary_value(&dvd, meta.binary_entry(0).unwrap(), 0).unwrap(),
            Some(&b"x"[..])
        );
        assert_eq!(
            doc_values::binary_value(&dvd, meta.binary_entry(1).unwrap(), 0).unwrap(),
            Some(&b"y"[..])
        );
    }

    // --- SORTED doc-values merging (ordinal remapping via term-bytes resolution) ---

    fn sorted_field(name: &str, number: i32) -> FieldInfo {
        let mut f = field(name, number);
        f.doc_values_type = DocValuesType::Sorted;
        f
    }

    /// Same idea as [`FlushedBinaryDv`], for SORTED doc values -- `values`
    /// is one raw term per doc (dense, `values.len() == max_doc`), exactly
    /// what [`doc_values::write_single_dense_sorted_field`] takes.
    struct FlushedSortedDv {
        data: Vec<u8>,
        entry: SortedEntry,
    }

    fn flush_sorted_dv(
        field_number: i32,
        values: &[Vec<u8>],
        segment_id: [u8; ID_LENGTH],
    ) -> FlushedSortedDv {
        let max_doc = values.len() as i32;
        let (meta, data, _skip) = doc_values::write_single_dense_sorted_field(
            field_number,
            values,
            max_doc,
            &segment_id,
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        )
        .unwrap();
        let field_infos = field_infos::FieldInfos {
            fields: vec![sorted_field("x", field_number)],
        };
        let (_version, parsed) = doc_values::parse_meta(
            &meta,
            &segment_id,
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            &field_infos,
        )
        .unwrap();
        let entry = parsed.sorted_entry(field_number).unwrap().clone();
        FlushedSortedDv { data, entry }
    }

    impl FlushedSortedDv {
        fn source(&self) -> SourceSortedDocValues<'_> {
            SourceSortedDocValues {
                data: &self.data,
                entry: self.entry.clone(),
            }
        }
    }

    /// Resolves every doc's merged SORTED term, doc by doc, through the
    /// *unmodified* reader stack (`parse_meta` + `sorted_ord` +
    /// `terms_dict::decode_all_terms`) -- the critical correctness check:
    /// not just "some valid ordinal", but the actual right term bytes per
    /// doc, read back exactly as a real caller would.
    fn read_back_sorted_terms(
        dvm: &[u8],
        dvd: &[u8],
        field_number: i32,
        doc_count: i32,
    ) -> Vec<Vec<u8>> {
        let merged_field_infos = field_infos::FieldInfos {
            fields: vec![sorted_field("x", field_number)],
        };
        let (_v, meta) = doc_values::parse_meta(
            dvm,
            &[9u8; ID_LENGTH],
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            &merged_field_infos,
        )
        .unwrap();
        let entry = meta.sorted_entry(field_number).unwrap();
        let dict = terms_dict::decode_all_terms(dvd, &entry.terms).unwrap();
        (0..doc_count)
            .map(|d| {
                let ord = doc_values::sorted_ord(dvd, entry, d).unwrap().unwrap();
                dict[ord as usize].clone()
            })
            .collect()
    }

    #[test]
    fn sorted_doc_values_merge_with_overlapping_terms_dedupes_into_one_shared_dictionary_entry() {
        // Source 0: docs "red", "blue"; source 1: docs "red", "green" -- both
        // sources independently assign "red" ordinal 0 (it's the
        // alphabetically-first of each source's own two-term dictionary).
        // Real bug case: if this merge naively concatenated ordinals without
        // resolving to bytes, source 1's "red" (ordinal 0 in its own dict)
        // could get merged as a *different* dictionary entry than source 0's
        // "red" (also ordinal 0) purely because they came from different
        // sources -- this test would catch that by checking actual resolved
        // term bytes, not just ordinal counts.
        let seg0_id = [1u8; ID_LENGTH];
        let seg1_id = [2u8; ID_LENGTH];
        let dv0 = flush_sorted_dv(0, &[b"red".to_vec(), b"blue".to_vec()], seg0_id);
        let dv1 = flush_sorted_dv(0, &[b"red".to_vec(), b"green".to_vec()], seg1_id);

        let fields = vec![sorted_field("color", 0)];
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
        let stored1 = flush(
            &dir,
            &tmp,
            "_1",
            seg1_id,
            &fields,
            &[doc_with(0, "c"), doc_with(0, "d")],
        );
        let reader0 = open_reader(&stored0);
        let reader1 = open_reader(&stored1);

        let dv0_source = [dv0.source()];
        let dv1_source = [dv1.source()];
        let source0 = MergeSource {
            field_infos: &stored0.fields,
            reader: &reader0,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &dv0_source,
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &[],
            vectors: None,
        };
        let source1 = MergeSource {
            field_infos: &stored1.fields,
            reader: &reader1,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &dv1_source,
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &[],
            vectors: None,
        };

        let tmp_dir = FsDirectory::open(&tmp);
        merge_stored_only_segments(
            &tmp_dir,
            &[source0, source1],
            "_merged_sorted_overlap",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();

        let dvm = std::fs::read(tmp.join(format!(
            "{}.dvm",
            per_field_segment("_merged_sorted_overlap", DOC_VALUES_FORMAT_NAME)
        )))
        .unwrap();
        let dvd = std::fs::read(tmp.join(format!(
            "{}.dvd",
            per_field_segment("_merged_sorted_overlap", DOC_VALUES_FORMAT_NAME)
        )))
        .unwrap();
        let merged_field_infos = field_infos::FieldInfos {
            fields: vec![sorted_field("color", 0)],
        };
        let (_v, meta) = doc_values::parse_meta(
            &dvm,
            &[9u8; ID_LENGTH],
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            &merged_field_infos,
        )
        .unwrap();
        let entry = meta.sorted_entry(0).unwrap();
        // "red" is shared across both sources -- the merged dictionary must
        // dedupe it into exactly one entry, so the distinct dictionary size
        // is 3 ("red", "blue", "green"), not 4.
        assert_eq!(entry.terms.terms_dict_size, 3);

        // And every doc must resolve to the RIGHT term, not just any valid
        // ordinal -- this is the actual correctness check.
        let terms = read_back_sorted_terms(&dvm, &dvd, 0, 4);
        assert_eq!(
            terms,
            vec![
                b"red".to_vec(),
                b"blue".to_vec(),
                b"red".to_vec(),
                b"green".to_vec(),
            ]
        );
    }

    #[test]
    fn sorted_doc_values_merge_with_disjoint_terms_contains_all_terms_from_both_sources() {
        let seg0_id = [1u8; ID_LENGTH];
        let seg1_id = [2u8; ID_LENGTH];
        let dv0 = flush_sorted_dv(0, &[b"apple".to_vec()], seg0_id);
        let dv1 = flush_sorted_dv(0, &[b"zebra".to_vec()], seg1_id);

        let fields = vec![sorted_field("word", 0)];
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let stored0 = flush(&dir, &tmp, "_0", seg0_id, &fields, &[doc_with(0, "a")]);
        let stored1 = flush(&dir, &tmp, "_1", seg1_id, &fields, &[doc_with(0, "b")]);
        let reader0 = open_reader(&stored0);
        let reader1 = open_reader(&stored1);

        let dv0_source = [dv0.source()];
        let dv1_source = [dv1.source()];
        let source0 = MergeSource {
            field_infos: &stored0.fields,
            reader: &reader0,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &dv0_source,
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &[],
            vectors: None,
        };
        let source1 = MergeSource {
            field_infos: &stored1.fields,
            reader: &reader1,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &dv1_source,
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &[],
            vectors: None,
        };

        merge_stored_only_segments(
            &dir,
            &[source0, source1],
            "_merged_sorted_disjoint",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();

        let dvm = std::fs::read(tmp.join(format!(
            "{}.dvm",
            per_field_segment("_merged_sorted_disjoint", DOC_VALUES_FORMAT_NAME)
        )))
        .unwrap();
        let dvd = std::fs::read(tmp.join(format!(
            "{}.dvd",
            per_field_segment("_merged_sorted_disjoint", DOC_VALUES_FORMAT_NAME)
        )))
        .unwrap();
        let terms = read_back_sorted_terms(&dvm, &dvd, 0, 2);
        assert_eq!(terms, vec![b"apple".to_vec(), b"zebra".to_vec()]);
    }

    #[test]
    fn sorted_doc_values_missing_in_a_live_contributing_source_is_an_error() {
        let seg0_id = [1u8; ID_LENGTH];
        let dv0 = flush_sorted_dv(0, &[b"x".to_vec()], seg0_id);
        let fields = vec![sorted_field("word", 0)];
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
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &dv0_source,
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &[],
            vectors: None,
        };
        // Source 1 has live docs but no SORTED doc-values entry at all for
        // field "word".
        let source1 = MergeSource::stored_only(&stored1.fields, &reader1, None);

        let result = merge_stored_only_segments(
            &dir,
            &[source0, source1],
            "_merged_sorted_err",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        );
        assert!(matches!(
            result,
            Err(Error::SortedDocValuesFieldMissingInSource {
                merged_field_number: 0
            })
        ));
    }

    #[test]
    fn two_sorted_doc_values_fields_land_in_one_merged_dvm() {
        let seg0_id = [1u8; ID_LENGTH];
        let dv_a = flush_sorted_dv(0, &[b"x".to_vec()], seg0_id);
        let dv_b = flush_sorted_dv(1, &[b"y".to_vec()], seg0_id);
        let fields = vec![sorted_field("a", 0), sorted_field("b", 1)];
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
        let sorted = vec![
            SourceSortedDocValues {
                data: sources_a.data,
                entry: sources_a.entry.clone(),
            },
            SourceSortedDocValues {
                data: sources_b.data,
                entry: sources_b.entry.clone(),
            },
        ];
        let source0 = MergeSource {
            field_infos: &stored0.fields,
            reader: &reader0,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &sorted,
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &[],
            vectors: None,
        };

        merge_stored_only_segments(
            &dir,
            &[source0],
            "_merged_sdv_two",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();

        // Each field rebuilds its *own* merged dictionary, so ordinal 0 of
        // field 0 and ordinal 0 of field 1 are different terms.
        let (dvd, meta) = open_merged_doc_values(
            tmp.path(),
            "_merged_sdv_two",
            [9u8; ID_LENGTH],
            vec![sorted_field("a", 0), sorted_field("b", 1)],
        );
        for (field_number, expected) in [(0, &b"x"[..]), (1, &b"y"[..])] {
            let entry = meta.sorted_entry(field_number).unwrap();
            let ord = doc_values::sorted_ord(&dvd, entry, 0).unwrap().unwrap();
            let dict = terms_dict::decode_all_terms(&dvd, &entry.terms).unwrap();
            assert_eq!(dict[ord as usize], expected);
        }
    }

    #[test]
    fn numeric_and_binary_doc_values_share_one_merged_dvm() {
        // Doc-values *types* mix freely in one segment: every merged field
        // goes into the same `.dvm`/`.dvd`/`.dvs` through
        // `doc_values::write_dense_fields`, which is what a real
        // `Lucene90DocValuesFormat` segment is. (Before this batch the merge
        // wrote one field triple per type and had to refuse the combination.)
        let seg0_id = [1u8; ID_LENGTH];
        let numeric_dv = flush_numeric_dv(0, &[1], seg0_id);
        let binary_dv = flush_binary_dv(1, &[b"v".to_vec()], seg0_id);
        let fields = vec![numeric_field("num", 0), binary_field("bin", 1)];
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
        let numeric_source = [numeric_dv.source()];
        let binary_source = [binary_dv.source()];
        let source0 = MergeSource {
            field_infos: &stored0.fields,
            reader: &reader0,
            live_docs: None,
            numeric_doc_values: &numeric_source,
            binary_doc_values: &binary_source,
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &[],
            vectors: None,
        };

        merge_stored_only_segments(
            &dir,
            &[source0],
            "_merged_mixed_dv",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();

        let (dvd, meta) = open_merged_doc_values(
            tmp.path(),
            "_merged_mixed_dv",
            [9u8; ID_LENGTH],
            vec![numeric_field("num", 0), binary_field("bin", 1)],
        );
        assert_eq!(
            doc_values::numeric_value(&dvd, meta.numeric_entry(0).unwrap(), 0).unwrap(),
            Some(1)
        );
        assert_eq!(
            doc_values::binary_value(&dvd, meta.binary_entry(1).unwrap(), 0).unwrap(),
            Some(&b"v"[..])
        );
    }

    #[test]
    fn sorted_numeric_and_sorted_set_doc_values_share_one_merged_dvm() {
        // Same rule as `numeric_and_binary_doc_values_share_one_merged_dvm`,
        // exercised for the pair whose readers are the most intertwined:
        // a SORTED_SET field that collapsed to one value per doc is stored
        // through the very same layout a SORTED_NUMERIC field uses, so
        // getting the two into one `.dvm` without crossing their entries is
        // worth its own case.
        let seg0_id = [1u8; ID_LENGTH];
        let sorted_numeric_dv = flush_sorted_numeric_dv(0, &[vec![1]], seg0_id);
        let sorted_set_dv = flush_sorted_set_dv(1, &[vec![b"v".to_vec()]], seg0_id);
        let fields = vec![sorted_numeric_field("num", 0), sorted_set_field("set", 1)];
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
        let sorted_numeric_source = [sorted_numeric_dv.source()];
        let sorted_set_source = [sorted_set_dv.source()];
        let source0 = MergeSource {
            field_infos: &stored0.fields,
            reader: &reader0,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &sorted_numeric_source,
            sorted_set_doc_values: &sorted_set_source,
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &[],
            vectors: None,
        };

        merge_stored_only_segments(
            &dir,
            &[source0],
            "_merged_mixed_dv2",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();

        let (dvd, meta) = open_merged_doc_values(
            tmp.path(),
            "_merged_mixed_dv2",
            [9u8; ID_LENGTH],
            vec![sorted_numeric_field("num", 0), sorted_set_field("set", 1)],
        );
        assert_eq!(
            doc_values::sorted_numeric_values(&dvd, meta.sorted_numeric_entry(0).unwrap(), 0)
                .unwrap(),
            vec![1]
        );
        let set_entry = meta.sorted_set_entry(1).unwrap();
        let ords = sorted_set_doc_ordinals(&dvd, set_entry, 0).unwrap();
        let dict = sorted_set_source_dict(&dvd, set_entry).unwrap();
        assert_eq!(
            ords.iter()
                .map(|&o| dict[o as usize].clone())
                .collect::<Vec<_>>(),
            vec![b"v".to_vec()]
        );
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
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &[],
            vectors: None,
        };
        let source1 = MergeSource {
            field_infos: &stored1.fields,
            reader: &reader1,
            live_docs: Some(&all_deleted),
            numeric_doc_values: &numeric1,
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &[],
            vectors: None,
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

        let merged_reader = std::fs::read(tmp.join(format!("{}.fdt", sci.segment_name)));
        assert!(
            merged_reader.is_ok(),
            "merge should succeed, not reject on source 1's unrelated deleted-only field"
        );
    }

    #[test]
    fn a_fully_deleted_sources_unrelated_binary_field_does_not_trigger_too_many_fields() {
        // Same regression shape as the NUMERIC version above, for BINARY:
        // a 100%-deleted source's own binary-dv field must not count toward
        // the "more than one field" limit, since it contributes zero live
        // docs to the merge.
        let seg0_id = [1u8; ID_LENGTH];
        let seg1_id = [2u8; ID_LENGTH];
        let dv_a = flush_binary_dv(0, &[b"aa".to_vec(), b"bb".to_vec()], seg0_id);
        let dv_junk = flush_binary_dv(0, &[b"zz".to_vec()], seg1_id);
        let fields0 = vec![binary_field("a", 0)];
        let fields1 = vec![binary_field("junk", 0)];
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
        let binary0 = vec![dv_a.source()];
        let binary1 = vec![dv_junk.source()];
        let all_deleted = FixedBitSet::new(1); // source 1: nothing live
        let source0 = MergeSource {
            field_infos: &stored0.fields,
            reader: &reader0,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &binary0,
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &[],
            vectors: None,
        };
        let source1 = MergeSource {
            field_infos: &stored1.fields,
            reader: &reader1,
            live_docs: Some(&all_deleted),
            numeric_doc_values: &[],
            binary_doc_values: &binary1,
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &[],
            vectors: None,
        };

        let sci = merge_stored_only_segments(
            &dir,
            &[source0, source1],
            "_merged_binary_dv_deleted_unrelated",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();

        let merged_reader = std::fs::read(tmp.join(format!("{}.fdt", sci.segment_name)));
        assert!(
            merged_reader.is_ok(),
            "merge should succeed, not reject on source 1's unrelated deleted-only binary field"
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
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &norms0_source,
            term_vectors: None,
            postings: &[],
            points: &[],
            vectors: None,
        };
        let source1 = MergeSource {
            field_infos: &stored1.fields,
            reader: &reader1,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &norms1_source,
            term_vectors: None,
            postings: &[],
            points: &[],
            vectors: None,
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

        let nvd = std::fs::read(tmp.join("_merged_norms.nvd")).unwrap();
        let nvm = std::fs::read(tmp.join("_merged_norms.nvm")).unwrap();
        let (_v, parsed) = norms::parse_meta(&nvm, &[9u8; ID_LENGTH], "").unwrap();
        let entry = parsed.entry(0).unwrap();
        let values: Vec<i64> = (0..2)
            .map(|d| norms::norm_value(&nvd, entry, d).unwrap().unwrap())
            .collect();
        assert_eq!(values, vec![2, 3]);
    }

    #[test]
    fn term_vectors_merge_across_two_sources_with_deletions_and_a_source_with_none() {
        // Source 1 contributes a live doc but has no term-vectors reader at
        // all, while source 0 does. Real Lucene's `TermVectorsWriter.merge`
        // handles exactly this (`mergeState.termVectorsReaders[i]` may be
        // null -> `vectors = null` -> `addAllDocVectors(null, ...)` writes a
        // vector-less document), which is what happens when term vectors
        // are turned on for an index that already has segments. So source
        // 1's doc must come through as an empty term-vectors document, and
        // source 0's surviving doc must keep its vectors.
        let seg0_id = [1u8; ID_LENGTH];
        // Source 0: 2 docs, both with a term-vectors field 0 ("id"->0).
        let tv0 = flush_term_vectors(&[tv_doc(0, &[("a", 0)]), tv_doc(0, &[("b", 0)])], seg0_id);
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
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: Some(&tv0_reader),
            postings: &[],
            points: &[],
            vectors: None,
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

        let tvd = std::fs::read(tmp.join("_merged_tv.tvd")).unwrap();
        let tvx = std::fs::read(tmp.join("_merged_tv.tvx")).unwrap();
        let tvm = std::fs::read(tmp.join("_merged_tv.tvm")).unwrap();
        let merged = term_vectors::open(&tvd, &tvx, &tvm, &[9u8; ID_LENGTH], "").unwrap();
        assert_eq!(merged.max_doc(), 2);

        // Merged doc 0 == source 0's surviving doc ("b"), vectors intact.
        let doc0 = merged.document(0).unwrap().unwrap();
        assert_eq!(doc0.fields.len(), 1);
        assert_eq!(doc0.fields[0].field_number, 0);
        assert_eq!(doc0.fields[0].terms[0].term, b"b".to_vec());

        // Merged doc 1 == source 1's doc, which had no term vectors: an
        // empty term-vectors document, not an error.
        let doc1 = merged.document(1).unwrap();
        assert!(
            doc1.is_none() || doc1.as_ref().unwrap().fields.is_empty(),
            "a source without term vectors contributes a vector-less doc: {doc1:?}"
        );
    }

    /// Builds a single-field, single-term term-vector document with
    /// POSITIONS+OFFSETS+PAYLOADS all populated -- unlike [`tv_doc`] (which
    /// only exercises positions), this is the shape
    /// [`term_vectors_merge_carries_offsets_and_payloads_through`] needs to
    /// prove offsets/payloads survive [`write_merged_term_vectors`] unchanged.
    fn tv_doc_with_offsets_and_payloads(
        field_number: i32,
        term: &str,
        position: i32,
        start_offset: i32,
        end_offset: i32,
        payload: &[u8],
    ) -> TermVectorsDocument {
        TermVectorsDocument {
            fields: vec![TermVectorField {
                field_number,
                has_positions: true,
                has_offsets: true,
                has_payloads: true,
                terms: vec![TermVectorTerm {
                    term: term.as_bytes().to_vec(),
                    freq: 1,
                    positions: Some(vec![position]),
                    start_offsets: Some(vec![start_offset]),
                    end_offsets: Some(vec![end_offset]),
                    payloads: Some(vec![payload.to_vec()]),
                }],
            }],
        }
    }

    // ---------------------------------------------------------------
    // `Lucene90CompressingTermVectorsWriter.merge`'s bulk path (`c8`).
    // ---------------------------------------------------------------

    /// Enough documents to span several 128-document term-vector chunks, so
    /// the bulk path has whole chunks to copy and a ragged dirty tail to
    /// re-encode.
    const TV_BULK_DOCS: usize = 300;

    fn tv_bulk_docs(tag: &str) -> Vec<TermVectorsDocument> {
        (0..TV_BULK_DOCS)
            .map(|n| tv_doc(0, &[(&format!("{tag}{n:04}"), (n % 5) as i32)]))
            .collect()
    }

    fn tv_bulk_stored(tag: &str) -> Vec<Document> {
        (0..TV_BULK_DOCS)
            .map(|n| doc_with(0, &format!("{tag}{n:04}")))
            .collect()
    }

    /// Merges two term-vector-bearing sources and returns the merged
    /// `(tvd, tvx, tvm)`; `live` and `renumber` control which merge strategy
    /// each source ends up on.
    #[allow(clippy::type_complexity)]
    fn merge_two_tv_sources(
        tmp: &std::path::Path,
        live1: Option<&FixedBitSet>,
        renumber_source_1: bool,
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let dir = FsDirectory::open(tmp);
        let seg0_id = [1u8; ID_LENGTH];
        let seg1_id = [2u8; ID_LENGTH];
        let tv0 = flush_term_vectors(&tv_bulk_docs("a"), seg0_id);
        let tv1 = flush_term_vectors(&tv_bulk_docs("b"), seg1_id);
        let fields0 = vec![tv_field("body", 0)];
        // A source whose field number the merge has to remap is not a
        // `MatchingReaders` match, so it cannot be bulk-copied.
        let fields1 = if renumber_source_1 {
            vec![tv_field("other", 0), tv_field("body", 1)]
        } else {
            vec![tv_field("body", 0)]
        };
        let stored0 = flush(&dir, tmp, "_0", seg0_id, &fields0, &tv_bulk_stored("a"));
        let stored1 = flush(&dir, tmp, "_1", seg1_id, &fields1, &tv_bulk_stored("b"));
        let reader0 = open_reader(&stored0);
        let reader1 = open_reader(&stored1);
        let tv0_reader = tv0.reader();
        let tv1_reader = tv1.reader();
        let source0 = MergeSource {
            term_vectors: Some(&tv0_reader),
            ..MergeSource::stored_only(&stored0.fields, &reader0, None)
        };
        let source1 = MergeSource {
            term_vectors: Some(&tv1_reader),
            ..MergeSource::stored_only(&stored1.fields, &reader1, live1)
        };
        merge_stored_only_segments(
            &dir,
            &[source0, source1],
            "_m",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();
        let read = |ext: &str| std::fs::read(tmp.join(format!("_m.{ext}"))).unwrap();
        (read("tvd"), read("tvx"), read("tvm"))
    }

    #[test]
    fn matching_deletion_free_term_vector_sources_are_bulk_copied_verbatim() {
        let tmp = tempdir();
        let (tvd, tvx, tvm) = merge_two_tv_sources(&tmp, None, false);
        let merged = term_vectors::open(&tvd, &tvx, &tvm, &[9u8; ID_LENGTH], "").unwrap();
        assert_eq!(merged.max_doc(), 2 * TV_BULK_DOCS as i32);

        // Each source contributed two whole 128-doc chunks copied verbatim
        // plus its own 44-doc dirty tail, and the tails came across *as*
        // dirty chunks -- which is the observable signature of a byte copy
        // (a re-encoding merge would have coalesced all 600 documents into
        // 4 clean chunks and one dirty tail).
        assert_eq!(merged.num_chunks(), 6);
        assert_eq!(merged.num_dirty_chunks(), 2);
        assert_eq!(merged.num_dirty_docs(), 2 * 44);

        // ... and every document still reads back, in order, with its own
        // vectors.
        let expected: Vec<TermVectorsDocument> = tv_bulk_docs("a")
            .into_iter()
            .chain(tv_bulk_docs("b"))
            .collect();
        for (i, want) in expected.iter().enumerate() {
            let got = merged.document(i as i32).unwrap().unwrap();
            assert_eq!(&got.fields[0].terms, &want.fields[0].terms, "doc {i}");
        }
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn a_term_vector_source_with_deletions_is_re_encoded_not_copied() {
        let tmp = tempdir();
        let mut live1 = FixedBitSet::new(TV_BULK_DOCS);
        for i in 0..TV_BULK_DOCS {
            if !i.is_multiple_of(3) {
                live1.set(i);
            }
        }
        let surviving1 = TV_BULK_DOCS - TV_BULK_DOCS.div_ceil(3);
        let (tvd, tvx, tvm) = merge_two_tv_sources(&tmp, Some(&live1), false);
        let merged = term_vectors::open(&tvd, &tvx, &tvm, &[9u8; ID_LENGTH], "").unwrap();
        assert_eq!(merged.max_doc(), (TV_BULK_DOCS + surviving1) as i32);

        // Source 0 is still bulk-copied (2 clean chunks + its dirty tail);
        // source 1's surviving documents are re-encoded, so they pack into
        // fresh full chunks with exactly one dirty tail at the very end.
        assert_eq!(merged.num_dirty_chunks(), 2);

        let src0 = tv_bulk_docs("a");
        let src1 = tv_bulk_docs("b");
        for (i, want) in src0.iter().enumerate() {
            let got = merged.document(i as i32).unwrap().unwrap();
            assert_eq!(&got.fields[0].terms, &want.fields[0].terms, "doc {i}");
        }
        let mut merged_id = TV_BULK_DOCS as i32;
        for (i, want) in src1.iter().enumerate() {
            if !live1.get(i) {
                continue;
            }
            let got = merged.document(merged_id).unwrap().unwrap();
            assert_eq!(
                &got.fields[0].terms, &want.fields[0].terms,
                "source 1 doc {i}"
            );
            merged_id += 1;
        }
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn a_renumbered_term_vector_source_is_re_encoded_with_remapped_field_numbers() {
        let tmp = tempdir();
        let (tvd, tvx, tvm) = merge_two_tv_sources(&tmp, None, true);
        let merged = term_vectors::open(&tvd, &tvx, &tvm, &[9u8; ID_LENGTH], "").unwrap();
        assert_eq!(merged.max_doc(), 2 * TV_BULK_DOCS as i32);
        // `reconcile_field_numbers` keeps source 0's "body" at 0 and gives
        // source 1's "other" the next free number, 1 -- so source 1's map is
        // {0 -> 1, 1 -> 0}, not the identity, and `MatchingReaders` rules it
        // out of the bulk path. Its term-vector data sits on its own field 0
        // ("other"), which must come through as merged field 1. A bulk copy
        // would have carried the source's own number through verbatim and
        // put every one of those vectors under the wrong field name -- the
        // exact failure Java's "bulk merge is scary" comment is about, so
        // check it for *every* document, not just the first.
        for i in 0..TV_BULK_DOCS {
            let got = merged.document((TV_BULK_DOCS + i) as i32).unwrap().unwrap();
            assert_eq!(got.fields[0].field_number, 1, "source 1 doc {i}");
            assert_eq!(
                got.fields[0].terms[0].term,
                format!("b{i:04}").into_bytes(),
                "source 1 doc {i}"
            );
        }
        // Source 0 *is* a match, so it is still bulk-copied: its own dirty
        // tail chunk comes across as dirty, alongside the merge's own final
        // flush.
        assert_eq!(merged.num_dirty_chunks(), 2);
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn a_term_vector_source_whose_tvd_fails_its_own_checksum_is_never_bulk_copied() {
        // The hazard behind Java's "bulk merge is scary" comment: the bulk
        // path copies compressed bytes verbatim and then writes a freshly
        // computed, valid footer over them, so without
        // `reader.checkIntegrity(...)` a corrupt source would be laundered
        // into a merged segment that passes every checksum from then on.
        // The flipped byte leaves every length, pointer and footer field
        // intact -- exactly the corruption `retrieve_checksum` cannot see.
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let seg0_id = [1u8; ID_LENGTH];
        let mut tv0 = flush_term_vectors(&tv_bulk_docs("a"), seg0_id);
        let victim = tv0.tvd.len() - lucene_store::codec_util::FOOTER_LENGTH - 20;
        tv0.tvd[victim] ^= 0x40;
        let fields0 = vec![tv_field("body", 0)];
        let stored0 = flush(&dir, &tmp, "_0", seg0_id, &fields0, &tv_bulk_stored("a"));
        let reader0 = open_reader(&stored0);
        let tv0_reader = tv0.reader();
        let source0 = MergeSource {
            term_vectors: Some(&tv0_reader),
            ..MergeSource::stored_only(&stored0.fields, &reader0, None)
        };
        let err = merge_stored_only_segments(
            &dir,
            &[source0],
            "_m",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap_err();
        assert!(
            matches!(err, Error::TermVectors(_)),
            "expected the term-vectors checksum to reject the merge, got {err:?}"
        );
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn term_vectors_merge_carries_offsets_and_payloads_through() {
        let seg0_id = [1u8; ID_LENGTH];
        let seg1_id = [2u8; ID_LENGTH];

        // Source 0: 2 docs, both with offsets+payloads term vectors.
        let tv0 = flush_term_vectors(
            &[
                tv_doc_with_offsets_and_payloads(0, "alpha", 0, 0, 5, &[0xAA]),
                tv_doc_with_offsets_and_payloads(0, "beta", 0, 0, 4, &[0xBB, 0xCC]),
            ],
            seg0_id,
        );
        // Source 1: 1 doc, also with offsets+payloads term vectors.
        let tv1 = flush_term_vectors(
            &[tv_doc_with_offsets_and_payloads(
                0,
                "gamma",
                0,
                0,
                5,
                &[0xDD],
            )],
            seg1_id,
        );
        let tv0_reader = tv0.reader();
        let tv1_reader = tv1.reader();

        // Sanity-check both flush side actually round-trips offsets/payloads
        // before using them to exercise the merge.
        let doc0 = tv0_reader.document(0).unwrap().unwrap();
        assert!(doc0.fields[0].has_offsets && doc0.fields[0].has_payloads);
        assert_eq!(doc0.fields[0].terms[0].start_offsets, Some(vec![0]));
        assert_eq!(doc0.fields[0].terms[0].end_offsets, Some(vec![5]));
        assert_eq!(doc0.fields[0].terms[0].payloads, Some(vec![vec![0xAA]]));

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
            &[doc_with(0, "alpha"), doc_with(0, "beta")],
        );
        let stored1 = flush(&dir, &tmp, "_1", seg1_id, &fields1, &[doc_with(0, "gamma")]);
        let reader0 = open_reader(&stored0);
        let reader1 = open_reader(&stored1);

        let source0 = MergeSource {
            field_infos: &stored0.fields,
            reader: &reader0,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: Some(&tv0_reader),
            postings: &[],
            points: &[],
            vectors: None,
        };
        let source1 = MergeSource {
            field_infos: &stored1.fields,
            reader: &reader1,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: Some(&tv1_reader),
            postings: &[],
            points: &[],
            vectors: None,
        };

        merge_stored_only_segments(
            &dir,
            &[source0, source1],
            "_merged_tv_offsets_payloads",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();

        let merged_tvd = std::fs::read(tmp.join("_merged_tv_offsets_payloads.tvd")).unwrap();
        let merged_tvx = std::fs::read(tmp.join("_merged_tv_offsets_payloads.tvx")).unwrap();
        let merged_tvm = std::fs::read(tmp.join("_merged_tv_offsets_payloads.tvm")).unwrap();
        let merged_reader =
            term_vectors::open(&merged_tvd, &merged_tvx, &merged_tvm, &[9u8; ID_LENGTH], "")
                .unwrap();

        let merged0 = merged_reader.document(0).unwrap().unwrap();
        assert!(merged0.fields[0].has_offsets && merged0.fields[0].has_payloads);
        assert_eq!(merged0.fields[0].terms[0].term, b"alpha");
        assert_eq!(merged0.fields[0].terms[0].start_offsets, Some(vec![0]));
        assert_eq!(merged0.fields[0].terms[0].end_offsets, Some(vec![5]));
        assert_eq!(merged0.fields[0].terms[0].payloads, Some(vec![vec![0xAA]]));

        let merged1 = merged_reader.document(1).unwrap().unwrap();
        assert_eq!(merged1.fields[0].terms[0].term, b"beta");
        assert_eq!(merged1.fields[0].terms[0].start_offsets, Some(vec![0]));
        assert_eq!(merged1.fields[0].terms[0].end_offsets, Some(vec![4]));
        assert_eq!(
            merged1.fields[0].terms[0].payloads,
            Some(vec![vec![0xBB, 0xCC]])
        );

        // Doc 2 comes from source 1 (seg1), merged after source 0's 2 docs.
        let merged2 = merged_reader.document(2).unwrap().unwrap();
        assert_eq!(merged2.fields[0].terms[0].term, b"gamma");
        assert_eq!(merged2.fields[0].terms[0].start_offsets, Some(vec![0]));
        assert_eq!(merged2.fields[0].terms[0].end_offsets, Some(vec![5]));
        assert_eq!(merged2.fields[0].terms[0].payloads, Some(vec![vec![0xDD]]));
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
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &norms0_source,
            term_vectors: Some(&tv0_reader),
            postings: &[],
            points: &[],
            vectors: None,
        };
        let source1 = MergeSource {
            field_infos: &stored1.fields,
            reader: &reader1,
            live_docs: None,
            numeric_doc_values: &dv1_source,
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &norms1_source,
            term_vectors: Some(&tv1_reader),
            postings: &[],
            points: &[],
            vectors: None,
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
        let merged_fdt = std::fs::read(tmp.join("_merged_all.fdt")).unwrap();
        let merged_fdx = std::fs::read(tmp.join("_merged_all.fdx")).unwrap();
        let merged_fdm = std::fs::read(tmp.join("_merged_all.fdm")).unwrap();
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
        let dvd = std::fs::read(tmp.join(format!(
            "{}.dvd",
            per_field_segment("_merged_all", DOC_VALUES_FORMAT_NAME)
        )))
        .unwrap();
        let dvm = std::fs::read(tmp.join(format!(
            "{}.dvm",
            per_field_segment("_merged_all", DOC_VALUES_FORMAT_NAME)
        )))
        .unwrap();
        let merged_field_infos = field_infos::FieldInfos {
            fields: vec![numeric_field("body", 0)],
        };
        let (_v, dv_meta) = doc_values::parse_meta(
            &dvm,
            &merged_id,
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            &merged_field_infos,
        )
        .unwrap();
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
        let nvd = std::fs::read(tmp.join("_merged_all.nvd")).unwrap();
        let nvm = std::fs::read(tmp.join("_merged_all.nvm")).unwrap();
        let (_v, norms_meta) = norms::parse_meta(&nvm, &merged_id, "").unwrap();
        let norms_entry = norms_meta.entry(0).unwrap();
        let norms_values: Vec<i64> = (0..3)
            .map(|d| norms::norm_value(&nvd, norms_entry, d).unwrap().unwrap())
            .collect();
        assert_eq!(norms_values, vec![1, 2, 3]);

        // Term vectors.
        let tvd = std::fs::read(tmp.join("_merged_all.tvd")).unwrap();
        let tvx = std::fs::read(tmp.join("_merged_all.tvx")).unwrap();
        let tvm = std::fs::read(tmp.join("_merged_all.tvm")).unwrap();
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
        let si_bytes = std::fs::read(tmp.join("_merged_all.si")).unwrap();
        let si = segment_info::parse(&si_bytes, &merged_id).unwrap();
        for ext in [
            "fdt", "fdx", "fdm", "fnm", "nvm", "nvd", "tvd", "tvx", "tvm",
        ] {
            let name = format!("_merged_all.{ext}");
            assert!(si.files.contains(&name), "missing {name} in .si files list");
        }
        // Doc values are a per-field format, so their files carry the
        // `_<format>_<suffix>` segment name real Lucene resolves them by.
        for ext in ["dvm", "dvd", "dvs"] {
            let name = format!(
                "{}.{ext}",
                per_field_segment("_merged_all", DOC_VALUES_FORMAT_NAME)
            );
            assert!(si.files.contains(&name), "missing {name} in .si files list");
        }
    }

    #[test]
    fn full_round_trip_merges_stored_fields_binary_doc_values_norms_and_term_vectors_together() {
        // Same shape as
        // `full_round_trip_merges_stored_fields_doc_values_norms_and_term_vectors_together`
        // above, but with a BINARY doc-values field instead of NUMERIC
        // (can't combine both in one call -- see
        // `numeric_and_binary_doc_values_in_the_same_call_is_rejected`).
        let seg0_id = [1u8; ID_LENGTH];
        let seg1_id = [2u8; ID_LENGTH];

        let dv0 = flush_binary_dv(0, &[b"pp".to_vec(), b"qq".to_vec()], seg0_id);
        let dv1 = flush_binary_dv(0, &[b"rr".to_vec()], seg1_id);
        let norms0 = flush_norms(0, &[1, 2], seg0_id);
        let norms1 = flush_norms(0, &[3], seg1_id);
        let tv0 = flush_term_vectors(&[tv_doc(0, &[("x", 0)]), tv_doc(0, &[("y", 0)])], seg0_id);
        let tv1 = flush_term_vectors(&[tv_doc(0, &[("z", 0)])], seg1_id);

        let mut field0 = binary_field("body", 0);
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
            numeric_doc_values: &[],
            binary_doc_values: &dv0_source,
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &norms0_source,
            term_vectors: Some(&tv0_reader),
            postings: &[],
            points: &[],
            vectors: None,
        };
        let source1 = MergeSource {
            field_infos: &stored1.fields,
            reader: &reader1,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &dv1_source,
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &norms1_source,
            term_vectors: Some(&tv1_reader),
            postings: &[],
            points: &[],
            vectors: None,
        };

        let merged_id = [9u8; ID_LENGTH];
        merge_stored_only_segments(
            &dir,
            &[source0, source1],
            "_merged_all_bin",
            merged_id,
            "Lucene104",
            version(),
        )
        .unwrap();

        // Stored fields.
        let merged_fdt = std::fs::read(tmp.join("_merged_all_bin.fdt")).unwrap();
        let merged_fdx = std::fs::read(tmp.join("_merged_all_bin.fdx")).unwrap();
        let merged_fdm = std::fs::read(tmp.join("_merged_all_bin.fdm")).unwrap();
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
        let dvd = std::fs::read(tmp.join(format!(
            "{}.dvd",
            per_field_segment("_merged_all_bin", DOC_VALUES_FORMAT_NAME)
        )))
        .unwrap();
        let dvm = std::fs::read(tmp.join(format!(
            "{}.dvm",
            per_field_segment("_merged_all_bin", DOC_VALUES_FORMAT_NAME)
        )))
        .unwrap();
        let merged_field_infos = field_infos::FieldInfos {
            fields: vec![binary_field("body", 0)],
        };
        let (_v, dv_meta) = doc_values::parse_meta(
            &dvm,
            &merged_id,
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            &merged_field_infos,
        )
        .unwrap();
        let dv_entry = dv_meta.binary_entry(0).unwrap();
        let dv_values: Vec<Vec<u8>> = (0..3)
            .map(|d| {
                doc_values::binary_value(&dvd, dv_entry, d)
                    .unwrap()
                    .unwrap()
                    .to_vec()
            })
            .collect();
        assert_eq!(
            dv_values,
            vec![b"pp".to_vec(), b"qq".to_vec(), b"rr".to_vec()]
        );

        // Norms.
        let nvd = std::fs::read(tmp.join("_merged_all_bin.nvd")).unwrap();
        let nvm = std::fs::read(tmp.join("_merged_all_bin.nvm")).unwrap();
        let (_v, norms_meta) = norms::parse_meta(&nvm, &merged_id, "").unwrap();
        let norms_entry = norms_meta.entry(0).unwrap();
        let norms_values: Vec<i64> = (0..3)
            .map(|d| norms::norm_value(&nvd, norms_entry, d).unwrap().unwrap())
            .collect();
        assert_eq!(norms_values, vec![1, 2, 3]);

        // Term vectors.
        let tvd = std::fs::read(tmp.join("_merged_all_bin.tvd")).unwrap();
        let tvx = std::fs::read(tmp.join("_merged_all_bin.tvx")).unwrap();
        let tvm = std::fs::read(tmp.join("_merged_all_bin.tvm")).unwrap();
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
        let si_bytes = std::fs::read(tmp.join("_merged_all_bin.si")).unwrap();
        let si = segment_info::parse(&si_bytes, &merged_id).unwrap();
        for ext in [
            "fdt", "fdx", "fdm", "fnm", "nvm", "nvd", "tvd", "tvx", "tvm",
        ] {
            let name = format!("_merged_all_bin.{ext}");
            assert!(si.files.contains(&name), "missing {name} in .si files list");
        }
        // Doc values are a per-field format, so their files carry the
        // `_<format>_<suffix>` segment name real Lucene resolves them by.
        for ext in ["dvm", "dvd", "dvs"] {
            let name = format!(
                "{}.{ext}",
                per_field_segment("_merged_all_bin", DOC_VALUES_FORMAT_NAME)
            );
            assert!(si.files.contains(&name), "missing {name} in .si files list");
        }
    }

    #[test]
    fn full_round_trip_merges_stored_fields_sorted_doc_values_norms_and_term_vectors_together() {
        // Same shape as
        // `full_round_trip_merges_stored_fields_binary_doc_values_norms_and_term_vectors_together`
        // above, but with a SORTED doc-values field (with an overlapping
        // term across sources, to exercise dictionary dedup end to end
        // alongside stored fields/norms/term vectors in one real merge call).
        let seg0_id = [1u8; ID_LENGTH];
        let seg1_id = [2u8; ID_LENGTH];

        let dv0 = flush_sorted_dv(0, &[b"red".to_vec(), b"blue".to_vec()], seg0_id);
        let dv1 = flush_sorted_dv(0, &[b"red".to_vec()], seg1_id);
        let norms0 = flush_norms(0, &[1, 2], seg0_id);
        let norms1 = flush_norms(0, &[3], seg1_id);
        let tv0 = flush_term_vectors(&[tv_doc(0, &[("x", 0)]), tv_doc(0, &[("y", 0)])], seg0_id);
        let tv1 = flush_term_vectors(&[tv_doc(0, &[("z", 0)])], seg1_id);

        let mut field0 = sorted_field("color", 0);
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
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &dv0_source,
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &norms0_source,
            term_vectors: Some(&tv0_reader),
            postings: &[],
            points: &[],
            vectors: None,
        };
        let source1 = MergeSource {
            field_infos: &stored1.fields,
            reader: &reader1,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &dv1_source,
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &norms1_source,
            term_vectors: Some(&tv1_reader),
            postings: &[],
            points: &[],
            vectors: None,
        };

        let merged_id = [9u8; ID_LENGTH];
        merge_stored_only_segments(
            &dir,
            &[source0, source1],
            "_merged_all_sorted",
            merged_id,
            "Lucene104",
            version(),
        )
        .unwrap();

        // Stored fields.
        let merged_fdt = std::fs::read(tmp.join("_merged_all_sorted.fdt")).unwrap();
        let merged_fdx = std::fs::read(tmp.join("_merged_all_sorted.fdx")).unwrap();
        let merged_fdm = std::fs::read(tmp.join("_merged_all_sorted.fdm")).unwrap();
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

        // Doc values: resolved through the unmodified reader stack, checked
        // against the actual expected term per doc (not just ordinal shape)
        // -- "red" (docs 0 and 2, from different sources) must dedupe to the
        // same merged dictionary entry.
        let dvd = std::fs::read(tmp.join(format!(
            "{}.dvd",
            per_field_segment("_merged_all_sorted", DOC_VALUES_FORMAT_NAME)
        )))
        .unwrap();
        let dvm = std::fs::read(tmp.join(format!(
            "{}.dvm",
            per_field_segment("_merged_all_sorted", DOC_VALUES_FORMAT_NAME)
        )))
        .unwrap();
        let terms = read_back_sorted_terms(&dvm, &dvd, 0, 3);
        assert_eq!(
            terms,
            vec![b"red".to_vec(), b"blue".to_vec(), b"red".to_vec()]
        );

        // Norms.
        let nvd = std::fs::read(tmp.join("_merged_all_sorted.nvd")).unwrap();
        let nvm = std::fs::read(tmp.join("_merged_all_sorted.nvm")).unwrap();
        let (_v, norms_meta) = norms::parse_meta(&nvm, &merged_id, "").unwrap();
        let norms_entry = norms_meta.entry(0).unwrap();
        let norms_values: Vec<i64> = (0..3)
            .map(|d| norms::norm_value(&nvd, norms_entry, d).unwrap().unwrap())
            .collect();
        assert_eq!(norms_values, vec![1, 2, 3]);

        // Term vectors.
        let tvd = std::fs::read(tmp.join("_merged_all_sorted.tvd")).unwrap();
        let tvx = std::fs::read(tmp.join("_merged_all_sorted.tvx")).unwrap();
        let tvm = std::fs::read(tmp.join("_merged_all_sorted.tvm")).unwrap();
        let tv_reader = term_vectors::open(&tvd, &tvx, &tvm, &merged_id, "").unwrap();
        assert_eq!(tv_reader.max_doc(), 3);
        let tv_terms: Vec<Vec<u8>> = (0..3)
            .map(|d| {
                tv_reader.document(d).unwrap().unwrap().fields[0].terms[0]
                    .term
                    .clone()
            })
            .collect();
        assert_eq!(tv_terms, vec![b"x".to_vec(), b"y".to_vec(), b"z".to_vec()]);

        // Segment info lists every file.
        let si_bytes = std::fs::read(tmp.join("_merged_all_sorted.si")).unwrap();
        let si = segment_info::parse(&si_bytes, &merged_id).unwrap();
        for ext in [
            "fdt", "fdx", "fdm", "fnm", "nvm", "nvd", "tvd", "tvx", "tvm",
        ] {
            let name = format!("_merged_all_sorted.{ext}");
            assert!(si.files.contains(&name), "missing {name} in .si files list");
        }
        // Doc values are a per-field format, so their files carry the
        // `_<format>_<suffix>` segment name real Lucene resolves them by.
        for ext in ["dvm", "dvd", "dvs"] {
            let name = format!(
                "{}.{ext}",
                per_field_segment("_merged_all_sorted", DOC_VALUES_FORMAT_NAME)
            );
            assert!(si.files.contains(&name), "missing {name} in .si files list");
        }
    }

    // --- merge_sorted_stored_only_segments (k-way sort-preserving merge) ---

    /// Reads back the merged segment's stored "id" field (a String) for every
    /// doc, in doc order -- the assertion helper every k-way-merge test below
    /// uses to confirm both order and content.
    fn read_merged_ids(
        tmp: &std::path::Path,
        segment_name: &str,
        segment_id: [u8; ID_LENGTH],
    ) -> Vec<String> {
        let fdt = std::fs::read(tmp.join(format!("{segment_name}.fdt"))).unwrap();
        let fdx = std::fs::read(tmp.join(format!("{segment_name}.fdx"))).unwrap();
        let fdm = std::fs::read(tmp.join(format!("{segment_name}.fdm"))).unwrap();
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
        let si_bytes = std::fs::read(tmp.join("_merged_sorted.si")).unwrap();
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

        let fdt = std::fs::read(tmp.join("_merged_payload.fdt")).unwrap();
        let fdx = std::fs::read(tmp.join("_merged_payload.fdx")).unwrap();
        let fdm = std::fs::read(tmp.join("_merged_payload.fdm")).unwrap();
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
    fn doc_values_norms_and_term_vectors_follow_the_sort_key_not_source_concatenation() {
        // Task #205 (the doc-values/norms/term-vectors long-tail item from
        // PLAN.md's Phase 8 backlog): confirms `merge_sorted_stored_only_
        // segments` physically reorders BINARY doc-values, norms, and term
        // vectors by sort key -- not just stored fields -- exactly like
        // `three_sources_k_way_merge_by_sort_key` proves for stored fields
        // above. Interleaved sort keys mean naive source concatenation
        // would produce a visibly wrong order at the first cross-source
        // boundary; a real k-way merge must interleave.
        //
        // Source 0 (already sorted by "num"): docs with keys 10, 30.
        // Source 1 (already sorted by "num"): docs with keys 20, 40.
        // Correct global order: 10, 20, 30, 40 (source0/1/0/1 interleaved).
        let seg0_id = [1u8; ID_LENGTH];
        let seg1_id = [2u8; ID_LENGTH];

        // BINARY doc-values: value encodes the key ("v10", "v30" / "v20",
        // "v40") so the merged order can be read straight back off it.
        let bin0 = flush_binary_dv(1, &[b"v10".to_vec(), b"v30".to_vec()], seg0_id);
        let bin1 = flush_binary_dv(1, &[b"v20".to_vec(), b"v40".to_vec()], seg1_id);

        // Norms: same encoding scheme, divided by 10 (1, 3 / 2, 4).
        let norms0 = flush_norms(2, &[1, 3], seg0_id);
        let norms1 = flush_norms(2, &[2, 4], seg1_id);

        // Term vectors: one field, one term "t", whose position encodes the
        // key divided by 10 (same scheme as norms) -- `tv_doc`'s second
        // tuple element is a position, not a freq.
        let tv0 = flush_term_vectors(&[tv_doc(3, &[("t", 1)]), tv_doc(3, &[("t", 3)])], seg0_id);
        let tv1 = flush_term_vectors(&[tv_doc(3, &[("t", 2)]), tv_doc(3, &[("t", 4)])], seg1_id);

        let fields = vec![
            field("id", 0),
            binary_field("bin", 1),
            norms_field("nrm", 2),
            tv_field("tv", 3),
        ];
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let stored0 = flush(
            &dir,
            &tmp,
            "_0",
            seg0_id,
            &fields,
            &[doc_with(0, "10"), doc_with(0, "30")],
        );
        let stored1 = flush(
            &dir,
            &tmp,
            "_1",
            seg1_id,
            &fields,
            &[doc_with(0, "20"), doc_with(0, "40")],
        );
        let reader0 = open_reader(&stored0);
        let reader1 = open_reader(&stored1);
        let tv0_reader = tv0.reader();
        let tv1_reader = tv1.reader();

        let bin0_source = [bin0.source()];
        let bin1_source = [bin1.source()];
        let norms0_source = [norms0.source()];
        let norms1_source = [norms1.source()];

        let source0 = MergeSource {
            field_infos: &stored0.fields,
            reader: &reader0,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &bin0_source,
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &norms0_source,
            term_vectors: Some(&tv0_reader),
            postings: &[],
            points: &[],
            vectors: None,
        };
        let source1 = MergeSource {
            field_infos: &stored1.fields,
            reader: &reader1,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &bin1_source,
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &norms1_source,
            term_vectors: Some(&tv1_reader),
            postings: &[],
            points: &[],
            vectors: None,
        };

        let keys0: Vec<Option<i64>> = vec![Some(10), Some(30)];
        let keys1: Vec<Option<i64>> = vec![Some(20), Some(40)];
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
            &[source0, source1],
            &sort_fields,
            "_merged_dv_norms_tv_sorted",
            merged_id,
            "Lucene104",
            version(),
        )
        .unwrap();

        // Stored "id" field confirms the expected doc order up front.
        let ids = read_merged_ids(&tmp, "_merged_dv_norms_tv_sorted", merged_id);
        assert_eq!(ids, vec!["10", "20", "30", "40"]);

        // BINARY doc-values: must read back "v10","v20","v30","v40" -- the
        // sorted order -- not "v10","v30","v20","v40" (source concatenation).
        let dvd = std::fs::read(tmp.join(format!(
            "{}.dvd",
            per_field_segment("_merged_dv_norms_tv_sorted", DOC_VALUES_FORMAT_NAME)
        )))
        .unwrap();
        let dvm = std::fs::read(tmp.join(format!(
            "{}.dvm",
            per_field_segment("_merged_dv_norms_tv_sorted", DOC_VALUES_FORMAT_NAME)
        )))
        .unwrap();
        let merged_field_infos = field_infos::FieldInfos {
            fields: vec![binary_field("bin", 1)],
        };
        let (_v, meta) = doc_values::parse_meta(
            &dvm,
            &merged_id,
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            &merged_field_infos,
        )
        .unwrap();
        let entry = meta.binary_entry(1).unwrap();
        let bin_values: Vec<Vec<u8>> = (0..4)
            .map(|d| {
                doc_values::binary_value(&dvd, entry, d)
                    .unwrap()
                    .unwrap()
                    .to_vec()
            })
            .collect();
        assert_eq!(
            bin_values,
            vec![
                b"v10".to_vec(),
                b"v20".to_vec(),
                b"v30".to_vec(),
                b"v40".to_vec(),
            ]
        );

        // Norms: must read back 1,2,3,4 (sorted), not 1,3,2,4 (concatenated).
        let nvd = std::fs::read(tmp.join("_merged_dv_norms_tv_sorted.nvd")).unwrap();
        let nvm = std::fs::read(tmp.join("_merged_dv_norms_tv_sorted.nvm")).unwrap();
        let (_v, parsed) = norms::parse_meta(&nvm, &merged_id, "").unwrap();
        let norm_entry = parsed.entry(2).unwrap();
        let norm_values: Vec<i64> = (0..4)
            .map(|d| norms::norm_value(&nvd, norm_entry, d).unwrap().unwrap())
            .collect();
        assert_eq!(norm_values, vec![1, 2, 3, 4]);

        // Term vectors: doc 0's term "t" must have position 1 (key 10),
        // doc 1's position 2 (key 20), doc 2's position 3 (key 30), doc 3's
        // position 4 (key 40) -- sorted order, not concatenation order
        // (which would put position 3 at doc 1 and position 2 at doc 2).
        let tvd = std::fs::read(tmp.join("_merged_dv_norms_tv_sorted.tvd")).unwrap();
        let tvx = std::fs::read(tmp.join("_merged_dv_norms_tv_sorted.tvx")).unwrap();
        let tvm = std::fs::read(tmp.join("_merged_dv_norms_tv_sorted.tvm")).unwrap();
        let tv_reader = term_vectors::open(&tvd, &tvx, &tvm, &merged_id, "").unwrap();
        let tv_positions: Vec<i32> = (0..4)
            .map(|d| {
                tv_reader.document(d).unwrap().unwrap().fields[0].terms[0]
                    .positions
                    .as_ref()
                    .unwrap()[0]
            })
            .collect();
        assert_eq!(tv_positions, vec![1, 2, 3, 4]);
    }

    #[test]
    fn numeric_sorted_sorted_numeric_and_sorted_set_doc_values_follow_the_sort_key() {
        // Sibling of `doc_values_norms_and_term_vectors_follow_the_sort_key_
        // not_source_concatenation` above, covering the four doc-values
        // types that test doesn't exercise (NUMERIC/SORTED/SORTED_NUMERIC/
        // SORTED_SET) -- reusing the exact same interleaved-sort-key setup
        // (source0: keys 10,30; source1: keys 20,40; correct order
        // 10,20,30,40) so a reorder bug in any one of these four merge
        // helpers specifically would fail here, not just leave them "not
        // dropped" (which the pre-existing concatenation-merge tests for
        // these types already prove, but can't catch a reorder bug since
        // they merge via `merge_stored_only_segments`, not the sort-aware
        // k-way path).
        let seg0_id = [1u8; ID_LENGTH];
        let seg1_id = [2u8; ID_LENGTH];

        let num0 = flush_numeric_dv(1, &[10, 30], seg0_id);
        let num1 = flush_numeric_dv(1, &[20, 40], seg1_id);
        let sorted0 = flush_sorted_dv(2, &[b"v10".to_vec(), b"v30".to_vec()], seg0_id);
        let sorted1 = flush_sorted_dv(2, &[b"v20".to_vec(), b"v40".to_vec()], seg1_id);
        let sn0 = flush_sorted_numeric_dv(3, &[vec![10], vec![30]], seg0_id);
        let sn1 = flush_sorted_numeric_dv(3, &[vec![20], vec![40]], seg1_id);
        let ss0 = flush_sorted_set_dv(4, &[vec![b"v10".to_vec()], vec![b"v30".to_vec()]], seg0_id);
        let ss1 = flush_sorted_set_dv(4, &[vec![b"v20".to_vec()], vec![b"v40".to_vec()]], seg1_id);

        let fields = vec![
            field("id", 0),
            numeric_field("num", 1),
            sorted_field("srt", 2),
            sorted_numeric_field("sn", 3),
            sorted_set_field("ss", 4),
        ];
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let stored0 = flush(
            &dir,
            &tmp,
            "_0",
            seg0_id,
            &fields,
            &[doc_with(0, "10"), doc_with(0, "30")],
        );
        let stored1 = flush(
            &dir,
            &tmp,
            "_1",
            seg1_id,
            &fields,
            &[doc_with(0, "20"), doc_with(0, "40")],
        );
        let reader0 = open_reader(&stored0);
        let reader1 = open_reader(&stored1);

        let num0_source = [num0.source()];
        let num1_source = [num1.source()];
        let sorted0_source = [sorted0.source()];
        let sorted1_source = [sorted1.source()];
        let sn0_source = [sn0.source()];
        let sn1_source = [sn1.source()];
        let ss0_source = [ss0.source()];
        let ss1_source = [ss1.source()];

        let keys0: Vec<Option<i64>> = vec![Some(10), Some(30)];
        let keys1: Vec<Option<i64>> = vec![Some(20), Some(40)];
        let per_source_keys: Vec<&[Option<i64>]> = vec![&keys0, &keys1];
        let sort_fields = vec![MergeSortKeySpec {
            field: "id",
            reverse: false,
            missing: SortMissingValue::Last,
            per_source_keys: &per_source_keys,
        }];

        // This port's doc-values merge is single-doc-values-type-per-merge
        // (pre-existing limitation, unrelated to sorting -- see the
        // `MultipleDocValuesTypesInOneMerge` error), so each type is merged
        // in its own separate call rather than all four in one `MergeSource`
        // pair; each still shares the same interleaved-sort-key setup.
        let merged_id = [9u8; ID_LENGTH];

        let source0_num = MergeSource {
            field_infos: &stored0.fields,
            reader: &reader0,
            live_docs: None,
            numeric_doc_values: &num0_source,
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &[],
            vectors: None,
        };
        let source1_num = MergeSource {
            field_infos: &stored1.fields,
            reader: &reader1,
            live_docs: None,
            numeric_doc_values: &num1_source,
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &[],
            vectors: None,
        };
        merge_sorted_stored_only_segments(
            &dir,
            &[source0_num, source1_num],
            &sort_fields,
            "_merged_num_sorted",
            merged_id,
            "Lucene104",
            version(),
        )
        .unwrap();
        let ids = read_merged_ids(&tmp, "_merged_num_sorted", merged_id);
        assert_eq!(ids, vec!["10", "20", "30", "40"]);
        let dvd = std::fs::read(tmp.join(format!(
            "{}.dvd",
            per_field_segment("_merged_num_sorted", DOC_VALUES_FORMAT_NAME)
        )))
        .unwrap();
        let dvm = std::fs::read(tmp.join(format!(
            "{}.dvm",
            per_field_segment("_merged_num_sorted", DOC_VALUES_FORMAT_NAME)
        )))
        .unwrap();
        let merged_numeric_field_infos = field_infos::FieldInfos {
            fields: vec![numeric_field("x", 1)],
        };
        let (_v, meta) = doc_values::parse_meta(
            &dvm,
            &merged_id,
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            &merged_numeric_field_infos,
        )
        .unwrap();
        let num_entry = meta.numeric_entry(1).unwrap();
        let numeric_values: Vec<i64> = (0..4)
            .map(|d| {
                doc_values::numeric_value(&dvd, num_entry, d)
                    .unwrap()
                    .unwrap()
            })
            .collect();
        // NUMERIC: 10,20,30,40 sorted, not 10,30,20,40 concatenated.
        assert_eq!(numeric_values, vec![10, 20, 30, 40]);

        let source0_sorted = MergeSource {
            field_infos: &stored0.fields,
            reader: &reader0,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &sorted0_source,
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &[],
            vectors: None,
        };
        let source1_sorted = MergeSource {
            field_infos: &stored1.fields,
            reader: &reader1,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &sorted1_source,
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &[],
            vectors: None,
        };
        merge_sorted_stored_only_segments(
            &dir,
            &[source0_sorted, source1_sorted],
            &sort_fields,
            "_merged_sorted_sorted",
            merged_id,
            "Lucene104",
            version(),
        )
        .unwrap();
        let dvd = std::fs::read(tmp.join(format!(
            "{}.dvd",
            per_field_segment("_merged_sorted_sorted", DOC_VALUES_FORMAT_NAME)
        )))
        .unwrap();
        let dvm = std::fs::read(tmp.join(format!(
            "{}.dvm",
            per_field_segment("_merged_sorted_sorted", DOC_VALUES_FORMAT_NAME)
        )))
        .unwrap();
        // SORTED: "v10","v20","v30","v40" sorted, not concatenated.
        let sorted_terms = read_back_sorted_terms(&dvm, &dvd, 2, 4);
        assert_eq!(
            sorted_terms,
            vec![
                b"v10".to_vec(),
                b"v20".to_vec(),
                b"v30".to_vec(),
                b"v40".to_vec(),
            ]
        );

        let source0_sn = MergeSource {
            field_infos: &stored0.fields,
            reader: &reader0,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &sn0_source,
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &[],
            vectors: None,
        };
        let source1_sn = MergeSource {
            field_infos: &stored1.fields,
            reader: &reader1,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &sn1_source,
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &[],
            vectors: None,
        };
        merge_sorted_stored_only_segments(
            &dir,
            &[source0_sn, source1_sn],
            &sort_fields,
            "_merged_sn_sorted",
            merged_id,
            "Lucene104",
            version(),
        )
        .unwrap();
        let dvd = std::fs::read(tmp.join(format!(
            "{}.dvd",
            per_field_segment("_merged_sn_sorted", DOC_VALUES_FORMAT_NAME)
        )))
        .unwrap();
        let dvm = std::fs::read(tmp.join(format!(
            "{}.dvm",
            per_field_segment("_merged_sn_sorted", DOC_VALUES_FORMAT_NAME)
        )))
        .unwrap();
        // SORTED_NUMERIC: [10],[20],[30],[40] sorted, not concatenated.
        let sn_values = read_back_sorted_numeric_values(&dvm, &dvd, 3, 4);
        assert_eq!(sn_values, vec![vec![10], vec![20], vec![30], vec![40]]);

        let source0_ss = MergeSource {
            field_infos: &stored0.fields,
            reader: &reader0,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &ss0_source,
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &[],
            vectors: None,
        };
        let source1_ss = MergeSource {
            field_infos: &stored1.fields,
            reader: &reader1,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &ss1_source,
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &[],
            vectors: None,
        };
        merge_sorted_stored_only_segments(
            &dir,
            &[source0_ss, source1_ss],
            &sort_fields,
            "_merged_ss_sorted",
            merged_id,
            "Lucene104",
            version(),
        )
        .unwrap();
        let dvd = std::fs::read(tmp.join(format!(
            "{}.dvd",
            per_field_segment("_merged_ss_sorted", DOC_VALUES_FORMAT_NAME)
        )))
        .unwrap();
        let dvm = std::fs::read(tmp.join(format!(
            "{}.dvm",
            per_field_segment("_merged_ss_sorted", DOC_VALUES_FORMAT_NAME)
        )))
        .unwrap();
        // SORTED_SET: ["v10"],["v20"],["v30"],["v40"] sorted, not concatenated.
        let ss_values = read_back_sorted_set_values(&dvm, &dvd, 4, 4);
        assert_eq!(
            ss_values,
            vec![
                vec![b"v10".to_vec()],
                vec![b"v20".to_vec()],
                vec![b"v30".to_vec()],
                vec![b"v40".to_vec()],
            ]
        );
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

        let fdt = std::fs::read(tmp.join("_merged_empty_sorted.fdt")).unwrap();
        let fdx = std::fs::read(tmp.join("_merged_empty_sorted.fdx")).unwrap();
        let fdm = std::fs::read(tmp.join("_merged_empty_sorted.fdm")).unwrap();
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

        let merged_fnm = std::fs::read(tmp.join("_merged_reconcile.fnm")).unwrap();
        let merged_fields = lucene_codecs::field_infos::parse(&merged_fnm, &merged_id, "").unwrap();
        let id_number = merged_fields
            .fields
            .iter()
            .find(|f| f.name == "id")
            .unwrap()
            .number;

        let fdt = std::fs::read(tmp.join("_merged_reconcile.fdt")).unwrap();
        let fdx = std::fs::read(tmp.join("_merged_reconcile.fdx")).unwrap();
        let fdm = std::fs::read(tmp.join("_merged_reconcile.fdm")).unwrap();
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

    // --- SORTED_NUMERIC ---

    fn sorted_numeric_field(name: &str, number: i32) -> FieldInfo {
        let mut f = field(name, number);
        f.doc_values_type = DocValuesType::SortedNumeric;
        f
    }

    /// Same idea as [`FlushedNumericDv`], for SORTED_NUMERIC doc values --
    /// `values` is one doc's whole value list per entry (dense,
    /// `values.len() == max_doc`, every doc non-empty), exactly what
    /// [`doc_values::write_single_dense_sorted_numeric_field`] takes.
    struct FlushedSortedNumericDv {
        data: Vec<u8>,
        entry: SortedNumericEntry,
    }

    fn flush_sorted_numeric_dv(
        field_number: i32,
        values: &[Vec<i64>],
        segment_id: [u8; ID_LENGTH],
    ) -> FlushedSortedNumericDv {
        let (meta, data, _skip) = doc_values::write_single_dense_sorted_numeric_field(
            field_number,
            values,
            &segment_id,
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        )
        .unwrap();
        let field_infos = field_infos::FieldInfos {
            fields: vec![sorted_numeric_field("x", field_number)],
        };
        let (_version, parsed) = doc_values::parse_meta(
            &meta,
            &segment_id,
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            &field_infos,
        )
        .unwrap();
        let entry = parsed.sorted_numeric_entry(field_number).unwrap().clone();
        FlushedSortedNumericDv { data, entry }
    }

    impl FlushedSortedNumericDv {
        fn source(&self) -> SourceSortedNumericDocValues<'_> {
            SourceSortedNumericDocValues {
                data: &self.data,
                entry: self.entry.clone(),
            }
        }
    }

    /// Resolves every merged doc's whole value list, doc by doc, through the
    /// *unmodified* reader stack (`parse_meta` + `sorted_numeric_values`) --
    /// the correctness check that a doc's full multi-value list survived the
    /// merge, not just its value count.
    fn read_back_sorted_numeric_values(
        dvm: &[u8],
        dvd: &[u8],
        field_number: i32,
        doc_count: i32,
    ) -> Vec<Vec<i64>> {
        let merged_field_infos = field_infos::FieldInfos {
            fields: vec![sorted_numeric_field("x", field_number)],
        };
        let (_v, meta) = doc_values::parse_meta(
            dvm,
            &[9u8; ID_LENGTH],
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            &merged_field_infos,
        )
        .unwrap();
        let entry = meta.sorted_numeric_entry(field_number).unwrap();
        (0..doc_count)
            .map(|d| doc_values::sorted_numeric_values(dvd, entry, d).unwrap())
            .collect()
    }

    #[test]
    fn sorted_numeric_doc_values_merge_across_two_sources_with_deletions() {
        // Source 0: 2 docs, doc 0 has 2 values, doc 1 (deleted) has 1 value --
        // confirms a surviving multi-value doc keeps *both* of its values
        // after merge, not just its first/last.
        let seg0_id = [1u8; ID_LENGTH];
        let seg1_id = [2u8; ID_LENGTH];
        let dv0 = flush_sorted_numeric_dv(0, &[vec![10, 11], vec![20]], seg0_id);
        // Source 1: 1 doc, single value.
        let dv1 = flush_sorted_numeric_dv(0, &[vec![30]], seg1_id);

        let fields = vec![sorted_numeric_field("nums", 0)];
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
        live0.set(0); // keep doc 0 (values [10, 11]), drop doc 1 ([20])

        let dv0_source = [dv0.source()];
        let dv1_source = [dv1.source()];
        let source0 = MergeSource {
            field_infos: &stored0.fields,
            reader: &reader0,
            live_docs: Some(&live0),
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &dv0_source,
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &[],
            vectors: None,
        };
        let source1 = MergeSource {
            field_infos: &stored1.fields,
            reader: &reader1,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &dv1_source,
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &[],
            vectors: None,
        };

        merge_stored_only_segments(
            &dir,
            &[source0, source1],
            "_merged_sorted_numeric",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();

        let dvm = std::fs::read(tmp.join(format!(
            "{}.dvm",
            per_field_segment("_merged_sorted_numeric", DOC_VALUES_FORMAT_NAME)
        )))
        .unwrap();
        let dvd = std::fs::read(tmp.join(format!(
            "{}.dvd",
            per_field_segment("_merged_sorted_numeric", DOC_VALUES_FORMAT_NAME)
        )))
        .unwrap();
        let values = read_back_sorted_numeric_values(&dvm, &dvd, 0, 2);
        assert_eq!(values, vec![vec![10, 11], vec![30]]);
    }

    #[test]
    fn sorted_numeric_doc_values_missing_in_a_live_contributing_source_is_an_error() {
        let seg0_id = [1u8; ID_LENGTH];
        let dv0 = flush_sorted_numeric_dv(0, &[vec![10]], seg0_id);
        let fields = vec![sorted_numeric_field("nums", 0)];
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
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &dv0_source,
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &[],
            vectors: None,
        };
        // Source 1 has live docs but no SORTED_NUMERIC doc-values entry at
        // all for field "nums".
        let source1 = MergeSource::stored_only(&stored1.fields, &reader1, None);

        let result = merge_stored_only_segments(
            &dir,
            &[source0, source1],
            "_merged_sorted_numeric_err",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        );
        assert!(matches!(
            result,
            Err(Error::SortedNumericDocValuesFieldMissingInSource {
                merged_field_number: 0
            })
        ));
    }

    #[test]
    fn two_sorted_numeric_doc_values_fields_land_in_one_merged_dvm() {
        let seg0_id = [1u8; ID_LENGTH];
        let dv_a = flush_sorted_numeric_dv(0, &[vec![1]], seg0_id);
        let dv_b = flush_sorted_numeric_dv(1, &[vec![2]], seg0_id);
        let fields = vec![sorted_numeric_field("a", 0), sorted_numeric_field("b", 1)];
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
        let sorted_numeric = vec![dv_a.source(), dv_b.source()];
        let source0 = MergeSource {
            field_infos: &stored0.fields,
            reader: &reader0,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &sorted_numeric,
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &[],
            vectors: None,
        };

        merge_stored_only_segments(
            &dir,
            &[source0],
            "_merged_sndv_two",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();

        let (dvd, meta) = open_merged_doc_values(
            tmp.path(),
            "_merged_sndv_two",
            [9u8; ID_LENGTH],
            vec![sorted_numeric_field("a", 0), sorted_numeric_field("b", 1)],
        );
        assert_eq!(
            doc_values::sorted_numeric_values(&dvd, meta.sorted_numeric_entry(0).unwrap(), 0)
                .unwrap(),
            vec![1]
        );
        assert_eq!(
            doc_values::sorted_numeric_values(&dvd, meta.sorted_numeric_entry(1).unwrap(), 0)
                .unwrap(),
            vec![2]
        );
    }

    // --- SORTED_SET ---

    fn sorted_set_field(name: &str, number: i32) -> FieldInfo {
        let mut f = field(name, number);
        f.doc_values_type = DocValuesType::SortedSet;
        f
    }

    /// Same idea as [`FlushedSortedDv`], for SORTED_SET doc values --
    /// `values` is one doc's whole (possibly multi-valued, possibly
    /// duplicate-containing) raw value set per entry (dense,
    /// `values.len() == max_doc`, every doc non-empty), exactly what
    /// [`doc_values::write_single_dense_sorted_set_field`] takes.
    struct FlushedSortedSetDv {
        data: Vec<u8>,
        entry: SortedSetEntry,
    }

    fn flush_sorted_set_dv(
        field_number: i32,
        values: &[Vec<Vec<u8>>],
        segment_id: [u8; ID_LENGTH],
    ) -> FlushedSortedSetDv {
        let max_doc = values.len() as i32;
        let (meta, data, _skip) = doc_values::write_single_dense_sorted_set_field(
            field_number,
            values,
            max_doc,
            &segment_id,
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        )
        .unwrap();
        let field_infos = field_infos::FieldInfos {
            fields: vec![sorted_set_field("x", field_number)],
        };
        let (_version, parsed) = doc_values::parse_meta(
            &meta,
            &segment_id,
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            &field_infos,
        )
        .unwrap();
        let entry = parsed.sorted_set_entry(field_number).unwrap().clone();
        FlushedSortedSetDv { data, entry }
    }

    impl FlushedSortedSetDv {
        fn source(&self) -> SourceSortedSetDocValues<'_> {
            SourceSortedSetDocValues {
                data: &self.data,
                entry: self.entry.clone(),
            }
        }
    }

    /// Resolves every merged doc's full (sorted, deduped) value set, doc by
    /// doc, through the *unmodified* reader stack ([`sorted_set_doc_ordinals`]
    /// and [`sorted_set_source_dict`], the same helpers
    /// [`merge_sorted_set_doc_values`] itself uses) -- the critical
    /// correctness check: not just "some valid ordinals", but the actual
    /// right resolved terms per doc, read back exactly as a real caller
    /// would.
    fn read_back_sorted_set_values(
        dvm: &[u8],
        dvd: &[u8],
        field_number: i32,
        doc_count: i32,
    ) -> Vec<Vec<Vec<u8>>> {
        let merged_field_infos = field_infos::FieldInfos {
            fields: vec![sorted_set_field("x", field_number)],
        };
        let (_v, meta) = doc_values::parse_meta(
            dvm,
            &[9u8; ID_LENGTH],
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            &merged_field_infos,
        )
        .unwrap();
        let entry = meta.sorted_set_entry(field_number).unwrap();
        let dict = sorted_set_source_dict(dvd, entry).unwrap();
        (0..doc_count)
            .map(|d| {
                sorted_set_doc_ordinals(dvd, entry, d)
                    .unwrap()
                    .into_iter()
                    .map(|ord| dict[ord as usize].clone())
                    .collect()
            })
            .collect()
    }

    #[test]
    fn sorted_set_doc_values_merge_with_overlapping_terms_dedupes_into_one_shared_dictionary_entry()
    {
        // Source 0: one doc with ["red", "blue"]; source 1: one doc with
        // ["red", "green"] -- both sources independently assign "red"
        // ordinal 1 (alphabetically after "blue"/"green" respectively in
        // each source's own 2-term dictionary... actually "blue" < "red" and
        // "green" < "red", so "red" is ordinal 1 in both). Real bug case: if
        // this merge naively concatenated ordinals without resolving to
        // bytes, source 1's "red" could land on a different merged
        // dictionary entry than source 0's "red" purely because they came
        // from different sources -- this test catches that by checking the
        // actual resolved term *sets*, not just dictionary size.
        let seg0_id = [1u8; ID_LENGTH];
        let seg1_id = [2u8; ID_LENGTH];
        let dv0 = flush_sorted_set_dv(0, &[vec![b"red".to_vec(), b"blue".to_vec()]], seg0_id);
        let dv1 = flush_sorted_set_dv(0, &[vec![b"red".to_vec(), b"green".to_vec()]], seg1_id);

        let fields = vec![sorted_set_field("colors", 0)];
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let stored0 = flush(&dir, &tmp, "_0", seg0_id, &fields, &[doc_with(0, "a")]);
        let stored1 = flush(&dir, &tmp, "_1", seg1_id, &fields, &[doc_with(0, "b")]);
        let reader0 = open_reader(&stored0);
        let reader1 = open_reader(&stored1);

        let dv0_source = [dv0.source()];
        let dv1_source = [dv1.source()];
        let source0 = MergeSource {
            field_infos: &stored0.fields,
            reader: &reader0,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &dv0_source,
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &[],
            vectors: None,
        };
        let source1 = MergeSource {
            field_infos: &stored1.fields,
            reader: &reader1,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &dv1_source,
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &[],
            vectors: None,
        };

        merge_stored_only_segments(
            &dir,
            &[source0, source1],
            "_merged_sorted_set_overlap",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();

        let dvm = std::fs::read(tmp.join(format!(
            "{}.dvm",
            per_field_segment("_merged_sorted_set_overlap", DOC_VALUES_FORMAT_NAME)
        )))
        .unwrap();
        let dvd = std::fs::read(tmp.join(format!(
            "{}.dvd",
            per_field_segment("_merged_sorted_set_overlap", DOC_VALUES_FORMAT_NAME)
        )))
        .unwrap();
        let merged_field_infos = field_infos::FieldInfos {
            fields: vec![sorted_set_field("colors", 0)],
        };
        let (_v, meta) = doc_values::parse_meta(
            &dvm,
            &[9u8; ID_LENGTH],
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            &merged_field_infos,
        )
        .unwrap();
        let entry = meta.sorted_set_entry(0).unwrap();
        let dict = sorted_set_source_dict(&dvd, entry).unwrap();
        // "red" is shared across both sources -- the merged dictionary must
        // dedupe it into exactly one entry, so the distinct dictionary size
        // is 3 ("red", "blue", "green"), not 4.
        assert_eq!(dict.len(), 3);

        // And each doc must resolve to its RIGHT value set, not just any
        // valid ordinals -- this is the actual correctness check.
        let mut values = read_back_sorted_set_values(&dvm, &dvd, 0, 2);
        for doc_values in &mut values {
            doc_values.sort();
        }
        assert_eq!(
            values,
            vec![
                {
                    let mut v = vec![b"red".to_vec(), b"blue".to_vec()];
                    v.sort();
                    v
                },
                {
                    let mut v = vec![b"red".to_vec(), b"green".to_vec()];
                    v.sort();
                    v
                },
            ]
        );
    }

    #[test]
    fn sorted_set_doc_values_merge_with_disjoint_terms_contains_all_terms_from_both_sources() {
        let seg0_id = [1u8; ID_LENGTH];
        let seg1_id = [2u8; ID_LENGTH];
        let dv0 = flush_sorted_set_dv(0, &[vec![b"apple".to_vec()]], seg0_id);
        let dv1 = flush_sorted_set_dv(0, &[vec![b"zebra".to_vec()]], seg1_id);

        let fields = vec![sorted_set_field("word", 0)];
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let stored0 = flush(&dir, &tmp, "_0", seg0_id, &fields, &[doc_with(0, "a")]);
        let stored1 = flush(&dir, &tmp, "_1", seg1_id, &fields, &[doc_with(0, "b")]);
        let reader0 = open_reader(&stored0);
        let reader1 = open_reader(&stored1);

        let dv0_source = [dv0.source()];
        let dv1_source = [dv1.source()];
        let source0 = MergeSource {
            field_infos: &stored0.fields,
            reader: &reader0,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &dv0_source,
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &[],
            vectors: None,
        };
        let source1 = MergeSource {
            field_infos: &stored1.fields,
            reader: &reader1,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &dv1_source,
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &[],
            vectors: None,
        };

        merge_stored_only_segments(
            &dir,
            &[source0, source1],
            "_merged_sorted_set_disjoint",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();

        let dvm = std::fs::read(tmp.join(format!(
            "{}.dvm",
            per_field_segment("_merged_sorted_set_disjoint", DOC_VALUES_FORMAT_NAME)
        )))
        .unwrap();
        let dvd = std::fs::read(tmp.join(format!(
            "{}.dvd",
            per_field_segment("_merged_sorted_set_disjoint", DOC_VALUES_FORMAT_NAME)
        )))
        .unwrap();
        let values = read_back_sorted_set_values(&dvm, &dvd, 0, 2);
        assert_eq!(
            values,
            vec![vec![b"apple".to_vec()], vec![b"zebra".to_vec()]]
        );
    }

    #[test]
    fn sorted_set_doc_values_missing_in_a_live_contributing_source_is_an_error() {
        let seg0_id = [1u8; ID_LENGTH];
        let dv0 = flush_sorted_set_dv(0, &[vec![b"x".to_vec()]], seg0_id);
        let fields = vec![sorted_set_field("word", 0)];
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
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &dv0_source,
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &[],
            vectors: None,
        };
        // Source 1 has live docs but no SORTED_SET doc-values entry at all
        // for field "word".
        let source1 = MergeSource::stored_only(&stored1.fields, &reader1, None);

        let result = merge_stored_only_segments(
            &dir,
            &[source0, source1],
            "_merged_sorted_set_err",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        );
        assert!(matches!(
            result,
            Err(Error::SortedSetDocValuesFieldMissingInSource {
                merged_field_number: 0
            })
        ));
    }

    #[test]
    fn two_sorted_set_doc_values_fields_land_in_one_merged_dvm() {
        let seg0_id = [1u8; ID_LENGTH];
        let dv_a = flush_sorted_set_dv(0, &[vec![b"x".to_vec()]], seg0_id);
        let dv_b = flush_sorted_set_dv(1, &[vec![b"y".to_vec()]], seg0_id);
        let fields = vec![sorted_set_field("a", 0), sorted_set_field("b", 1)];
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
        let sorted_set = vec![dv_a.source(), dv_b.source()];
        let source0 = MergeSource {
            field_infos: &stored0.fields,
            reader: &reader0,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &sorted_set,
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &[],
            vectors: None,
        };

        merge_stored_only_segments(
            &dir,
            &[source0],
            "_merged_ssdv_two",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();

        let (dvd, meta) = open_merged_doc_values(
            tmp.path(),
            "_merged_ssdv_two",
            [9u8; ID_LENGTH],
            vec![sorted_set_field("a", 0), sorted_set_field("b", 1)],
        );
        for (field_number, expected) in [(0, &b"x"[..]), (1, &b"y"[..])] {
            let entry = meta.sorted_set_entry(field_number).unwrap();
            let ords = sorted_set_doc_ordinals(&dvd, entry, 0).unwrap();
            let dict = sorted_set_source_dict(&dvd, entry).unwrap();
            assert_eq!(
                ords.iter()
                    .map(|&o| dict[o as usize].clone())
                    .collect::<Vec<_>>(),
                vec![expected.to_vec()]
            );
        }
    }

    // --- the sorted merge's document order, and the doc maps derived from it ---

    /// `build_doc_id_maps` must be the exact inverse of `doc_order`: merged
    /// doc `m` is `doc_order[m] == (s, d)` iff `maps[s][d] == m`.
    ///
    /// This is the single most load-bearing invariant in a sorted merge.
    /// Stored fields, doc values, norms and term vectors are written by
    /// *walking* `doc_order`; postings, points and vectors are written by
    /// *looking up* the map. If the two disagree by so much as one document,
    /// every file is still well-formed, every checksum still valid, every doc
    /// id still in range -- and a term, a point or a vector is attached to a
    /// different document than its stored fields.
    #[test]
    fn the_doc_id_maps_invert_the_doc_order_for_both_merge_orders() {
        let live = vec![vec![0i32, 2, 3], vec![1], vec![], vec![0, 1]];
        let max_doc = [4i32, 3, 2, 2];
        let concat = concat_doc_order(&live);

        // A two-tier sort whose keys interleave the sources, with a missing
        // value and a tie the second tier has to break.
        let rank: Vec<Vec<Option<i64>>> = vec![
            vec![Some(5), Some(9), Some(1), None],
            vec![Some(9), Some(3), Some(7)],
            vec![Some(0), Some(0)],
            vec![Some(1), Some(5)],
        ];
        let tie: Vec<Vec<Option<i64>>> = vec![
            vec![Some(0), Some(1), Some(2), Some(3)],
            vec![Some(0), Some(1), Some(2)],
            vec![Some(0), Some(1)],
            vec![Some(9), Some(8)],
        ];
        let rank_slices: Vec<&[Option<i64>]> = rank.iter().map(|v| v.as_slice()).collect();
        let tie_slices: Vec<&[Option<i64>]> = tie.iter().map(|v| v.as_slice()).collect();
        let specs = vec![
            MergeSortKeySpec {
                field: "rank",
                reverse: true,
                missing: SortMissingValue::Last,
                per_source_keys: &rank_slices,
            },
            MergeSortKeySpec {
                field: "tie",
                reverse: false,
                missing: SortMissingValue::First,
                per_source_keys: &tie_slices,
            },
        ];
        let sorted = sorted_doc_order(&specs, &live);

        for order in [&concat, &sorted] {
            assert_eq!(order.len(), 6, "every live document appears exactly once");
            let maps = build_doc_id_maps(&max_doc, order);
            for (merged, &(src, doc)) in order.iter().enumerate() {
                assert_eq!(
                    mapped_doc_id(&maps[src], doc),
                    Some(merged as i32),
                    "source {src} doc {doc}"
                );
            }
            // Every doc the order does *not* name maps to nothing.
            let named: HashSet<(usize, i32)> = order.iter().copied().collect();
            for (src, &max) in max_doc.iter().enumerate() {
                for doc in 0..max {
                    if !named.contains(&(src, doc)) {
                        assert_eq!(mapped_doc_id(&maps[src], doc), None, "src {src} doc {doc}");
                    }
                }
            }
            // Within one source the map is increasing -- what lets
            // `merge_postings` treat a source's contribution to a term as
            // already ascending, and what `merge_one_flat_vector_field`
            // enforces.
            for (src, &max) in max_doc.iter().enumerate() {
                let mapped: Vec<i32> = (0..max)
                    .filter_map(|d| mapped_doc_id(&maps[src], d))
                    .collect();
                assert!(
                    mapped.windows(2).all(|w| w[0] < w[1]),
                    "source {src}'s doc map is not increasing: {mapped:?}"
                );
            }
        }
        // ...and the two orders really are different, or the check above
        // would be vacuous.
        assert_ne!(concat, sorted);
    }

    /// `MultiSorter.sort`'s comparator, tier by tier: the reversed first tier
    /// takes its missing documents to the *front* (the sentinel is
    /// `Long.MAX_VALUE` and reverse applies to it too), the second tier
    /// breaks ties, and source index then doc id break what is left.
    #[test]
    fn the_sorted_doc_order_is_multi_tier_sentinel_reversing_and_stable() {
        // Each source is already in the merged order (the precondition):
        // source 0 is [missing, 5], source 1 is [7, 5], both descending once
        // the missing value takes its `Long.MAX_VALUE` sentinel.
        let live = vec![vec![0i32, 1], vec![0, 1]];
        let rank: Vec<Vec<Option<i64>>> = vec![vec![None, Some(5)], vec![Some(7), Some(5)]];
        let rank_slices: Vec<&[Option<i64>]> = rank.iter().map(|v| v.as_slice()).collect();
        let specs = vec![MergeSortKeySpec {
            field: "rank",
            reverse: true,
            missing: SortMissingValue::Last,
            per_source_keys: &rank_slices,
        }];
        assert_eq!(
            sorted_doc_order(&specs, &live),
            // missing (MAX, and `reverse` applies to the sentinel too, so it
            // comes *first* under a missing-**last** descending sort), then
            // 7, then the two 5s with the source index breaking the tie.
            vec![(0, 0), (1, 0), (0, 1), (1, 1)]
        );
    }

    /// A mis-shaped sort-key table is a named error, not a panic: the entry
    /// point is `pub`, and the shapes are easy to get wrong (the outer list is
    /// per *source*, each inner slice per *document of that source* -- not per
    /// live document, because it is indexed by pre-merge doc id).
    #[test]
    fn a_mis_shaped_sort_key_table_is_reported_rather_than_panicking() {
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
        let reader0 = open_reader(&seg0);
        let sources = vec![MergeSource::stored_only(&seg0.fields, &reader0, None)];

        let merge = |specs: &[MergeSortKeySpec<'_>]| {
            merge_segments(
                &dir,
                &sources,
                Some(specs),
                &MergeOptions::default(),
                "_merged_badsort",
                [9u8; ID_LENGTH],
                "Lucene104",
                version(),
            )
        };
        assert!(matches!(merge(&[]), Err(Error::EmptySortFields)));

        // The outer list has no entry for the one source.
        let empty: Vec<&[Option<i64>]> = Vec::new();
        assert!(matches!(
            merge(&[MergeSortKeySpec {
                field: "rank",
                reverse: false,
                missing: SortMissingValue::Last,
                per_source_keys: &empty,
            }]),
            Err(Error::SortKeysWrongLength {
                source_index: None,
                expected: 1,
                found: 0,
                ..
            })
        ));

        // One entry per *live* document instead of per document -- the shape
        // a caller who filtered deletions first would naturally build.
        let short: Vec<Option<i64>> = vec![Some(1)];
        let short_slices: Vec<&[Option<i64>]> = vec![short.as_slice()];
        assert!(matches!(
            merge(&[MergeSortKeySpec {
                field: "rank",
                reverse: false,
                missing: SortMissingValue::Last,
                per_source_keys: &short_slices,
            }]),
            Err(Error::SortKeysWrongLength {
                source_index: Some(0),
                expected: 2,
                found: 1,
                ..
            })
        ));
    }

    /// `take_permuted` moves rather than clones, and applies the permutation
    /// in the "which element becomes position n" direction -- the same
    /// direction confusion `segment_writer::permute_in_place` was once caught
    /// by, so it is pinned with a 3-cycle, which no involution could catch.
    #[test]
    fn take_permuted_moves_each_element_to_its_named_position() {
        let mut values = vec![vec![1], vec![2], vec![3]];
        assert_eq!(
            take_permuted(&mut values, &[2, 0, 1]),
            vec![vec![3], vec![1], vec![2]]
        );
    }

    // --- postings ---

    fn postings_field(name: &str, number: i32) -> FieldInfo {
        let mut f = field(name, number);
        f.index_options = IndexOptions::DocsAndFreqs;
        f
    }

    #[test]
    fn two_sources_no_deletions_merge_postings_correctly() {
        let seg0_id = [1u8; ID_LENGTH];
        let seg1_id = [2u8; ID_LENGTH];

        // Source 0: 2 docs -- doc 0 has "apple" (freq 2), doc 1 has "banana"
        // (freq 1).
        let terms0 = vec![
            TermPostings {
                term: b"apple".to_vec(),
                docs: vec![(0, 2)],
                ..Default::default()
            },
            TermPostings {
                term: b"banana".to_vec(),
                docs: vec![(1, 1)],
                ..Default::default()
            },
        ];
        let input0 = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: 2,
            has_payloads: false,
            terms: &terms0,
        };
        let output0 = postings_writer::write_single_field(
            &input0,
            &seg0_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let field_infos0 = field_infos::FieldInfos {
            fields: vec![postings_field("body", 0)],
        };
        let fields0 = lucene_codecs::blocktree::open(
            &output0.tim,
            &output0.tip,
            &output0.tmd,
            &field_infos0,
            &seg0_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
            2,
        )
        .unwrap();
        let doc_in0 = DocInput::open(
            &output0.doc,
            &seg0_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let field_terms0 = fields0.field("body").unwrap();

        // Source 1: 1 doc -- doc 0 has "cherry" (freq 3).
        let terms1 = vec![TermPostings {
            term: b"cherry".to_vec(),
            docs: vec![(0, 3)],
            ..Default::default()
        }];
        let input1 = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: 1,
            has_payloads: false,
            terms: &terms1,
        };
        let output1 = postings_writer::write_single_field(
            &input1,
            &seg1_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let field_infos1 = field_infos::FieldInfos {
            fields: vec![postings_field("body", 0)],
        };
        let fields1 = lucene_codecs::blocktree::open(
            &output1.tim,
            &output1.tip,
            &output1.tmd,
            &field_infos1,
            &seg1_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
            1,
        )
        .unwrap();
        let doc_in1 = DocInput::open(
            &output1.doc,
            &seg1_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let field_terms1 = fields1.field("body").unwrap();

        let fields = vec![postings_field("body", 0)];
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

        let src_postings0 = [SourcePostings {
            field_number: 0,
            field_terms: field_terms0,
            doc_in: Some(&doc_in0),
            pos_in: None,
            pay_in: None,
        }];
        let src_postings1 = [SourcePostings {
            field_number: 0,
            field_terms: field_terms1,
            doc_in: Some(&doc_in1),
            pos_in: None,
            pay_in: None,
        }];
        let source0 = MergeSource {
            field_infos: &stored0.fields,
            reader: &reader0,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &src_postings0,
            points: &[],
            vectors: None,
        };
        let source1 = MergeSource {
            field_infos: &stored1.fields,
            reader: &reader1,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &src_postings1,
            points: &[],
            vectors: None,
        };

        let sci = merge_stored_only_segments(
            &dir,
            &[source0, source1],
            "_merged_postings",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();
        assert_eq!(sci.segment_name, "_merged_postings");

        let tim = std::fs::read(tmp.join(format!(
            "{}.tim",
            per_field_segment("_merged_postings", POSTINGS_FORMAT_NAME)
        )))
        .unwrap();
        let tip = std::fs::read(tmp.join(format!(
            "{}.tip",
            per_field_segment("_merged_postings", POSTINGS_FORMAT_NAME)
        )))
        .unwrap();
        let tmd = std::fs::read(tmp.join(format!(
            "{}.tmd",
            per_field_segment("_merged_postings", POSTINGS_FORMAT_NAME)
        )))
        .unwrap();
        let doc = std::fs::read(tmp.join(format!(
            "{}.doc",
            per_field_segment("_merged_postings", POSTINGS_FORMAT_NAME)
        )))
        .unwrap();
        let merged_field_infos = field_infos::FieldInfos {
            fields: vec![postings_field("body", 0)],
        };
        let merged_fields = lucene_codecs::blocktree::open(
            &tim,
            &tip,
            &tmd,
            &merged_field_infos,
            &[9u8; ID_LENGTH],
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
            3,
        )
        .unwrap();
        let merged_doc_in = DocInput::open(
            &doc,
            &[9u8; ID_LENGTH],
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let merged_terms = merged_fields.field("body").unwrap();

        let apple = merged_terms
            .postings(b"apple", Some(&merged_doc_in))
            .unwrap()
            .unwrap();
        assert_eq!(apple.docs, vec![0]);
        assert_eq!(apple.freqs, vec![2]);

        let banana = merged_terms
            .postings(b"banana", Some(&merged_doc_in))
            .unwrap()
            .unwrap();
        assert_eq!(banana.docs, vec![1]);
        assert_eq!(banana.freqs, vec![1]);

        // "cherry" only existed in source 1's doc 0, which is renumbered to
        // merged doc 2 (after source 0's 2 docs).
        let cherry = merged_terms
            .postings(b"cherry", Some(&merged_doc_in))
            .unwrap()
            .unwrap();
        assert_eq!(cherry.docs, vec![2]);
        assert_eq!(cherry.freqs, vec![3]);
    }

    /// Task #212's coverage/scope review flagged that `IndexOptions::
    /// DocsAndCustomFreqs` -- accepted by `merge_postings`'s supported-list
    /// alongside `Docs`/`DocsAndFreqs`, wire-identical to `DocsAndFreqs` --
    /// had zero merge test coverage despite being a real, reachable field
    /// type since `IndexWriter::set_custom_freq_postings_field`. This proves
    /// merging preserves the caller's opaque custom-freq values verbatim
    /// (not re-derived as an occurrence count), the same way
    /// [`two_sources_no_deletions_merge_postings_correctly`] proves it for
    /// `DocsAndFreqs`.
    #[test]
    fn two_sources_no_deletions_merge_docs_and_custom_freqs_correctly() {
        let seg0_id = [1u8; ID_LENGTH];
        let seg1_id = [2u8; ID_LENGTH];

        // Source 0: 1 doc -- "score" with a custom freq of 50 (not an
        // occurrence count; the term appears once in the doc's stored text
        // below, so a bug that silently re-derived freq from occurrences
        // would produce 1, not 50).
        let terms0 = vec![TermPostings {
            term: b"score".to_vec(),
            docs: vec![(0, 50)],
            ..Default::default()
        }];
        let input0 = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndCustomFreqs,
            doc_count: 1,
            has_payloads: false,
            terms: &terms0,
        };
        let output0 = postings_writer::write_single_field(
            &input0,
            &seg0_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let mut field0 = postings_field("body", 0);
        field0.index_options = IndexOptions::DocsAndCustomFreqs;
        let field_infos0 = field_infos::FieldInfos {
            fields: vec![field0.clone()],
        };
        let fields0 = lucene_codecs::blocktree::open(
            &output0.tim,
            &output0.tip,
            &output0.tmd,
            &field_infos0,
            &seg0_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
            1,
        )
        .unwrap();
        let doc_in0 = DocInput::open(
            &output0.doc,
            &seg0_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let field_terms0 = fields0.field("body").unwrap();

        // Source 1: 1 doc -- same term "score", custom freq of 5.
        let terms1 = vec![TermPostings {
            term: b"score".to_vec(),
            docs: vec![(0, 5)],
            ..Default::default()
        }];
        let input1 = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndCustomFreqs,
            doc_count: 1,
            has_payloads: false,
            terms: &terms1,
        };
        let output1 = postings_writer::write_single_field(
            &input1,
            &seg1_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let fields1 = lucene_codecs::blocktree::open(
            &output1.tim,
            &output1.tip,
            &output1.tmd,
            &field_infos0,
            &seg1_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
            1,
        )
        .unwrap();
        let doc_in1 = DocInput::open(
            &output1.doc,
            &seg1_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let field_terms1 = fields1.field("body").unwrap();

        let fields = vec![field0];
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let stored0 = flush(&dir, &tmp, "_0", seg0_id, &fields, &[doc_with(0, "a")]);
        let stored1 = flush(&dir, &tmp, "_1", seg1_id, &fields, &[doc_with(0, "b")]);
        let reader0 = open_reader(&stored0);
        let reader1 = open_reader(&stored1);

        let src_postings0 = [SourcePostings {
            field_number: 0,
            field_terms: field_terms0,
            doc_in: Some(&doc_in0),
            pos_in: None,
            pay_in: None,
        }];
        let src_postings1 = [SourcePostings {
            field_number: 0,
            field_terms: field_terms1,
            doc_in: Some(&doc_in1),
            pos_in: None,
            pay_in: None,
        }];
        let source0 = MergeSource {
            field_infos: &stored0.fields,
            reader: &reader0,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &src_postings0,
            points: &[],
            vectors: None,
        };
        let source1 = MergeSource {
            field_infos: &stored1.fields,
            reader: &reader1,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &src_postings1,
            points: &[],
            vectors: None,
        };

        merge_stored_only_segments(
            &dir,
            &[source0, source1],
            "_merged_custom_freqs",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();

        let tim = std::fs::read(tmp.join(format!(
            "{}.tim",
            per_field_segment("_merged_custom_freqs", POSTINGS_FORMAT_NAME)
        )))
        .unwrap();
        let tip = std::fs::read(tmp.join(format!(
            "{}.tip",
            per_field_segment("_merged_custom_freqs", POSTINGS_FORMAT_NAME)
        )))
        .unwrap();
        let tmd = std::fs::read(tmp.join(format!(
            "{}.tmd",
            per_field_segment("_merged_custom_freqs", POSTINGS_FORMAT_NAME)
        )))
        .unwrap();
        let doc = std::fs::read(tmp.join(format!(
            "{}.doc",
            per_field_segment("_merged_custom_freqs", POSTINGS_FORMAT_NAME)
        )))
        .unwrap();
        let merged_fields = lucene_codecs::blocktree::open(
            &tim,
            &tip,
            &tmd,
            &field_infos0,
            &[9u8; ID_LENGTH],
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
            2,
        )
        .unwrap();
        let merged_doc_in = DocInput::open(
            &doc,
            &[9u8; ID_LENGTH],
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let merged_terms = merged_fields.field("body").unwrap();

        let score = merged_terms
            .postings(b"score", Some(&merged_doc_in))
            .unwrap()
            .unwrap();
        assert_eq!(score.docs, vec![0, 1]);
        // The merged freqs are the two sources' custom values verbatim, in
        // merged-doc-id order -- not re-derived from occurrence counts
        // (which would both be 1).
        assert_eq!(score.freqs, vec![50, 5]);
    }

    #[test]
    fn term_across_multiple_sources_merges_in_doc_id_order() {
        let seg0_id = [1u8; ID_LENGTH];
        let seg1_id = [2u8; ID_LENGTH];

        // Both sources index "the": source 0 doc 0 (freq 1), source 1 doc 0
        // (freq 2) -- the merged term must contain both docs in ascending
        // merged-doc-id order (source 0's docs first).
        let terms0 = vec![TermPostings {
            term: b"the".to_vec(),
            docs: vec![(0, 1)],
            ..Default::default()
        }];
        let input0 = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: 1,
            has_payloads: false,
            terms: &terms0,
        };
        let output0 = postings_writer::write_single_field(
            &input0,
            &seg0_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let field_infos0 = field_infos::FieldInfos {
            fields: vec![postings_field("body", 0)],
        };
        let fields0 = lucene_codecs::blocktree::open(
            &output0.tim,
            &output0.tip,
            &output0.tmd,
            &field_infos0,
            &seg0_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
            1,
        )
        .unwrap();
        let doc_in0 = DocInput::open(
            &output0.doc,
            &seg0_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let field_terms0 = fields0.field("body").unwrap();

        let terms1 = vec![TermPostings {
            term: b"the".to_vec(),
            docs: vec![(0, 2)],
            ..Default::default()
        }];
        let input1 = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: 1,
            has_payloads: false,
            terms: &terms1,
        };
        let output1 = postings_writer::write_single_field(
            &input1,
            &seg1_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let field_infos1 = field_infos::FieldInfos {
            fields: vec![postings_field("body", 0)],
        };
        let fields1 = lucene_codecs::blocktree::open(
            &output1.tim,
            &output1.tip,
            &output1.tmd,
            &field_infos1,
            &seg1_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
            1,
        )
        .unwrap();
        let doc_in1 = DocInput::open(
            &output1.doc,
            &seg1_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let field_terms1 = fields1.field("body").unwrap();

        let fields = vec![postings_field("body", 0)];
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let stored0 = flush(&dir, &tmp, "_0", seg0_id, &fields, &[doc_with(0, "a")]);
        let stored1 = flush(&dir, &tmp, "_1", seg1_id, &fields, &[doc_with(0, "b")]);
        let reader0 = open_reader(&stored0);
        let reader1 = open_reader(&stored1);

        let src_postings0 = [SourcePostings {
            field_number: 0,
            field_terms: field_terms0,
            doc_in: Some(&doc_in0),
            pos_in: None,
            pay_in: None,
        }];
        let src_postings1 = [SourcePostings {
            field_number: 0,
            field_terms: field_terms1,
            doc_in: Some(&doc_in1),
            pos_in: None,
            pay_in: None,
        }];
        let source0 = MergeSource {
            field_infos: &stored0.fields,
            reader: &reader0,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &src_postings0,
            points: &[],
            vectors: None,
        };
        let source1 = MergeSource {
            field_infos: &stored1.fields,
            reader: &reader1,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &src_postings1,
            points: &[],
            vectors: None,
        };

        let tmp2 = tmp.path().to_path_buf();
        merge_stored_only_segments(
            &dir,
            &[source0, source1],
            "_merged_the",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();

        let tim = std::fs::read(std::path::Path::new(&tmp2).join(format!(
            "{}.tim",
            per_field_segment("_merged_the", POSTINGS_FORMAT_NAME)
        )))
        .unwrap();
        let tip = std::fs::read(std::path::Path::new(&tmp2).join(format!(
            "{}.tip",
            per_field_segment("_merged_the", POSTINGS_FORMAT_NAME)
        )))
        .unwrap();
        let tmd = std::fs::read(std::path::Path::new(&tmp2).join(format!(
            "{}.tmd",
            per_field_segment("_merged_the", POSTINGS_FORMAT_NAME)
        )))
        .unwrap();
        let doc = std::fs::read(std::path::Path::new(&tmp2).join(format!(
            "{}.doc",
            per_field_segment("_merged_the", POSTINGS_FORMAT_NAME)
        )))
        .unwrap();
        let merged_field_infos = field_infos::FieldInfos {
            fields: vec![postings_field("body", 0)],
        };
        let merged_fields = lucene_codecs::blocktree::open(
            &tim,
            &tip,
            &tmd,
            &merged_field_infos,
            &[9u8; ID_LENGTH],
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
            2,
        )
        .unwrap();
        let merged_doc_in = DocInput::open(
            &doc,
            &[9u8; ID_LENGTH],
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let merged_terms = merged_fields.field("body").unwrap();

        let the = merged_terms
            .postings(b"the", Some(&merged_doc_in))
            .unwrap()
            .unwrap();
        assert_eq!(the.docs, vec![0, 1]);
        assert_eq!(the.freqs, vec![1, 2]);
    }

    #[test]
    fn deletions_drop_docs_from_merged_postings() {
        let seg0_id = [1u8; ID_LENGTH];

        // Source 0: 2 docs -- doc 0 has "apple", doc 1 has "banana"; doc 1 is
        // deleted, so "banana" must not survive the merge at all.
        let terms0 = vec![
            TermPostings {
                term: b"apple".to_vec(),
                docs: vec![(0, 1)],
                ..Default::default()
            },
            TermPostings {
                term: b"banana".to_vec(),
                docs: vec![(1, 1)],
                ..Default::default()
            },
        ];
        let input0 = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: 2,
            has_payloads: false,
            terms: &terms0,
        };
        let output0 = postings_writer::write_single_field(
            &input0,
            &seg0_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let field_infos0 = field_infos::FieldInfos {
            fields: vec![postings_field("body", 0)],
        };
        let fields0 = lucene_codecs::blocktree::open(
            &output0.tim,
            &output0.tip,
            &output0.tmd,
            &field_infos0,
            &seg0_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
            2,
        )
        .unwrap();
        let doc_in0 = DocInput::open(
            &output0.doc,
            &seg0_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let field_terms0 = fields0.field("body").unwrap();

        let fields = vec![postings_field("body", 0)];
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
        let reader0 = open_reader(&stored0);

        let mut live0 = FixedBitSet::new(2);
        live0.set(0); // keep doc 0 only

        let src_postings0 = [SourcePostings {
            field_number: 0,
            field_terms: field_terms0,
            doc_in: Some(&doc_in0),
            pos_in: None,
            pay_in: None,
        }];
        let source0 = MergeSource {
            field_infos: &stored0.fields,
            reader: &reader0,
            live_docs: Some(&live0),
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &src_postings0,
            points: &[],
            vectors: None,
        };

        merge_stored_only_segments(
            &dir,
            &[source0],
            "_merged_del",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();

        let tim = std::fs::read(tmp.join(format!(
            "{}.tim",
            per_field_segment("_merged_del", POSTINGS_FORMAT_NAME)
        )))
        .unwrap();
        let tip = std::fs::read(tmp.join(format!(
            "{}.tip",
            per_field_segment("_merged_del", POSTINGS_FORMAT_NAME)
        )))
        .unwrap();
        let tmd = std::fs::read(tmp.join(format!(
            "{}.tmd",
            per_field_segment("_merged_del", POSTINGS_FORMAT_NAME)
        )))
        .unwrap();
        let doc = std::fs::read(tmp.join(format!(
            "{}.doc",
            per_field_segment("_merged_del", POSTINGS_FORMAT_NAME)
        )))
        .unwrap();
        let merged_field_infos = field_infos::FieldInfos {
            fields: vec![postings_field("body", 0)],
        };
        let merged_fields = lucene_codecs::blocktree::open(
            &tim,
            &tip,
            &tmd,
            &merged_field_infos,
            &[9u8; ID_LENGTH],
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
            1,
        )
        .unwrap();
        let merged_doc_in = DocInput::open(
            &doc,
            &[9u8; ID_LENGTH],
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let merged_terms = merged_fields.field("body").unwrap();

        let apple = merged_terms
            .postings(b"apple", Some(&merged_doc_in))
            .unwrap()
            .unwrap();
        assert_eq!(apple.docs, vec![0]);

        assert!(merged_terms.seek_exact(b"banana").is_none());
    }

    #[test]
    fn fully_deleted_source_contributes_nothing_to_postings() {
        let seg0_id = [1u8; ID_LENGTH];
        let seg1_id = [2u8; ID_LENGTH];

        // Source 0: 1 doc, fully deleted -- its "ghost" term must not survive.
        let terms0 = vec![TermPostings {
            term: b"ghost".to_vec(),
            docs: vec![(0, 1)],
            ..Default::default()
        }];
        let input0 = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: 1,
            has_payloads: false,
            terms: &terms0,
        };
        let output0 = postings_writer::write_single_field(
            &input0,
            &seg0_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let field_infos0 = field_infos::FieldInfos {
            fields: vec![postings_field("body", 0)],
        };
        let fields0 = lucene_codecs::blocktree::open(
            &output0.tim,
            &output0.tip,
            &output0.tmd,
            &field_infos0,
            &seg0_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
            1,
        )
        .unwrap();
        let doc_in0 = DocInput::open(
            &output0.doc,
            &seg0_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let field_terms0 = fields0.field("body").unwrap();

        // Source 1: 1 doc, alive -- "alive" survives.
        let terms1 = vec![TermPostings {
            term: b"alive".to_vec(),
            docs: vec![(0, 1)],
            ..Default::default()
        }];
        let input1 = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: 1,
            has_payloads: false,
            terms: &terms1,
        };
        let output1 = postings_writer::write_single_field(
            &input1,
            &seg1_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let field_infos1 = field_infos::FieldInfos {
            fields: vec![postings_field("body", 0)],
        };
        let fields1 = lucene_codecs::blocktree::open(
            &output1.tim,
            &output1.tip,
            &output1.tmd,
            &field_infos1,
            &seg1_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
            1,
        )
        .unwrap();
        let doc_in1 = DocInput::open(
            &output1.doc,
            &seg1_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let field_terms1 = fields1.field("body").unwrap();

        let fields = vec![postings_field("body", 0)];
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let stored0 = flush(&dir, &tmp, "_0", seg0_id, &fields, &[doc_with(0, "a")]);
        let stored1 = flush(&dir, &tmp, "_1", seg1_id, &fields, &[doc_with(0, "b")]);
        let reader0 = open_reader(&stored0);
        let reader1 = open_reader(&stored1);

        let live0 = FixedBitSet::new(1); // no bits set -- fully deleted

        let src_postings0 = [SourcePostings {
            field_number: 0,
            field_terms: field_terms0,
            doc_in: Some(&doc_in0),
            pos_in: None,
            pay_in: None,
        }];
        let src_postings1 = [SourcePostings {
            field_number: 0,
            field_terms: field_terms1,
            doc_in: Some(&doc_in1),
            pos_in: None,
            pay_in: None,
        }];
        let source0 = MergeSource {
            field_infos: &stored0.fields,
            reader: &reader0,
            live_docs: Some(&live0),
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &src_postings0,
            points: &[],
            vectors: None,
        };
        let source1 = MergeSource {
            field_infos: &stored1.fields,
            reader: &reader1,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &src_postings1,
            points: &[],
            vectors: None,
        };

        merge_stored_only_segments(
            &dir,
            &[source0, source1],
            "_merged_fully_deleted",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();

        let tim = std::fs::read(tmp.join(format!(
            "{}.tim",
            per_field_segment("_merged_fully_deleted", POSTINGS_FORMAT_NAME)
        )))
        .unwrap();
        let tip = std::fs::read(tmp.join(format!(
            "{}.tip",
            per_field_segment("_merged_fully_deleted", POSTINGS_FORMAT_NAME)
        )))
        .unwrap();
        let tmd = std::fs::read(tmp.join(format!(
            "{}.tmd",
            per_field_segment("_merged_fully_deleted", POSTINGS_FORMAT_NAME)
        )))
        .unwrap();
        let doc = std::fs::read(tmp.join(format!(
            "{}.doc",
            per_field_segment("_merged_fully_deleted", POSTINGS_FORMAT_NAME)
        )))
        .unwrap();
        let merged_field_infos = field_infos::FieldInfos {
            fields: vec![postings_field("body", 0)],
        };
        let merged_fields = lucene_codecs::blocktree::open(
            &tim,
            &tip,
            &tmd,
            &merged_field_infos,
            &[9u8; ID_LENGTH],
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
            1,
        )
        .unwrap();
        let merged_doc_in = DocInput::open(
            &doc,
            &[9u8; ID_LENGTH],
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let merged_terms = merged_fields.field("body").unwrap();

        assert!(merged_terms.seek_exact(b"ghost").is_none());
        let alive = merged_terms
            .postings(b"alive", Some(&merged_doc_in))
            .unwrap()
            .unwrap();
        assert_eq!(alive.docs, vec![0]);
    }

    #[test]
    fn a_source_that_never_saw_the_postings_field_simply_contributes_no_terms() {
        // `FieldsConsumer.merge`: `Terms terms = fields.terms(field); if
        // (terms == null) continue;`. A segment written before the field
        // existed has no `FieldInfo` for it and contributes no terms --
        // that must not fail the merge (declaring the same field with
        // *different* index_options is a separate matter, rejected up front
        // by `reconcile_field_numbers`).
        let seg0_id = [1u8; ID_LENGTH];
        let terms0 = vec![TermPostings {
            term: b"apple".to_vec(),
            docs: vec![(0, 1)],
            ..Default::default()
        }];
        let input0 = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: 1,
            has_payloads: false,
            terms: &terms0,
        };
        let output0 = postings_writer::write_single_field(
            &input0,
            &seg0_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let field_infos0 = field_infos::FieldInfos {
            fields: vec![postings_field("body", 0)],
        };
        let fields0 = lucene_codecs::blocktree::open(
            &output0.tim,
            &output0.tip,
            &output0.tmd,
            &field_infos0,
            &seg0_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
            1,
        )
        .unwrap();
        let doc_in0 = DocInput::open(
            &output0.doc,
            &seg0_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let field_terms0 = fields0.field("body").unwrap();

        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let stored0 = flush(
            &dir,
            &tmp,
            "_0",
            seg0_id,
            &[postings_field("body", 0)],
            &[doc_with(0, "a")],
        );
        // Source 1 never saw "body" at all.
        let stored1 = flush(
            &dir,
            &tmp,
            "_1",
            [2u8; ID_LENGTH],
            &[field("id", 0)],
            &[doc_with(0, "b")],
        );
        let reader0 = open_reader(&stored0);
        let reader1 = open_reader(&stored1);

        let src_postings0 = [SourcePostings {
            field_number: 0,
            field_terms: field_terms0,
            doc_in: Some(&doc_in0),
            pos_in: None,
            pay_in: None,
        }];
        let source0 = MergeSource {
            postings: &src_postings0,
            ..MergeSource::stored_only(&stored0.fields, &reader0, None)
        };
        let source1 = MergeSource::stored_only(&stored1.fields, &reader1, None);

        merge_stored_only_segments(
            &dir,
            &[source0, source1],
            "_merged_postings_absent_source",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();

        let read = |ext: &str| {
            std::fs::read(tmp.join(format!(
                "{}.{ext}",
                per_field_segment("_merged_postings_absent_source", POSTINGS_FORMAT_NAME)
            )))
            .unwrap()
        };
        let (tim, tip, tmd, doc) = (read("tim"), read("tip"), read("tmd"), read("doc"));
        let merged_field_infos = field_infos::FieldInfos {
            fields: vec![postings_field("body", 0)],
        };
        let merged_fields = lucene_codecs::blocktree::open(
            &tim,
            &tip,
            &tmd,
            &merged_field_infos,
            &[9u8; ID_LENGTH],
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
            1,
        )
        .unwrap();
        let merged_doc_in = DocInput::open(
            &doc,
            &[9u8; ID_LENGTH],
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let merged_terms = merged_fields.field("body").unwrap();
        let apple = merged_terms
            .postings(b"apple", Some(&merged_doc_in))
            .unwrap()
            .unwrap();
        assert_eq!(
            apple.docs,
            vec![0],
            "only source 0's doc has the term; source 1 contributes none"
        );
    }

    #[test]
    fn postings_field_missing_in_a_live_contributing_source_is_an_error() {
        let seg0_id = [1u8; ID_LENGTH];

        let terms0 = vec![TermPostings {
            term: b"apple".to_vec(),
            docs: vec![(0, 1)],
            ..Default::default()
        }];
        let input0 = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: 1,
            has_payloads: false,
            terms: &terms0,
        };
        let output0 = postings_writer::write_single_field(
            &input0,
            &seg0_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let field_infos0 = field_infos::FieldInfos {
            fields: vec![postings_field("body", 0)],
        };
        let fields0 = lucene_codecs::blocktree::open(
            &output0.tim,
            &output0.tip,
            &output0.tmd,
            &field_infos0,
            &seg0_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
            1,
        )
        .unwrap();
        let doc_in0 = DocInput::open(
            &output0.doc,
            &seg0_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let field_terms0 = fields0.field("body").unwrap();

        let fields = vec![postings_field("body", 0)];
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let stored0 = flush(&dir, &tmp, "_0", seg0_id, &fields, &[doc_with(0, "a")]);
        // Source 1 declares the same "body" field but supplies no postings
        // data for it at all -- a schema mismatch, since source 0 has live
        // docs indexing that field.
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

        let src_postings0 = [SourcePostings {
            field_number: 0,
            field_terms: field_terms0,
            doc_in: Some(&doc_in0),
            pos_in: None,
            pay_in: None,
        }];
        let source0 = MergeSource {
            field_infos: &stored0.fields,
            reader: &reader0,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &src_postings0,
            points: &[],
            vectors: None,
        };
        let source1 = MergeSource {
            field_infos: &stored1.fields,
            reader: &reader1,
            live_docs: None,
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
        };

        let result = merge_stored_only_segments(
            &dir,
            &[source0, source1],
            "_merged_missing_postings",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        );
        assert!(matches!(
            result,
            Err(Error::PostingsFieldMissingInSource { .. })
        ));
    }

    #[test]
    fn two_sources_merge_positions_offsets_and_payloads_correctly() {
        // End-to-end: two segments, each with a field indexing positions,
        // offsets, and payloads, merged into one -- regression test for the
        // gap flagged in the module's top doc comment ("only Docs/
        // DocsAndFreqs fields merge today"). Verifies the merged term
        // dictionary's positions/offsets/payloads are queryable post-merge
        // via `FieldTerms::positions`, the same read path phrase matching
        // uses.
        let seg0_id = [1u8; ID_LENGTH];
        let seg1_id = [2u8; ID_LENGTH];

        // Source 0: 1 doc -- doc 0 has "apple" at positions 0 and 2, with
        // offsets and a payload on the second occurrence only.
        let terms0 = vec![TermPostings {
            term: b"apple".to_vec(),
            docs: vec![(0, 2)],
            positions: vec![vec![0, 2]],
            offsets: vec![vec![(0, 5), (10, 15)]],
            payloads: vec![vec![Vec::new(), b"pay0".to_vec()]],
        }];
        let input0 = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
            doc_count: 1,
            has_payloads: true,
            terms: &terms0,
        };
        let output0 = postings_writer::write_single_field(
            &input0,
            &seg0_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let mut field_with_positions = postings_field("body", 0);
        field_with_positions.index_options = IndexOptions::DocsAndFreqsAndPositionsAndOffsets;
        field_with_positions.store_payloads = true;
        let field_infos0 = field_infos::FieldInfos {
            fields: vec![field_with_positions.clone()],
        };
        let fields0 = lucene_codecs::blocktree::open(
            &output0.tim,
            &output0.tip,
            &output0.tmd,
            &field_infos0,
            &seg0_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
            1,
        )
        .unwrap();
        let doc_in0 = DocInput::open(
            &output0.doc,
            &seg0_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let pos_in0 = lucene_codecs::postings::PosInput::open(
            &output0.pos,
            &seg0_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let pay_in0 = lucene_codecs::postings::PayInput::open(
            &output0.pay,
            &seg0_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let field_terms0 = fields0.field("body").unwrap();

        // Source 1: 1 doc -- doc 0 has "banana" at position 1, no payload.
        let terms1 = vec![TermPostings {
            term: b"banana".to_vec(),
            docs: vec![(0, 1)],
            positions: vec![vec![1]],
            offsets: vec![vec![(6, 12)]],
            payloads: vec![vec![Vec::new()]],
        }];
        let input1 = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
            doc_count: 1,
            has_payloads: true,
            terms: &terms1,
        };
        let output1 = postings_writer::write_single_field(
            &input1,
            &seg1_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let field_infos1 = field_infos::FieldInfos {
            fields: vec![field_with_positions.clone()],
        };
        let fields1 = lucene_codecs::blocktree::open(
            &output1.tim,
            &output1.tip,
            &output1.tmd,
            &field_infos1,
            &seg1_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
            1,
        )
        .unwrap();
        let doc_in1 = DocInput::open(
            &output1.doc,
            &seg1_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let pos_in1 = lucene_codecs::postings::PosInput::open(
            &output1.pos,
            &seg1_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let pay_in1 = lucene_codecs::postings::PayInput::open(
            &output1.pay,
            &seg1_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let field_terms1 = fields1.field("body").unwrap();

        let fields = vec![field_with_positions.clone()];
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let stored0 = flush(&dir, &tmp, "_0", seg0_id, &fields, &[doc_with(0, "a")]);
        let stored1 = flush(&dir, &tmp, "_1", seg1_id, &fields, &[doc_with(0, "b")]);
        let reader0 = open_reader(&stored0);
        let reader1 = open_reader(&stored1);

        let src_postings0 = [SourcePostings {
            field_number: 0,
            field_terms: field_terms0,
            doc_in: Some(&doc_in0),
            pos_in: Some(&pos_in0),
            pay_in: Some(&pay_in0),
        }];
        let src_postings1 = [SourcePostings {
            field_number: 0,
            field_terms: field_terms1,
            doc_in: Some(&doc_in1),
            pos_in: Some(&pos_in1),
            pay_in: Some(&pay_in1),
        }];
        let source0 = MergeSource {
            field_infos: &stored0.fields,
            reader: &reader0,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &src_postings0,
            points: &[],
            vectors: None,
        };
        let source1 = MergeSource {
            field_infos: &stored1.fields,
            reader: &reader1,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &src_postings1,
            points: &[],
            vectors: None,
        };

        let sci = merge_stored_only_segments(
            &dir,
            &[source0, source1],
            "_merged_positions",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();
        assert_eq!(sci.segment_name, "_merged_positions");

        let tim = std::fs::read(tmp.join(format!(
            "{}.tim",
            per_field_segment("_merged_positions", POSTINGS_FORMAT_NAME)
        )))
        .unwrap();
        let tip = std::fs::read(tmp.join(format!(
            "{}.tip",
            per_field_segment("_merged_positions", POSTINGS_FORMAT_NAME)
        )))
        .unwrap();
        let tmd = std::fs::read(tmp.join(format!(
            "{}.tmd",
            per_field_segment("_merged_positions", POSTINGS_FORMAT_NAME)
        )))
        .unwrap();
        let doc = std::fs::read(tmp.join(format!(
            "{}.doc",
            per_field_segment("_merged_positions", POSTINGS_FORMAT_NAME)
        )))
        .unwrap();
        let pos = std::fs::read(tmp.join(format!(
            "{}.pos",
            per_field_segment("_merged_positions", POSTINGS_FORMAT_NAME)
        )))
        .unwrap();
        let pay = std::fs::read(tmp.join(format!(
            "{}.pay",
            per_field_segment("_merged_positions", POSTINGS_FORMAT_NAME)
        )))
        .unwrap();
        let merged_field_infos = field_infos::FieldInfos {
            fields: vec![field_with_positions],
        };
        let merged_fields = lucene_codecs::blocktree::open(
            &tim,
            &tip,
            &tmd,
            &merged_field_infos,
            &[9u8; ID_LENGTH],
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
            2,
        )
        .unwrap();
        let merged_doc_in = DocInput::open(
            &doc,
            &[9u8; ID_LENGTH],
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let merged_pos_in = lucene_codecs::postings::PosInput::open(
            &pos,
            &[9u8; ID_LENGTH],
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let merged_pay_in = lucene_codecs::postings::PayInput::open(
            &pay,
            &[9u8; ID_LENGTH],
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let merged_terms = merged_fields.field("body").unwrap();

        let apple = merged_terms
            .positions(
                b"apple",
                Some(&merged_doc_in),
                &merged_pos_in,
                Some(&merged_pay_in),
            )
            .unwrap()
            .unwrap();
        assert_eq!(
            apple.len(),
            1,
            "apple occurs in exactly 1 doc (merged doc 0)"
        );
        assert_eq!(
            apple[0]
                .iter()
                .map(|p| (p.position, p.start_offset, p.end_offset, p.payload.clone()))
                .collect::<Vec<_>>(),
            vec![(0, 0, 5, Vec::new()), (2, 10, 15, b"pay0".to_vec()),]
        );

        // "banana" only existed in source 1's doc 0, renumbered to merged
        // doc 1 (after source 0's 1 doc).
        let banana = merged_terms
            .positions(
                b"banana",
                Some(&merged_doc_in),
                &merged_pos_in,
                Some(&merged_pay_in),
            )
            .unwrap()
            .unwrap();
        assert_eq!(
            banana.len(),
            1,
            "banana occurs in exactly 1 doc (merged doc 1)"
        );
        assert_eq!(
            banana[0]
                .iter()
                .map(|p| (p.position, p.start_offset, p.end_offset, p.payload.clone()))
                .collect::<Vec<_>>(),
            vec![(1, 6, 12, Vec::new())]
        );

        let apple_docs = merged_terms
            .postings(b"apple", Some(&merged_doc_in))
            .unwrap()
            .unwrap();
        assert_eq!(apple_docs.docs, vec![0]);
        let banana_docs = merged_terms
            .postings(b"banana", Some(&merged_doc_in))
            .unwrap()
            .unwrap();
        assert_eq!(banana_docs.docs, vec![1]);
    }

    #[test]
    fn a_source_with_positions_disagreeing_with_the_merged_docs_and_freqs_field_is_rejected() {
        // Source 0's "body" is Docs/DocsAndFreqs (first-seen, so it becomes
        // the merged field's canonical index_options via
        // `reconcile_field_numbers`). Source 1's own "body" indexes
        // positions. Without cross-source validation this would silently
        // drop source 1's positions data instead of erroring -- regression
        // test for `Error::PostingsIndexOptionsDisagreement`.
        let seg0_id = [1u8; ID_LENGTH];
        let seg1_id = [2u8; ID_LENGTH];

        let terms0 = vec![TermPostings {
            term: b"apple".to_vec(),
            docs: vec![(0, 1)],
            ..Default::default()
        }];
        let input0 = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: 1,
            has_payloads: false,
            terms: &terms0,
        };
        let output0 = postings_writer::write_single_field(
            &input0,
            &seg0_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let field_infos0 = field_infos::FieldInfos {
            fields: vec![postings_field("body", 0)],
        };
        let fields0 = lucene_codecs::blocktree::open(
            &output0.tim,
            &output0.tip,
            &output0.tmd,
            &field_infos0,
            &seg0_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
            1,
        )
        .unwrap();
        let doc_in0 = DocInput::open(
            &output0.doc,
            &seg0_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let field_terms0 = fields0.field("body").unwrap();

        let terms1 = vec![TermPostings {
            term: b"banana".to_vec(),
            docs: vec![(0, 1)],
            positions: vec![vec![0]],
            ..Default::default()
        }];
        let input1 = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            doc_count: 1,
            has_payloads: false,
            terms: &terms1,
        };
        let output1 = postings_writer::write_single_field(
            &input1,
            &seg1_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let mut field_with_positions = postings_field("body", 0);
        field_with_positions.index_options = IndexOptions::DocsAndFreqsAndPositions;
        let field_infos1 = field_infos::FieldInfos {
            fields: vec![field_with_positions.clone()],
        };
        let fields1 = lucene_codecs::blocktree::open(
            &output1.tim,
            &output1.tip,
            &output1.tmd,
            &field_infos1,
            &seg1_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
            1,
        )
        .unwrap();
        let doc_in1 = DocInput::open(
            &output1.doc,
            &seg1_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let field_terms1 = fields1.field("body").unwrap();

        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let stored0 = flush(
            &dir,
            &tmp,
            "_0",
            seg0_id,
            &[postings_field("body", 0)],
            &[doc_with(0, "a")],
        );
        let reader0 = open_reader(&stored0);
        let stored1 = flush(
            &dir,
            &tmp,
            "_1",
            seg1_id,
            &[field_with_positions],
            &[doc_with(0, "a")],
        );
        let reader1 = open_reader(&stored1);

        let src_postings0 = [SourcePostings {
            field_number: 0,
            field_terms: field_terms0,
            doc_in: Some(&doc_in0),
            pos_in: None,
            pay_in: None,
        }];
        let src_postings1 = [SourcePostings {
            field_number: 0,
            field_terms: field_terms1,
            doc_in: Some(&doc_in1),
            pos_in: None,
            pay_in: None,
        }];
        let source0 = MergeSource {
            field_infos: &stored0.fields,
            reader: &reader0,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &src_postings0,
            points: &[],
            vectors: None,
        };
        let source1 = MergeSource {
            field_infos: &stored1.fields,
            reader: &reader1,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &src_postings1,
            points: &[],
            vectors: None,
        };

        let result = merge_stored_only_segments(
            &dir,
            &[source0, source1],
            "_merged_disagreeing_index_options",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        );
        assert!(matches!(
            result,
            Err(Error::PostingsIndexOptionsDisagreement { .. })
        ));
    }

    #[test]
    fn a_source_with_payloads_ors_into_the_merged_fields_store_payloads() {
        // Both sources index positions (same index_options), but source 1's
        // "body" also stores payloads while source 0's (first-seen, so it
        // seeds the merged FieldInfo) doesn't. Real Lucene's
        // `FieldInfos.Builder.add` ORs `hasPayloads` rather than rejecting
        // the mismatch (`if (fi.hasPayloads()) curFi.setStorePayloads()`),
        // so the merged field must store payloads and source 1's real
        // payload bytes must survive, while source 0's payload-free
        // occurrences come through as empty payloads (which the postings
        // writer treats as "no payload at this occurrence").
        let seg0_id = [1u8; ID_LENGTH];
        let seg1_id = [2u8; ID_LENGTH];

        let terms0 = vec![TermPostings {
            term: b"apple".to_vec(),
            docs: vec![(0, 1)],
            positions: vec![vec![0]],
            ..Default::default()
        }];
        let input0 = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            doc_count: 1,
            has_payloads: false,
            terms: &terms0,
        };
        let output0 = postings_writer::write_single_field(
            &input0,
            &seg0_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let mut field0 = postings_field("body", 0);
        field0.index_options = IndexOptions::DocsAndFreqsAndPositions;
        let field_infos0 = field_infos::FieldInfos {
            fields: vec![field0.clone()],
        };
        let fields0 = lucene_codecs::blocktree::open(
            &output0.tim,
            &output0.tip,
            &output0.tmd,
            &field_infos0,
            &seg0_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
            1,
        )
        .unwrap();
        let doc_in0 = DocInput::open(
            &output0.doc,
            &seg0_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let pos_in0 = lucene_codecs::postings::PosInput::open(
            &output0.pos,
            &seg0_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let field_terms0 = fields0.field("body").unwrap();

        let terms1 = vec![TermPostings {
            term: b"banana".to_vec(),
            docs: vec![(0, 1)],
            positions: vec![vec![0]],
            payloads: vec![vec![b"pay".to_vec()]],
            ..Default::default()
        }];
        let input1 = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            doc_count: 1,
            has_payloads: true,
            terms: &terms1,
        };
        let output1 = postings_writer::write_single_field(
            &input1,
            &seg1_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let mut field1 = postings_field("body", 0);
        field1.index_options = IndexOptions::DocsAndFreqsAndPositions;
        field1.store_payloads = true;
        let field_infos1 = field_infos::FieldInfos {
            fields: vec![field1.clone()],
        };
        let fields1 = lucene_codecs::blocktree::open(
            &output1.tim,
            &output1.tip,
            &output1.tmd,
            &field_infos1,
            &seg1_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
            1,
        )
        .unwrap();
        let doc_in1 = DocInput::open(
            &output1.doc,
            &seg1_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let pos_in1 = lucene_codecs::postings::PosInput::open(
            &output1.pos,
            &seg1_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let field_terms1 = fields1.field("body").unwrap();

        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let stored0 = flush(&dir, &tmp, "_0", seg0_id, &[field0], &[doc_with(0, "a")]);
        let reader0 = open_reader(&stored0);
        let stored1 = flush(&dir, &tmp, "_1", seg1_id, &[field1], &[doc_with(0, "a")]);
        let reader1 = open_reader(&stored1);

        let src_postings0 = [SourcePostings {
            field_number: 0,
            field_terms: field_terms0,
            doc_in: Some(&doc_in0),
            pos_in: Some(&pos_in0),
            pay_in: None,
        }];
        let src_postings1 = [SourcePostings {
            field_number: 0,
            field_terms: field_terms1,
            doc_in: Some(&doc_in1),
            pos_in: Some(&pos_in1),
            pay_in: None,
        }];
        let source0 = MergeSource {
            field_infos: &stored0.fields,
            reader: &reader0,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &src_postings0,
            points: &[],
            vectors: None,
        };
        let source1 = MergeSource {
            field_infos: &stored1.fields,
            reader: &reader1,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &src_postings1,
            points: &[],
            vectors: None,
        };

        merge_stored_only_segments(
            &dir,
            &[source0, source1],
            "_merged_ored_payloads",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();

        // The merged .fnm must record store_payloads = true (the OR), not
        // the first-seen source's `false`.
        let fnm = std::fs::read(tmp.join("_merged_ored_payloads.fnm")).unwrap();
        let merged_infos = field_infos::parse(&fnm, &[9u8; ID_LENGTH], "").unwrap();
        let merged_body = merged_infos
            .fields
            .iter()
            .find(|f| f.name == "body")
            .unwrap();
        assert!(
            merged_body.store_payloads,
            "store_payloads must be ORed across sources"
        );

        let read = |ext: &str| {
            std::fs::read(tmp.join(format!(
                "{}.{ext}",
                per_field_segment("_merged_ored_payloads", POSTINGS_FORMAT_NAME)
            )))
            .unwrap()
        };
        let (tim, tip, tmd, doc, pos, pay) = (
            read("tim"),
            read("tip"),
            read("tmd"),
            read("doc"),
            read("pos"),
            read("pay"),
        );
        let merged_fields = lucene_codecs::blocktree::open(
            &tim,
            &tip,
            &tmd,
            &merged_infos,
            &[9u8; ID_LENGTH],
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
            2,
        )
        .unwrap();
        let merged_doc_in = DocInput::open(
            &doc,
            &[9u8; ID_LENGTH],
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let merged_pos_in = lucene_codecs::postings::PosInput::open(
            &pos,
            &[9u8; ID_LENGTH],
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let merged_pay_in = lucene_codecs::postings::PayInput::open(
            &pay,
            &[9u8; ID_LENGTH],
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let merged_terms = merged_fields.field("body").unwrap();

        // "apple" came from the payload-free source: empty payload.
        let apple = merged_terms
            .positions(
                b"apple",
                Some(&merged_doc_in),
                &merged_pos_in,
                Some(&merged_pay_in),
            )
            .unwrap()
            .unwrap();
        assert_eq!(
            apple[0]
                .iter()
                .map(|p| (p.position, p.payload.clone()))
                .collect::<Vec<_>>(),
            vec![(0, Vec::new())]
        );

        // "banana" came from the payload-bearing source: bytes preserved.
        let banana = merged_terms
            .positions(
                b"banana",
                Some(&merged_doc_in),
                &merged_pos_in,
                Some(&merged_pay_in),
            )
            .unwrap()
            .unwrap();
        assert_eq!(
            banana[0]
                .iter()
                .map(|p| (p.position, p.payload.clone()))
                .collect::<Vec<_>>(),
            vec![(0, b"pay".to_vec())]
        );
    }

    #[test]
    fn a_positions_indexing_source_missing_an_opened_pos_reader_is_a_clean_error_not_a_panic() {
        // Reuses source0's fixture from the payloads-disagreement test above
        // (a real DocsAndFreqsAndPositions field), but the caller "forgets"
        // to open pos_in -- regression test proving this is a typed
        // `Error::PostingsPositionsInputMissingInSource`, not an `expect()`
        // panic deep inside the per-term merge loop.
        let seg0_id = [1u8; ID_LENGTH];

        let terms0 = vec![TermPostings {
            term: b"apple".to_vec(),
            docs: vec![(0, 1)],
            positions: vec![vec![0]],
            ..Default::default()
        }];
        let input0 = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            doc_count: 1,
            has_payloads: false,
            terms: &terms0,
        };
        let output0 = postings_writer::write_single_field(
            &input0,
            &seg0_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let mut field0 = postings_field("body", 0);
        field0.index_options = IndexOptions::DocsAndFreqsAndPositions;
        let field_infos0 = field_infos::FieldInfos {
            fields: vec![field0.clone()],
        };
        let fields0 = lucene_codecs::blocktree::open(
            &output0.tim,
            &output0.tip,
            &output0.tmd,
            &field_infos0,
            &seg0_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
            1,
        )
        .unwrap();
        let doc_in0 = DocInput::open(
            &output0.doc,
            &seg0_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let field_terms0 = fields0.field("body").unwrap();

        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let stored0 = flush(&dir, &tmp, "_0", seg0_id, &[field0], &[doc_with(0, "a")]);
        let reader0 = open_reader(&stored0);

        // pos_in deliberately left None despite the field indexing positions.
        let src_postings0 = [SourcePostings {
            field_number: 0,
            field_terms: field_terms0,
            doc_in: Some(&doc_in0),
            pos_in: None,
            pay_in: None,
        }];
        let source0 = MergeSource {
            field_infos: &stored0.fields,
            reader: &reader0,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &src_postings0,
            points: &[],
            vectors: None,
        };

        let result = merge_stored_only_segments(
            &dir,
            &[source0],
            "_merged_missing_pos_input",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        );
        assert!(matches!(
            result,
            Err(Error::PostingsPositionsInputMissingInSource { .. })
        ));
    }

    fn points_field(name: &str, number: i32, num_dims: i32, bytes_per_dim: i32) -> FieldInfo {
        points_field_with_index_dims(name, number, num_dims, num_dims, bytes_per_dim)
    }

    fn points_field_with_index_dims(
        name: &str,
        number: i32,
        num_dims: i32,
        num_index_dims: i32,
        bytes_per_dim: i32,
    ) -> FieldInfo {
        let mut f = field(name, number);
        f.point_dimension_count = num_dims;
        f.point_index_dimension_count = num_index_dims;
        f.point_num_bytes = bytes_per_dim;
        f
    }

    fn packed4(v: u32) -> Vec<u8> {
        v.to_be_bytes().to_vec()
    }

    fn write_one_field_points(
        field_number: i32,
        num_dims: i32,
        bytes_per_dim: i32,
        points: Vec<(i32, Vec<u8>)>,
        segment_id: &[u8; ID_LENGTH],
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        write_one_field_points_with_index_dims(
            field_number,
            num_dims,
            num_dims,
            bytes_per_dim,
            points,
            segment_id,
        )
    }

    fn write_one_field_points_with_index_dims(
        field_number: i32,
        num_dims: i32,
        num_index_dims: i32,
        bytes_per_dim: i32,
        points: Vec<(i32, Vec<u8>)>,
        segment_id: &[u8; ID_LENGTH],
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let fields = vec![WritePointsField {
            field_number,
            num_dims,
            num_index_dims,
            bytes_per_dim,
            points,
        }];
        points::write(
            &fields,
            points::DEFAULT_MAX_POINTS_IN_LEAF_NODE,
            segment_id,
            "",
        )
        .unwrap()
    }

    #[test]
    fn two_sources_no_deletions_merge_points_correctly() {
        let seg0_id = [1u8; ID_LENGTH];
        let seg1_id = [2u8; ID_LENGTH];

        let (kdm0, kdi0, kdd0) =
            write_one_field_points(0, 1, 4, vec![(0, packed4(10)), (1, packed4(20))], &seg0_id);
        let points_reader0 = points::open(&kdm0, &kdi0, &kdd0, &seg0_id, "").unwrap();

        let (kdm1, kdi1, kdd1) = write_one_field_points(0, 1, 4, vec![(0, packed4(30))], &seg1_id);
        let points_reader1 = points::open(&kdm1, &kdi1, &kdd1, &seg1_id, "").unwrap();

        let fields = vec![points_field("loc", 0, 1, 4)];
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

        let src_points0 = [SourcePoints {
            field_number: 0,
            reader: &points_reader0,
        }];
        let src_points1 = [SourcePoints {
            field_number: 0,
            reader: &points_reader1,
        }];
        let source0 = MergeSource {
            field_infos: &stored0.fields,
            reader: &reader0,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &src_points0,
            vectors: None,
        };
        let source1 = MergeSource {
            field_infos: &stored1.fields,
            reader: &reader1,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &src_points1,
            vectors: None,
        };

        let sci = merge_stored_only_segments(
            &dir,
            &[source0, source1],
            "_merged_points",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();
        assert_eq!(sci.segment_name, "_merged_points");

        let kdm = std::fs::read(tmp.join("_merged_points.kdm")).unwrap();
        let kdi = std::fs::read(tmp.join("_merged_points.kdi")).unwrap();
        let kdd = std::fs::read(tmp.join("_merged_points.kdd")).unwrap();
        let merged_reader = points::open(&kdm, &kdi, &kdd, &[9u8; ID_LENGTH], "").unwrap();

        let mut merged_points = merged_reader.decode_all_points(0).unwrap();
        merged_points.sort_by_key(|p| p.doc_id);
        assert_eq!(
            merged_points
                .iter()
                .map(|p| (p.doc_id, p.packed_value.clone()))
                .collect::<Vec<_>>(),
            vec![
                (0, packed4(10)),
                (1, packed4(20)),
                // Source 1's only doc is renumbered to merged doc 2, after
                // source 0's 2 docs.
                (2, packed4(30)),
            ]
        );

        // Full round-trip: a range query through the unmodified points-range
        // resolver (the same one `lucene_search`'s points range query
        // composes) must return exactly the merged docs whose values fall in
        // range, using the real reader/decoder stack end to end.
        let in_range = crate::points_delete::resolve_points_range_doc_ids(
            &merged_reader,
            None,
            0,
            &packed4(15),
            &packed4(30),
        )
        .unwrap();
        assert_eq!(in_range, vec![1, 2]);
    }

    #[test]
    fn points_field_with_deletions_drops_non_live_docs() {
        let seg0_id = [1u8; ID_LENGTH];

        let (kdm0, kdi0, kdd0) = write_one_field_points(
            0,
            1,
            4,
            vec![(0, packed4(10)), (1, packed4(20)), (2, packed4(30))],
            &seg0_id,
        );
        let points_reader0 = points::open(&kdm0, &kdi0, &kdd0, &seg0_id, "").unwrap();

        let fields = vec![points_field("loc", 0, 1, 4)];
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let stored0 = flush(
            &dir,
            &tmp,
            "_0",
            seg0_id,
            &fields,
            &[doc_with(0, "a"), doc_with(0, "b"), doc_with(0, "c")],
        );
        let reader0 = open_reader(&stored0);

        // Drop doc 1 -- surviving docs 0 and 2 renumber to merged 0 and 1.
        let mut live0 = FixedBitSet::new(3);
        live0.set(0);
        live0.set(2);

        let src_points0 = [SourcePoints {
            field_number: 0,
            reader: &points_reader0,
        }];
        let source0 = MergeSource {
            field_infos: &stored0.fields,
            reader: &reader0,
            live_docs: Some(&live0),
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &src_points0,
            vectors: None,
        };

        merge_stored_only_segments(
            &dir,
            &[source0],
            "_merged_points_deletions",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();

        let kdm = std::fs::read(tmp.join("_merged_points_deletions.kdm")).unwrap();
        let kdi = std::fs::read(tmp.join("_merged_points_deletions.kdi")).unwrap();
        let kdd = std::fs::read(tmp.join("_merged_points_deletions.kdd")).unwrap();
        let merged_reader = points::open(&kdm, &kdi, &kdd, &[9u8; ID_LENGTH], "").unwrap();
        let mut merged_points = merged_reader.decode_all_points(0).unwrap();
        merged_points.sort_by_key(|p| p.doc_id);
        assert_eq!(
            merged_points
                .iter()
                .map(|p| (p.doc_id, p.packed_value.clone()))
                .collect::<Vec<_>>(),
            vec![(0, packed4(10)), (1, packed4(30))]
        );
    }

    #[test]
    fn fully_deleted_source_contributes_no_points() {
        let seg0_id = [1u8; ID_LENGTH];
        let seg1_id = [2u8; ID_LENGTH];

        let (kdm0, kdi0, kdd0) = write_one_field_points(0, 1, 4, vec![(0, packed4(10))], &seg0_id);
        let points_reader0 = points::open(&kdm0, &kdi0, &kdd0, &seg0_id, "").unwrap();
        let (kdm1, kdi1, kdd1) = write_one_field_points(0, 1, 4, vec![(0, packed4(99))], &seg1_id);
        let points_reader1 = points::open(&kdm1, &kdi1, &kdd1, &seg1_id, "").unwrap();

        let fields = vec![points_field("loc", 0, 1, 4)];
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let stored0 = flush(&dir, &tmp, "_0", seg0_id, &fields, &[doc_with(0, "a")]);
        let stored1 = flush(&dir, &tmp, "_1", seg1_id, &fields, &[doc_with(0, "b")]);
        let reader0 = open_reader(&stored0);
        let reader1 = open_reader(&stored1);

        let all_deleted = FixedBitSet::new(1); // source 1: nothing live

        let src_points0 = [SourcePoints {
            field_number: 0,
            reader: &points_reader0,
        }];
        let src_points1 = [SourcePoints {
            field_number: 0,
            reader: &points_reader1,
        }];
        let source0 = MergeSource {
            field_infos: &stored0.fields,
            reader: &reader0,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &src_points0,
            vectors: None,
        };
        let source1 = MergeSource {
            field_infos: &stored1.fields,
            reader: &reader1,
            live_docs: Some(&all_deleted),
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &src_points1,
            vectors: None,
        };

        merge_stored_only_segments(
            &dir,
            &[source0, source1],
            "_merged_points_fully_deleted",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();

        let kdm = std::fs::read(tmp.join("_merged_points_fully_deleted.kdm")).unwrap();
        let kdi = std::fs::read(tmp.join("_merged_points_fully_deleted.kdi")).unwrap();
        let kdd = std::fs::read(tmp.join("_merged_points_fully_deleted.kdd")).unwrap();
        let merged_reader = points::open(&kdm, &kdi, &kdd, &[9u8; ID_LENGTH], "").unwrap();
        let merged_points = merged_reader.decode_all_points(0).unwrap();
        assert_eq!(
            merged_points
                .iter()
                .map(|p| (p.doc_id, p.packed_value.clone()))
                .collect::<Vec<_>>(),
            vec![(0, packed4(10))]
        );
    }

    #[test]
    fn points_field_missing_in_a_live_contributing_source_is_an_error() {
        let seg0_id = [1u8; ID_LENGTH];

        let (kdm0, kdi0, kdd0) = write_one_field_points(0, 1, 4, vec![(0, packed4(10))], &seg0_id);
        let points_reader0 = points::open(&kdm0, &kdi0, &kdd0, &seg0_id, "").unwrap();

        let fields = vec![points_field("loc", 0, 1, 4)];
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let stored0 = flush(&dir, &tmp, "_0", seg0_id, &fields, &[doc_with(0, "a")]);
        // Source 1 declares the same "loc" field but supplies no points data
        // for it at all -- a schema mismatch, since source 0 has live docs
        // indexing that field.
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

        let src_points0 = [SourcePoints {
            field_number: 0,
            reader: &points_reader0,
        }];
        let source0 = MergeSource {
            field_infos: &stored0.fields,
            reader: &reader0,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &src_points0,
            vectors: None,
        };
        let source1 = MergeSource {
            field_infos: &stored1.fields,
            reader: &reader1,
            live_docs: None,
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
        };

        let result = merge_stored_only_segments(
            &dir,
            &[source0, source1],
            "_merged_missing_points",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        );
        assert!(matches!(
            result,
            Err(Error::PointsFieldMissingInSource { .. })
        ));
    }

    #[test]
    fn a_source_that_never_saw_the_points_field_simply_contributes_no_points() {
        // `PointsWriter.merge`: `if (readerFieldInfo == null) continue;`.
        // Adding a points field to an index that already has segments is
        // normal -- the older segments have no `FieldInfo` for it at all and
        // contribute nothing, rather than failing the merge.
        let seg0_id = [1u8; ID_LENGTH];
        let (kdm0, kdi0, kdd0) = write_one_field_points(0, 1, 4, vec![(0, packed4(10))], &seg0_id);
        let points_reader0 = points::open(&kdm0, &kdi0, &kdd0, &seg0_id, "").unwrap();

        let fields0 = vec![points_field("loc", 0, 1, 4)];
        // Source 1 has a different field entirely; "loc" is unknown to it.
        let fields1 = vec![field("id", 0)];
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let stored0 = flush(&dir, &tmp, "_0", seg0_id, &fields0, &[doc_with(0, "a")]);
        let stored1 = flush(
            &dir,
            &tmp,
            "_1",
            [2u8; ID_LENGTH],
            &fields1,
            &[doc_with(0, "b")],
        );
        let reader0 = open_reader(&stored0);
        let reader1 = open_reader(&stored1);

        let src_points0 = [SourcePoints {
            field_number: 0,
            reader: &points_reader0,
        }];
        let source0 = MergeSource {
            points: &src_points0,
            vectors: None,
            ..MergeSource::stored_only(&stored0.fields, &reader0, None)
        };
        let source1 = MergeSource::stored_only(&stored1.fields, &reader1, None);

        merge_stored_only_segments(
            &dir,
            &[source0, source1],
            "_merged_points_absent_source",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();

        let kdm = std::fs::read(tmp.join("_merged_points_absent_source.kdm")).unwrap();
        let kdi = std::fs::read(tmp.join("_merged_points_absent_source.kdi")).unwrap();
        let kdd = std::fs::read(tmp.join("_merged_points_absent_source.kdd")).unwrap();
        let merged = points::open(&kdm, &kdi, &kdd, &[9u8; ID_LENGTH], "").unwrap();
        let all = merged.decode_all_points(0).unwrap();
        assert_eq!(
            all.iter()
                .map(|p| (p.doc_id, p.packed_value.clone()))
                .collect::<Vec<_>>(),
            vec![(0, packed4(10))],
            "only source 0's point survives, at merged doc 0"
        );
    }

    #[test]
    fn points_shape_disagreement_across_sources_is_an_error() {
        let seg0_id = [1u8; ID_LENGTH];
        let seg1_id = [2u8; ID_LENGTH];

        let (kdm0, kdi0, kdd0) = write_one_field_points(0, 1, 4, vec![(0, packed4(10))], &seg0_id);
        let points_reader0 = points::open(&kdm0, &kdi0, &kdd0, &seg0_id, "").unwrap();
        // Source 1's own points data uses 8 bytes per dimension, disagreeing
        // with the merged field's declared shape (4 bytes per dimension,
        // taken from source 0's FieldInfo since it's first-seen).
        let (kdm1, kdi1, kdd1) = write_one_field_points(
            0,
            1,
            8,
            vec![(0, vec![0u8, 0, 0, 0, 0, 0, 0, 42])],
            &seg1_id,
        );
        let points_reader1 = points::open(&kdm1, &kdi1, &kdd1, &seg1_id, "").unwrap();

        let fields = vec![points_field("loc", 0, 1, 4)];
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let stored0 = flush(&dir, &tmp, "_0", seg0_id, &fields, &[doc_with(0, "a")]);
        let stored1 = flush(&dir, &tmp, "_1", seg1_id, &fields, &[doc_with(0, "b")]);
        let reader0 = open_reader(&stored0);
        let reader1 = open_reader(&stored1);

        let src_points0 = [SourcePoints {
            field_number: 0,
            reader: &points_reader0,
        }];
        let src_points1 = [SourcePoints {
            field_number: 0,
            reader: &points_reader1,
        }];
        let source0 = MergeSource {
            field_infos: &stored0.fields,
            reader: &reader0,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &src_points0,
            vectors: None,
        };
        let source1 = MergeSource {
            field_infos: &stored1.fields,
            reader: &reader1,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &src_points1,
            vectors: None,
        };

        let result = merge_stored_only_segments(
            &dir,
            &[source0, source1],
            "_merged_points_shape_mismatch",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        );
        assert!(matches!(result, Err(Error::PointsShapeDisagreement { .. })));
    }

    /// Two sources, both `num_dims=3`/`num_index_dims=2` (a
    /// `LatLonShape`-style bounding box shape with one trailing data-only
    /// dimension), merge cleanly: every point's full 3-dimension packed
    /// value -- including the non-indexed third dimension -- survives the
    /// merge unchanged, and doc ids are renumbered into the merged id space
    /// exactly like the `num_index_dims == num_dims` case already covered by
    /// `two_sources_no_deletions_merge_points_correctly`.
    #[test]
    fn two_sources_consistent_num_index_dims_less_than_num_dims_merge_correctly() {
        let seg0_id = [1u8; ID_LENGTH];
        let seg1_id = [2u8; ID_LENGTH];

        let packed12 = |a: u32, b: u32, c: u32| -> Vec<u8> {
            [a.to_be_bytes(), b.to_be_bytes(), c.to_be_bytes()].concat()
        };

        let (kdm0, kdi0, kdd0) = write_one_field_points_with_index_dims(
            0,
            3,
            2,
            4,
            vec![(0, packed12(1, 2, 100)), (1, packed12(3, 4, 200))],
            &seg0_id,
        );
        let points_reader0 = points::open(&kdm0, &kdi0, &kdd0, &seg0_id, "").unwrap();

        let (kdm1, kdi1, kdd1) = write_one_field_points_with_index_dims(
            0,
            3,
            2,
            4,
            vec![(0, packed12(5, 6, 300))],
            &seg1_id,
        );
        let points_reader1 = points::open(&kdm1, &kdi1, &kdd1, &seg1_id, "").unwrap();

        let fields = vec![points_field_with_index_dims("loc", 0, 3, 2, 4)];
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

        let src_points0 = [SourcePoints {
            field_number: 0,
            reader: &points_reader0,
        }];
        let src_points1 = [SourcePoints {
            field_number: 0,
            reader: &points_reader1,
        }];
        let source0 = MergeSource {
            field_infos: &stored0.fields,
            reader: &reader0,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &src_points0,
            vectors: None,
        };
        let source1 = MergeSource {
            field_infos: &stored1.fields,
            reader: &reader1,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &src_points1,
            vectors: None,
        };

        let sci = merge_stored_only_segments(
            &dir,
            &[source0, source1],
            "_merged_points_index_dims_lt_dims",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();
        assert_eq!(sci.segment_name, "_merged_points_index_dims_lt_dims");

        let base = tmp.join("_merged_points_index_dims_lt_dims");
        let kdm = std::fs::read(base.with_extension("kdm")).unwrap();
        let kdi = std::fs::read(base.with_extension("kdi")).unwrap();
        let kdd = std::fs::read(base.with_extension("kdd")).unwrap();
        let merged_reader = points::open(&kdm, &kdi, &kdd, &[9u8; ID_LENGTH], "").unwrap();

        let merged_field = merged_reader.field(0).unwrap();
        assert_eq!(merged_field.num_dims, 3);
        assert_eq!(merged_field.num_index_dims, 2);

        let mut merged_points = merged_reader.decode_all_points(0).unwrap();
        merged_points.sort_by_key(|p| p.doc_id);
        assert_eq!(
            merged_points
                .iter()
                .map(|p| (p.doc_id, p.packed_value.clone()))
                .collect::<Vec<_>>(),
            vec![
                (0, packed12(1, 2, 100)),
                (1, packed12(3, 4, 200)),
                // Source 1's only doc is renumbered to merged doc 2, after
                // source 0's 2 docs.
                (2, packed12(5, 6, 300)),
            ]
        );
    }

    /// Two sources whose points data agree on `num_dims`/`bytes_per_dim` but
    /// disagree on `num_index_dims` (source 0 is `num_index_dims=1`, source 1
    /// is `num_index_dims=2`, both `num_dims=2`) -- field-number
    /// reconciliation records source 0's `FieldInfo` (`num_index_dims=1`) as
    /// the merged field's declared shape, so source 1's own
    /// `num_index_dims=2` disagrees and must be rejected rather than having
    /// its second dimension silently reinterpreted as a data-only payload
    /// dimension.
    #[test]
    fn points_index_dims_disagreement_across_sources_is_rejected() {
        let seg0_id = [1u8; ID_LENGTH];
        let seg1_id = [2u8; ID_LENGTH];

        let packed8 = |a: u32, b: u32| -> Vec<u8> { [a.to_be_bytes(), b.to_be_bytes()].concat() };

        let (kdm0, kdi0, kdd0) = write_one_field_points_with_index_dims(
            0,
            2,
            1,
            4,
            vec![(0, packed8(10, 11))],
            &seg0_id,
        );
        let points_reader0 = points::open(&kdm0, &kdi0, &kdd0, &seg0_id, "").unwrap();

        let (kdm1, kdi1, kdd1) = write_one_field_points_with_index_dims(
            0,
            2,
            2,
            4,
            vec![(0, packed8(20, 21))],
            &seg1_id,
        );
        let points_reader1 = points::open(&kdm1, &kdi1, &kdd1, &seg1_id, "").unwrap();

        let fields0 = vec![points_field_with_index_dims("loc", 0, 2, 1, 4)];
        let fields1 = vec![points_field_with_index_dims("loc", 0, 2, 2, 4)];
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let stored0 = flush(&dir, &tmp, "_0", seg0_id, &fields0, &[doc_with(0, "a")]);
        let stored1 = flush(&dir, &tmp, "_1", seg1_id, &fields1, &[doc_with(0, "b")]);
        let reader0 = open_reader(&stored0);
        let reader1 = open_reader(&stored1);

        let src_points0 = [SourcePoints {
            field_number: 0,
            reader: &points_reader0,
        }];
        let src_points1 = [SourcePoints {
            field_number: 0,
            reader: &points_reader1,
        }];
        let source0 = MergeSource {
            field_infos: &stored0.fields,
            reader: &reader0,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &src_points0,
            vectors: None,
        };
        let source1 = MergeSource {
            field_infos: &stored1.fields,
            reader: &reader1,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &src_points1,
            vectors: None,
        };

        let result = merge_stored_only_segments(
            &dir,
            &[source0, source1],
            "_merged_points_index_dims_mismatch",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        );
        assert!(matches!(
            result,
            Err(Error::PointsIndexDimsDisagreement { .. })
        ));
    }

    // --- the three stored-fields merge strategies ---

    fn strategies_for(sources: &[MergeSource]) -> Vec<StoredFieldsMergeStrategy> {
        let sources_fields: Vec<&[FieldInfo]> = sources.iter().map(|s| s.field_infos).collect();
        let (_, maps) = reconcile_field_numbers(&sources_fields).unwrap();
        let matching = matching_readers(sources, &maps);
        let writer = stored_fields::StoredFieldsWriter::new(
            stored_fields::Mode::BestSpeed,
            &[0u8; ID_LENGTH],
            "",
        );
        sources
            .iter()
            .zip(&matching)
            .map(|(s, &m)| stored_fields_merge_strategy(&writer, s, m))
            .collect()
    }

    fn read_merged(
        tmp: &std::path::Path,
        name: &str,
        id: [u8; ID_LENGTH],
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let base = tmp;
        let _ = id;
        (
            std::fs::read(base.join(format!("{name}.fdt"))).unwrap(),
            std::fs::read(base.join(format!("{name}.fdx"))).unwrap(),
            std::fs::read(base.join(format!("{name}.fdm"))).unwrap(),
        )
    }

    #[test]
    fn a_matching_deletion_free_source_is_bulk_copied_verbatim() {
        // The BULK path, over several real chunk boundaries: the merged
        // segment must hold exactly the source documents, and -- because
        // nothing was recompressed -- exactly the sources' chunk counts.
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let fields = vec![field("id", 0)];
        const N: usize = 2_500; // > 2 * the 1024-doc chunk cap
        let docs_a: Vec<Document> = (0..N).map(|i| doc_with(0, &format!("a{i}"))).collect();
        let docs_b: Vec<Document> = (0..N).map(|i| doc_with(0, &format!("b{i}"))).collect();
        let seg0 = flush(&dir, &tmp, "_0", [1u8; ID_LENGTH], &fields, &docs_a);
        let seg1 = flush(&dir, &tmp, "_1", [2u8; ID_LENGTH], &fields, &docs_b);
        let reader0 = open_reader(&seg0);
        let reader1 = open_reader(&seg1);
        let sources = vec![
            MergeSource::stored_only(&seg0.fields, &reader0, None),
            MergeSource::stored_only(&seg1.fields, &reader1, None),
        ];
        assert_eq!(
            strategies_for(&sources),
            vec![
                StoredFieldsMergeStrategy::Bulk,
                StoredFieldsMergeStrategy::Bulk
            ]
        );

        merge_stored_only_segments(
            &dir,
            &sources,
            "_merged",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();

        let (fdt, fdx, fdm) = read_merged(&tmp, "_merged", [9u8; ID_LENGTH]);
        let merged = stored_fields::open(&fdt, &fdx, &fdm, &[9u8; ID_LENGTH], "").unwrap();
        assert_eq!(merged.max_doc(), 2 * N as i32);
        assert_eq!(
            merged.num_chunks(),
            reader0.num_chunks() + reader1.num_chunks(),
            "every chunk should have been copied, not rebuilt"
        );
        for i in 0..N {
            assert_eq!(
                merged.document(i as i32).unwrap().fields[0].value,
                FieldValue::String(format!("a{i}"))
            );
            assert_eq!(
                merged.document((N + i) as i32).unwrap().fields[0].value,
                FieldValue::String(format!("b{i}"))
            );
        }
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn a_source_with_deletions_takes_the_doc_path_and_drops_exactly_the_deleted_docs() {
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let fields = vec![field("id", 0)];
        const N: usize = 1_500;
        let docs: Vec<Document> = (0..N).map(|i| doc_with(0, &format!("d{i}"))).collect();
        let seg0 = flush(&dir, &tmp, "_0", [1u8; ID_LENGTH], &fields, &docs);
        let reader0 = open_reader(&seg0);

        // Every third document deleted, spanning both of the source's chunks.
        let mut live = FixedBitSet::new(N);
        for i in 0..N {
            if i % 3 != 0 {
                live.set(i);
            }
        }
        let sources = vec![MergeSource::stored_only(
            &seg0.fields,
            &reader0,
            Some(&live),
        )];
        assert_eq!(
            strategies_for(&sources),
            vec![StoredFieldsMergeStrategy::Doc]
        );

        merge_stored_only_segments(
            &dir,
            &sources,
            "_merged",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();

        let (fdt, fdx, fdm) = read_merged(&tmp, "_merged", [9u8; ID_LENGTH]);
        let merged = stored_fields::open(&fdt, &fdx, &fdm, &[9u8; ID_LENGTH], "").unwrap();
        let survivors: Vec<usize> = (0..N).filter(|i| i % 3 != 0).collect();
        assert_eq!(merged.max_doc(), survivors.len() as i32);
        for (merged_id, &source_id) in survivors.iter().enumerate() {
            assert_eq!(
                merged.document(merged_id as i32).unwrap().fields[0].value,
                FieldValue::String(format!("d{source_id}")),
                "merged doc {merged_id}"
            );
        }
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn a_source_whose_field_numbers_change_takes_the_visitor_path_and_is_renumbered() {
        // Source 1 numbers "body" 0 and "id" 1; the merge numbers them the
        // other way round (source 0 was seen first). Its serialized bytes
        // therefore cannot be copied -- they encode the source's own numbers.
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let fields0 = vec![field("id", 0), field("body", 1)];
        let fields1 = vec![field("body", 0), field("id", 1)];
        let docs0 = vec![Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("id-0".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::String("body-0".to_string()),
                },
            ],
        }];
        let docs1 = vec![Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("body-1".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::String("id-1".to_string()),
                },
            ],
        }];
        let seg0 = flush(&dir, &tmp, "_0", [1u8; ID_LENGTH], &fields0, &docs0);
        let seg1 = flush(&dir, &tmp, "_1", [2u8; ID_LENGTH], &fields1, &docs1);
        let reader0 = open_reader(&seg0);
        let reader1 = open_reader(&seg1);
        let sources = vec![
            MergeSource::stored_only(&seg0.fields, &reader0, None),
            MergeSource::stored_only(&seg1.fields, &reader1, None),
        ];
        assert_eq!(
            strategies_for(&sources),
            vec![
                StoredFieldsMergeStrategy::Bulk,
                StoredFieldsMergeStrategy::Visitor
            ]
        );

        merge_stored_only_segments(
            &dir,
            &sources,
            "_merged",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();

        let (fdt, fdx, fdm) = read_merged(&tmp, "_merged", [9u8; ID_LENGTH]);
        let merged = stored_fields::open(&fdt, &fdx, &fdm, &[9u8; ID_LENGTH], "").unwrap();
        assert_eq!(merged.max_doc(), 2);
        // Merged numbering: id = 0, body = 1, for both documents.
        for (doc_id, want) in [(0, "0"), (1, "1")] {
            let doc = merged.document(doc_id).unwrap();
            let id_field = doc.fields.iter().find(|f| f.field_number == 0).unwrap();
            let body_field = doc.fields.iter().find(|f| f.field_number == 1).unwrap();
            assert_eq!(id_field.value, FieldValue::String(format!("id-{want}")));
            assert_eq!(body_field.value, FieldValue::String(format!("body-{want}")));
        }
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn matching_readers_is_the_identity_field_number_test() {
        // Source 0's numbering survives; source 1's does not, because "body"
        // was already claimed number 1 by source 0; source 2's does, because
        // it agrees with source 0.
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let fields0 = vec![field("id", 0), field("body", 1)];
        let fields1 = vec![field("body", 0)];
        let fields2 = vec![field("id", 0), field("body", 1)];
        let seg0 = flush(
            &dir,
            &tmp,
            "_0",
            [1u8; ID_LENGTH],
            &fields0,
            &[doc_with(0, "a")],
        );
        let seg1 = flush(
            &dir,
            &tmp,
            "_1",
            [2u8; ID_LENGTH],
            &fields1,
            &[doc_with(0, "b")],
        );
        let seg2 = flush(
            &dir,
            &tmp,
            "_2",
            [3u8; ID_LENGTH],
            &fields2,
            &[doc_with(0, "c")],
        );
        let (r0, r1, r2) = (open_reader(&seg0), open_reader(&seg1), open_reader(&seg2));
        let sources = vec![
            MergeSource::stored_only(&seg0.fields, &r0, None),
            MergeSource::stored_only(&seg1.fields, &r1, None),
            MergeSource::stored_only(&seg2.fields, &r2, None),
        ];
        let sources_fields: Vec<&[FieldInfo]> = sources.iter().map(|s| s.field_infos).collect();
        let (_, maps) = reconcile_field_numbers(&sources_fields).unwrap();
        assert_eq!(maps[0].get(&0), Some(&0));
        assert_eq!(
            maps[1].get(&0),
            Some(&1),
            "source 1's \"body\" is renumbered"
        );
        assert_eq!(matching_readers(&sources, &maps), vec![true, false, true]);
        // ...and that is what decides BULK vs VISITOR.
        assert_eq!(
            strategies_for(&sources),
            vec![
                StoredFieldsMergeStrategy::Bulk,
                StoredFieldsMergeStrategy::Visitor,
                StoredFieldsMergeStrategy::Bulk
            ]
        );
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn a_source_whose_fdt_fails_its_own_checksum_is_never_bulk_copied() {
        // `checkIntegrity` before the strategy choice: the bulk path copies a
        // source's compressed bytes verbatim and writes a fresh, valid footer
        // over them, so without this the merge would launder a corrupt source
        // into a segment that passes every checksum from then on.
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let fields = vec![field("id", 0)];
        let docs: Vec<Document> = (0..50).map(|i| doc_with(0, &format!("v{i}"))).collect();
        let mut seg = flush(&dir, &tmp, "_0", [1u8; ID_LENGTH], &fields, &docs);
        // Flip one byte of the compressed payload, leaving every length,
        // pointer and footer field intact -- exactly the shape of corruption
        // `retrieve_checksum` cannot see.
        let mid = seg.fdt.len() / 2;
        seg.fdt[mid] ^= 0xFF;

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
        assert!(
            result.is_err(),
            "a source whose .fdt fails its checksum must not be merged"
        );
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn a_live_docs_bitset_shorter_than_max_doc_is_reported_not_indexed_past() {
        // `docs/arithmetic-gate.md`'s crate rule: *never index a
        // `FixedBitSet` with an index bounded against anything other than
        // that bitset's own `len()`.* The doc id here is bounded by the
        // source's stored-fields `maxDoc` (off the `.fdm`) and the bitset
        // comes off the `.liv` -- two independent files.
        //
        // Both failure modes c28's rule names are covered:
        //  * an **empty** bitset, where `words` itself is empty, so
        //    `words[index >> 6]` is a real index panic in release as well as
        //    debug (in a debug build `FixedBitSet::get`'s own bare
        //    `debug_assert!(index < self.num_bits)` fires first) -- either
        //    way this case aborts the test without the fix;
        //  * a bitset a few bits **short**, where the index still lands
        //    inside `words` and reads a ghost bit: five documents silently
        //    dropped from the merged segment, with nothing reported.
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let fields = vec![field("id", 0)];
        let docs: Vec<Document> = (0..50).map(|i| doc_with(0, &format!("v{i}"))).collect();
        let seg = flush(&dir, &tmp, "_0", [1u8; ID_LENGTH], &fields, &docs);
        let reader = open_reader(&seg);

        for short_len in [0usize, 45] {
            let mut bits = FixedBitSet::new(short_len);
            for d in 0..short_len {
                bits.set(d);
            }
            let sources = vec![MergeSource::stored_only(&seg.fields, &reader, Some(&bits))];
            let result = merge_stored_only_segments(
                &dir,
                &sources,
                "_merged",
                [9u8; ID_LENGTH],
                "Lucene104",
                version(),
            );
            assert!(
                matches!(
                    result,
                    Err(Error::LiveDocsLengthMismatch {
                        source_index: 0,
                        max_doc: 50,
                        live_docs_len,
                    }) if live_docs_len == short_len
                ),
                "a .liv covering {short_len} of 50 documents must be reported, got {result:?}"
            );
        }

        // The control: a bitset that *does* cover `maxDoc` merges, and the
        // deletion it records is honoured -- so the check above rejects a
        // disagreement rather than every live-docs merge.
        let mut exact = FixedBitSet::new(50);
        for d in 0..50 {
            exact.set(d);
        }
        exact.clear(7);
        let sources = vec![MergeSource::stored_only(&seg.fields, &reader, Some(&exact))];
        let merged = merge_stored_only_segments(
            &dir,
            &sources,
            "_merged",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .expect("a .liv the right length merges");
        let si_bytes = dir.open("_merged.si").unwrap().to_vec();
        let si = segment_info::parse(&si_bytes, &merged.segment_id).unwrap();
        assert_eq!(si.doc_count, 49, "the one deleted document was dropped");
        std::fs::remove_dir_all(&tmp).ok();
    }

    // --- the merged `.fnm` must describe the files the merge wrote ---

    #[test]
    fn a_merged_field_with_postings_gets_the_per_field_format_attributes() {
        let mut fields = vec![field("body", 0)];
        fields[0].index_options = IndexOptions::DocsAndFreqs;
        fields[0]
            .attributes
            .push(("keep".to_string(), "me".to_string()));
        describe_written_files(&mut fields, &[0], &[], &[], false, &[]);
        assert!(fields[0]
            .attributes
            .contains(&("keep".to_string(), "me".to_string())));
        assert!(fields[0].attributes.contains(&(
            "PerFieldPostingsFormat.format".to_string(),
            POSTINGS_FORMAT_NAME.to_string()
        )));
        assert!(fields[0].attributes.contains(&(
            "PerFieldPostingsFormat.suffix".to_string(),
            PER_FIELD_SUFFIX.to_string()
        )));
    }

    #[test]
    fn a_stale_per_field_attribute_inherited_from_a_source_is_stripped() {
        // The source's own `.fnm` names a postings format; this merge wrote no
        // postings, so a reader following the attribute would look for a file
        // that does not exist.
        let mut fields = vec![field("body", 0)];
        fields[0].index_options = IndexOptions::DocsAndFreqs;
        fields[0].attributes.push((
            "PerFieldPostingsFormat.format".to_string(),
            POSTINGS_FORMAT_NAME.to_string(),
        ));
        fields[0].attributes.push((
            "PerFieldDocValuesFormat.suffix".to_string(),
            "0".to_string(),
        ));
        describe_written_files(&mut fields, &[], &[], &[], false, &[]);
        assert!(fields[0].attributes.is_empty());
    }

    #[test]
    fn an_indexed_field_the_merge_wrote_no_norms_for_must_omit_them() {
        // `DirectoryReader.open` throws on the missing `.nvm` rather than
        // degrading, so this is the difference between an openable index and
        // an unopenable one.
        let mut fields = vec![field("body", 0), field("title", 1)];
        for f in fields.iter_mut() {
            f.index_options = IndexOptions::DocsAndFreqs;
            f.omit_norms = false;
        }
        describe_written_files(&mut fields, &[0, 1], &[], &[1], false, &[]);
        assert!(fields[0].omit_norms, "no norms written for field 0");
        assert!(!fields[1].omit_norms, "norms written for field 1");
    }

    #[test]
    fn a_field_claiming_term_vectors_loses_the_claim_when_none_were_written() {
        let mut fields = vec![field("body", 0)];
        fields[0].index_options = IndexOptions::DocsAndFreqs;
        fields[0].store_term_vectors = true;
        describe_written_files(&mut fields, &[], &[], &[], false, &[]);
        assert!(!fields[0].store_term_vectors);

        let mut fields = vec![field("body", 0)];
        fields[0].index_options = IndexOptions::DocsAndFreqs;
        fields[0].store_term_vectors = true;
        describe_written_files(&mut fields, &[], &[], &[], true, &[]);
        assert!(fields[0].store_term_vectors);
    }

    #[test]
    fn a_merged_doc_values_field_gets_its_per_field_format_attributes() {
        let mut fields = vec![field("score", 0)];
        fields[0].doc_values_type = DocValuesType::Numeric;
        describe_written_files(&mut fields, &[], &[0], &[], false, &[]);
        assert!(fields[0].attributes.contains(&(
            "PerFieldDocValuesFormat.format".to_string(),
            DOC_VALUES_FORMAT_NAME.to_string()
        )));
        assert!(fields[0].attributes.contains(&(
            "PerFieldDocValuesFormat.suffix".to_string(),
            PER_FIELD_SUFFIX.to_string()
        )));
        // A non-indexed field never gains an `omit_norms` rewrite.
        assert!(!fields[0].omit_norms);
    }

    // --- BKDWriter.merge's k-way point merge ---

    fn pt(doc: i32, v: u32) -> (i32, Vec<u8>) {
        (doc, v.to_be_bytes().to_vec())
    }

    #[test]
    fn one_dimension_point_streams_are_k_way_merged_into_one_sorted_stream() {
        // Three sorted sources, disjoint merged doc-id ranges (as
        // `build_doc_id_maps` guarantees), interleaved values.
        let a = vec![pt(0, 1), pt(1, 4), pt(2, 9)];
        let b = vec![pt(3, 2), pt(4, 5)];
        let c = vec![pt(5, 0), pt(6, 3), pt(7, 100)];
        let merged = merge_point_streams(vec![a, b, c], 1, 4);
        let values: Vec<u32> = merged
            .iter()
            .map(|(_, v)| u32::from_be_bytes(v[..4].try_into().unwrap()))
            .collect();
        assert_eq!(values, vec![0, 1, 2, 3, 4, 5, 9, 100]);
        assert_eq!(
            merged.iter().map(|(d, _)| *d).collect::<Vec<i32>>(),
            vec![5, 0, 3, 6, 1, 4, 2, 7]
        );
    }

    #[test]
    fn equal_point_values_are_ordered_by_document_id() {
        // `mergeComparator`'s `thenComparingInt(mr -> mr.docID)`.
        let a = vec![pt(7, 5)];
        let b = vec![pt(2, 5)];
        let c = vec![pt(4, 5)];
        let merged = merge_point_streams(vec![a, b, c], 1, 4);
        assert_eq!(
            merged.iter().map(|(d, _)| *d).collect::<Vec<i32>>(),
            vec![2, 4, 7]
        );
    }

    #[test]
    fn a_multi_index_dimension_field_is_concatenated_not_merged() {
        // No total order on values, so Java re-indexes -- and so does this.
        let a = vec![(0, vec![9, 9]), (1, vec![0, 0])];
        let b = vec![(2, vec![5, 5])];
        let merged = merge_point_streams(vec![a.clone(), b.clone()], 2, 1);
        assert_eq!(merged, [a, b].concat());
    }

    #[test]
    fn an_unsorted_source_stream_falls_back_to_concatenation() {
        // A hand-built `MergeSource`, or a segment some other writer produced:
        // the one-pass path's precondition is *checked*, never assumed, and
        // `points::write` sorts whatever it is handed.
        let a = vec![pt(0, 9), pt(1, 1)];
        let b = vec![pt(2, 5)];
        let merged = merge_point_streams(vec![a.clone(), b.clone()], 1, 4);
        assert_eq!(merged, [a, b].concat());
    }

    #[test]
    fn merging_empty_and_single_point_streams_is_well_defined() {
        assert!(merge_point_streams(Vec::new(), 1, 4).is_empty());
        assert!(merge_point_streams(vec![Vec::new(), Vec::new()], 1, 4).is_empty());
        let only = vec![pt(3, 7)];
        assert_eq!(
            merge_point_streams(vec![Vec::new(), only.clone(), Vec::new()], 1, 4),
            only
        );
    }
}
