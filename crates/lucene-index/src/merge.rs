//! Port of `org.apache.lucene.index.SegmentMerger` (plus the field-numbering
//! half of `FieldInfos.FieldNumbers`) -- merges N already-flushed segments
//! into one new segment, dropping deleted docs and renumbering doc ids to be
//! contiguous (`0..mergedDocCount`). Stored fields are always merged; doc
//! values, norms, term vectors, postings, and now BKD points are merged too
//! whenever a source supplies them (see "Doc values / norms / term vectors",
//! "Postings", and "Points" below for the honest scope of each part).
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
//! 3. merges any supplied doc-values/norms/term-vectors/postings/points data
//!    the same way (drop deleted docs, renumber contiguously, remap field
//!    numbers), then writes stored fields, field infos, segment info, and
//!    whichever of `.dvm`/`.dvd`/`.dvs`, `.nvm`/`.nvd`, `.tvd`/`.tvx`/`.tvm`,
//!    `.doc`/`.tim`/`.tip`/`.tmd`, `.kdm`/`.kdi`/`.kdd` the merge produced,
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
//! [`lucene_codecs::doc_values::write_single_dense_numeric_field`],
//! [`lucene_codecs::doc_values::write_single_dense_binary_field`],
//! [`lucene_codecs::doc_values::write_single_dense_sorted_field`],
//! [`lucene_codecs::doc_values::write_single_dense_sorted_numeric_field`],
//! [`lucene_codecs::doc_values::write_single_dense_sorted_set_field`], and
//! [`lucene_codecs::norms::write_single_dense_field`] each write a complete,
//! self-contained `.dvm`/`.dvd`/`.dvs` (or `.nvm`/`.nvd`) file pair/triple
//! for exactly **one field** -- these five are now thin one-field wrappers
//! over [`lucene_codecs::doc_values::write_dense_fields`], which *can* write
//! multiple distinct fields (of the same or different doc-values types) into
//! one multi-field `.dvd`/`.dvm`/`.dvs` container, the real on-disk shape
//! where every field's data shares one file. **This merge module does not
//! yet consume that capability**: it's a documented, deliberate scope
//! boundary, not a silent gap -- wiring multi-field merges through would mean
//! reworking [`MergeSource`]'s one-`Option<...>`-per-type shape into
//! per-source field lists, which is out of scope for the task that added
//! `write_dense_fields`. So this merge still inherits the old limit: at most
//! one numeric-doc-values field, at most one BINARY-doc-
//! values field, at most one SORTED-doc-values field, at most one
//! SORTED_NUMERIC-doc-values field, at most one SORTED_SET-doc-values field,
//! and at most one norms field may be merged per call
//! ([`Error::TooManyNumericDocValuesFields`] /
//! [`Error::TooManyBinaryDocValuesFields`] /
//! [`Error::TooManySortedDocValuesFields`] /
//! [`Error::TooManySortedNumericDocValuesFields`] /
//! [`Error::TooManySortedSetDocValuesFields`] /
//! [`Error::TooManyNormsFields`] otherwise) -- and, since this port's
//! numeric, BINARY, SORTED, SORTED_NUMERIC, and SORTED_SET writers all land
//! on the same `.dvm`/`.dvd`/`.dvs` extensions, at most one of those five
//! doc-values types may be merged in the same call
//! ([`Error::MultipleDocValuesTypesInOneMerge`]). Term vectors have no such
//! limit (`write_best_speed` already handles any number of fields per doc).
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
//! [`Error::BinaryDocValuesFieldMissingInSource`] /
//! [`Error::SortedDocValuesFieldMissingInSource`] /
//! [`Error::NormsFieldMissingInSource`] rather than silently dropping the
//! field or a doc's value.
//!
//! Term vectors have no such constraint: a source with no term-vectors
//! reader for a doc, or a doc with none, simply contributes an empty
//! [`lucene_codecs::term_vectors::TermVectorsDocument`] (matches the real
//! per-doc "this doc has none" case `write_best_speed` already handles).
//! Term vectors do have one different constraint, though: `write_best_speed`
//! only supports fields with positions (no offsets, no payloads -- see its
//! own doc comment), so [`merge_term_vectors`] validates every source's term-
//! vector fields up front and returns
//! [`Error::TermVectorOffsetsOrPayloadsNotSupported`] rather than letting an
//! offsets/payloads field reach `write_best_speed`'s internal `assert!` and
//! panic. Positions-only term vectors (with or without positions at all)
//! merge and round-trip correctly through the real reader/writer stack; this
//! is otherwise the same "reuse the existing decoder/encoder verbatim" story
//! as postings.
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
//! for postings. The same "sparse across sources" philosophy still applies
//! at the field level: a source that contributes live docs but has no
//! postings field at all for a name another live-doc-contributing source
//! does is a hard error ([`Error::PostingsFieldMissingInSource`]), not a
//! silent drop -- ordinary per-doc/per-term sparsity (most docs don't
//! contain most terms) is not an error, since that's exactly what a term
//! dictionary already models.
//!
//! **Scope: `IndexOptions::Docs`/`DocsAndFreqs` only.** Positions, offsets,
//! and payloads (`.pos`/`.pay`) are not merged -- a field whose merged
//! `index_options` indexes positions is rejected with
//! [`Error::PostingsIndexOptionsNotSupported`] rather than silently
//! dropping that data. This mirrors the doc-values/norms merge's own
//! "start with the smallest defensible slice, be honest about the rest"
//! precedent; extending it to positions/offsets/payloads is a documented
//! follow-up (see `docs/parity.md`). Because field-number reconciliation
//! only records the *first-seen* source's `FieldInfo` as the merged one and
//! never checks agreement across sources sharing a field name, every other
//! live-doc-contributing source's own `index_options` for that field is
//! independently checked against the merged choice
//! ([`Error::PostingsIndexOptionsDisagreement`]) -- otherwise a source with
//! positions could have them silently dropped whenever an earlier,
//! positions-free source happened to be picked as canonical.
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
//! A merged field with points data in at least one live-doc-contributing
//! source but not in every such source is a hard error
//! ([`Error::PointsFieldMissingInSource`]), matching the "sparse across
//! sources" philosophy applied everywhere else in this module -- but unlike
//! doc-values/norms, a field has no per-doc denseness requirement of its
//! own here: a live doc contributing zero points for a field (or a field
//! that ends up with zero surviving points overall, e.g. every
//! contributing doc's point belonged to a deleted doc) is not an error --
//! points are naturally sparse the same way postings are. A field that ends
//! up with zero points after the merge is simply omitted from the merged
//! segment (matching real Lucene's `finish()` returning `null`/omitting the
//! field, and [`lucene_codecs::points::write`]'s own documented
//! `EmptyField` restriction).
//!
//! **Scope: `num_index_dims == num_dims` only, single packed-value shape per
//! field across all sources.** [`lucene_codecs::points::write`] always
//! treats `num_index_dims` as equal to `num_dims` (see its own doc comment)
//! -- a source field with extra, non-indexed data-only dimensions
//! (`num_index_dims != num_dims`) is rejected with
//! [`Error::PointsIndexDimsNotSupported`] rather than silently dropping the
//! non-indexed dimensions. And because field-number reconciliation only
//! records the *first-seen* source's `FieldInfo` as the merged one, every
//! other live-doc-contributing source's own BKD tree shape
//! (`num_dims`/`bytes_per_dim`) is independently checked against the merged
//! field's declared shape and rejected with
//! [`Error::PointsShapeDisagreement`] on a mismatch -- otherwise a source
//! using, say, 2 dimensions could have its points silently misinterpreted as
//! 1-dimensional (or vice versa) whenever an earlier, differently-shaped
//! source happened to be picked as canonical. Multi-dimension points (e.g.
//! `LatLonPoint`-shaped 2D fields) and multi-valued points (multiple points
//! per doc for the same field) are both supported -- this is exactly what
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
//! See `docs/parity.md` and `PLAN.md`'s Phase 5 section for the exact,
//! currently-true scope line.

use std::collections::{BTreeSet, HashMap, HashSet};

use crate::segment_info::{self, IndexSortField, LuceneVersion, SegmentInfo, SortMissingValue};
use crate::segment_infos::SegmentCommitInfo;
use lucene_codecs::blocktree::FieldTerms;
use lucene_codecs::doc_values::{
    self, BinaryEntry, NumericEntry, SortedEntry, SortedNumericEntry, SortedSetEntry, SortedSetKind,
};
use lucene_codecs::field_infos::{self, FieldInfo, IndexOptions};
use lucene_codecs::norms::{self, NormsEntry};
use lucene_codecs::points::{self, WritePointsField};
use lucene_codecs::postings::DocInput;
use lucene_codecs::postings_writer::{self, FieldPostingsInput, TermPostings};
use lucene_codecs::stored_fields::{self, Document};
use lucene_codecs::term_vectors::{self, TermVectorsDocument, TermVectorsReader};
use lucene_codecs::terms_dict;
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
    /// Same limit as [`Error::TooManyNumericDocValuesFields`], for BINARY
    /// doc values.
    #[error(
        "merging binary doc values for more than one field per call isn't supported yet (found fields {0:?})"
    )]
    TooManyBinaryDocValuesFields(Vec<i32>),
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
    /// More than one field across the merged sources has SORTED doc-values
    /// data -- same single-field limit as
    /// [`Error::TooManyNumericDocValuesFields`], for SORTED.
    #[error(
        "merging sorted doc values for more than one field per call isn't supported yet (found fields {0:?})"
    )]
    TooManySortedDocValuesFields(Vec<i32>),
    /// Same as [`Error::DocValuesFieldMissingInSource`], for SORTED doc
    /// values.
    #[error(
        "merged field number {merged_field_number} has sorted doc values in some sources but not in every source that contributes live docs (or not for every one of that source's live docs)"
    )]
    SortedDocValuesFieldMissingInSource { merged_field_number: i32 },
    /// More than one field across the merged sources has SORTED_NUMERIC
    /// doc-values data -- same single-field limit as
    /// [`Error::TooManyNumericDocValuesFields`], for SORTED_NUMERIC.
    #[error(
        "merging sorted-numeric doc values for more than one field per call isn't supported yet (found fields {0:?})"
    )]
    TooManySortedNumericDocValuesFields(Vec<i32>),
    /// Same as [`Error::DocValuesFieldMissingInSource`], for SORTED_NUMERIC
    /// doc values -- also raised for a live doc whose resolved value list
    /// came back empty, since
    /// [`lucene_codecs::doc_values::write_single_dense_sorted_numeric_field`]
    /// requires every doc to have at least one value.
    #[error(
        "merged field number {merged_field_number} has sorted-numeric doc values in some sources but not in every source that contributes live docs (or not for every one of that source's live docs)"
    )]
    SortedNumericDocValuesFieldMissingInSource { merged_field_number: i32 },
    /// More than one field across the merged sources has SORTED_SET
    /// doc-values data -- same single-field limit as
    /// [`Error::TooManyNumericDocValuesFields`], for SORTED_SET.
    #[error(
        "merging sorted-set doc values for more than one field per call isn't supported yet (found fields {0:?})"
    )]
    TooManySortedSetDocValuesFields(Vec<i32>),
    /// Same as [`Error::DocValuesFieldMissingInSource`], for SORTED_SET doc
    /// values -- also raised for a live doc whose resolved value set came
    /// back empty, since
    /// [`lucene_codecs::doc_values::write_single_dense_sorted_set_field`]
    /// requires every doc to have at least one value.
    #[error(
        "merged field number {merged_field_number} has sorted-set doc values in some sources but not in every source that contributes live docs (or not for every one of that source's live docs)"
    )]
    SortedSetDocValuesFieldMissingInSource { merged_field_number: i32 },
    /// This port's numeric, BINARY, SORTED, SORTED_NUMERIC, and SORTED_SET
    /// doc-values writers all produce single-field `.dvm`/`.dvd`/`.dvs` files
    /// with no multi-field on-disk layout (see this module's doc comment) --
    /// a merge that has more than one of these doc-values types present at
    /// once would silently overwrite one file triple with another, so this
    /// is rejected outright rather than corrupting the merged segment.
    #[error(
        "merging more than one doc values type in one call isn't supported yet (found fields: numeric={numeric_field_number:?}, binary={binary_field_number:?}, sorted={sorted_field_number:?}, sorted_numeric={sorted_numeric_field_number:?}, sorted_set={sorted_set_field_number:?})"
    )]
    MultipleDocValuesTypesInOneMerge {
        numeric_field_number: Option<i32>,
        binary_field_number: Option<i32>,
        sorted_field_number: Option<i32>,
        sorted_numeric_field_number: Option<i32>,
        sorted_set_field_number: Option<i32>,
    },
    /// A field has postings data in at least one source that contributes
    /// live docs, but not in every such source -- see this module's doc
    /// comment on the "sparse across sources" rule (postings within one
    /// source are naturally sparse per-doc/per-term; this error is only
    /// about a whole source missing the *field* entirely).
    #[error(
        "merged field number {merged_field_number} has postings in some sources but not in every source that contributes live docs"
    )]
    PostingsFieldMissingInSource { merged_field_number: i32 },
    /// This port's postings merge only handles `IndexOptions::Docs` and
    /// `IndexOptions::DocsAndFreqs` fields -- positions/offsets/payloads
    /// merging isn't implemented yet (see this module's doc comment).
    #[error(
        "merging postings for merged field number {merged_field_number} isn't supported: index_options {index_options:?} indexes positions, but this port's postings merge only supports IndexOptions::Docs/DocsAndFreqs so far"
    )]
    PostingsIndexOptionsNotSupported {
        merged_field_number: i32,
        index_options: IndexOptions,
    },
    /// Field-number reconciliation only records the *first-seen* source's
    /// `FieldInfo` as the merged one (see `reconcile_field_numbers`) and
    /// never checks that every other live-doc-contributing source agrees on
    /// `index_options` for that field name. Without this check, a source
    /// whose own `index_options` indexes positions/offsets/payloads could
    /// have that data silently dropped whenever an earlier source in the
    /// list happens to be Docs/DocsAndFreqs-only -- this is the hard error
    /// instead.
    #[error(
        "merged field number {merged_field_number} has disagreeing index_options across sources: source claims {source_index_options:?} but the merged field is {merged_index_options:?}"
    )]
    PostingsIndexOptionsDisagreement {
        merged_field_number: i32,
        merged_index_options: IndexOptions,
        source_index_options: IndexOptions,
    },
    /// [`lucene_codecs::term_vectors::write_best_speed`] only supports
    /// term-vector fields with positions (no offsets, no payloads) --
    /// passing it a field with offsets or payloads trips an internal
    /// `assert!`, not a `Result`. Without this check, a merge source whose
    /// term vectors have offsets/payloads would panic deep inside the
    /// writer instead of failing cleanly, so this validates every source's
    /// term-vector fields up front and rejects the unsupported case loudly.
    #[error(
        "merged field number {merged_field_number} has term vectors with offsets ({has_offsets}) or payloads ({has_payloads}), but this port's term-vectors write side (write_best_speed) only supports positions"
    )]
    TermVectorOffsetsOrPayloadsNotSupported {
        merged_field_number: i32,
        has_offsets: bool,
        has_payloads: bool,
    },
    #[error(transparent)]
    Points(#[from] lucene_codecs::points::Error),
    /// A field has BKD points data in at least one source that contributes
    /// live docs, but not in every such source -- see this module's doc
    /// comment on the "sparse across sources" rule. Unlike postings, points
    /// have no per-doc sparsity of their own to model (a doc either has a
    /// point for a field or it doesn't), so this is the only "missing"
    /// failure mode for points.
    #[error(
        "merged field number {merged_field_number} has BKD points in some sources but not in every source that contributes live docs"
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
    /// [`lucene_codecs::points::write`] always treats `num_index_dims` as
    /// equal to `num_dims` (no support for extra, non-indexed data-only
    /// dimensions) -- a source field whose own `num_index_dims != num_dims`
    /// can't be re-encoded by this port's write side, so it's rejected
    /// outright rather than silently dropping the non-indexed dimensions or
    /// mis-encoding the tree.
    #[error(
        "merged field number {merged_field_number} has BKD points with num_index_dims ({num_index_dims}) != num_dims ({num_dims}) in a contributing source, which this port's points write side does not support"
    )]
    PointsIndexDimsNotSupported {
        merged_field_number: i32,
        num_dims: i32,
        num_index_dims: i32,
    },
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
/// # Scope: Docs/DocsAndFreqs only, no positions/offsets/payloads
///
/// [`merge_postings`] (driven from this field) only merges fields whose
/// merged [`FieldInfo::index_options`] is [`IndexOptions::Docs`] or
/// [`IndexOptions::DocsAndFreqs`] -- merging positions/offsets/payloads
/// (`.pos`/`.pay`) is a documented follow-up, not implemented here (see
/// this module's top doc comment and `docs/parity.md`). A field whose
/// merged `index_options` indexes positions is rejected with
/// [`Error::PostingsIndexOptionsNotSupported`] rather than silently
/// dropping its positions/offsets/payloads data.
pub struct SourcePostings<'a> {
    pub field_number: i32,
    pub field_terms: &'a FieldTerms,
    pub doc_in: Option<&'a DocInput<'a>>,
}

/// One source's BKD points (`.kdm`/`.kdi`/`.kdd`) data for a single field --
/// `field_number` is that source's own original field number (pre-merge,
/// same convention as [`SourcePostings::field_number`]). `reader` is that
/// source's already-opened [`lucene_codecs::points::PointsReader`] (via
/// [`lucene_codecs::points::open`]) -- the exact same read path
/// `lucene_search`'s points range query already uses, reused verbatim here
/// rather than re-deriving points decoding.
///
/// # Scope: one packed value per dimension count/width, no data-only dims
///
/// [`lucene_codecs::points::write`] always treats `num_index_dims` as equal
/// to `num_dims` (see its own doc comment) -- a field with extra, non-indexed
/// data-only dimensions on the *read* side (`num_index_dims != num_dims`)
/// can't be re-encoded by this port's write side, so [`merge_points`] rejects
/// that shape with [`Error::PointsIndexDimsNotSupported`] rather than
/// silently truncating or corrupting the merged tree.
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
    let binary_dv = merge_binary_doc_values(sources, &per_source_maps, &per_source_live_ids)?;
    let sorted_dv = merge_sorted_doc_values(sources, &per_source_maps, &per_source_live_ids)?;
    let sorted_numeric_dv =
        merge_sorted_numeric_doc_values(sources, &per_source_maps, &per_source_live_ids)?;
    let sorted_set_dv =
        merge_sorted_set_doc_values(sources, &per_source_maps, &per_source_live_ids)?;
    let present_count = [
        numeric_dv.is_some(),
        binary_dv.is_some(),
        sorted_dv.is_some(),
        sorted_numeric_dv.is_some(),
        sorted_set_dv.is_some(),
    ]
    .into_iter()
    .filter(|&present| present)
    .count();
    if present_count > 1 {
        return Err(Error::MultipleDocValuesTypesInOneMerge {
            numeric_field_number: numeric_dv.as_ref().map(|(n, _)| *n),
            binary_field_number: binary_dv.as_ref().map(|(n, _)| *n),
            sorted_field_number: sorted_dv.as_ref().map(|(n, _)| *n),
            sorted_numeric_field_number: sorted_numeric_dv.as_ref().map(|(n, _)| *n),
            sorted_set_field_number: sorted_set_dv.as_ref().map(|(n, _)| *n),
        });
    }
    let merged_norms = merge_norms(sources, &per_source_maps, &per_source_live_ids)?;
    let tv_docs = merge_term_vectors(sources, &per_source_maps, &per_source_live_ids)?;
    let merged_postings_fields = merge_postings(
        sources,
        &per_source_maps,
        &per_source_live_ids,
        &merged_fields,
    )?;
    let merged_points_fields = merge_points(
        sources,
        &per_source_maps,
        &per_source_live_ids,
        &merged_fields,
    )?;

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

    if let Some((field_number, values)) = binary_dv {
        let (dvm, dvd, dvs) = doc_values::write_single_dense_binary_field(
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

    if let Some((field_number, values)) = sorted_dv {
        let (dvm, dvd, dvs) = doc_values::write_single_dense_sorted_field(
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

    if let Some((field_number, values)) = sorted_numeric_dv {
        let (dvm, dvd, dvs) = doc_values::write_single_dense_sorted_numeric_field(
            field_number,
            &values,
            &merged_segment_id,
            "",
        )?;
        for (ext, bytes) in [("dvm", &dvm), ("dvd", &dvd), ("dvs", &dvs)] {
            let name = format!("{merged_segment_name}.{ext}");
            write_file(dir, &name, bytes)?;
            files.push(name);
        }
    }

    if let Some((field_number, values)) = sorted_set_dv {
        let (dvm, dvd, dvs) = doc_values::write_single_dense_sorted_set_field(
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

    if !merged_postings_fields.is_empty() {
        let inputs: Vec<FieldPostingsInput<'_>> = merged_postings_fields
            .iter()
            .map(|f| FieldPostingsInput {
                field_number: f.field_number,
                index_options: f.index_options,
                doc_count: f.doc_count,
                has_payloads: false,
                terms: &f.terms,
            })
            .collect();
        let output = postings_writer::write_fields(&inputs, &merged_segment_id, "")?;
        let mut exts: Vec<(&str, &[u8])> = vec![
            ("doc", &output.doc),
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
            let name = format!("{merged_segment_name}.{ext}");
            write_file(dir, &name, bytes)?;
            files.push(name);
        }
    }

    // A merged field with zero surviving points (every contributing live
    // doc happened to have none) is simply omitted -- `points::write`
    // doesn't support empty fields (see its own doc comment), and this
    // matches real Lucene's `finish()` returning `null`/omitting the field
    // entirely in that case.
    let non_empty_points_fields: Vec<&MergedPointsField> = merged_points_fields
        .iter()
        .filter(|f| !f.points.is_empty())
        .collect();
    if !non_empty_points_fields.is_empty() {
        let inputs: Vec<WritePointsField> = non_empty_points_fields
            .iter()
            .map(|f| WritePointsField {
                field_number: f.field_number,
                num_dims: f.num_dims,
                bytes_per_dim: f.bytes_per_dim,
                points: f.points.clone(),
            })
            .collect();
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
/// side has no path that reorders doc-values/norms/term-vectors/postings/
/// points data during a merge (see this module's top doc comment). Any
/// doc-values/norms/term-vectors/postings/points data attached to a
/// `MergeSource` is silently ignored by this function; use
/// [`merge_stored_only_segments`] instead if that data needs
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

/// Merges BINARY doc-values data across `sources` into one `(merged_field_
/// number, per_doc_values)` pair, contiguous in the same doc order
/// `merged_docs` was built in -- or `Ok(None)` if no source has any BINARY
/// doc-values data at all. Same single-field limit and "sparse across
/// sources" rule as [`merge_numeric_doc_values`].
fn merge_binary_doc_values(
    sources: &[MergeSource],
    per_source_maps: &[HashMap<i32, i32>],
    per_source_live_ids: &[Vec<i32>],
) -> Result<Option<(i32, Vec<Vec<u8>>)>> {
    let mut candidates: Vec<i32> = Vec::new();
    for ((source, map), live_ids) in sources.iter().zip(per_source_maps).zip(per_source_live_ids) {
        if live_ids.is_empty() {
            // Same "fully-deleted source can't affect the merged output"
            // exemption as merge_numeric_doc_values.
            continue;
        }
        for bf in source.binary_doc_values {
            if let Some(&merged_number) = map.get(&bf.entry.field_number) {
                if !candidates.contains(&merged_number) {
                    candidates.push(merged_number);
                }
            }
        }
    }
    if candidates.len() > 1 {
        return Err(Error::TooManyBinaryDocValuesFields(candidates));
    }
    let Some(merged_field_number) = candidates.into_iter().next() else {
        return Ok(None);
    };

    let mut values: Vec<Vec<u8>> = Vec::new();
    for ((source, map), live_ids) in sources.iter().zip(per_source_maps).zip(per_source_live_ids) {
        if live_ids.is_empty() {
            continue;
        }
        let original_number = map
            .iter()
            .find(|&(_, &merged)| merged == merged_field_number)
            .map(|(&orig, _)| orig);
        let Some(original_number) = original_number else {
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
        for &doc_id in live_ids {
            let value = doc_values::binary_value(entry.data, &entry.entry, doc_id)?.ok_or(
                Error::BinaryDocValuesFieldMissingInSource {
                    merged_field_number,
                },
            )?;
            values.push(value.to_vec());
        }
    }
    Ok(Some((merged_field_number, values)))
}

/// Merges SORTED doc-values data across `sources` into one `(merged_field_
/// number, per_doc_term_bytes)` pair, contiguous in the same doc order
/// `merged_docs` was built in -- or `Ok(None)` if no source has any SORTED
/// doc-values data at all. Same single-field limit and "sparse across
/// sources" rule as [`merge_numeric_doc_values`].
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
/// [`doc_values::write_single_dense_sorted_field`] takes raw per-doc term
/// bytes and rebuilds the merged, deduplicated, sorted dictionary (and this
/// merge's ordinals) itself, so there's no separate remapping step to get
/// wrong: two sources' docs that happen to share a term end up pointing at
/// the exact same merged dictionary entry purely because
/// `write_single_dense_sorted_field`'s dictionary-building sorts and dedups
/// by term *bytes*, not by ordinal.
fn merge_sorted_doc_values(
    sources: &[MergeSource],
    per_source_maps: &[HashMap<i32, i32>],
    per_source_live_ids: &[Vec<i32>],
) -> Result<Option<(i32, Vec<Vec<u8>>)>> {
    let mut candidates: Vec<i32> = Vec::new();
    for ((source, map), live_ids) in sources.iter().zip(per_source_maps).zip(per_source_live_ids) {
        if live_ids.is_empty() {
            // Same "fully-deleted source can't affect the merged output"
            // exemption as merge_numeric_doc_values.
            continue;
        }
        for sf in source.sorted_doc_values {
            if let Some(&merged_number) = map.get(&sf.entry.field_number) {
                if !candidates.contains(&merged_number) {
                    candidates.push(merged_number);
                }
            }
        }
    }
    if candidates.len() > 1 {
        return Err(Error::TooManySortedDocValuesFields(candidates));
    }
    let Some(merged_field_number) = candidates.into_iter().next() else {
        return Ok(None);
    };

    let mut values: Vec<Vec<u8>> = Vec::new();
    for ((source, map), live_ids) in sources.iter().zip(per_source_maps).zip(per_source_live_ids) {
        if live_ids.is_empty() {
            continue;
        }
        let original_number = map
            .iter()
            .find(|&(_, &merged)| merged == merged_field_number)
            .map(|(&orig, _)| orig);
        let Some(original_number) = original_number else {
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
        // This source's own dictionary, in ordinal order -- resolves this
        // source's ordinals to term bytes without needing any other
        // source's dictionary.
        let source_dict = terms_dict::decode_all_terms(sf.data, &sf.entry.terms)?;
        for &doc_id in live_ids {
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
    }
    Ok(Some((merged_field_number, values)))
}

/// Merges SORTED_NUMERIC doc-values data across `sources` into one
/// `(merged_field_number, per_doc_values)` pair, contiguous in the same doc
/// order `merged_docs` was built in -- or `Ok(None)` if no source has any
/// SORTED_NUMERIC doc-values data at all. Same single-field limit and
/// "sparse across sources" rule as [`merge_numeric_doc_values`].
///
/// Unlike SORTED, SORTED_NUMERIC has no shared dictionary to reconcile: each
/// live doc simply contributes its own `Vec<i64>` of values (in whatever
/// order/count the source has), so merging is concatenation, exactly like
/// [`merge_numeric_doc_values`] generalized from one value per doc to a list
/// per doc.
/// [`lucene_codecs::doc_values::write_single_dense_sorted_numeric_field`]
/// requires every doc to have at least one value, so a live doc whose
/// resolved list comes back empty is treated the same as a field missing
/// from its source entirely.
fn merge_sorted_numeric_doc_values(
    sources: &[MergeSource],
    per_source_maps: &[HashMap<i32, i32>],
    per_source_live_ids: &[Vec<i32>],
) -> Result<Option<(i32, Vec<Vec<i64>>)>> {
    let mut candidates: Vec<i32> = Vec::new();
    for ((source, map), live_ids) in sources.iter().zip(per_source_maps).zip(per_source_live_ids) {
        if live_ids.is_empty() {
            // Same "fully-deleted source can't affect the merged output"
            // exemption as merge_numeric_doc_values.
            continue;
        }
        for snf in source.sorted_numeric_doc_values {
            if let Some(&merged_number) = map.get(&snf.entry.field_number) {
                if !candidates.contains(&merged_number) {
                    candidates.push(merged_number);
                }
            }
        }
    }
    if candidates.len() > 1 {
        return Err(Error::TooManySortedNumericDocValuesFields(candidates));
    }
    let Some(merged_field_number) = candidates.into_iter().next() else {
        return Ok(None);
    };

    let mut values: Vec<Vec<i64>> = Vec::new();
    for ((source, map), live_ids) in sources.iter().zip(per_source_maps).zip(per_source_live_ids) {
        if live_ids.is_empty() {
            continue;
        }
        let original_number = map
            .iter()
            .find(|&(_, &merged)| merged == merged_field_number)
            .map(|(&orig, _)| orig);
        let Some(original_number) = original_number else {
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
        for &doc_id in live_ids {
            let doc_values = doc_values::sorted_numeric_values(entry.data, &entry.entry, doc_id)?;
            if doc_values.is_empty() {
                return Err(Error::SortedNumericDocValuesFieldMissingInSource {
                    merged_field_number,
                });
            }
            values.push(doc_values);
        }
    }
    Ok(Some((merged_field_number, values)))
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

/// One merged field's SORTED_SET output: a `merged_field_number` paired with
/// one resolved (possibly-duplicate, unsorted) term-bytes value set per doc,
/// exactly what
/// [`lucene_codecs::doc_values::write_single_dense_sorted_set_field`] takes
/// (it does its own per-doc dedup/sort). A type alias purely to keep
/// [`merge_sorted_set_doc_values`]'s signature legible.
type SortedSetMergeResult = Option<(i32, Vec<Vec<Vec<u8>>>)>;

/// Merges SORTED_SET doc-values data across `sources` into one
/// `(merged_field_number, per_doc_term_bytes_sets)` pair, contiguous in the
/// same doc order `merged_docs` was built in -- or `Ok(None)` if no source
/// has any SORTED_SET doc-values data at all. Same single-field limit and
/// "sparse across sources" rule as [`merge_numeric_doc_values`].
///
/// Exactly [`merge_sorted_doc_values`]'s "resolve to bytes, let the writer
/// dedupe" approach, applied per-*value* instead of per-doc: each live doc's
/// own source's ordinals ([`sorted_set_doc_ordinals`]) are resolved to term
/// bytes via that source's own dictionary
/// ([`sorted_set_source_dict`]), producing a `Vec<Vec<u8>>` per doc, which
/// [`lucene_codecs::doc_values::write_single_dense_sorted_set_field`] then
/// deduplicates (both within a doc and across docs/sources) into the merged
/// dictionary itself -- so, same as SORTED, there is no separate
/// ordinal-remapping table to get wrong; two sources' docs that share a term
/// land on the same merged dictionary entry purely because the merged
/// dictionary is deduplicated by term bytes.
/// [`lucene_codecs::doc_values::write_single_dense_sorted_set_field`]
/// requires every doc to have at least one value, so a live doc whose
/// resolved value set comes back empty is treated the same as a field
/// missing from its source entirely.
fn merge_sorted_set_doc_values(
    sources: &[MergeSource],
    per_source_maps: &[HashMap<i32, i32>],
    per_source_live_ids: &[Vec<i32>],
) -> Result<SortedSetMergeResult> {
    let mut candidates: Vec<i32> = Vec::new();
    for ((source, map), live_ids) in sources.iter().zip(per_source_maps).zip(per_source_live_ids) {
        if live_ids.is_empty() {
            // Same "fully-deleted source can't affect the merged output"
            // exemption as merge_numeric_doc_values.
            continue;
        }
        for ssf in source.sorted_set_doc_values {
            if let Some(&merged_number) = map.get(&ssf.entry.field_number) {
                if !candidates.contains(&merged_number) {
                    candidates.push(merged_number);
                }
            }
        }
    }
    if candidates.len() > 1 {
        return Err(Error::TooManySortedSetDocValuesFields(candidates));
    }
    let Some(merged_field_number) = candidates.into_iter().next() else {
        return Ok(None);
    };

    let mut values: Vec<Vec<Vec<u8>>> = Vec::new();
    for ((source, map), live_ids) in sources.iter().zip(per_source_maps).zip(per_source_live_ids) {
        if live_ids.is_empty() {
            continue;
        }
        let original_number = map
            .iter()
            .find(|&(_, &merged)| merged == merged_field_number)
            .map(|(&orig, _)| orig);
        let Some(original_number) = original_number else {
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
        // This source's own dictionary, in ordinal order -- resolves this
        // source's ordinals to term bytes without needing any other
        // source's dictionary.
        let source_dict = sorted_set_source_dict(ssf.data, &ssf.entry)?;
        for &doc_id in live_ids {
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
                if field.has_offsets || field.has_payloads {
                    return Err(Error::TermVectorOffsetsOrPayloadsNotSupported {
                        merged_field_number: field.field_number,
                        has_offsets: field.has_offsets,
                        has_payloads: field.has_payloads,
                    });
                }
            }
            merged_docs.push(doc);
        }
    }
    Ok(Some(merged_docs))
}

/// Builds, per source, a map from that source's own (pre-merge) live doc ids
/// to the merged, contiguous doc id space -- the postings-merge analogue of
/// `merge_stored_only_segments`'s own doc-concatenation loop, factored out
/// here since [`merge_postings`] needs random-access lookup (a term's
/// postings can reference any live doc in any order a source's `.doc` file
/// happens to store them in -- already ascending per source, see below --
/// not just a linear walk), unlike the linear `per_source_live_ids` iteration
/// every other `merge_*` function does. Source `i`'s live docs land, in
/// order, immediately after source `i-1`'s (matching
/// `merge_stored_only_segments`'s concatenation order), so within one
/// source the resulting map is order-preserving: a term's ascending
/// (docID) list from one source's `.doc` file maps to an ascending merged-
/// docID list too, and since sources occupy disjoint, increasing merged-id
/// ranges, concatenating sources in order for a given term also yields a
/// fully ascending merged-docID list overall -- no separate sort step
/// needed.
fn build_doc_id_maps(per_source_live_ids: &[Vec<i32>]) -> Vec<HashMap<i32, i32>> {
    let mut maps = Vec::with_capacity(per_source_live_ids.len());
    let mut merged_offset: i32 = 0;
    for live_ids in per_source_live_ids {
        let mut map = HashMap::with_capacity(live_ids.len());
        for (i, &doc_id) in live_ids.iter().enumerate() {
            map.insert(doc_id, merged_offset + i as i32);
        }
        merged_offset += live_ids.len() as i32;
        maps.push(map);
    }
    maps
}

/// One merged field's postings, ready to hand to
/// [`lucene_codecs::postings_writer::write_fields`] (via a borrowed
/// [`FieldPostingsInput`] built from `terms`).
struct MergedPostingsField {
    field_number: i32,
    index_options: IndexOptions,
    doc_count: i32,
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
/// # Scope: Docs/DocsAndFreqs only
///
/// See [`SourcePostings`]'s doc comment: a candidate field whose merged
/// `index_options` indexes positions is rejected with
/// [`Error::PostingsIndexOptionsNotSupported`].
fn merge_postings(
    sources: &[MergeSource],
    per_source_maps: &[HashMap<i32, i32>],
    per_source_live_ids: &[Vec<i32>],
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

    let doc_id_maps = build_doc_id_maps(per_source_live_ids);
    let mut result = Vec::with_capacity(candidates.len());

    for merged_field_number in candidates {
        let merged_field = merged_fields
            .iter()
            .find(|f| f.number == merged_field_number)
            .expect("merged_field_number came from reconcile_field_numbers over these same sources, so it must have an entry in merged_fields");
        let index_options = merged_field.index_options;
        if !matches!(
            index_options,
            IndexOptions::Docs | IndexOptions::DocsAndFreqs
        ) {
            return Err(Error::PostingsIndexOptionsNotSupported {
                merged_field_number,
                index_options,
            });
        }

        // Per-source (in source order) this field's `SourcePostings`, or
        // `None` for a fully-deleted source (exempt, same as elsewhere) --
        // any other missing source is the hard "sparse across sources"
        // error.
        let mut per_source_field: Vec<Option<&SourcePostings<'_>>> =
            Vec::with_capacity(sources.len());
        for ((source, map), live_ids) in
            sources.iter().zip(per_source_maps).zip(per_source_live_ids)
        {
            if live_ids.is_empty() {
                per_source_field.push(None);
                continue;
            }
            let original_number = map
                .iter()
                .find(|&(_, &merged)| merged == merged_field_number)
                .map(|(&orig, _)| orig);
            let Some(original_number) = original_number else {
                return Err(Error::PostingsFieldMissingInSource {
                    merged_field_number,
                });
            };
            if let Some(source_field) = source
                .field_infos
                .iter()
                .find(|f| f.number == original_number)
            {
                if source_field.index_options != index_options {
                    return Err(Error::PostingsIndexOptionsDisagreement {
                        merged_field_number,
                        merged_index_options: index_options,
                        source_index_options: source_field.index_options,
                    });
                }
            }
            let Some(pf) = source
                .postings
                .iter()
                .find(|pf| pf.field_number == original_number)
            else {
                return Err(Error::PostingsFieldMissingInSource {
                    merged_field_number,
                });
            };
            per_source_field.push(Some(pf));
        }

        // Union of every contributing source's own term dictionary, by
        // bytes -- resolves each source's terms independently, same
        // "let the merged structure dedupe by bytes" approach
        // merge_sorted_doc_values uses for ordinals.
        let mut all_terms: BTreeSet<Vec<u8>> = BTreeSet::new();
        for pf in per_source_field.iter().flatten() {
            let mut it = pf.field_terms.iter();
            while let Some((term, _stats)) = it.next() {
                all_terms.insert(term.to_vec());
            }
        }

        let mut terms_out: Vec<TermPostings> = Vec::with_capacity(all_terms.len());
        for term in all_terms {
            let mut docs: Vec<(i32, i32)> = Vec::new();
            for (src_idx, pf) in per_source_field.iter().enumerate() {
                let Some(pf) = pf else { continue };
                let Some(source_postings) = pf.field_terms.postings(&term, pf.doc_in)? else {
                    continue;
                };
                let doc_id_map = &doc_id_maps[src_idx];
                for (&doc_id, &freq) in source_postings
                    .docs
                    .iter()
                    .zip(source_postings.freqs.iter())
                {
                    if let Some(&merged_doc_id) = doc_id_map.get(&doc_id) {
                        docs.push((merged_doc_id, freq));
                    }
                }
            }
            if !docs.is_empty() {
                terms_out.push(TermPostings {
                    term,
                    docs,
                    positions: Vec::new(),
                    offsets: Vec::new(),
                    payloads: Vec::new(),
                });
            }
        }

        let mut doc_set: HashSet<i32> = HashSet::new();
        for t in &terms_out {
            for &(doc_id, _) in &t.docs {
                doc_set.insert(doc_id);
            }
        }
        let doc_count = doc_set.len() as i32;

        result.push(MergedPostingsField {
            field_number: merged_field_number,
            index_options,
            doc_count,
            terms: terms_out,
        });
    }

    Ok(result)
}

/// One merged field's BKD points, ready to hand to
/// [`lucene_codecs::points::write`] (via a [`WritePointsField`] built from
/// `points`).
struct MergedPointsField {
    field_number: i32,
    num_dims: i32,
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
/// a mismatch, and any source field whose `num_index_dims != num_dims` is
/// rejected with [`Error::PointsIndexDimsNotSupported`] (see [`SourcePoints`]
/// for why).
fn merge_points(
    sources: &[MergeSource],
    per_source_maps: &[HashMap<i32, i32>],
    per_source_live_ids: &[Vec<i32>],
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

    let doc_id_maps = build_doc_id_maps(per_source_live_ids);
    let mut result = Vec::with_capacity(candidates.len());

    for merged_field_number in candidates {
        let merged_field = merged_fields
            .iter()
            .find(|f| f.number == merged_field_number)
            .expect("merged_field_number came from reconcile_field_numbers over these same sources, so it must have an entry in merged_fields");
        let merged_num_dims = merged_field.point_dimension_count;
        let merged_bytes_per_dim = merged_field.point_num_bytes;

        let mut points: Vec<(i32, Vec<u8>)> = Vec::new();
        for (src_idx, ((source, map), live_ids)) in sources
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
                return Err(Error::PointsFieldMissingInSource {
                    merged_field_number,
                });
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
            if field_meta.num_index_dims != field_meta.num_dims {
                return Err(Error::PointsIndexDimsNotSupported {
                    merged_field_number,
                    num_dims: field_meta.num_dims,
                    num_index_dims: field_meta.num_index_dims,
                });
            }
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

            let doc_id_map = &doc_id_maps[src_idx];
            for point in sp.reader.decode_all_points(original_number)? {
                if let Some(&merged_doc_id) = doc_id_map.get(&point.doc_id) {
                    points.push((merged_doc_id, point.packed_value));
                }
            }
        }

        result.push(MergedPointsField {
            field_number: merged_field_number,
            num_dims: merged_num_dims,
            bytes_per_dim: merged_bytes_per_dim,
            points,
        });
    }

    Ok(result)
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
            "",
        )
        .unwrap();
        let field_infos = field_infos::FieldInfos {
            fields: vec![binary_field("x", field_number)],
        };
        let (_version, parsed) =
            doc_values::parse_meta(&meta, &segment_id, "", &field_infos).unwrap();
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
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &[],
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
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &[],
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

        let dvd = std::fs::read(std::path::Path::new(&tmp).join("_merged_bdv.dvd")).unwrap();
        let dvm = std::fs::read(std::path::Path::new(&tmp).join("_merged_bdv.dvm")).unwrap();
        let merged_field_infos = field_infos::FieldInfos {
            fields: vec![binary_field("bin", 0)],
        };
        let (_v, meta) =
            doc_values::parse_meta(&dvm, &[9u8; ID_LENGTH], "", &merged_field_infos).unwrap();
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
    fn more_than_one_binary_doc_values_field_is_rejected() {
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
        };

        let result = merge_stored_only_segments(
            &dir,
            &[source0],
            "_merged_bdv_toomany",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        );
        assert!(matches!(
            result,
            Err(Error::TooManyBinaryDocValuesFields(_))
        ));
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
            "",
        )
        .unwrap();
        let field_infos = field_infos::FieldInfos {
            fields: vec![sorted_field("x", field_number)],
        };
        let (_version, parsed) =
            doc_values::parse_meta(&meta, &segment_id, "", &field_infos).unwrap();
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
        let (_v, meta) =
            doc_values::parse_meta(dvm, &[9u8; ID_LENGTH], "", &merged_field_infos).unwrap();
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

        let dvm =
            std::fs::read(std::path::Path::new(&tmp).join("_merged_sorted_overlap.dvm")).unwrap();
        let dvd =
            std::fs::read(std::path::Path::new(&tmp).join("_merged_sorted_overlap.dvd")).unwrap();
        let merged_field_infos = field_infos::FieldInfos {
            fields: vec![sorted_field("color", 0)],
        };
        let (_v, meta) =
            doc_values::parse_meta(&dvm, &[9u8; ID_LENGTH], "", &merged_field_infos).unwrap();
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

        let dvm =
            std::fs::read(std::path::Path::new(&tmp).join("_merged_sorted_disjoint.dvm")).unwrap();
        let dvd =
            std::fs::read(std::path::Path::new(&tmp).join("_merged_sorted_disjoint.dvd")).unwrap();
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
    fn more_than_one_sorted_doc_values_field_is_rejected() {
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
        };

        let result = merge_stored_only_segments(
            &dir,
            &[source0],
            "_merged_sdv_toomany",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        );
        assert!(matches!(
            result,
            Err(Error::TooManySortedDocValuesFields(_))
        ));
    }

    #[test]
    fn numeric_and_binary_doc_values_in_the_same_call_is_rejected() {
        // This port's numeric and BINARY doc-values writers both produce
        // single-field `.dvm`/`.dvd`/`.dvs` files -- merging one field of
        // each in the same call would silently overwrite one file triple
        // with the other, so it must be a hard error instead.
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
        };

        let result = merge_stored_only_segments(
            &dir,
            &[source0],
            "_merged_mixed_dv",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        );
        assert!(matches!(
            result,
            Err(Error::MultipleDocValuesTypesInOneMerge { .. })
        ));
    }

    #[test]
    fn sorted_numeric_and_sorted_set_doc_values_in_the_same_call_is_rejected() {
        // Same rule as `numeric_and_binary_doc_values_in_the_same_call_is_rejected`,
        // exercised for a different pair: SORTED_NUMERIC and SORTED_SET also
        // both land on `.dvm`/`.dvd`/`.dvs`, so mixing them in one call must
        // be rejected too, not just the numeric/BINARY pair.
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
        };

        let result = merge_stored_only_segments(
            &dir,
            &[source0],
            "_merged_mixed_dv2",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        );
        assert!(matches!(
            result,
            Err(Error::MultipleDocValuesTypesInOneMerge { .. })
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
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: &[],
            points: &[],
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

        let merged_reader =
            std::fs::read(std::path::Path::new(&tmp).join(format!("{}.fdt", sci.segment_name)));
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
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: Some(&tv0_reader),
            postings: &[],
            points: &[],
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

    fn write_tv_vint(out: &mut Vec<u8>, mut v: i32) {
        loop {
            let mut b = (v & 0x7f) as u8;
            v = ((v as u32) >> 7) as i32;
            if v != 0 {
                b |= 0x80;
                out.push(b);
            } else {
                out.push(b);
                break;
            }
        }
    }

    fn write_tv_string(out: &mut Vec<u8>, s: &str) {
        write_tv_vint(out, s.len() as i32);
        out.extend_from_slice(s.as_bytes());
    }

    /// Hand-encodes a single-doc, single-chunk `.tvd`/`.tvx`/`.tvm` triple
    /// with one field (number 0) that has POSITIONS+OFFSETS+PAYLOADS and one
    /// term "a" (freq 1) -- mirrors
    /// `lucene_codecs::term_vectors::tests::build_single_doc_chunk`'s shape,
    /// trimmed to a single term, since this port's write side
    /// (`write_best_speed`) can't produce offsets/payloads itself (that's
    /// exactly the gap [`Error::TermVectorOffsetsOrPayloadsNotSupported`]
    /// guards against) -- a merge source with such data has to be hand-built
    /// to exercise the check.
    fn build_offsets_payloads_term_vectors(
        segment_id: [u8; ID_LENGTH],
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        use lucene_store::codec_util;

        const DATA_CODEC: &str = "Lucene90TermVectorsData";
        const INDEX_CODEC: &str = "Lucene90TermVectorsIndexIdx";
        const META_CODEC: &str = "Lucene90TermVectorsIndexMeta";

        let mut tvd = Vec::new();
        tvd.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
        write_tv_string(&mut tvd, DATA_CODEC);
        tvd.extend_from_slice(&0u32.to_be_bytes()); // version
        tvd.extend_from_slice(&segment_id);
        tvd.push(0); // empty suffix
        let chunk_start = tvd.len() as i64;

        write_tv_vint(&mut tvd, 0); // docBase
        write_tv_vint(&mut tvd, 1 << 1); // token: chunkDocs=1, dirty=0
        write_tv_vint(&mut tvd, 1); // numFields = totalFields = 1

        // fieldNums: 1 distinct field (number 0), 8 bits/value.
        tvd.push(8);
        tvd.push(0);

        // allFieldNumOffs: 1 field, offset 0, 1 bit/value.
        write_tv_vint(&mut tvd, 1);
        tvd.push(0x00);

        // flags: selector=1 (direct array), 1 field, value=7
        // (POSITIONS|OFFSETS|PAYLOADS).
        write_tv_vint(&mut tvd, 1);
        write_tv_vint(&mut tvd, 1);
        tvd.push(0x07);

        // numTerms: 1 field, value=1.
        write_tv_vint(&mut tvd, 1); // bitsRequired
        write_tv_vint(&mut tvd, 1); // slice byte length
        tvd.push(1);

        // prefixLengths [0] (bpv=0, constant).
        tvd.push(0x01);
        // suffixLengths [1] (bpv=0, constant, min=1): token, minValue vlong.
        tvd.extend_from_slice(&[0x00, 0x01]);
        // termFreqsMinus1 [0] (bpv=0, constant).
        tvd.push(0x01);

        // positions_flat [0] (bpv=0, constant).
        tvd.push(0x01);

        // charsPerTerm: 1 distinct field, value 1.0.
        tvd.extend_from_slice(&1.0f32.to_bits().to_le_bytes());
        // start_offsets_flat [0] (bpv=0, constant).
        tvd.push(0x01);
        // lengths_flat [1] (bpv=0, constant, min=1).
        tvd.extend_from_slice(&[0x00, 0x01]);
        // payload_lengths_flat [1] (bpv=0, constant, min=1).
        tvd.extend_from_slice(&[0x00, 0x01]);

        // LZ4 (CompressionMode.FAST, no dictionary): literal-only unit
        // wrapping "a" (term suffix) then payload byte 0xAA.
        let payload = [b'a', 0xAA];
        tvd.push((payload.len() as u8) << 4);
        tvd.extend_from_slice(&payload);

        tvd.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
        tvd.extend_from_slice(&0u32.to_be_bytes());
        let checksum = crc32fast::hash(&tvd) as u64;
        tvd.extend_from_slice(&checksum.to_be_bytes());

        let mut tvx = Vec::new();
        tvx.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
        write_tv_string(&mut tvx, INDEX_CODEC);
        tvx.extend_from_slice(&0u32.to_be_bytes());
        tvx.extend_from_slice(&segment_id);
        tvx.push(0);
        let docs_start = tvx.len() as i64;
        let docs_end = tvx.len() as i64;
        let start_pointers_end = tvx.len() as i64;
        tvx.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
        tvx.extend_from_slice(&0u32.to_be_bytes());
        let checksum = crc32fast::hash(&tvx) as u64;
        tvx.extend_from_slice(&checksum.to_be_bytes());

        let max_doc = 1i32;
        let max_pointer = (tvd.len() - codec_util::FOOTER_LENGTH) as i64;
        let mut tvm = Vec::new();
        tvm.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
        write_tv_string(&mut tvm, META_CODEC);
        tvm.extend_from_slice(&0u32.to_be_bytes());
        tvm.extend_from_slice(&segment_id);
        tvm.push(0);
        write_tv_vint(&mut tvm, 0); // packedIntsVersion
        write_tv_vint(&mut tvm, 4096); // chunkSize
        tvm.extend_from_slice(&max_doc.to_le_bytes());
        tvm.extend_from_slice(&0i32.to_le_bytes()); // blockShift
        tvm.extend_from_slice(&2i32.to_le_bytes()); // index_num_chunks
        tvm.extend_from_slice(&docs_start.to_le_bytes());
        for min in [0i64, max_doc as i64] {
            tvm.extend_from_slice(&min.to_le_bytes());
            tvm.extend_from_slice(&0i32.to_le_bytes());
            tvm.extend_from_slice(&0i64.to_le_bytes());
            tvm.push(0);
        }
        tvm.extend_from_slice(&docs_end.to_le_bytes());
        for min in [chunk_start, max_pointer] {
            tvm.extend_from_slice(&min.to_le_bytes());
            tvm.extend_from_slice(&0i32.to_le_bytes());
            tvm.extend_from_slice(&0i64.to_le_bytes());
            tvm.push(0);
        }
        tvm.extend_from_slice(&start_pointers_end.to_le_bytes());
        tvm.extend_from_slice(&max_pointer.to_le_bytes());
        write_tv_vint(&mut tvm, 1); // numChunks (outer)
        write_tv_vint(&mut tvm, 0); // numDirtyChunks
        write_tv_vint(&mut tvm, 0); // numDirtyDocs
        tvm.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
        tvm.extend_from_slice(&0u32.to_be_bytes());
        let checksum = crc32fast::hash(&tvm) as u64;
        tvm.extend_from_slice(&checksum.to_be_bytes());

        (tvd, tvx, tvm)
    }

    #[test]
    fn term_vectors_merge_rejects_offsets_and_payloads() {
        let seg0_id = [1u8; ID_LENGTH];
        let (tvd, tvx, tvm) = build_offsets_payloads_term_vectors(seg0_id);
        let tv0_reader = term_vectors::open(&tvd, &tvx, &tvm, &seg0_id, "").unwrap();
        // Sanity-check the hand-built bytes actually decode to
        // POSITIONS+OFFSETS+PAYLOADS before using them to exercise the
        // merge-time rejection.
        let doc0 = tv0_reader.document(0).unwrap().unwrap();
        assert!(doc0.fields[0].has_offsets && doc0.fields[0].has_payloads);
        assert_eq!(doc0.fields[0].terms[0].term, b"a");
        assert_eq!(doc0.fields[0].terms[0].start_offsets, Some(vec![0]));
        assert_eq!(doc0.fields[0].terms[0].payloads, Some(vec![vec![0xAA]]));

        let fields0 = vec![tv_field("id", 0)];
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let stored0 = flush(&dir, &tmp, "_0", seg0_id, &fields0, &[doc_with(0, "a")]);
        let reader0 = open_reader(&stored0);

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
        };

        let err = merge_stored_only_segments(
            &dir,
            &[source0],
            "_merged_tv_rejected",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            Error::TermVectorOffsetsOrPayloadsNotSupported {
                merged_field_number: 0,
                has_offsets: true,
                has_payloads: true,
            }
        ));
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
        let merged_fdt =
            std::fs::read(std::path::Path::new(&tmp).join("_merged_all_bin.fdt")).unwrap();
        let merged_fdx =
            std::fs::read(std::path::Path::new(&tmp).join("_merged_all_bin.fdx")).unwrap();
        let merged_fdm =
            std::fs::read(std::path::Path::new(&tmp).join("_merged_all_bin.fdm")).unwrap();
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
        let dvd = std::fs::read(std::path::Path::new(&tmp).join("_merged_all_bin.dvd")).unwrap();
        let dvm = std::fs::read(std::path::Path::new(&tmp).join("_merged_all_bin.dvm")).unwrap();
        let merged_field_infos = field_infos::FieldInfos {
            fields: vec![binary_field("body", 0)],
        };
        let (_v, dv_meta) =
            doc_values::parse_meta(&dvm, &merged_id, "", &merged_field_infos).unwrap();
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
        let nvd = std::fs::read(std::path::Path::new(&tmp).join("_merged_all_bin.nvd")).unwrap();
        let nvm = std::fs::read(std::path::Path::new(&tmp).join("_merged_all_bin.nvm")).unwrap();
        let (_v, norms_meta) = norms::parse_meta(&nvm, &merged_id, "").unwrap();
        let norms_entry = norms_meta.entry(0).unwrap();
        let norms_values: Vec<i64> = (0..3)
            .map(|d| norms::norm_value(&nvd, norms_entry, d).unwrap().unwrap())
            .collect();
        assert_eq!(norms_values, vec![1, 2, 3]);

        // Term vectors.
        let tvd = std::fs::read(std::path::Path::new(&tmp).join("_merged_all_bin.tvd")).unwrap();
        let tvx = std::fs::read(std::path::Path::new(&tmp).join("_merged_all_bin.tvx")).unwrap();
        let tvm = std::fs::read(std::path::Path::new(&tmp).join("_merged_all_bin.tvm")).unwrap();
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
        let si_bytes =
            std::fs::read(std::path::Path::new(&tmp).join("_merged_all_bin.si")).unwrap();
        let si = segment_info::parse(&si_bytes, &merged_id).unwrap();
        for ext in [
            "fdt", "fdx", "fdm", "fnm", "dvm", "dvd", "dvs", "nvm", "nvd", "tvd", "tvx", "tvm",
        ] {
            let name = format!("_merged_all_bin.{ext}");
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
        let merged_fdt =
            std::fs::read(std::path::Path::new(&tmp).join("_merged_all_sorted.fdt")).unwrap();
        let merged_fdx =
            std::fs::read(std::path::Path::new(&tmp).join("_merged_all_sorted.fdx")).unwrap();
        let merged_fdm =
            std::fs::read(std::path::Path::new(&tmp).join("_merged_all_sorted.fdm")).unwrap();
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
        let dvd = std::fs::read(std::path::Path::new(&tmp).join("_merged_all_sorted.dvd")).unwrap();
        let dvm = std::fs::read(std::path::Path::new(&tmp).join("_merged_all_sorted.dvm")).unwrap();
        let terms = read_back_sorted_terms(&dvm, &dvd, 0, 3);
        assert_eq!(
            terms,
            vec![b"red".to_vec(), b"blue".to_vec(), b"red".to_vec()]
        );

        // Norms.
        let nvd = std::fs::read(std::path::Path::new(&tmp).join("_merged_all_sorted.nvd")).unwrap();
        let nvm = std::fs::read(std::path::Path::new(&tmp).join("_merged_all_sorted.nvm")).unwrap();
        let (_v, norms_meta) = norms::parse_meta(&nvm, &merged_id, "").unwrap();
        let norms_entry = norms_meta.entry(0).unwrap();
        let norms_values: Vec<i64> = (0..3)
            .map(|d| norms::norm_value(&nvd, norms_entry, d).unwrap().unwrap())
            .collect();
        assert_eq!(norms_values, vec![1, 2, 3]);

        // Term vectors.
        let tvd = std::fs::read(std::path::Path::new(&tmp).join("_merged_all_sorted.tvd")).unwrap();
        let tvx = std::fs::read(std::path::Path::new(&tmp).join("_merged_all_sorted.tvx")).unwrap();
        let tvm = std::fs::read(std::path::Path::new(&tmp).join("_merged_all_sorted.tvm")).unwrap();
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
        let si_bytes =
            std::fs::read(std::path::Path::new(&tmp).join("_merged_all_sorted.si")).unwrap();
        let si = segment_info::parse(&si_bytes, &merged_id).unwrap();
        for ext in [
            "fdt", "fdx", "fdm", "fnm", "dvm", "dvd", "dvs", "nvm", "nvd", "tvd", "tvx", "tvm",
        ] {
            let name = format!("_merged_all_sorted.{ext}");
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
            "",
        )
        .unwrap();
        let field_infos = field_infos::FieldInfos {
            fields: vec![sorted_numeric_field("x", field_number)],
        };
        let (_version, parsed) =
            doc_values::parse_meta(&meta, &segment_id, "", &field_infos).unwrap();
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
        let (_v, meta) =
            doc_values::parse_meta(dvm, &[9u8; ID_LENGTH], "", &merged_field_infos).unwrap();
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

        let dvm =
            std::fs::read(std::path::Path::new(&tmp).join("_merged_sorted_numeric.dvm")).unwrap();
        let dvd =
            std::fs::read(std::path::Path::new(&tmp).join("_merged_sorted_numeric.dvd")).unwrap();
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
    fn more_than_one_sorted_numeric_doc_values_field_is_rejected() {
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
        };

        let result = merge_stored_only_segments(
            &dir,
            &[source0],
            "_merged_sndv_toomany",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        );
        assert!(matches!(
            result,
            Err(Error::TooManySortedNumericDocValuesFields(_))
        ));
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
            "",
        )
        .unwrap();
        let field_infos = field_infos::FieldInfos {
            fields: vec![sorted_set_field("x", field_number)],
        };
        let (_version, parsed) =
            doc_values::parse_meta(&meta, &segment_id, "", &field_infos).unwrap();
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
        let (_v, meta) =
            doc_values::parse_meta(dvm, &[9u8; ID_LENGTH], "", &merged_field_infos).unwrap();
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

        let dvm = std::fs::read(std::path::Path::new(&tmp).join("_merged_sorted_set_overlap.dvm"))
            .unwrap();
        let dvd = std::fs::read(std::path::Path::new(&tmp).join("_merged_sorted_set_overlap.dvd"))
            .unwrap();
        let merged_field_infos = field_infos::FieldInfos {
            fields: vec![sorted_set_field("colors", 0)],
        };
        let (_v, meta) =
            doc_values::parse_meta(&dvm, &[9u8; ID_LENGTH], "", &merged_field_infos).unwrap();
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

        let dvm = std::fs::read(std::path::Path::new(&tmp).join("_merged_sorted_set_disjoint.dvm"))
            .unwrap();
        let dvd = std::fs::read(std::path::Path::new(&tmp).join("_merged_sorted_set_disjoint.dvd"))
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
    fn more_than_one_sorted_set_doc_values_field_is_rejected() {
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
        };

        let result = merge_stored_only_segments(
            &dir,
            &[source0],
            "_merged_ssdv_toomany",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        );
        assert!(matches!(
            result,
            Err(Error::TooManySortedSetDocValuesFields(_))
        ));
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
        let output0 = postings_writer::write_single_field(&input0, &seg0_id, "").unwrap();
        let field_infos0 = field_infos::FieldInfos {
            fields: vec![postings_field("body", 0)],
        };
        let fields0 = lucene_codecs::blocktree::open(
            &output0.tim,
            &output0.tip,
            &output0.tmd,
            &field_infos0,
            &seg0_id,
            "",
            2,
        )
        .unwrap();
        let doc_in0 = DocInput::open(&output0.doc, &seg0_id, "").unwrap();
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
        let output1 = postings_writer::write_single_field(&input1, &seg1_id, "").unwrap();
        let field_infos1 = field_infos::FieldInfos {
            fields: vec![postings_field("body", 0)],
        };
        let fields1 = lucene_codecs::blocktree::open(
            &output1.tim,
            &output1.tip,
            &output1.tmd,
            &field_infos1,
            &seg1_id,
            "",
            1,
        )
        .unwrap();
        let doc_in1 = DocInput::open(&output1.doc, &seg1_id, "").unwrap();
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
        }];
        let src_postings1 = [SourcePostings {
            field_number: 0,
            field_terms: field_terms1,
            doc_in: Some(&doc_in1),
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

        let tim = std::fs::read(std::path::Path::new(&tmp).join("_merged_postings.tim")).unwrap();
        let tip = std::fs::read(std::path::Path::new(&tmp).join("_merged_postings.tip")).unwrap();
        let tmd = std::fs::read(std::path::Path::new(&tmp).join("_merged_postings.tmd")).unwrap();
        let doc = std::fs::read(std::path::Path::new(&tmp).join("_merged_postings.doc")).unwrap();
        let merged_field_infos = field_infos::FieldInfos {
            fields: vec![postings_field("body", 0)],
        };
        let merged_fields = lucene_codecs::blocktree::open(
            &tim,
            &tip,
            &tmd,
            &merged_field_infos,
            &[9u8; ID_LENGTH],
            "",
            3,
        )
        .unwrap();
        let merged_doc_in = DocInput::open(&doc, &[9u8; ID_LENGTH], "").unwrap();
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
        let output0 = postings_writer::write_single_field(&input0, &seg0_id, "").unwrap();
        let field_infos0 = field_infos::FieldInfos {
            fields: vec![postings_field("body", 0)],
        };
        let fields0 = lucene_codecs::blocktree::open(
            &output0.tim,
            &output0.tip,
            &output0.tmd,
            &field_infos0,
            &seg0_id,
            "",
            1,
        )
        .unwrap();
        let doc_in0 = DocInput::open(&output0.doc, &seg0_id, "").unwrap();
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
        let output1 = postings_writer::write_single_field(&input1, &seg1_id, "").unwrap();
        let field_infos1 = field_infos::FieldInfos {
            fields: vec![postings_field("body", 0)],
        };
        let fields1 = lucene_codecs::blocktree::open(
            &output1.tim,
            &output1.tip,
            &output1.tmd,
            &field_infos1,
            &seg1_id,
            "",
            1,
        )
        .unwrap();
        let doc_in1 = DocInput::open(&output1.doc, &seg1_id, "").unwrap();
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
        }];
        let src_postings1 = [SourcePostings {
            field_number: 0,
            field_terms: field_terms1,
            doc_in: Some(&doc_in1),
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
        };

        let tmp2 = tmp.clone();
        merge_stored_only_segments(
            &dir,
            &[source0, source1],
            "_merged_the",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();

        let tim = std::fs::read(std::path::Path::new(&tmp2).join("_merged_the.tim")).unwrap();
        let tip = std::fs::read(std::path::Path::new(&tmp2).join("_merged_the.tip")).unwrap();
        let tmd = std::fs::read(std::path::Path::new(&tmp2).join("_merged_the.tmd")).unwrap();
        let doc = std::fs::read(std::path::Path::new(&tmp2).join("_merged_the.doc")).unwrap();
        let merged_field_infos = field_infos::FieldInfos {
            fields: vec![postings_field("body", 0)],
        };
        let merged_fields = lucene_codecs::blocktree::open(
            &tim,
            &tip,
            &tmd,
            &merged_field_infos,
            &[9u8; ID_LENGTH],
            "",
            2,
        )
        .unwrap();
        let merged_doc_in = DocInput::open(&doc, &[9u8; ID_LENGTH], "").unwrap();
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
        let output0 = postings_writer::write_single_field(&input0, &seg0_id, "").unwrap();
        let field_infos0 = field_infos::FieldInfos {
            fields: vec![postings_field("body", 0)],
        };
        let fields0 = lucene_codecs::blocktree::open(
            &output0.tim,
            &output0.tip,
            &output0.tmd,
            &field_infos0,
            &seg0_id,
            "",
            2,
        )
        .unwrap();
        let doc_in0 = DocInput::open(&output0.doc, &seg0_id, "").unwrap();
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

        let tim = std::fs::read(std::path::Path::new(&tmp).join("_merged_del.tim")).unwrap();
        let tip = std::fs::read(std::path::Path::new(&tmp).join("_merged_del.tip")).unwrap();
        let tmd = std::fs::read(std::path::Path::new(&tmp).join("_merged_del.tmd")).unwrap();
        let doc = std::fs::read(std::path::Path::new(&tmp).join("_merged_del.doc")).unwrap();
        let merged_field_infos = field_infos::FieldInfos {
            fields: vec![postings_field("body", 0)],
        };
        let merged_fields = lucene_codecs::blocktree::open(
            &tim,
            &tip,
            &tmd,
            &merged_field_infos,
            &[9u8; ID_LENGTH],
            "",
            1,
        )
        .unwrap();
        let merged_doc_in = DocInput::open(&doc, &[9u8; ID_LENGTH], "").unwrap();
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
        let output0 = postings_writer::write_single_field(&input0, &seg0_id, "").unwrap();
        let field_infos0 = field_infos::FieldInfos {
            fields: vec![postings_field("body", 0)],
        };
        let fields0 = lucene_codecs::blocktree::open(
            &output0.tim,
            &output0.tip,
            &output0.tmd,
            &field_infos0,
            &seg0_id,
            "",
            1,
        )
        .unwrap();
        let doc_in0 = DocInput::open(&output0.doc, &seg0_id, "").unwrap();
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
        let output1 = postings_writer::write_single_field(&input1, &seg1_id, "").unwrap();
        let field_infos1 = field_infos::FieldInfos {
            fields: vec![postings_field("body", 0)],
        };
        let fields1 = lucene_codecs::blocktree::open(
            &output1.tim,
            &output1.tip,
            &output1.tmd,
            &field_infos1,
            &seg1_id,
            "",
            1,
        )
        .unwrap();
        let doc_in1 = DocInput::open(&output1.doc, &seg1_id, "").unwrap();
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
        }];
        let src_postings1 = [SourcePostings {
            field_number: 0,
            field_terms: field_terms1,
            doc_in: Some(&doc_in1),
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

        let tim =
            std::fs::read(std::path::Path::new(&tmp).join("_merged_fully_deleted.tim")).unwrap();
        let tip =
            std::fs::read(std::path::Path::new(&tmp).join("_merged_fully_deleted.tip")).unwrap();
        let tmd =
            std::fs::read(std::path::Path::new(&tmp).join("_merged_fully_deleted.tmd")).unwrap();
        let doc =
            std::fs::read(std::path::Path::new(&tmp).join("_merged_fully_deleted.doc")).unwrap();
        let merged_field_infos = field_infos::FieldInfos {
            fields: vec![postings_field("body", 0)],
        };
        let merged_fields = lucene_codecs::blocktree::open(
            &tim,
            &tip,
            &tmd,
            &merged_field_infos,
            &[9u8; ID_LENGTH],
            "",
            1,
        )
        .unwrap();
        let merged_doc_in = DocInput::open(&doc, &[9u8; ID_LENGTH], "").unwrap();
        let merged_terms = merged_fields.field("body").unwrap();

        assert!(merged_terms.seek_exact(b"ghost").is_none());
        let alive = merged_terms
            .postings(b"alive", Some(&merged_doc_in))
            .unwrap()
            .unwrap();
        assert_eq!(alive.docs, vec![0]);
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
        let output0 = postings_writer::write_single_field(&input0, &seg0_id, "").unwrap();
        let field_infos0 = field_infos::FieldInfos {
            fields: vec![postings_field("body", 0)],
        };
        let fields0 = lucene_codecs::blocktree::open(
            &output0.tim,
            &output0.tip,
            &output0.tmd,
            &field_infos0,
            &seg0_id,
            "",
            1,
        )
        .unwrap();
        let doc_in0 = DocInput::open(&output0.doc, &seg0_id, "").unwrap();
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
    fn postings_field_with_positions_is_rejected_as_out_of_scope() {
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
        let output0 = postings_writer::write_single_field(&input0, &seg0_id, "").unwrap();
        let mut field_with_positions = postings_field("body", 0);
        field_with_positions.index_options = IndexOptions::DocsAndFreqsAndPositions;
        let field_infos0 = field_infos::FieldInfos {
            fields: vec![field_with_positions.clone()],
        };
        let fields0 = lucene_codecs::blocktree::open(
            &output0.tim,
            &output0.tip,
            &output0.tmd,
            &field_infos0,
            &seg0_id,
            "",
            1,
        )
        .unwrap();
        let doc_in0 = DocInput::open(&output0.doc, &seg0_id, "").unwrap();
        let field_terms0 = fields0.field("body").unwrap();

        let fields = vec![field_with_positions];
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let stored0 = flush(&dir, &tmp, "_0", seg0_id, &fields, &[doc_with(0, "a")]);
        let reader0 = open_reader(&stored0);

        let src_postings0 = [SourcePostings {
            field_number: 0,
            field_terms: field_terms0,
            doc_in: Some(&doc_in0),
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
        };

        let result = merge_stored_only_segments(
            &dir,
            &[source0],
            "_merged_positions_rejected",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        );
        assert!(matches!(
            result,
            Err(Error::PostingsIndexOptionsNotSupported { .. })
        ));
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
        let output0 = postings_writer::write_single_field(&input0, &seg0_id, "").unwrap();
        let field_infos0 = field_infos::FieldInfos {
            fields: vec![postings_field("body", 0)],
        };
        let fields0 = lucene_codecs::blocktree::open(
            &output0.tim,
            &output0.tip,
            &output0.tmd,
            &field_infos0,
            &seg0_id,
            "",
            1,
        )
        .unwrap();
        let doc_in0 = DocInput::open(&output0.doc, &seg0_id, "").unwrap();
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
        let output1 = postings_writer::write_single_field(&input1, &seg1_id, "").unwrap();
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
            "",
            1,
        )
        .unwrap();
        let doc_in1 = DocInput::open(&output1.doc, &seg1_id, "").unwrap();
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
        }];
        let src_postings1 = [SourcePostings {
            field_number: 0,
            field_terms: field_terms1,
            doc_in: Some(&doc_in1),
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

    fn points_field(name: &str, number: i32, num_dims: i32, bytes_per_dim: i32) -> FieldInfo {
        let mut f = field(name, number);
        f.point_dimension_count = num_dims;
        f.point_index_dimension_count = num_dims;
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
        let fields = vec![WritePointsField {
            field_number,
            num_dims,
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

        let kdm = std::fs::read(std::path::Path::new(&tmp).join("_merged_points.kdm")).unwrap();
        let kdi = std::fs::read(std::path::Path::new(&tmp).join("_merged_points.kdi")).unwrap();
        let kdd = std::fs::read(std::path::Path::new(&tmp).join("_merged_points.kdd")).unwrap();
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

        let kdm =
            std::fs::read(std::path::Path::new(&tmp).join("_merged_points_deletions.kdm")).unwrap();
        let kdi =
            std::fs::read(std::path::Path::new(&tmp).join("_merged_points_deletions.kdi")).unwrap();
        let kdd =
            std::fs::read(std::path::Path::new(&tmp).join("_merged_points_deletions.kdd")).unwrap();
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

        let kdm =
            std::fs::read(std::path::Path::new(&tmp).join("_merged_points_fully_deleted.kdm"))
                .unwrap();
        let kdi =
            std::fs::read(std::path::Path::new(&tmp).join("_merged_points_fully_deleted.kdi"))
                .unwrap();
        let kdd =
            std::fs::read(std::path::Path::new(&tmp).join("_merged_points_fully_deleted.kdd"))
                .unwrap();
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

    /// Hand-builds a single-field, single-leaf `.kdm`/`.kdi`/`.kdd` triple
    /// whose `num_index_dims` differs from `num_dims` -- a shape
    /// [`lucene_codecs::points::write`] itself can never produce (it always
    /// treats them as equal), but one the read side already tolerates, so
    /// this exercises [`Error::PointsIndexDimsNotSupported`] without relying
    /// on a write path that can't create the input in the first place.
    fn build_index_dims_mismatch_points(
        segment_id: &[u8; ID_LENGTH],
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        use lucene_store::codec_util;

        let num_dims: i32 = 2;
        let num_index_dims: i32 = 1;
        let bytes_per_dim: i32 = 4;
        let field_number: i32 = 0;

        // -- kdd (data): index header, then one leaf block, then footer.
        let mut kdd: Vec<u8> = Vec::new();
        codec_util::write_index_header(&mut kdd, "Lucene90PointsFormatData", 1, segment_id, "");
        let leaf_fp = kdd.len() as i64;
        kdd.write_vint(1); // count
        kdd.write_byte((-2i8) as u8); // CONTINUOUS_IDS
        kdd.write_vint(0); // start doc id
        kdd.write_vint(0); // common prefix, dim 0
        kdd.write_vint(0); // common prefix, dim 1
        kdd.write_byte((-2i8) as u8); // compressedDim marker: sparse run encoding
                                      // num_index_dims == 1, so no per-leaf bounding box is written/read.
        kdd.write_vint(1); // sub-block run length
        kdd.write_bytes(&[0, 0, 0, 11]); // dim 0 value
        kdd.write_bytes(&[0, 0, 0, 22]); // dim 1 value
        codec_util::write_footer(&mut kdd);

        // -- kdi (index): index header, then this field's packed index (a
        // single leaf needs only its own file-pointer delta, matching
        // `pack_index`'s `num_leaves == 1` top-level case).
        let mut kdi: Vec<u8> = Vec::new();
        codec_util::write_index_header(&mut kdi, "Lucene90PointsFormatIndex", 1, segment_id, "");
        let index_start_pointer = kdi.len() as i64;
        kdi.write_vlong(leaf_fp);
        let num_index_bytes = (kdi.len() as i64 - index_start_pointer) as i32;
        codec_util::write_footer(&mut kdi);

        // -- kdm (meta): index header, one field entry, terminator, lengths,
        // footer.
        let mut kdm: Vec<u8> = Vec::new();
        codec_util::write_index_header(&mut kdm, "Lucene90PointsFormatMeta", 1, segment_id, "");
        kdm.write_i32(field_number);
        codec_util::write_header(&mut kdm, "BKD", 10);
        kdm.write_vint(num_dims);
        kdm.write_vint(num_index_dims);
        kdm.write_vint(512); // max_points_in_leaf_node
        kdm.write_vint(bytes_per_dim);
        kdm.write_vint(1); // num_leaves
        kdm.write_bytes(&[0, 0, 0, 11]); // min_packed_value (num_index_dims * bytes_per_dim)
        kdm.write_bytes(&[0, 0, 0, 11]); // max_packed_value
        kdm.write_vlong(1); // point_count
        kdm.write_vint(1); // doc_count
        kdm.write_vint(num_index_bytes);
        kdm.write_i64(leaf_fp); // min_leaf_block_fp (discarded on read)
        kdm.write_i64(index_start_pointer);
        kdm.write_i32(-1); // field-loop terminator
        kdm.write_i64(kdi.len() as i64);
        kdm.write_i64(kdd.len() as i64);
        codec_util::write_footer(&mut kdm);

        (kdm, kdi, kdd)
    }

    #[test]
    fn points_index_dims_not_supported_is_rejected() {
        let seg0_id = [1u8; ID_LENGTH];
        let (kdm0, kdi0, kdd0) = build_index_dims_mismatch_points(&seg0_id);
        let points_reader0 = points::open(&kdm0, &kdi0, &kdd0, &seg0_id, "").unwrap();

        // The field's FieldInfo declares 2 dims/2 index dims to match the
        // hand-built data's `num_dims`; only the points data itself has the
        // unsupported `num_index_dims != num_dims` shape.
        let fields = vec![points_field("loc", 0, 2, 4)];
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let stored0 = flush(&dir, &tmp, "_0", seg0_id, &fields, &[doc_with(0, "a")]);
        let reader0 = open_reader(&stored0);

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
        };

        let result = merge_stored_only_segments(
            &dir,
            &[source0],
            "_merged_points_index_dims",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        );
        assert!(matches!(
            result,
            Err(Error::PointsIndexDimsNotSupported { .. })
        ));
    }
}
