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
//! `write_best_speed` supports offsets and payloads as well as positions,
//! and [`merge_term_vectors`] passes them through unchanged: unlike postings
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
//! for postings. The same "sparse across sources" philosophy still applies
//! at the field level: a source that contributes live docs but has no
//! postings field at all for a name another live-doc-contributing source
//! does is a hard error ([`Error::PostingsFieldMissingInSource`]), not a
//! silent drop -- ordinary per-doc/per-term sparsity (most docs don't
//! contain most terms) is not an error, since that's exactly what a term
//! dictionary already models.
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
    /// At least one source contributing live docs to this merge has a
    /// term-vectors reader (`MergeSource::term_vectors`), but another
    /// live-doc-contributing source does not -- same "sparse across
    /// sources" philosophy as [`Error::PostingsFieldMissingInSource`]: a
    /// whole source missing term vectors entirely, while a sibling source
    /// has them, would otherwise be silently treated as "every doc in
    /// that source has an empty term-vectors document" rather than
    /// surfaced as the caller-side wiring mismatch it actually is.
    #[error(
        "term vectors are present in some merge sources but source index {source_index} (which contributes live docs) has no term-vectors reader"
    )]
    TermVectorsReaderMissingInSource { source_index: usize },
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
    /// Same hazard as [`Error::PostingsIndexOptionsDisagreement`] but for
    /// `FieldInfo::store_payloads`: the merged field's `has_payloads` is
    /// taken from the first-seen source only, so a later source that
    /// genuinely stores payloads (identical `index_options`, so the check
    /// above wouldn't catch it) would have its real payload bytes silently
    /// dropped from the merged output if a disagreeing source went
    /// unchecked -- this is the hard error instead.
    #[error(
        "merged field number {merged_field_number} has disagreeing store_payloads across sources: source claims {source_has_payloads} but the merged field is {merged_has_payloads}"
    )]
    PostingsPayloadsDisagreement {
        merged_field_number: i32,
        merged_has_payloads: bool,
        source_has_payloads: bool,
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

/// Builds the "concatenate sources in order" doc-visit order every
/// `merge_*_doc_values`/`merge_norms`/`merge_term_vectors` helper walks for
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
    let doc_order = concat_doc_order(&per_source_live_ids);

    let numeric_dv =
        merge_numeric_doc_values(sources, &per_source_maps, &per_source_live_ids, &doc_order)?;
    let binary_dv =
        merge_binary_doc_values(sources, &per_source_maps, &per_source_live_ids, &doc_order)?;
    let sorted_dv =
        merge_sorted_doc_values(sources, &per_source_maps, &per_source_live_ids, &doc_order)?;
    let sorted_numeric_dv = merge_sorted_numeric_doc_values(
        sources,
        &per_source_maps,
        &per_source_live_ids,
        &doc_order,
    )?;
    let sorted_set_dv =
        merge_sorted_set_doc_values(sources, &per_source_maps, &per_source_live_ids, &doc_order)?;
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
    let merged_norms = merge_norms(sources, &per_source_maps, &per_source_live_ids, &doc_order)?;
    let tv_docs = merge_term_vectors(sources, &per_source_maps, &per_source_live_ids, &doc_order)?;
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
                has_payloads: f.has_payloads,
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
                num_index_dims: f.num_index_dims,
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
/// Stored fields, doc-values (NUMERIC/BINARY/SORTED/SORTED_NUMERIC/
/// SORTED_SET), norms, and term vectors are all reordered by sort key here,
/// via the same `doc_order` (source_index, original_doc_id) pairs the
/// k-way merge above produces -- each is resolved through the exact same
/// `merge_numeric_doc_values`/`merge_binary_doc_values`/
/// `merge_sorted_doc_values`/`merge_sorted_numeric_doc_values`/
/// `merge_sorted_set_doc_values`/`merge_norms`/`merge_term_vectors` helpers
/// [`merge_stored_only_segments`] uses, just driven by this function's
/// sorted `doc_order` instead of that function's per-source-concatenated
/// one -- so the same single-field-per-type limits, "sparse across
/// sources" errors, and per-source-dictionary-resolution scope notes on
/// this module's top doc comment apply here too. **Postings and points are
/// still not merged by this function** -- a `MergeSource`'s `postings`/
/// `points` fields are silently ignored here, same as before; use
/// [`merge_stored_only_segments`] if that data needs to be merged
/// (concatenation order, no re-sort, and no index-sort metadata in the
/// resulting `.si`, since that merge doesn't preserve sort order).
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
    // Same (source_index, original_doc_id) pairs the k-way merge below
    // visits stored fields in, kept around so doc-values/norms/term-vectors
    // can be resolved in the exact same physical sort order rather than
    // concatenated by source (see `merge_*_doc_values`/`merge_norms`/
    // `merge_term_vectors`'s shared `doc_order` parameter).
    let mut doc_order: Vec<(usize, i32)> = Vec::new();
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
        doc_order.push((src_idx, doc_id));
    }
    let doc_count = merged_docs.len() as i32;

    // Doc-values/norms/term-vectors, resolved in the same global sort order
    // as `merged_docs` above (`doc_order`), not concatenated by source --
    // see this function's updated "Scope" doc comment.
    let numeric_dv =
        merge_numeric_doc_values(sources, &per_source_maps, &per_source_live_ids, &doc_order)?;
    let binary_dv =
        merge_binary_doc_values(sources, &per_source_maps, &per_source_live_ids, &doc_order)?;
    let sorted_dv =
        merge_sorted_doc_values(sources, &per_source_maps, &per_source_live_ids, &doc_order)?;
    let sorted_numeric_dv = merge_sorted_numeric_doc_values(
        sources,
        &per_source_maps,
        &per_source_live_ids,
        &doc_order,
    )?;
    let sorted_set_dv =
        merge_sorted_set_doc_values(sources, &per_source_maps, &per_source_live_ids, &doc_order)?;
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
    let merged_norms = merge_norms(sources, &per_source_maps, &per_source_live_ids, &doc_order)?;
    let tv_docs = merge_term_vectors(sources, &per_source_maps, &per_source_live_ids, &doc_order)?;

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
    doc_order: &[(usize, i32)],
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

    // Resolve each contributing source's entry once, up front -- `doc_order`
    // may interleave sources in any order (a k-way sorted merge does), so
    // the per-source resolution can no longer be folded into a single linear
    // pass over `per_source_live_ids` the way concatenation could.
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
        per_source_entry[idx] = Some(entry);
    }

    let mut values: Vec<i64> = Vec::with_capacity(doc_order.len());
    for &(src_idx, doc_id) in doc_order {
        let entry = per_source_entry[src_idx].ok_or(Error::DocValuesFieldMissingInSource {
            merged_field_number,
        })?;
        let value = doc_values::numeric_value(entry.data, &entry.entry, doc_id)?.ok_or(
            Error::DocValuesFieldMissingInSource {
                merged_field_number,
            },
        )?;
        values.push(value);
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
    doc_order: &[(usize, i32)],
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
    doc_order: &[(usize, i32)],
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

    // This source's own dictionary, in ordinal order -- resolves this
    // source's ordinals to term bytes without needing any other source's
    // dictionary. Resolved once per source up front (see
    // `merge_numeric_doc_values`'s `per_source_entry` for why `doc_order`
    // rules out a single linear pass here).
    type SortedDvResolved<'a> = Option<(&'a SourceSortedDocValues<'a>, Vec<Vec<u8>>)>;
    let mut per_source_resolved: Vec<SortedDvResolved> = Vec::with_capacity(sources.len());
    for ((source, map), live_ids) in sources.iter().zip(per_source_maps).zip(per_source_live_ids) {
        if live_ids.is_empty() {
            per_source_resolved.push(None);
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
        let term =
            source_dict
                .get(ord as usize)
                .ok_or(Error::SortedDocValuesFieldMissingInSource {
                    merged_field_number,
                })?;
        values.push(term.clone());
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
    doc_order: &[(usize, i32)],
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
        per_source_entry[idx] = Some(entry);
    }

    let mut values: Vec<Vec<i64>> = Vec::with_capacity(doc_order.len());
    for &(src_idx, doc_id) in doc_order {
        let entry =
            per_source_entry[src_idx].ok_or(Error::SortedNumericDocValuesFieldMissingInSource {
                merged_field_number,
            })?;
        let doc_values = doc_values::sorted_numeric_values(entry.data, &entry.entry, doc_id)?;
        if doc_values.is_empty() {
            return Err(Error::SortedNumericDocValuesFieldMissingInSource {
                merged_field_number,
            });
        }
        values.push(doc_values);
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
    doc_order: &[(usize, i32)],
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

    // This source's own dictionary, in ordinal order -- resolves this
    // source's ordinals to term bytes without needing any other source's
    // dictionary. Resolved once per source up front, same reason as
    // `merge_sorted_doc_values`.
    type SortedSetDvResolved<'a> = Option<(&'a SourceSortedSetDocValues<'a>, Vec<Vec<u8>>)>;
    let mut per_source_resolved: Vec<SortedSetDvResolved> = Vec::with_capacity(sources.len());
    for ((source, map), live_ids) in sources.iter().zip(per_source_maps).zip(per_source_live_ids) {
        if live_ids.is_empty() {
            per_source_resolved.push(None);
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
    Ok(Some((merged_field_number, values)))
}

/// Same shape and same rules as [`merge_numeric_doc_values`], for norms.
fn merge_norms(
    sources: &[MergeSource],
    per_source_maps: &[HashMap<i32, i32>],
    per_source_live_ids: &[Vec<i32>],
    doc_order: &[(usize, i32)],
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
    doc_order: &[(usize, i32)],
) -> Result<Option<Vec<TermVectorsDocument>>> {
    if sources.iter().all(|s| s.term_vectors.is_none()) {
        return Ok(None);
    }

    for (source_index, (source, live_ids)) in sources.iter().zip(per_source_live_ids).enumerate() {
        if !live_ids.is_empty() && source.term_vectors.is_none() {
            return Err(Error::TermVectorsReaderMissingInSource { source_index });
        }
    }

    let mut merged_docs: Vec<TermVectorsDocument> = Vec::with_capacity(doc_order.len());
    for &(src_idx, doc_id) in doc_order {
        let source = &sources[src_idx];
        let map = &per_source_maps[src_idx];
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
                if source_field.store_payloads != has_payloads {
                    return Err(Error::PostingsPayloadsDisagreement {
                        merged_field_number,
                        merged_has_payloads: has_payloads,
                        source_has_payloads: source_field.store_payloads,
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
            if has_positions && pf.pos_in.is_none() {
                return Err(Error::PostingsPositionsInputMissingInSource {
                    merged_field_number,
                    source_index: per_source_field.len(),
                });
            }
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
            let mut positions: Vec<Vec<i32>> = Vec::new();
            let mut offsets: Vec<Vec<(i32, i32)>> = Vec::new();
            let mut payloads: Vec<Vec<Vec<u8>>> = Vec::new();
            for (src_idx, pf) in per_source_field.iter().enumerate() {
                let Some(pf) = pf else { continue };
                let Some(source_postings) = pf.field_terms.postings(&term, pf.doc_in)? else {
                    continue;
                };
                // Positions (and offsets/payloads, bundled into `Position`)
                // for every live doc this source has for the term, in the
                // same doc order as `source_postings.docs`/`freqs` --
                // `FieldTerms::positions` re-derives docs/freqs itself
                // (cheap relative to the position/offset/payload decode it
                // also does), so this reuses that single read path rather
                // than re-deriving position decoding here.
                let source_positions = if has_positions {
                    Some(
                        pf.field_terms
                            .positions(
                                &term,
                                pf.doc_in,
                                pf.pos_in.expect(
                                    "checked as Error::PostingsPositionsInputMissingInSource above",
                                ),
                                pf.pay_in,
                            )?
                            .expect(
                                "term found via postings() above, so positions() must find it too",
                            ),
                    )
                } else {
                    None
                };
                let doc_id_map = &doc_id_maps[src_idx];
                for (doc_idx, (&doc_id, &freq)) in source_postings
                    .docs
                    .iter()
                    .zip(source_postings.freqs.iter())
                    .enumerate()
                {
                    if let Some(&merged_doc_id) = doc_id_map.get(&doc_id) {
                        docs.push((merged_doc_id, freq));
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
            }
            if !docs.is_empty() {
                terms_out.push(TermPostings {
                    term,
                    docs,
                    positions,
                    offsets,
                    payloads,
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
            has_payloads,
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
        let merged_num_index_dims = merged_field.point_index_dimension_count;
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
            for point in sp.reader.decode_all_points(original_number)? {
                if let Some(&merged_doc_id) = doc_id_map.get(&point.doc_id) {
                    points.push((merged_doc_id, point.packed_value));
                }
            }
        }

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
mod tests {}
