//! A unifying facade over this port's already-built write-side primitives --
//! analogous in spirit to real Lucene's `org.apache.lucene.index.IndexWriter`
//! as the single entry point for add/update/delete/commit, **not** a
//! reimplementation of any of the pieces it composes.
//!
//! # What this is
//!
//! Every write-side capability this facade exposes already existed as a
//! standalone primitive before this module:
//! - [`crate::segment_writer::flush_stored_only_segment`] -- flush a batch
//!   of buffered documents to a new segment.
//! - [`crate::update_document::update_document`] -- atomic delete-by-term +
//!   add-document, committed as one `segments_N`.
//! - [`crate::term_delete::resolve_and_apply_term_delete`] -- resolve a term
//!   to live doc IDs in one segment and apply the delete.
//! - [`crate::segment_infos::write`]/[`crate::segment_infos::read_latest`] --
//!   the `segments_N` commit file itself.
//!
//! What none of those modules provide on their own is a single stateful
//! object a caller can hold onto across several `add_document`/`commit`
//! calls without hand-threading a `SegmentInfos`, a segment-name counter, and
//! a buffered-document list through every call itself. [`IndexWriter`] is
//! exactly that: it owns the buffered-document list, the current committed
//! [`SegmentInfos`], and the next segment-name counter, and calls the
//! existing functions above in the right order with the right state at each
//! lifecycle point (`add_document` buffers; `commit` flushes the buffer via
//! `flush_stored_only_segment` and appends the result to `segment_infos`
//! before writing it; `update_document`/`delete_documents` delegate straight
//! to the existing atomic primitives).
//!
//! # Automatic merge triggering
//!
//! [`IndexWriter::set_merge_policy`] lets a caller opt this writer into
//! automatic merging: once a [`MergePolicyConfig`] is set, every
//! [`IndexWriter::commit`] call, right after writing its own `segments_N`,
//! synchronously asks [`crate::merge_policy::find_merges`] whether any of
//! this writer's *now-committed* segments should merge, and if so executes
//! each proposed group via [`crate::merge::merge_stored_only_segments`] and
//! folds the result in via [`IndexWriter::apply_merge`] -- reusing exactly
//! those two existing functions, not reimplementing either one. This repeats
//! (re-querying `find_merges` against the post-merge segment list) until it
//! proposes nothing further; each merge strictly reduces the total segment
//! count by at least one, so this loop is guaranteed to terminate. By
//! default (no [`MergePolicyConfig`] set), `commit()` behaves exactly as
//! before: no merge-policy consultation at all, matching every existing
//! caller of `commit()` from before this feature existed.
//!
//! This is deliberately synchronous, inside `commit()` itself -- this port
//! has no background-thread/`ConcurrentMergeScheduler`-equivalent
//! infrastructure, so "run it right there" is the only shape that fits
//! everything else here. [`IndexWriter::apply_merge`] remains public and
//! usable on its own for a caller that wants to drive a merge manually
//! instead (e.g. with different sources, or a policy this module doesn't
//! model).
//!
//! Still out of scope: no per-writer merge-policy *tuning* beyond whatever
//! [`MergePolicyConfig`] itself exposes, no concurrent/background merging, no
//! merge-scheduling across many tiers beyond what [`crate::merge_policy`]
//! itself already does in one [`crate::merge_policy::find_merges`] call, and
//! [`IndexWriter::update_document`]/[`IndexWriter::delete_documents`] do not
//! trigger this check (only [`IndexWriter::commit`] does, matching where
//! this port's flush/commit work already lived before this feature).
//!
//! # What this deliberately is not
//!
//! - **No RAM-based flush triggering.** Real `IndexWriter` auto-flushes once
//!   buffered documents exceed `ramBufferSizeMB`; this facade only flushes
//!   on an explicit [`IndexWriter::commit`] call, matching
//!   `segment_writer.rs`'s own documented stance that this port has "no RAM
//!   accounting or automatic flush-triggering" yet.
//! - **No multi-threaded `DocumentsWriterPerThread` pooling, no
//!   `IndexWriterConfig`-style tunable object** -- one caller, one
//!   `Directory`, sequential calls, exactly like every primitive this
//!   facade composes.
//! - **`update_document`/`delete_documents` only resolve against segments
//!   the caller explicitly supplies an opened [`SegmentDeleteSource`] for**
//!   (same limitation `update_document.rs` already documents) -- there is no
//!   reader pool that automatically opens every existing segment's postings
//!   for the caller. In particular, a document sitting only in this
//!   writer's own *unflushed* buffer can never be matched by a delete/update
//!   term (it isn't a segment yet), matching real Lucene's own
//!   `BufferedUpdates` timing (a delete only ever resolves against segments
//!   that exist *at delete time*).
//! - **No two-phase commit (`prepareCommit`/`commit`/`rollback`)** -- each
//!   [`IndexWriter::commit`] is a single, already-atomic
//!   [`crate::segment_infos::write`] call (see that function's own doc
//!   comment on why it bakes `Directory::sync` in); there is no separate
//!   "prepare" step to roll back before the final rename the way real
//!   Lucene's two-phase commit protocol has.
//!
//! # Segment/commit-file lifecycle
//!
//! [`IndexWriter::open`] looks for an existing `segments_N` in `dir` (via
//! [`lucene_store::directory::last_commit_generation`], not
//! [`crate::segment_infos::read_latest`] directly, so "no commit yet" is
//! distinguished from "a commit file exists but is corrupt" -- the latter
//! still surfaces as an `Err`, matching this port's stance elsewhere of
//! never treating corruption as an empty index). If none is found, it starts
//! from a fresh, empty [`SegmentInfos`] (generation/version/counter all `0`,
//! no segments) -- the first [`IndexWriter::commit`] then writes `segments_1`
//! (`SegmentInfos::write` picks a `generation` field the caller controls;
//! this facade always writes the *next* generation, matching real Lucene's
//! monotonic commit-generation counter). Segment names follow the real
//! `_0`, `_1`, ... convention (`IndexFileNames.segmentFileName`'s counter),
//! driven off `segment_infos.counter` so a writer resumed on an
//! already-committed directory doesn't collide with segment names an
//! earlier writer session already used.

use crate::deletes;
use crate::indexing_chain::invert_documents;
use crate::merge;
use crate::merge_policy;
use crate::segment_info::{self, LuceneVersion};
use crate::segment_infos::{self, SegmentCommitInfo, SegmentInfos};
use crate::segment_writer::{self, flush_stored_only_segment};
use crate::term_delete;
use crate::update_document::{self, SegmentDeleteSource};

use lucene_analysis::Analyzer;
use lucene_codecs::doc_values;
use lucene_codecs::field_infos::{DocValuesType, FieldInfo, IndexOptions};
use lucene_codecs::postings_writer::{self, FieldPostingsInput, TermPostings};
use lucene_codecs::stored_fields::{self, Document, FieldValue};
use lucene_codecs::term_vectors::{self, TermVectorField, TermVectorTerm, TermVectorsDocument};
use lucene_store::codec_util::ID_LENGTH;
use lucene_store::data_output::DataOutput;
use lucene_store::directory::Directory;
use lucene_util::fixed_bit_set::FixedBitSet;

pub use crate::merge_policy::MergePolicyConfig;
pub use crate::update_document::SegmentDeleteSource as DeleteSource;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    #[error(transparent)]
    SegmentWriter(#[from] segment_writer::Error),
    #[error(transparent)]
    SegmentInfos(#[from] segment_infos::Error),
    #[error(transparent)]
    UpdateDocument(#[from] update_document::Error),
    #[error(transparent)]
    TermDelete(#[from] term_delete::Error),
    #[error(transparent)]
    Deletes(#[from] deletes::Error),
    #[error(transparent)]
    Merge(#[from] merge::Error),
    #[error(transparent)]
    SegmentInfo(#[from] segment_info::Error),
    #[error(transparent)]
    StoredFields(#[from] lucene_codecs::stored_fields::Error),
    #[error(transparent)]
    LiveDocs(#[from] lucene_codecs::live_docs::Error),
    #[error(transparent)]
    PostingsWriter(#[from] postings_writer::Error),
    #[error(transparent)]
    TermVectors(#[from] term_vectors::Error),
    #[error(transparent)]
    DocValues(#[from] doc_values::WriteError),
    #[error("set_postings_field: no field named {0:?} in this writer's field list")]
    UnknownPostingsField(String),
    #[error(
        "set_postings_field: field {0:?} has index_options {1:?}; only Docs/DocsAndFreqs \
         is supported by this writer's postings write-side"
    )]
    UnsupportedPostingsIndexOptions(String, IndexOptions),
    #[error("set_term_vector_field: no field named {0:?} in this writer's field list")]
    UnknownTermVectorField(String),
    #[error(
        "set_term_vector_field: field {0:?} does not have store_term_vectors set; \
         this writer's term-vector write-side only builds term vectors for a field \
         whose FieldInfo already advertises them"
    )]
    UnsupportedTermVectorField(String),
    #[error("set_doc_values_field: no field named {0:?} in this writer's field list")]
    UnknownDocValuesField(String),
    #[error(
        "set_doc_values_field: field {0:?} has doc_values_type {1:?}; only \
         DocValuesType::Numeric is supported by this writer's doc-values write-side"
    )]
    UnsupportedDocValuesType(String, DocValuesType),
    #[error(
        "commit: doc-values field {0:?} is dense-only (write_single_dense_numeric_field has \
         no missing-value encoding) but pending doc {1} has no value for it"
    )]
    MissingDenseDocValue(String, usize),
    #[error(
        "commit: doc-values field {0:?} requires FieldValue::Int or FieldValue::Long on every \
         pending doc, but doc {1} has a {2} value"
    )]
    NonNumericDocValue(String, usize, &'static str),
}

pub type Result<T> = std::result::Result<T, Error>;

/// A single, coherent entry point over this port's write-side primitives.
/// See the module doc comment for the exact lifecycle and scope.
pub struct IndexWriter<'d> {
    dir: &'d dyn Directory,
    fields: Vec<FieldInfo>,
    codec_name: String,
    lucene_version: LuceneVersion,
    segment_infos: SegmentInfos,
    pending_docs: Vec<Document>,
    merge_policy: Option<MergePolicyConfig>,
    postings_field: Option<PostingsFieldConfig>,
    term_vector_field: Option<TermVectorFieldConfig>,
    doc_values_field: Option<DocValuesFieldConfig>,
}

/// One field this writer has been opted into also indexing real postings
/// for, resolved once by [`IndexWriter::set_postings_field`] against this
/// writer's fixed `fields` list (see that method's doc comment for the exact
/// scope this mirrors from [`postings_writer::write_single_field`]).
#[derive(Debug, Clone)]
struct PostingsFieldConfig {
    name: String,
    field_number: i32,
    index_options: IndexOptions,
}

/// One field this writer has been opted into also building real term
/// vectors for, resolved once by [`IndexWriter::set_term_vector_field`]
/// against this writer's fixed `fields` list -- same "resolve once, reuse
/// every commit" shape as [`PostingsFieldConfig`].
#[derive(Debug, Clone)]
struct TermVectorFieldConfig {
    name: String,
    field_number: i32,
}

/// One field this writer has been opted into also building real NUMERIC doc
/// values for, resolved once by [`IndexWriter::set_doc_values_field`] against
/// this writer's fixed `fields` list -- same "resolve once, reuse every
/// commit" shape as [`PostingsFieldConfig`]/[`TermVectorFieldConfig`].
#[derive(Debug, Clone)]
struct DocValuesFieldConfig {
    name: String,
    field_number: i32,
}

impl<'d> IndexWriter<'d> {
    /// Opens a writer over `dir`: resumes the latest existing commit if one
    /// is present, or starts a brand-new, empty index otherwise. `fields`
    /// describes every field [`IndexWriter::add_document`]/
    /// [`IndexWriter::update_document`] documents may use (this facade has
    /// no per-call schema reconciliation the way [`crate::merge`] does
    /// across sources -- every document flushed through one writer shares
    /// one fixed field list, same as every existing caller of
    /// [`flush_stored_only_segment`]). `codec_name`/`lucene_version` are
    /// recorded on every segment this writer flushes, same meaning as the
    /// identically-named parameters on [`flush_stored_only_segment`].
    pub fn open(
        dir: &'d dyn Directory,
        fields: Vec<FieldInfo>,
        codec_name: impl Into<String>,
        lucene_version: LuceneVersion,
    ) -> Result<Self> {
        let files = dir.list_all()?;
        let generation = lucene_store::directory::last_commit_generation(&files);
        let segment_infos = if generation < 0 {
            empty_segment_infos(lucene_version)
        } else {
            segment_infos::read_latest(dir)?
        };

        Ok(IndexWriter {
            dir,
            fields,
            codec_name: codec_name.into(),
            lucene_version,
            segment_infos,
            pending_docs: Vec::new(),
            merge_policy: None,
            postings_field: None,
            term_vector_field: None,
            doc_values_field: None,
        })
    }

    /// Opts this writer into also building and writing real postings
    /// (`.doc`/`.tim`/`.tip`/`.tmd`, via
    /// [`postings_writer::write_single_field`]) for one field of every
    /// segment [`IndexWriter::commit`] flushes from here on -- mirroring
    /// real Lucene's per-field `FieldType.setIndexOptions`, except this
    /// facade only ever indexes **one** field at a time (see
    /// [`postings_writer::write_single_field`]'s own "one field per call"
    /// scope note; there is no per-field file-suffix machinery here to fan
    /// that out to more than one field within a single segment).
    ///
    /// `Some(field_name)` looks `field_name` up in this writer's fixed
    /// `fields` list (from [`IndexWriter::open`]) and requires its
    /// `index_options` to already be `IndexOptions::Docs` or
    /// `IndexOptions::DocsAndFreqs` (an `Err` otherwise) -- the same
    /// analyzed-field-text convention real Lucene's own `FieldType` uses to
    /// mark a field indexable, and the same `index_options` restriction
    /// [`postings_writer::write_single_field`] itself enforces (no
    /// positions/offsets/payloads yet). `None` (the default a freshly
    /// [`IndexWriter::open`]ed writer starts with) turns this back off --
    /// `commit()` then behaves exactly as it did before this feature
    /// existed (stored fields only, matching every pre-existing caller).
    ///
    /// Only [`FieldValue::String`] values contribute indexable text for the
    /// opted-in field -- a document with no value, or a non-`String` value,
    /// for that field contributes no postings for that document (same "best
    /// effort per document" shape [`crate::indexing_chain::invert_documents`]
    /// already has for a missing `(doc_id, field, text)` triple).
    pub fn set_postings_field(&mut self, field_name: Option<&str>) -> Result<()> {
        self.postings_field = match field_name {
            None => None,
            Some(name) => {
                let info = self
                    .fields
                    .iter()
                    .find(|f| f.name == name)
                    .ok_or_else(|| Error::UnknownPostingsField(name.to_string()))?;
                if !matches!(
                    info.index_options,
                    IndexOptions::Docs | IndexOptions::DocsAndFreqs
                ) {
                    return Err(Error::UnsupportedPostingsIndexOptions(
                        name.to_string(),
                        info.index_options,
                    ));
                }
                Some(PostingsFieldConfig {
                    name: name.to_string(),
                    field_number: info.number,
                    index_options: info.index_options,
                })
            }
        };
        Ok(())
    }

    /// Opts this writer into also building and writing real term vectors
    /// (`.tvd`/`.tvx`/`.tvm`, via [`term_vectors::write_best_speed`]) for one
    /// field of every segment [`IndexWriter::commit`] flushes from here on --
    /// same "one field per call" scope as [`IndexWriter::set_postings_field`]
    /// (there is no per-field file-suffix machinery here to fan this out to
    /// more than one field within a single segment, and
    /// [`term_vectors::write_best_speed`] itself is a single-chunk writer,
    /// same as [`postings_writer::write_single_field`]).
    ///
    /// `Some(field_name)` looks `field_name` up in this writer's fixed
    /// `fields` list and requires its `store_term_vectors` flag to already be
    /// `true` (an `Err` otherwise) -- matching real Lucene's own
    /// `FieldType.setStoreTermVectors` convention, and this crate's own
    /// [`lucene_codecs::field_infos::FieldInfo::check_consistency`]
    /// invariant that a non-indexed field can never set that flag. `None`
    /// (the default a freshly [`IndexWriter::open`]ed writer starts with)
    /// turns this back off -- `commit()` then behaves exactly as it did
    /// before this feature existed.
    ///
    /// Only [`FieldValue::String`] values contribute indexable text for the
    /// opted-in field -- a document with no value, or a non-`String` value,
    /// for that field contributes no term vector for that document (same
    /// "best effort per document" shape [`IndexWriter::set_postings_field`]
    /// already has). This is independent of
    /// [`IndexWriter::set_postings_field`] -- a writer may have both set at
    /// once (to the same field or different fields); each is built and
    /// written from its own in-memory pass over `pending_docs` before
    /// anything reaches `dir`.
    pub fn set_term_vector_field(&mut self, field_name: Option<&str>) -> Result<()> {
        self.term_vector_field = match field_name {
            None => None,
            Some(name) => {
                let info = self
                    .fields
                    .iter()
                    .find(|f| f.name == name)
                    .ok_or_else(|| Error::UnknownTermVectorField(name.to_string()))?;
                if !info.store_term_vectors {
                    return Err(Error::UnsupportedTermVectorField(name.to_string()));
                }
                Some(TermVectorFieldConfig {
                    name: name.to_string(),
                    field_number: info.number,
                })
            }
        };
        Ok(())
    }

    /// Opts this writer into also building and writing real NUMERIC doc
    /// values (`.dvd`/`.dvm`/`.dvs`, via
    /// [`doc_values::write_single_dense_numeric_field`]) for one field of
    /// every segment [`IndexWriter::commit`] flushes from here on -- same
    /// "one field per call" scope as [`IndexWriter::set_postings_field`]/
    /// [`IndexWriter::set_term_vector_field`] (no per-field file-suffix
    /// machinery here to fan this out to more than one field within a single
    /// segment).
    ///
    /// **Only NUMERIC doc values are wired up by this writer.** BINARY and
    /// SORTED_NUMERIC write-side functions already exist in
    /// [`lucene_codecs::doc_values`] but are not wired into this facade --
    /// see that module's parity notes and `docs/parity.md` for the exact
    /// scope split.
    ///
    /// `Some(field_name)` looks `field_name` up in this writer's fixed
    /// `fields` list and requires its `doc_values_type` to already be
    /// `DocValuesType::Numeric` (an `Err` otherwise) -- matching real
    /// Lucene's own `FieldType.setDocValuesType(DocValuesType.NUMERIC)`
    /// convention. `None` (the default a freshly [`IndexWriter::open`]ed
    /// writer starts with) turns this back off -- `commit()` then behaves
    /// exactly as it did before this feature existed.
    ///
    /// **Dense-only, enforced fail-fast at `commit()` time, not here:**
    /// [`doc_values::write_single_dense_numeric_field`] has no missing-value
    /// encoding, so *every* pending doc at the time of a `commit()` call
    /// must carry a [`FieldValue::Int`] or [`FieldValue::Long`] value for
    /// the opted-in field, or the whole `commit()` call fails with
    /// [`Error::MissingDenseDocValue`]/[`Error::NonNumericDocValue`] and
    /// leaves `dir`/`pending_docs`/`segment_infos` unchanged -- unlike
    /// [`IndexWriter::set_postings_field`]/
    /// [`IndexWriter::set_term_vector_field`], a missing or wrong-typed value
    /// for this field is not "best effort, skip that doc" (there is no
    /// sparse/missing-value doc-values shape to fall back to here).
    pub fn set_doc_values_field(&mut self, field_name: Option<&str>) -> Result<()> {
        self.doc_values_field = match field_name {
            None => None,
            Some(name) => {
                let info = self
                    .fields
                    .iter()
                    .find(|f| f.name == name)
                    .ok_or_else(|| Error::UnknownDocValuesField(name.to_string()))?;
                if info.doc_values_type != DocValuesType::Numeric {
                    return Err(Error::UnsupportedDocValuesType(
                        name.to_string(),
                        info.doc_values_type,
                    ));
                }
                Some(DocValuesFieldConfig {
                    name: name.to_string(),
                    field_number: info.number,
                })
            }
        };
        Ok(())
    }

    /// Opts this writer into automatic merge triggering (see module doc
    /// comment): `Some(config)` makes every subsequent
    /// [`IndexWriter::commit`] call consult
    /// [`crate::merge_policy::find_merges`] with `config` after writing its
    /// own commit, and execute/fold in whatever it proposes. `None` (the
    /// default a freshly [`IndexWriter::open`]ed writer starts with) turns
    /// this back off -- `commit()` then behaves exactly as it did before
    /// this feature existed.
    pub fn set_merge_policy(&mut self, config: Option<MergePolicyConfig>) {
        self.merge_policy = config;
    }

    /// Buffers `doc` for the next [`IndexWriter::commit`] -- real Lucene's
    /// `IndexWriter.addDocument`, minus the RAM-threshold auto-flush this
    /// port doesn't have (see module doc comment). Nothing is written to
    /// `dir` until `commit` is called.
    pub fn add_document(&mut self, doc: Document) {
        self.pending_docs.push(doc);
    }

    /// The atomic delete-by-term + add-document real Lucene calls
    /// `updateDocument`: delegates directly to
    /// [`update_document::update_document`], flushing `doc` as a brand-new
    /// segment and applying the term delete to every segment `delete_sources`
    /// supplies an opened source for, all in one commit. Unlike
    /// [`IndexWriter::add_document`], this is **not** buffered -- it commits
    /// immediately (matching [`update_document::update_document`]'s own
    /// all-or-nothing atomicity, which only makes sense as an immediate
    /// commit; buffering it would let a later `commit()` observe a
    /// half-applied update if that call somehow failed in between).
    ///
    /// Bumps this writer's in-memory [`IndexWriter::segment_infos`] to the
    /// new commit on success and returns it; on `Err`, nothing was written
    /// (see [`update_document::update_document`]'s own atomicity guarantee)
    /// and this writer's state is unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn update_document(
        &mut self,
        delete_sources: &[SegmentDeleteSource],
        field: &str,
        term: &[u8],
        new_doc: Document,
    ) -> Result<&SegmentInfos> {
        let new_segment_name = self.next_segment_name();
        let new_segment_id = generate_segment_id(self.segment_infos.counter);

        let updated = update_document::update_document(
            self.dir,
            &self.segment_infos,
            delete_sources,
            field,
            term,
            &new_segment_name,
            new_segment_id,
            &self.codec_name,
            self.lucene_version,
            &self.fields,
            std::slice::from_ref(&new_doc),
        )?;
        self.segment_infos = updated;
        self.segment_infos.counter += 1;
        Ok(&self.segment_infos)
    }

    /// Deletes every live doc matching `(field, term)` in whichever of this
    /// writer's current segments `delete_sources` supplies an opened source
    /// for -- delegates to
    /// [`term_delete::resolve_and_apply_term_delete`] per matching segment,
    /// then commits the whole updated segment list as one new `segments_N`
    /// generation (same atomicity shape as
    /// [`IndexWriter::update_document`]: either every targeted segment's
    /// `.liv` update lands in the same commit, or -- on the first failure --
    /// nothing commits and this writer's state is unchanged).
    ///
    /// A segment with no matching entry in `delete_sources` is left
    /// untouched (same "caller supplies whatever it has open" scope as
    /// [`update_document::SegmentDeleteSource`]'s own doc comment).
    pub fn delete_documents(
        &mut self,
        delete_sources: &[SegmentDeleteSource],
        field: &str,
        term: &[u8],
    ) -> Result<&SegmentInfos> {
        let mut updated_segments = Vec::with_capacity(self.segment_infos.segments.len());
        for sci in &self.segment_infos.segments {
            match delete_sources
                .iter()
                .find(|src| src.segment_name == sci.segment_name)
            {
                Some(src) => {
                    let updated = term_delete::resolve_and_apply_term_delete(
                        self.dir,
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

        let mut new_segment_infos = self.segment_infos.clone();
        new_segment_infos.generation += 1;
        new_segment_infos.version += 1;
        new_segment_infos.segments = updated_segments;
        segment_infos::write(&new_segment_infos, self.dir)?;

        self.segment_infos = new_segment_infos;
        Ok(&self.segment_infos)
    }

    /// Flushes every currently-buffered [`IndexWriter::add_document`] call
    /// (if any) to a brand-new segment via
    /// [`flush_stored_only_segment`], appends it to this writer's segment
    /// list, and writes the whole updated list as the next `segments_N`
    /// generation via [`crate::segment_infos::write`] -- real Lucene's
    /// `IndexWriter.commit()` after one or more buffered
    /// `DocumentsWriterPerThread.flush()`-worth of documents. If this writer
    /// has a [`MergePolicyConfig`] set via [`IndexWriter::set_merge_policy`],
    /// this also performs real `commit()`'s automatic merge-triggering step
    /// (see module doc comment) right after the flush commits; with no
    /// merge policy set (the default), this method is unchanged from before
    /// that feature existed.
    ///
    /// A `commit()` with an empty pending-document buffer still writes the
    /// next `segments_N` generation (bumping `version`) with no new
    /// segment appended -- matches real Lucene's `commit()` being a valid,
    /// if unusual, no-op-content commit rather than a special "nothing to do"
    /// case that skips writing. Returns the new committed [`SegmentInfos`].
    ///
    /// If [`IndexWriter::set_postings_field`] has opted this writer into
    /// postings for one field, this also builds and writes that field's real
    /// `.doc`/`.tim`/`.tip`/`.tmd` for the flushed segment (see
    /// [`IndexWriter::build_postings_output`]/
    /// [`IndexWriter::write_postings_files`]) -- entirely in memory *before*
    /// anything is written to `dir`, so a docFreq >= 256 term (this writer's
    /// documented single-`.tim`-block limit, see
    /// [`postings_writer::write_single_field`]) makes the **whole** `commit()`
    /// call fail with `Err` and leaves `dir`/`pending_docs`/`segment_infos`
    /// completely unchanged, exactly like [`IndexWriter::update_document`]'s
    /// own atomicity guarantee -- never a partially-written segment.
    pub fn commit(&mut self) -> Result<&SegmentInfos> {
        let mut new_segment_infos = self.segment_infos.clone();
        new_segment_infos.generation += 1;
        new_segment_infos.version += 1;

        if !self.pending_docs.is_empty() {
            let segment_name = self.next_segment_name();
            let segment_id = generate_segment_id(self.segment_infos.counter);

            // Built and validated entirely in memory before anything is
            // written to `dir` -- see this method's own doc comment on why
            // that ordering is what makes a docFreq-too-large rejection
            // atomic.
            let postings_output = match &self.postings_field {
                Some(cfg) => Self::build_postings_output(&self.pending_docs, cfg, &segment_id)?,
                None => None,
            };
            let term_vectors_output = match &self.term_vector_field {
                Some(cfg) => Self::build_term_vectors_output(&self.pending_docs, cfg)
                    .map(|docs| term_vectors::write_best_speed(&docs, &segment_id, "")),
                None => None,
            };
            let doc_values_output = match &self.doc_values_field {
                Some(cfg) => Some(Self::build_doc_values_output(
                    &self.pending_docs,
                    cfg,
                    &segment_id,
                )?),
                None => None,
            };

            let sci = flush_stored_only_segment(
                self.dir,
                &segment_name,
                segment_id,
                &self.codec_name,
                self.lucene_version,
                &self.fields,
                &self.pending_docs,
                false,
            )?;

            if let Some(output) = postings_output {
                Self::write_postings_files(self.dir, &segment_name, &segment_id, &output)?;
            }
            if let Some((tvd, tvx, tvm)) = term_vectors_output {
                Self::write_term_vector_files(
                    self.dir,
                    &segment_name,
                    &segment_id,
                    &tvd,
                    &tvx,
                    &tvm,
                )?;
            }
            if let Some((dvm, dvd, dvs)) = doc_values_output {
                Self::write_doc_values_files(
                    self.dir,
                    &segment_name,
                    &segment_id,
                    &dvm,
                    &dvd,
                    &dvs,
                )?;
            }

            new_segment_infos.segments.push(sci);
            new_segment_infos.counter += 1;
            self.pending_docs.clear();
        }

        segment_infos::write(&new_segment_infos, self.dir)?;
        self.segment_infos = new_segment_infos;

        if self.merge_policy.is_some() {
            self.auto_merge()?;
        }

        Ok(&self.segment_infos)
    }

    /// Builds [`postings_writer::write_single_field`]'s input from `docs`'
    /// [`FieldValue::String`] values for `config.field_number` (each pending
    /// doc's index into `docs` becomes its doc ID in the new segment,
    /// matching [`flush_stored_only_segment`]'s own doc-ordering), tokenizes
    /// via [`crate::indexing_chain::invert_documents`] with a plain
    /// [`Analyzer::standard`] (no stopwords -- this facade has no
    /// per-field-analyzer configuration yet, see module doc comment's scope
    /// notes elsewhere in this crate), and calls
    /// [`postings_writer::write_single_field`] to actually encode the bytes.
    ///
    /// Returns `Ok(None)` when no pending doc has any indexable text for this
    /// field (nothing to write -- not an error; matches
    /// [`postings_writer::write_single_field`]'s own `Error::EmptyTerms`
    /// being a caller-input problem, not a "commit anyway" outcome we want to
    /// force on every commit that happens to have no postings content).
    /// Returns `Err` on [`postings_writer::write_single_field`]'s own
    /// validation failures, in particular
    /// [`postings_writer::Error::DocFreqTooLarge`] once any one term in this
    /// commit's batch occurs in `>= BLOCK_SIZE` (256) pending docs -- this
    /// writer has no multi-block `.tim` support, so that case is rejected
    /// rather than silently producing wrong bytes (see module doc comment).
    fn build_postings_output(
        docs: &[Document],
        config: &PostingsFieldConfig,
        segment_id: &[u8; ID_LENGTH],
    ) -> Result<Option<postings_writer::Output>> {
        let mut triples: Vec<(i32, &str, &str)> = Vec::new();
        for (doc_id, doc) in docs.iter().enumerate() {
            let text = doc
                .fields
                .iter()
                .find(|f| f.field_number == config.field_number)
                .and_then(|f| match &f.value {
                    FieldValue::String(s) => Some(s.as_str()),
                    _ => None,
                });
            if let Some(text) = text {
                triples.push((doc_id as i32, config.name.as_str(), text));
            }
        }
        if triples.is_empty() {
            return Ok(None);
        }

        let analyzer = Analyzer::standard(None);
        let inverted = invert_documents(&triples, &analyzer);

        // Every triple built above shares `config.name` as its field, so
        // `inverted.terms` (keyed by `(field, term)`) only ever has entries
        // for this one field -- no need to filter by field here. Its
        // `BTreeMap` iteration order is therefore already ascending by term
        // bytes (the ordering `postings_writer::write_single_field`
        // requires), so no separate sort is needed either.
        let mut doc_ids = std::collections::BTreeSet::new();
        let mut terms: Vec<TermPostings> = Vec::new();
        for ((_, term), entries) in &inverted.terms {
            let term_docs: Vec<(i32, i32)> = entries
                .iter()
                .map(|entry| {
                    doc_ids.insert(entry.doc_id);
                    (entry.doc_id, entry.term_freq())
                })
                .collect();
            terms.push(TermPostings {
                term: term.as_bytes().to_vec(),
                docs: term_docs,
            });
        }
        if terms.is_empty() {
            return Ok(None);
        }

        let input = FieldPostingsInput {
            field_number: config.field_number,
            index_options: config.index_options,
            doc_count: doc_ids.len() as i32,
            terms: &terms,
        };
        let output = postings_writer::write_single_field(&input, segment_id, "")?;
        Ok(Some(output))
    }

    /// Builds one [`TermVectorsDocument`] per entry in `docs` (in the same
    /// doc-ID order [`flush_stored_only_segment`] uses -- index into `docs`
    /// == doc ID in the new segment), sourced from `config.field_number`'s
    /// [`FieldValue::String`] values, tokenized via
    /// [`crate::indexing_chain::invert_documents`] with a plain
    /// [`Analyzer::standard`] (same analyzer/scope notes as
    /// [`IndexWriter::build_postings_output`]).
    ///
    /// [`crate::indexing_chain::invert_documents`] returns a *term-keyed*
    /// inverted index (postings grouped by `(field, term)`, each entry a
    /// doc-ID-sorted list) -- the shape a postings writer wants. Term
    /// vectors need the transpose: *per-document* `term -> (freq,
    /// positions)`, so this regroups that same inverted index by doc ID
    /// rather than reimplementing tokenization a second time.
    ///
    /// Returns `Ok(None)` when no pending doc has any indexable text for this
    /// field at all (mirrors [`IndexWriter::build_postings_output`]'s own
    /// "nothing to write this commit" outcome). Otherwise returns exactly
    /// `docs.len()` entries, one per doc ID -- a doc with no term-vector data
    /// of its own still gets an entry with an empty `fields` list (a
    /// legitimate, readable "no term vectors for this doc" shape; see
    /// [`term_vectors::write_best_speed`]'s own tests for this exact case),
    /// never a shorter vector, since [`term_vectors::write_best_speed`]
    /// derives `max_doc` directly from `docs.len()`.
    fn build_term_vectors_output(
        docs: &[Document],
        config: &TermVectorFieldConfig,
    ) -> Option<Vec<TermVectorsDocument>> {
        let mut triples: Vec<(i32, &str, &str)> = Vec::new();
        for (doc_id, doc) in docs.iter().enumerate() {
            let text = doc
                .fields
                .iter()
                .find(|f| f.field_number == config.field_number)
                .and_then(|f| match &f.value {
                    FieldValue::String(s) => Some(s.as_str()),
                    _ => None,
                });
            if let Some(text) = text {
                triples.push((doc_id as i32, config.name.as_str(), text));
            }
        }
        if triples.is_empty() {
            return None;
        }

        let analyzer = Analyzer::standard(None);
        let inverted = invert_documents(&triples, &analyzer);
        if inverted.terms.is_empty() {
            // Every doc's text tokenized to zero terms (e.g. only
            // whitespace) -- same "nothing to write this commit" outcome as
            // an empty `triples`, not an error.
            return None;
        }

        // Regroup the term-keyed inverted index by doc ID: for each doc,
        // collect every term it occurs in (ascending term-byte order, since
        // `inverted.terms` -- a `BTreeMap` keyed only by this one field's
        // `(field, term)` pairs here -- already iterates that way) into one
        // `TermVectorField`.
        let mut per_doc: Vec<Vec<TermVectorTerm>> = vec![Vec::new(); docs.len()];
        for ((_, term), entries) in &inverted.terms {
            for entry in entries {
                per_doc[entry.doc_id as usize].push(TermVectorTerm {
                    term: term.as_bytes().to_vec(),
                    freq: entry.term_freq(),
                    positions: Some(entry.positions()),
                    start_offsets: None,
                    end_offsets: None,
                    payloads: None,
                });
            }
        }

        let tv_docs = per_doc
            .into_iter()
            .map(|terms| TermVectorsDocument {
                fields: if terms.is_empty() {
                    Vec::new()
                } else {
                    vec![TermVectorField {
                        field_number: config.field_number,
                        has_positions: true,
                        has_offsets: false,
                        has_payloads: false,
                        terms,
                    }]
                },
            })
            .collect();
        Some(tv_docs)
    }

    /// Builds [`doc_values::write_single_dense_numeric_field`]'s input from
    /// `docs`' values for `config.field_number` (each pending doc's index
    /// into `docs` becomes its doc ID in the new segment, matching
    /// [`flush_stored_only_segment`]'s own doc-ordering) and calls it to
    /// actually encode the bytes.
    ///
    /// Unlike [`IndexWriter::build_postings_output`]/
    /// [`IndexWriter::build_term_vectors_output`], this has no "best effort,
    /// skip that doc" fallback: [`doc_values::write_single_dense_numeric_field`]
    /// is dense-only, so a pending doc with no value at all for
    /// `config.field_number` fails the whole `commit()` call with
    /// [`Error::MissingDenseDocValue`], and a pending doc whose value isn't
    /// [`FieldValue::Int`]/[`FieldValue::Long`] fails it with
    /// [`Error::NonNumericDocValue`] -- see [`IndexWriter::set_doc_values_field`]'s
    /// doc comment. Called only when `docs` is non-empty (see `commit`'s own
    /// `!self.pending_docs.is_empty()` guard), so this never has to decide
    /// what an empty-batch doc-values write even means.
    fn build_doc_values_output(
        docs: &[Document],
        config: &DocValuesFieldConfig,
        segment_id: &[u8; ID_LENGTH],
    ) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
        let mut values: Vec<i64> = Vec::with_capacity(docs.len());
        for (doc_id, doc) in docs.iter().enumerate() {
            let field = doc
                .fields
                .iter()
                .find(|f| f.field_number == config.field_number)
                .ok_or_else(|| Error::MissingDenseDocValue(config.name.clone(), doc_id))?;
            let value = match &field.value {
                FieldValue::Int(v) => *v as i64,
                FieldValue::Long(v) => *v,
                other => {
                    return Err(Error::NonNumericDocValue(
                        config.name.clone(),
                        doc_id,
                        field_value_kind(other),
                    ));
                }
            };
            values.push(value);
        }

        let max_doc = docs.len() as i32;
        let output = doc_values::write_single_dense_numeric_field(
            config.field_number,
            &values,
            max_doc,
            segment_id,
            "",
        )?;
        Ok(output)
    }

    /// Writes [`IndexWriter::build_doc_values_output`]'s three files
    /// (`<segment_name>.dvm`/`.dvd`/`.dvs`) into `dir` and patches the
    /// already-written `<segment_name>.si` to list them -- same
    /// read-modify-write-then-resync pattern
    /// [`IndexWriter::write_term_vector_files`]/
    /// [`IndexWriter::write_postings_files`] already use.
    fn write_doc_values_files(
        dir: &dyn Directory,
        segment_name: &str,
        segment_id: &[u8; ID_LENGTH],
        dvm: &[u8],
        dvd: &[u8],
        dvs: &[u8],
    ) -> Result<()> {
        let dvm_name = format!("{segment_name}.dvm");
        let dvd_name = format!("{segment_name}.dvd");
        let dvs_name = format!("{segment_name}.dvs");

        for (name, bytes) in [(&dvm_name, dvm), (&dvd_name, dvd), (&dvs_name, dvs)] {
            write_file(dir, name, bytes)?;
        }

        let si_name = format!("{segment_name}.si");
        let si_bytes: Vec<u8> = dir.open(&si_name)?.to_vec();
        let mut si = segment_info::parse(&si_bytes, segment_id)?;
        si.files
            .extend([dvm_name.clone(), dvd_name.clone(), dvs_name.clone()]);
        let si_bytes = segment_info::write(&si, "");
        write_file(dir, &si_name, &si_bytes)?;

        dir.sync(&[dvm_name, dvd_name, dvs_name, si_name])?;
        Ok(())
    }

    /// Writes [`IndexWriter::build_term_vectors_output`]'s three files
    /// (`<segment_name>.tvd`/`.tvx`/`.tvm`) into `dir` and patches the
    /// already-written `<segment_name>.si` to list them -- same
    /// read-modify-write-then-resync pattern
    /// [`IndexWriter::write_postings_files`] already uses, reused rather than
    /// duplicated here so a segment with both postings and term vectors
    /// configured ends up with one `.si` correctly listing all seven files
    /// (whichever write happened first is read back and extended by the
    /// second, not overwritten).
    fn write_term_vector_files(
        dir: &dyn Directory,
        segment_name: &str,
        segment_id: &[u8; ID_LENGTH],
        tvd: &[u8],
        tvx: &[u8],
        tvm: &[u8],
    ) -> Result<()> {
        let tvd_name = format!("{segment_name}.tvd");
        let tvx_name = format!("{segment_name}.tvx");
        let tvm_name = format!("{segment_name}.tvm");

        for (name, bytes) in [(&tvd_name, tvd), (&tvx_name, tvx), (&tvm_name, tvm)] {
            write_file(dir, name, bytes)?;
        }

        let si_name = format!("{segment_name}.si");
        let si_bytes: Vec<u8> = dir.open(&si_name)?.to_vec();
        let mut si = segment_info::parse(&si_bytes, segment_id)?;
        si.files
            .extend([tvd_name.clone(), tvx_name.clone(), tvm_name.clone()]);
        let si_bytes = segment_info::write(&si, "");
        write_file(dir, &si_name, &si_bytes)?;

        dir.sync(&[tvd_name, tvx_name, tvm_name, si_name])?;
        Ok(())
    }

    /// Writes [`IndexWriter::build_postings_output`]'s four files
    /// (`<segment_name>.doc`/`.tim`/`.tip`/`.tmd`) into `dir` and patches the
    /// already-written `<segment_name>.si` (from
    /// [`flush_stored_only_segment`], called just before this) to list them
    /// in [`crate::segment_info::SegmentInfo::files`] -- same
    /// read-modify-write-then-resync pattern
    /// [`crate::segment_writer::flush_sorted_stored_only_segment`] already
    /// uses to patch a `.si` after the fact, rather than duplicating
    /// [`flush_stored_only_segment`]'s own file-writing sequence here.
    fn write_postings_files(
        dir: &dyn Directory,
        segment_name: &str,
        segment_id: &[u8; ID_LENGTH],
        output: &postings_writer::Output,
    ) -> Result<()> {
        let doc_name = format!("{segment_name}.doc");
        let tim_name = format!("{segment_name}.tim");
        let tip_name = format!("{segment_name}.tip");
        let tmd_name = format!("{segment_name}.tmd");

        for (name, bytes) in [
            (&doc_name, &output.doc),
            (&tim_name, &output.tim),
            (&tip_name, &output.tip),
            (&tmd_name, &output.tmd),
        ] {
            write_file(dir, name, bytes)?;
        }

        let si_name = format!("{segment_name}.si");
        let si_bytes: Vec<u8> = dir.open(&si_name)?.to_vec();
        let mut si = segment_info::parse(&si_bytes, segment_id)?;
        si.files.extend([
            doc_name.clone(),
            tim_name.clone(),
            tip_name.clone(),
            tmd_name.clone(),
        ]);
        let si_bytes = segment_info::write(&si, "");
        write_file(dir, &si_name, &si_bytes)?;

        dir.sync(&[doc_name, tim_name, tip_name, tmd_name, si_name])?;
        Ok(())
    }

    /// The automatic-merge-triggering step [`IndexWriter::commit`] runs when
    /// a [`MergePolicyConfig`] is set (see module doc comment). Repeatedly
    /// asks [`crate::merge_policy::find_merges`] for merge candidates among
    /// this writer's current committed segments and, for each proposed
    /// group, executes it via [`crate::merge::merge_stored_only_segments`]
    /// and folds the result in via [`IndexWriter::apply_merge`], until
    /// `find_merges` proposes nothing further. Terminates because every
    /// executed merge strictly reduces this writer's segment count by at
    /// least one (merging >= 2 segments into exactly 1).
    fn auto_merge(&mut self) -> Result<()> {
        let config = self
            .merge_policy
            .clone()
            .expect("auto_merge only called when merge_policy is Some");

        loop {
            let stats = self.segment_stats()?;
            let groups = merge_policy::find_merges(&stats, &config);
            if groups.is_empty() {
                break;
            }
            for group in groups {
                self.execute_merge(&group)?;
            }
        }
        Ok(())
    }

    /// Builds the [`crate::merge_policy::SegmentStat`] list
    /// [`IndexWriter::auto_merge`] feeds to
    /// [`crate::merge_policy::find_merges`], sourced from this writer's
    /// current committed segments: `doc_count`/on-disk size come from each
    /// segment's own `.si` file (via [`crate::segment_info::parse`] and
    /// [`crate::merge_policy::segment_byte_size`], the byte-accurate path
    /// that module's doc comment describes), `del_count` from this writer's
    /// own [`SegmentCommitInfo`] (already in memory, no re-read needed).
    ///
    /// **Segments with postings files (`.doc`/`.tim`/`.tip`/`.tmd`, written
    /// when [`IndexWriter::set_postings_field`] is configured), term-vector
    /// files (`.tvd`/`.tvx`/`.tvm`, written when
    /// [`IndexWriter::set_term_vector_field`] is configured), or doc-values
    /// files (`.dvd`/`.dvm`/`.dvs`, written when
    /// [`IndexWriter::set_doc_values_field`] is configured) are excluded
    /// entirely** -- [`execute_merge`](IndexWriter::execute_merge) only
    /// merges stored fields via
    /// [`crate::merge::merge_stored_only_segments`], which has no knowledge
    /// of postings, term vectors, or doc values at all. Feeding such a
    /// segment into `find_merges` would let an automatic merge silently drop
    /// that segment's postings/term vectors/doc values (the merged segment's
    /// `.si` would list only stored-fields files, and the source segment's
    /// real `.doc`/`.tim`/`.tip`/`.tmd`/`.tvd`/`.tvx`/`.tvm`/`.dvd`/`.dvm`/
    /// `.dvs` would become orphaned on disk) with no error surfaced --
    /// excluding these segments from consideration keeps them permanently
    /// un-mergeable rather than mergeable-with-silent-data-loss, until
    /// postings/term-vector/doc-values-aware merging exists.
    fn segment_stats(&self) -> Result<Vec<merge_policy::SegmentStat>> {
        let mut stats = Vec::with_capacity(self.segment_infos.segments.len());
        for sci in &self.segment_infos.segments {
            let si_bytes = self.dir.open(&format!("{}.si", sci.segment_name))?.to_vec();
            let si = segment_info::parse(&si_bytes, &sci.segment_id)?;
            if si
                .files
                .iter()
                .any(|f| f.ends_with(".doc") || f.ends_with(".tvd") || f.ends_with(".dvd"))
            {
                continue;
            }
            let size_bytes = merge_policy::segment_byte_size(self.dir, &si);
            stats.push(merge_policy::SegmentStat {
                name: sci.segment_name.clone(),
                doc_count: si.doc_count,
                del_count: sci.del_count,
                size_bytes,
            });
        }
        Ok(stats)
    }

    /// Executes one merge group `names` proposed by
    /// [`crate::merge_policy::find_merges`]: opens each named segment's
    /// stored fields (and live-docs bitset, if it has deletions) straight off
    /// `dir`, merges them via [`crate::merge::merge_stored_only_segments`]
    /// into a brand-new segment, and folds the result into this writer's
    /// committed state via [`IndexWriter::apply_merge`] (which itself writes
    /// the next `segments_N` generation -- each executed merge group is its
    /// own commit, same as a caller manually driving
    /// [`IndexWriter::apply_merge`] would produce).
    fn execute_merge(&mut self, names: &[String]) -> Result<()> {
        struct OpenedSegment {
            sci: SegmentCommitInfo,
            fdt: Vec<u8>,
            fdx: Vec<u8>,
            fdm: Vec<u8>,
            live_docs: Option<FixedBitSet>,
        }

        let mut opened = Vec::with_capacity(names.len());
        for name in names {
            let sci = self
                .segment_infos
                .segments
                .iter()
                .find(|s| &s.segment_name == name)
                .expect("merge_policy::find_merges only proposes segment names this writer currently has committed")
                .clone();

            let fdt = self.dir.open(&format!("{name}.fdt"))?.to_vec();
            let fdx = self.dir.open(&format!("{name}.fdx"))?.to_vec();
            let fdm = self.dir.open(&format!("{name}.fdm"))?.to_vec();

            let live_docs = if sci.del_gen >= 0 {
                let liv = self.dir.open(&deletes::liv_file_name(name, sci.del_gen))?;
                let reader = stored_fields::open(&fdt, &fdx, &fdm, &sci.segment_id, "")?;
                Some(lucene_codecs::live_docs::parse(
                    &liv,
                    &sci.segment_id,
                    sci.del_gen,
                    reader.max_doc() as usize,
                    sci.del_count as usize,
                )?)
            } else {
                None
            };

            opened.push(OpenedSegment {
                sci,
                fdt,
                fdx,
                fdm,
                live_docs,
            });
        }

        let readers: Vec<_> = opened
            .iter()
            .map(|o| stored_fields::open(&o.fdt, &o.fdx, &o.fdm, &o.sci.segment_id, ""))
            .collect::<std::result::Result<Vec<_>, _>>()?;

        let sources: Vec<merge::MergeSource> = opened
            .iter()
            .zip(readers.iter())
            .map(|(o, reader)| {
                merge::MergeSource::stored_only(&self.fields, reader, o.live_docs.as_ref())
            })
            .collect();

        let merged_segment_name = self.next_segment_name();
        let merged_segment_id = generate_segment_id(self.segment_infos.counter);
        let merged_sci = merge::merge_stored_only_segments(
            self.dir,
            &sources,
            &merged_segment_name,
            merged_segment_id,
            &self.codec_name,
            self.lucene_version,
        )?;
        self.segment_infos.counter += 1;

        let source_names: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        self.apply_merge(&source_names, merged_sci)?;
        Ok(())
    }

    /// This writer's most recently committed [`SegmentInfos`] -- does not
    /// reflect any not-yet-`commit()`ed [`IndexWriter::add_document`] calls.
    /// Exposed so a caller can drive [`crate::merge_policy::find_merges`]/
    /// [`crate::merge::merge_stored_only_segments`] manually (see module doc
    /// comment: merging is not automatically triggered by this facade).
    pub fn segment_infos(&self) -> &SegmentInfos {
        &self.segment_infos
    }

    /// Number of documents buffered by [`IndexWriter::add_document`] but not
    /// yet written to disk by a [`IndexWriter::commit`] call.
    pub fn pending_doc_count(&self) -> usize {
        self.pending_docs.len()
    }

    /// Replaces this writer's committed segment list with `merged` in place
    /// of `source_segment_names` -- the composition point for a caller that
    /// has just run [`crate::merge::merge_stored_only_segments`] against
    /// segments from [`IndexWriter::segment_infos`] and wants the result
    /// folded back into this writer's own view of the index (so a later
    /// `add_document`/`commit` builds on top of the merged state instead of
    /// the pre-merge one). Writes the updated segment list as the next
    /// `segments_N` generation, same commit shape as
    /// [`IndexWriter::commit`]/[`IndexWriter::update_document`].
    ///
    /// This does **not** call [`crate::merge_policy::find_merges`] or
    /// [`crate::merge::merge_stored_only_segments`] itself -- see the module
    /// doc comment's "no automatic merge triggering" scope note. It is
    /// purely the bookkeeping half: fold an already-completed merge's result
    /// into this writer's committed state.
    pub fn apply_merge(
        &mut self,
        source_segment_names: &[&str],
        merged: SegmentCommitInfo,
    ) -> Result<&SegmentInfos> {
        let mut new_segment_infos = self.segment_infos.clone();
        new_segment_infos.generation += 1;
        new_segment_infos.version += 1;
        new_segment_infos
            .segments
            .retain(|s| !source_segment_names.contains(&s.segment_name.as_str()));
        new_segment_infos.segments.push(merged);

        segment_infos::write(&new_segment_infos, self.dir)?;
        self.segment_infos = new_segment_infos;
        Ok(&self.segment_infos)
    }

    /// Real `IndexFileNames.segmentFileName`'s `_<counter in base 36>`
    /// convention, driven off this writer's current `segment_infos.counter`
    /// so segment names never collide with an earlier session's, even when
    /// resuming an already-committed directory.
    fn next_segment_name(&self) -> String {
        format!(
            "_{}",
            lucene_util::base36::to_base36(self.segment_infos.counter)
        )
    }
}

/// A brand-new, empty [`SegmentInfos`] for a directory with no existing
/// commit -- generation/version/counter all start at `0`, no segments, a
/// freshly generated commit id (see [`generate_segment_id`]'s doc comment on
/// why this facade doesn't use a real CSPRNG here).
fn empty_segment_infos(lucene_version: LuceneVersion) -> SegmentInfos {
    SegmentInfos {
        id: generate_segment_id(0),
        generation: 0,
        format_version: segment_infos::VERSION_86,
        lucene_version: to_segment_infos_version(lucene_version),
        index_created_version_major: lucene_version.major,
        version: 0,
        counter: 0,
        min_segment_lucene_version: None,
        segments: Vec::new(),
        user_data: Vec::new(),
    }
}

fn to_segment_infos_version(v: LuceneVersion) -> segment_infos::LuceneVersion {
    segment_infos::LuceneVersion {
        major: v.major,
        minor: v.minor,
        bugfix: v.bugfix,
    }
}

/// Generates a 16-byte segment/commit id from `salt` (this writer's current
/// segment-name counter) plus the current time -- **not** a
/// cryptographically random id the way real Lucene's
/// `StringHelper.randomId()` (backed by a `SecureRandom`) is. This
/// workspace has no `rand`-family dependency (see `Cargo.toml`'s
/// `[workspace.dependencies]`), and the only property this port's readers
/// actually rely on (verified: `.si`/`segments_N` parsing checks a
/// referenced id *matches*, never that it looks statistically random) is
/// "distinct segments get distinct ids" -- which salting a hash with a
/// monotonically increasing counter already guarantees deterministically,
/// without pulling in a new dependency for a property this scope doesn't
/// need.
/// Same minimal `create_output`/`write_bytes`/`close` sequence
/// `crate::segment_writer`'s own private `write_file` helper uses -- kept as
/// a separate copy here rather than made `pub(crate)` there, since this is
/// the only other module that currently needs it and the function is a
/// three-line wrapper, not shared logic worth a cross-module dependency for.
fn write_file(dir: &dyn Directory, name: &str, bytes: &[u8]) -> Result<()> {
    let mut out = dir.create_output(name)?;
    out.write_bytes(bytes);
    out.close()?;
    Ok(())
}

/// The `&'static str` kind name [`Error::NonNumericDocValue`] reports for a
/// [`FieldValue`] that isn't [`FieldValue::Int`]/[`FieldValue::Long`].
fn field_value_kind(value: &FieldValue) -> &'static str {
    match value {
        FieldValue::String(_) => "String",
        FieldValue::Binary(_) => "Binary",
        FieldValue::Int(_) => "Int",
        FieldValue::Long(_) => "Long",
        FieldValue::Float(_) => "Float",
        FieldValue::Double(_) => "Double",
    }
}

fn generate_segment_id(salt: i64) -> [u8; ID_LENGTH] {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::{SystemTime, UNIX_EPOCH};

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();

    let mut h1 = DefaultHasher::new();
    (nanos, salt, 1u8).hash(&mut h1);
    let mut h2 = DefaultHasher::new();
    (nanos, salt, 2u8).hash(&mut h2);

    let mut id = [0u8; ID_LENGTH];
    id[0..8].copy_from_slice(&h1.finish().to_le_bytes());
    id[8..16].copy_from_slice(&h2.finish().to_le_bytes());
    id
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_codecs::blocktree;
    use lucene_codecs::field_infos::{
        self as fi, DocValuesSkipIndexType, DocValuesType, IndexOptions, VectorEncoding,
        VectorSimilarityFunction,
    };
    use lucene_codecs::postings::DocInput;
    use lucene_codecs::stored_fields::{self, FieldValue, StoredField};
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

    fn doc_value(d: &Document) -> String {
        match &d.fields[0].value {
            FieldValue::String(s) => s.clone(),
            other => panic!("unexpected field value shape: {other:?}"),
        }
    }

    fn tempdir(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "lucene-rust-index-writer-test-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    /// Reads every document out of every segment `segment_infos` lists, in
    /// commit order -- the minimal "is this readable/searchable" check this
    /// crate can do on its own (the real reader/searcher stack lives one
    /// layer up in `lucene-search`, which already depends on `lucene-index`
    /// and so cannot be depended on back from here -- see the
    /// `architecture` skill).
    fn read_all_docs(dir: &FsDirectory, segment_infos: &SegmentInfos) -> Vec<String> {
        let mut out = Vec::new();
        for sci in &segment_infos.segments {
            let fdt = dir.open(&format!("{}.fdt", sci.segment_name)).unwrap();
            let fdx = dir.open(&format!("{}.fdx", sci.segment_name)).unwrap();
            let fdm = dir.open(&format!("{}.fdm", sci.segment_name)).unwrap();
            let reader = stored_fields::open(&fdt, &fdx, &fdm, &sci.segment_id, "").unwrap();
            let live = if sci.del_gen >= 0 {
                let liv = dir
                    .open(&deletes::liv_file_name(&sci.segment_name, sci.del_gen))
                    .unwrap();
                Some(
                    lucene_codecs::live_docs::parse(
                        &liv,
                        &sci.segment_id,
                        sci.del_gen,
                        reader.max_doc() as usize,
                        sci.del_count as usize,
                    )
                    .unwrap(),
                )
            } else {
                None
            };
            for doc_id in 0..reader.max_doc() {
                let is_live = live
                    .as_ref()
                    .map(|bits| bits.get(doc_id as usize))
                    .unwrap_or(true);
                if is_live {
                    out.push(doc_value(&reader.document(doc_id).unwrap()));
                }
            }
        }
        out
    }

    #[test]
    fn open_on_a_fresh_directory_starts_with_no_segments() {
        let tmp = tempdir("fresh");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        assert!(writer.segment_infos().segments.is_empty());
        assert_eq!(writer.pending_doc_count(), 0);
    }

    #[test]
    fn add_documents_then_commit_produces_one_readable_segment() {
        let tmp = tempdir("add-commit");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        writer.add_document(doc("a"));
        writer.add_document(doc("b"));
        writer.add_document(doc("c"));
        assert_eq!(writer.pending_doc_count(), 3);

        let sis = writer.commit().unwrap().clone();
        assert_eq!(sis.segments.len(), 1);
        assert_eq!(writer.pending_doc_count(), 0);

        // Readable back through the on-disk segments_N this call wrote --
        // not just through the returned struct.
        let reopened = segment_infos::read_latest(&dir).unwrap();
        assert_eq!(reopened.generation, sis.generation);
        assert_eq!(read_all_docs(&dir, &reopened), vec!["a", "b", "c"]);
    }

    #[test]
    fn commit_with_no_pending_documents_is_a_valid_no_op_content_commit() {
        let tmp = tempdir("empty-commit");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        let sis = writer.commit().unwrap().clone();
        assert_eq!(sis.generation, 1);
        assert!(sis.segments.is_empty());

        let reopened = segment_infos::read_latest(&dir).unwrap();
        assert_eq!(reopened.generation, 1);
    }

    #[test]
    fn multiple_commits_produce_multiple_independent_segments() {
        let tmp = tempdir("multi-commit");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        writer.add_document(doc("a"));
        writer.commit().unwrap();

        writer.add_document(doc("b"));
        writer.add_document(doc("c"));
        writer.commit().unwrap();

        let sis = writer.segment_infos().clone();
        assert_eq!(sis.segments.len(), 2);
        assert_ne!(sis.segments[0].segment_name, sis.segments[1].segment_name);

        let reopened = segment_infos::read_latest(&dir).unwrap();
        assert_eq!(reopened.segments.len(), 2);
        assert_eq!(read_all_docs(&dir, &reopened), vec!["a", "b", "c"]);
    }

    #[test]
    fn reopening_an_existing_directory_resumes_its_committed_state() {
        let tmp = tempdir("resume");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];

        {
            let mut writer =
                IndexWriter::open(&dir, fields.clone(), "Lucene104", version()).unwrap();
            writer.add_document(doc("a"));
            writer.commit().unwrap();
        }

        let mut writer2 = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        assert_eq!(writer2.segment_infos().segments.len(), 1);

        writer2.add_document(doc("b"));
        writer2.commit().unwrap();

        let sis = writer2.segment_infos();
        assert_eq!(sis.segments.len(), 2);
        // The second session's segment name must not collide with the
        // first's.
        assert_ne!(sis.segments[0].segment_name, sis.segments[1].segment_name);
    }

    // --- update_document/delete_documents: needs a real postings fixture,
    // same one term_delete.rs/update_document.rs's own tests already use. ---

    struct Fixture {
        fields: blocktree::BlockTreeFields,
        doc_bytes: Vec<u8>,
        segment_id: [u8; ID_LENGTH],
        suffix: String,
        max_doc: usize,
    }

    impl Fixture {
        fn doc_in(&self) -> DocInput<'_> {
            DocInput::open(&self.doc_bytes, &self.segment_id, &self.suffix).expect("open .doc")
        }
    }

    fn open_fixture() -> Fixture {
        let dir = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/blocktree_index/"
        );
        let manifest = std::fs::read_to_string(format!("{dir}manifest.properties"))
            .expect("run fixtures generator first (GenBlockTree)");
        let get = |key: &str| -> String {
            manifest
                .lines()
                .find_map(|l| l.strip_prefix(&format!("{key}=")))
                .unwrap_or_else(|| panic!("manifest key {key} missing"))
                .to_string()
        };
        let id_hex = get("id_hex");
        let mut segment_id = [0u8; ID_LENGTH];
        for (i, slot) in segment_id.iter_mut().enumerate() {
            *slot = u8::from_str_radix(&id_hex[i * 2..i * 2 + 2], 16).unwrap();
        }
        let suffix = get("segment_suffix");
        let max_doc: i32 = get("max_doc").parse().unwrap();

        let read_raw = |name: &str| -> Vec<u8> {
            std::fs::read(format!("{dir}{name}.raw")).unwrap_or_else(|_| panic!("missing {name}"))
        };
        let fnm = read_raw(&get("fnm_file_name"));
        let field_infos = fi::parse(&fnm, &segment_id, "").expect("parse .fnm");
        let tim = read_raw(&get("tim_file_name"));
        let tip = read_raw(&get("tip_file_name"));
        let tmd = read_raw(&get("tmd_file_name"));
        let fields = blocktree::open(
            &tim,
            &tip,
            &tmd,
            &field_infos,
            &segment_id,
            &suffix,
            max_doc,
        )
        .expect("open blocktree");
        let doc_bytes = read_raw(&get("doc_file_name"));

        Fixture {
            fields,
            doc_bytes,
            segment_id,
            suffix,
            max_doc: max_doc as usize,
        }
    }

    /// Seeds a writer's committed state with the real-Lucene fixture segment
    /// as segment `_0`, without going through `add_document`/`commit`
    /// (the fixture already has real postings; this facade's own
    /// `flush_stored_only_segment` path never writes any).
    fn writer_seeded_with_fixture<'d>(
        dir: &'d FsDirectory,
        fx: &Fixture,
        fields: Vec<FieldInfo>,
    ) -> IndexWriter<'d> {
        let mut writer = IndexWriter::open(dir, fields, "Lucene104", version()).unwrap();
        writer.segment_infos.segments.push(SegmentCommitInfo {
            segment_name: "_0".to_string(),
            segment_id: fx.segment_id,
            codec_name: "Lucene104".to_string(),
            del_gen: -1,
            del_count: 0,
            field_infos_gen: -1,
            doc_values_gen: -1,
            soft_del_count: 0,
            sci_id: None,
            field_infos_files: vec![],
            dv_update_files: vec![],
        });
        writer.segment_infos.counter = 1;
        writer
    }

    #[test]
    fn update_document_replaces_a_matched_doc_and_is_visible_after_commit() {
        let fx = open_fixture();
        let tmp = tempdir("update-doc");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = writer_seeded_with_fixture(&dir, &fx, fields);

        let sources = [SegmentDeleteSource {
            segment_name: "_0",
            fields: &fx.fields,
            doc_in: None, // "id"->"id0" is a singleton term, no .doc needed
            live_docs: None,
            max_doc: fx.max_doc,
        }];

        let sis = writer
            .update_document(&sources, "id", b"id0", doc("replacement"))
            .unwrap()
            .clone();
        assert_eq!(sis.segments.len(), 2);
        assert_eq!(sis.segments[0].del_count, 1);

        let reopened = segment_infos::read_latest(&dir).unwrap();
        // The old segment's real postings can't be read back through this
        // crate's stored-fields-only reader helper, so just confirm the new
        // segment (this writer's own flush) is visible and correct.
        let new_sci = reopened
            .segments
            .iter()
            .find(|s| s.segment_name != "_0")
            .unwrap();
        let fdt = dir.open(&format!("{}.fdt", new_sci.segment_name)).unwrap();
        let fdx = dir.open(&format!("{}.fdx", new_sci.segment_name)).unwrap();
        let fdm = dir.open(&format!("{}.fdm", new_sci.segment_name)).unwrap();
        let reader = stored_fields::open(&fdt, &fdx, &fdm, &new_sci.segment_id, "").unwrap();
        assert_eq!(doc_value(&reader.document(0).unwrap()), "replacement");
    }

    #[test]
    fn delete_documents_marks_matching_docs_dead_and_is_visible_after_commit() {
        let fx = open_fixture();
        let doc_in = fx.doc_in();
        let tmp = tempdir("delete-doc");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = writer_seeded_with_fixture(&dir, &fx, fields);

        let sources = [SegmentDeleteSource {
            segment_name: "_0",
            fields: &fx.fields,
            doc_in: Some(&doc_in),
            live_docs: None,
            max_doc: fx.max_doc,
        }];

        // "body" -> "cat" matches docs [0, 2] per the checked-in fixture
        // (same contents `term_delete.rs`'s own tests document).
        let sis = writer
            .delete_documents(&sources, "body", b"cat")
            .unwrap()
            .clone();
        assert_eq!(sis.segments.len(), 1);
        assert_eq!(sis.segments[0].del_count, 2);
        assert_eq!(sis.segments[0].del_gen, 1);

        let reopened = segment_infos::read_latest(&dir).unwrap();
        assert_eq!(reopened.segments[0].del_count, 2);

        let liv = dir.open("_0_1.liv").unwrap();
        let parsed =
            lucene_codecs::live_docs::parse(&liv, &fx.segment_id, 1, fx.max_doc, 2).unwrap();
        assert!(!parsed.get(0));
        assert!(parsed.get(1));
        assert!(!parsed.get(2));
    }

    #[test]
    fn delete_documents_with_no_matching_source_leaves_segment_untouched() {
        let fx = open_fixture();
        let tmp = tempdir("delete-doc-no-match");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = writer_seeded_with_fixture(&dir, &fx, fields);

        // No source targets "_0" at all, so the segment must pass through
        // unmodified (no .liv written, del_count stays 0) even though
        // segment_infos itself still advances a generation/version.
        let sources: [SegmentDeleteSource; 0] = [];
        let sis = writer
            .delete_documents(&sources, "body", b"cat")
            .unwrap()
            .clone();
        assert_eq!(sis.segments.len(), 1);
        assert_eq!(sis.segments[0].segment_name, "_0");
        assert_eq!(sis.segments[0].del_count, 0);
        assert_eq!(sis.segments[0].del_gen, -1);
    }

    #[test]
    fn a_failing_update_document_leaves_the_writer_state_unchanged() {
        let fx = open_fixture();
        let doc_in = fx.doc_in();
        let tmp = tempdir("update-fail");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = writer_seeded_with_fixture(&dir, &fx, fields);
        let before = writer.segment_infos().clone();

        let sources = [SegmentDeleteSource {
            segment_name: "_0",
            fields: &fx.fields,
            doc_in: Some(&doc_in),
            live_docs: None,
            max_doc: 1, // bogus: doc id 2 (from "cat") is out of range
        }];

        let result = writer.update_document(&sources, "body", b"cat", doc("nope"));
        assert!(result.is_err());
        assert_eq!(writer.segment_infos(), &before);
        assert!(!tmp.join("segments_1").exists());
    }

    /// A tight [`MergePolicyConfig`] whose threshold (`segments_per_tier`)
    /// this test suite deliberately crosses/stays-under, so `commit()`'s
    /// automatic-merge behavior is exercised deterministically rather than
    /// relying on the (much larger) real-Lucene-shaped defaults.
    fn tight_merge_policy() -> MergePolicyConfig {
        MergePolicyConfig {
            max_merge_at_once: 10,
            segments_per_tier: 2,
            max_merged_segment_size: 1_000_000,
            reclaim_weight: 1.0,
        }
    }

    #[test]
    fn commit_with_no_merge_policy_set_never_auto_merges() {
        let tmp = tempdir("no-merge-policy");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        for doc_id in 0..5 {
            writer.add_document(doc(&doc_id.to_string()));
            writer.commit().unwrap();
        }

        // 5 commits, no merge policy set => still 5 independent segments,
        // exactly as before automatic merge triggering existed.
        assert_eq!(writer.segment_infos().segments.len(), 5);
    }

    #[test]
    fn commit_below_merge_threshold_stays_unmerged() {
        let tmp = tempdir("below-threshold");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_merge_policy(Some(tight_merge_policy()));

        // tight_merge_policy's segments_per_tier is 2, so 2 commits (2
        // segments) must stay below/at threshold and remain unmerged.
        writer.add_document(doc("a"));
        writer.commit().unwrap();
        writer.add_document(doc("b"));
        writer.commit().unwrap();

        assert_eq!(writer.segment_infos().segments.len(), 2);
        let reopened = segment_infos::read_latest(&dir).unwrap();
        assert_eq!(reopened.segments.len(), 2);
        assert_eq!(read_all_docs(&dir, &reopened), vec!["a", "b"]);
    }

    #[test]
    fn commit_above_merge_threshold_automatically_merges_and_stays_readable() {
        let tmp = tempdir("above-threshold");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_merge_policy(Some(tight_merge_policy()));

        // tight_merge_policy's segments_per_tier is 2; crossing it (5
        // one-document commits) must trigger at least one automatic merge,
        // ending with fewer segments than commits.
        let ids = ["a", "b", "c", "d", "e"];
        for id in ids {
            writer.add_document(doc(id));
            writer.commit().unwrap();
        }

        let final_count = writer.segment_infos().segments.len();
        assert!(
            final_count < ids.len(),
            "expected automatic merging to reduce segment count below {}, got {final_count}",
            ids.len()
        );

        let reopened = segment_infos::read_latest(&dir).unwrap();
        assert_eq!(reopened.segments.len(), final_count);
        let mut docs = read_all_docs(&dir, &reopened);
        docs.sort();
        assert_eq!(docs, vec!["a", "b", "c", "d", "e"]);
    }

    #[test]
    fn repeated_commits_with_auto_merge_converge_without_panicking_or_looping_forever() {
        let tmp = tempdir("converge");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_merge_policy(Some(tight_merge_policy()));

        // Many small single-document commits in a row: each commit() call
        // must return (no infinite auto-merge loop), and the segment count
        // must never run away unboundedly.
        for i in 0..20 {
            writer.add_document(doc(&i.to_string()));
            writer.commit().unwrap();
            assert!(
                writer.segment_infos().segments.len() <= 20,
                "segment count should never exceed the number of commits made so far"
            );
        }

        let reopened = segment_infos::read_latest(&dir).unwrap();
        let mut docs = read_all_docs(&dir, &reopened);
        docs.sort();
        let mut expected: Vec<String> = (0..20).map(|i| i.to_string()).collect();
        expected.sort();
        assert_eq!(docs, expected);
    }

    #[test]
    fn auto_merge_correctly_carries_forward_a_segments_existing_deletions() {
        let tmp = tempdir("auto-merge-with-deletions");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields.clone(), "Lucene104", version()).unwrap();

        // Two ordinary commits (no merge policy yet), so segment "_0" has a
        // real, on-disk, flushed segment to apply a deletion to directly via
        // `deletes::apply_deletes` (the same primitive `delete_documents`
        // itself calls), independent of term resolution.
        writer.add_document(doc("a"));
        writer.add_document(doc("b"));
        writer.commit().unwrap();

        let sci = writer.segment_infos().segments[0].clone();
        assert_eq!(sci.segment_name, "_0");
        let fdt = dir.open("_0.fdt").unwrap();
        let fdx = dir.open("_0.fdx").unwrap();
        let fdm = dir.open("_0.fdm").unwrap();
        let reader = stored_fields::open(&fdt, &fdx, &fdm, &sci.segment_id, "").unwrap();
        let max_doc = reader.max_doc() as usize;

        // Delete doc 0 ("a") directly, bypassing term resolution entirely --
        // this is exactly what `execute_merge`'s `sci.del_gen >= 0` branch
        // must read back correctly during an automatic merge.
        let updated_sci = deletes::apply_deletes(&dir, &sci, None, max_doc, [0]).unwrap();
        assert_eq!(updated_sci.del_gen, 1);
        assert_eq!(updated_sci.del_count, 1);

        let mut new_segment_infos = writer.segment_infos().clone();
        new_segment_infos.segments[0] = updated_sci;
        new_segment_infos.generation += 1;
        new_segment_infos.version += 1;
        segment_infos::write(&new_segment_infos, &dir).unwrap();

        // Reopen the writer against this on-disk state (one segment with a
        // real deletion already applied), enable the merge policy, and cross
        // its threshold so the deleted segment gets folded into an automatic
        // merge.
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_merge_policy(Some(tight_merge_policy()));
        writer.add_document(doc("c"));
        writer.commit().unwrap();
        writer.add_document(doc("d"));
        writer.commit().unwrap();

        let final_count = writer.segment_infos().segments.len();
        assert!(
            final_count < 3,
            "expected the deleted-doc segment to merge away, got {final_count} segments"
        );

        let reopened = segment_infos::read_latest(&dir).unwrap();
        let mut docs = read_all_docs(&dir, &reopened);
        docs.sort();
        // "a" was deleted before the merge, so only "b", "c", "d" survive.
        assert_eq!(docs, vec!["b", "c", "d"]);
    }

    #[test]
    fn apply_merge_folds_a_merge_result_into_the_writers_committed_state() {
        let tmp = tempdir("apply-merge");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields.clone(), "Lucene104", version()).unwrap();

        writer.add_document(doc("a"));
        writer.commit().unwrap();
        writer.add_document(doc("b"));
        writer.commit().unwrap();
        assert_eq!(writer.segment_infos().segments.len(), 2);

        let seg0_name = writer.segment_infos().segments[0].segment_name.clone();
        let seg1_name = writer.segment_infos().segments[1].segment_name.clone();

        let fdt0 = dir.open(&format!("{seg0_name}.fdt")).unwrap().to_vec();
        let fdx0 = dir.open(&format!("{seg0_name}.fdx")).unwrap().to_vec();
        let fdm0 = dir.open(&format!("{seg0_name}.fdm")).unwrap().to_vec();
        let fdt1 = dir.open(&format!("{seg1_name}.fdt")).unwrap().to_vec();
        let fdx1 = dir.open(&format!("{seg1_name}.fdx")).unwrap().to_vec();
        let fdm1 = dir.open(&format!("{seg1_name}.fdm")).unwrap().to_vec();

        // Segment ids are generated internally, so re-derive them from the
        // committed SegmentInfos rather than hard-coding a value.
        let seg0_id = writer.segment_infos().segments[0].segment_id;
        let seg1_id = writer.segment_infos().segments[1].segment_id;
        let reader0 = stored_fields::open(&fdt0, &fdx0, &fdm0, &seg0_id, "").unwrap();
        let reader1 = stored_fields::open(&fdt1, &fdx1, &fdm1, &seg1_id, "").unwrap();

        let sources = vec![
            merge::MergeSource::stored_only(&fields, &reader0, None),
            merge::MergeSource::stored_only(&fields, &reader1, None),
        ];
        let merged_sci = merge::merge_stored_only_segments(
            &dir,
            &sources,
            "_merged",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();

        let sis = writer
            .apply_merge(&[seg0_name.as_str(), seg1_name.as_str()], merged_sci)
            .unwrap()
            .clone();
        assert_eq!(sis.segments.len(), 1);
        assert_eq!(sis.segments[0].segment_name, "_merged");

        let reopened = segment_infos::read_latest(&dir).unwrap();
        assert_eq!(reopened.segments.len(), 1);
        assert_eq!(read_all_docs(&dir, &reopened), vec!["a", "b"]);
    }

    // --- set_postings_field / commit()'s postings-writing path ---

    fn body_field(number: i32) -> FieldInfo {
        FieldInfo {
            index_options: IndexOptions::DocsAndFreqs,
            ..stored_only_field("body", number)
        }
    }

    fn doc_with_body(id: &str, body: &str) -> Document {
        Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String(id.to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::String(body.to_string()),
                },
            ],
        }
    }

    #[test]
    fn set_postings_field_rejects_an_unknown_field_name() {
        let tmp = tempdir("unknown-postings-field");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        let err = writer.set_postings_field(Some("nonexistent")).unwrap_err();
        assert!(matches!(err, Error::UnknownPostingsField(name) if name == "nonexistent"));
    }

    #[test]
    fn set_postings_field_rejects_a_field_with_no_index_options() {
        let tmp = tempdir("unindexed-postings-field");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        let err = writer.set_postings_field(Some("id")).unwrap_err();
        assert!(matches!(
            err,
            Error::UnsupportedPostingsIndexOptions(name, IndexOptions::None) if name == "id"
        ));
    }

    #[test]
    fn commit_with_postings_field_writes_readable_postings_for_multiple_docs_and_terms() {
        let tmp = tempdir("postings-commit");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("body")).unwrap();

        writer.add_document(doc_with_body("a", "the quick fox"));
        writer.add_document(doc_with_body("b", "the lazy fox"));
        writer.add_document(doc_with_body("c", "the fox runs"));
        let sis = writer.commit().unwrap().clone();
        assert_eq!(sis.segments.len(), 1);
        let sci = &sis.segments[0];

        // Stored fields are still intact (backward-compatible).
        assert_eq!(read_all_docs(&dir, &sis), vec!["a", "b", "c"]);

        // The postings files exist and are listed in `.si`.
        let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
        let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
        for ext in ["doc", "tim", "tip", "tmd"] {
            let name = format!("{}.{ext}", sci.segment_name);
            assert!(si.files.contains(&name), "missing {name} in .si files");
            assert!(
                dir.list_all().unwrap().contains(&name),
                "missing {name} on disk"
            );
        }

        // Readable via the existing, unmodified read side: `fox` occurs in
        // all 3 docs, `quick`/`lazy`/`runs` are singletons, `the` occurs in
        // all 3 too but is not a singleton either.
        let tim = dir.open(&format!("{}.tim", sci.segment_name)).unwrap();
        let tip = dir.open(&format!("{}.tip", sci.segment_name)).unwrap();
        let tmd = dir.open(&format!("{}.tmd", sci.segment_name)).unwrap();
        let doc_bytes = dir.open(&format!("{}.doc", sci.segment_name)).unwrap();
        let field_infos = fi::FieldInfos {
            fields: vec![
                fi::FieldInfo {
                    index_options: IndexOptions::None,
                    ..stored_only_field("id", 0)
                },
                body_field(1),
            ],
        };
        let block_fields = blocktree::open(&tim, &tip, &tmd, &field_infos, &sci.segment_id, "", 3)
            .expect("blocktree::open on IndexWriter-produced .tim/.tip/.tmd");
        let doc_in = DocInput::open(&doc_bytes, &sci.segment_id, "").expect("open .doc");
        let field = block_fields.field("body").unwrap();

        let postings = field.postings(b"fox", Some(&doc_in)).unwrap().unwrap();
        assert_eq!(postings.docs, vec![0, 1, 2]);
        let postings = field.postings(b"the", Some(&doc_in)).unwrap().unwrap();
        assert_eq!(postings.docs, vec![0, 1, 2]);
        let postings = field.postings(b"quick", Some(&doc_in)).unwrap().unwrap();
        assert_eq!(postings.docs, vec![0]);
        let postings = field.postings(b"lazy", Some(&doc_in)).unwrap().unwrap();
        assert_eq!(postings.docs, vec![1]);
        let postings = field.postings(b"runs", Some(&doc_in)).unwrap().unwrap();
        assert_eq!(postings.docs, vec![2]);
        assert!(field.seek_exact(b"missing").is_none());
    }

    #[test]
    fn commit_with_no_postings_field_configured_stays_stored_only() {
        // Backward compatibility: a writer that never calls
        // `set_postings_field` must produce exactly the same on-disk shape
        // as before this feature existed -- no `.doc`/`.tim`/`.tip`/`.tmd`
        // files at all.
        let tmp = tempdir("no-postings-field");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        writer.add_document(doc_with_body("a", "the quick fox"));
        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let files = dir.list_all().unwrap();
        for ext in ["doc", "tim", "tip", "tmd"] {
            assert!(!files.contains(&format!("{}.{ext}", sci.segment_name)));
        }
    }

    /// The documented `docFreq >= BLOCK_SIZE (256)` boundary: this writer
    /// has no multi-block `.tim` support, so a term occurring in 256+ pending
    /// docs must reject the *whole* `commit()` call atomically, leaving
    /// `dir`/`pending_docs`/`segment_infos` completely unchanged -- never a
    /// partially-written segment.
    #[test]
    fn commit_rejects_and_leaves_state_unchanged_when_a_term_reaches_doc_freq_256() {
        let tmp = tempdir("postings-docfreq-too-large");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("body")).unwrap();

        for i in 0..256 {
            writer.add_document(doc_with_body(&i.to_string(), "shared"));
        }
        let before = writer.segment_infos().clone();
        let before_pending = writer.pending_doc_count();

        let err = writer.commit().unwrap_err();
        assert!(matches!(
            err,
            Error::PostingsWriter(postings_writer::Error::DocFreqTooLarge {
                index: 0,
                doc_freq: 256
            })
        ));

        // Nothing committed: state and pending buffer both unchanged, and no
        // segments_1 was ever written.
        assert_eq!(writer.segment_infos(), &before);
        assert_eq!(writer.pending_doc_count(), before_pending);
        assert!(!tmp.join("segments_1").exists());
    }

    /// A term under the 256 boundary must still commit successfully -- the
    /// boundary is `>=`, not `>`. Capped at 100 docs (well under 256) rather
    /// than the tightest possible "255" case, because
    /// `flush_stored_only_segment`'s own `write_best_speed` has a separate,
    /// pre-existing, unrelated cap of `< 128` docs per flush (its bulk
    /// per-doc-array encoding only implements the scalar-tail path, not the
    /// 128-value transposed-block path -- see that assert's own message);
    /// this test only needs to prove the postings-side boundary isn't
    /// off-by-one in the "too eager" direction, which 100 already does.
    #[test]
    fn commit_succeeds_below_the_doc_freq_boundary() {
        let tmp = tempdir("postings-docfreq-just-under");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("body")).unwrap();

        for i in 0..100 {
            writer.add_document(doc_with_body(&i.to_string(), "shared"));
        }
        let sis = writer.commit().unwrap().clone();
        assert_eq!(sis.segments.len(), 1);
    }

    #[test]
    fn commit_with_postings_field_but_no_pending_docs_writes_no_postings_files() {
        let tmp = tempdir("postings-empty-commit");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("body")).unwrap();

        let sis = writer.commit().unwrap().clone();
        assert!(sis.segments.is_empty());
    }

    #[test]
    fn commit_with_postings_field_but_no_doc_has_that_fields_text_skips_postings() {
        // A document that omits the opted-in postings field entirely (no
        // `StoredField` for its `field_number`) contributes no postings --
        // this must not be an error, just "nothing to index this commit".
        let tmp = tempdir("postings-no-text");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("body")).unwrap();

        writer.add_document(doc("a")); // only field_number 0 ("id"), no "body"
        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let files = dir.list_all().unwrap();
        assert!(!files.contains(&format!("{}.tim", sci.segment_name)));
    }

    #[test]
    fn commit_with_postings_field_holding_a_non_string_value_skips_that_doc() {
        // A doc whose stored value for the opted-in postings field isn't a
        // `FieldValue::String` (e.g. `Int`) contributes no indexable text --
        // matches `set_postings_field`'s own doc comment.
        let tmp = tempdir("postings-non-string-value");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("body")).unwrap();

        writer.add_document(Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("a".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::Int(42), // not a String -- must be skipped
                },
            ],
        });
        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let files = dir.list_all().unwrap();
        assert!(!files.contains(&format!("{}.tim", sci.segment_name)));
    }

    #[test]
    fn commit_with_postings_field_text_that_tokenizes_to_nothing_skips_postings() {
        // The opted-in field has a `String` value on every doc, but that
        // text tokenizes to zero terms (e.g. only whitespace) -- still not
        // an error, just nothing to index this commit, distinct from the
        // "field missing/non-String" case above.
        let tmp = tempdir("postings-empty-tokenization");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("body")).unwrap();

        writer.add_document(doc_with_body("a", "   "));
        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let files = dir.list_all().unwrap();
        assert!(!files.contains(&format!("{}.tim", sci.segment_name)));
    }

    #[test]
    fn setting_postings_field_back_to_none_restores_stored_only_behavior() {
        let tmp = tempdir("postings-field-reset");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("body")).unwrap();
        writer.set_postings_field(None).unwrap();

        writer.add_document(doc_with_body("a", "the quick fox"));
        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let files = dir.list_all().unwrap();
        assert!(!files.contains(&format!("{}.tim", sci.segment_name)));
    }

    #[test]
    fn segments_with_postings_are_never_automatically_merged_away() {
        // Enabling both set_postings_field and set_merge_policy at once must
        // not let automatic merging silently drop a segment's postings --
        // execute_merge only knows how to merge stored fields
        // (merge_stored_only_segments has no .doc/.tim/.tip/.tmd awareness at
        // all), so segment_stats() excludes any segment carrying postings
        // files from find_merges' candidate pool entirely, keeping it
        // un-mergeable rather than mergeable-with-silent-data-loss.
        let tmp = tempdir("postings-and-merge-policy");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("body")).unwrap();
        writer.set_merge_policy(Some(tight_merge_policy()));

        // tight_merge_policy's segments_per_tier is 2 -- three one-doc
        // commits, each producing a segment with real postings, must cross
        // that threshold and would normally trigger a merge.
        for id in ["a", "b", "c"] {
            writer.add_document(doc_with_body(id, "shared text"));
            writer.commit().unwrap();
        }

        let final_count = writer.segment_infos().segments.len();
        assert_eq!(
            final_count, 3,
            "segments carrying postings must never be automatically merged"
        );

        // Every segment's real postings files must still be present and
        // correctly listed in its own .si -- nothing was silently dropped.
        for sci in &writer.segment_infos().segments.clone() {
            let files = dir.list_all().unwrap();
            assert!(files.contains(&format!("{}.tim", sci.segment_name)));
            let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
            let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
            assert!(si.files.iter().any(|f| f.ends_with(".tim")));
        }
    }

    // --- set_term_vector_field / commit()'s term-vector-writing path ---

    fn tv_body_field(number: i32) -> FieldInfo {
        FieldInfo {
            index_options: IndexOptions::DocsAndFreqs,
            store_term_vectors: true,
            ..stored_only_field("body", number)
        }
    }

    #[test]
    fn set_term_vector_field_rejects_an_unknown_field_name() {
        let tmp = tempdir("unknown-tv-field");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        let err = writer
            .set_term_vector_field(Some("nonexistent"))
            .unwrap_err();
        assert!(matches!(err, Error::UnknownTermVectorField(name) if name == "nonexistent"));
    }

    #[test]
    fn set_term_vector_field_rejects_a_field_without_store_term_vectors() {
        let tmp = tempdir("unflagged-tv-field");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), body_field(1)]; // body_field: store_term_vectors == false
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        let err = writer.set_term_vector_field(Some("body")).unwrap_err();
        assert!(matches!(err, Error::UnsupportedTermVectorField(name) if name == "body"));
    }

    #[test]
    fn commit_with_term_vector_field_writes_readable_term_vectors_for_multiple_docs_and_terms() {
        let tmp = tempdir("tv-commit");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), tv_body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_term_vector_field(Some("body")).unwrap();

        writer.add_document(doc_with_body("a", "the quick fox"));
        writer.add_document(doc_with_body("b", "the lazy fox"));
        writer.add_document(doc_with_body("c", "the fox runs"));
        let sis = writer.commit().unwrap().clone();
        assert_eq!(sis.segments.len(), 1);
        let sci = &sis.segments[0];

        // Stored fields are still intact (backward-compatible).
        assert_eq!(read_all_docs(&dir, &sis), vec!["a", "b", "c"]);

        // The term-vector files exist and are listed in `.si`.
        let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
        let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
        for ext in ["tvd", "tvx", "tvm"] {
            let name = format!("{}.{ext}", sci.segment_name);
            assert!(si.files.contains(&name), "missing {name} in .si files");
            assert!(
                dir.list_all().unwrap().contains(&name),
                "missing {name} on disk"
            );
        }

        // Readable via the existing, unmodified read side.
        let tvd = dir.open(&format!("{}.tvd", sci.segment_name)).unwrap();
        let tvx = dir.open(&format!("{}.tvx", sci.segment_name)).unwrap();
        let tvm = dir.open(&format!("{}.tvm", sci.segment_name)).unwrap();
        let reader = lucene_codecs::term_vectors::open(&tvd, &tvx, &tvm, &sci.segment_id, "")
            .expect("term_vectors::open on IndexWriter-produced .tvd/.tvx/.tvm");
        assert_eq!(reader.max_doc(), 3);

        let doc0 = reader.document(0).unwrap().unwrap();
        assert_eq!(doc0.fields.len(), 1);
        assert_eq!(doc0.fields[0].field_number, 1);
        assert!(doc0.fields[0].has_positions);
        let mut terms0: Vec<String> = doc0.fields[0]
            .terms
            .iter()
            .map(|t| String::from_utf8(t.term.clone()).unwrap())
            .collect();
        terms0.sort();
        assert_eq!(terms0, vec!["fox", "quick", "the"]);

        let doc2 = reader.document(2).unwrap().unwrap();
        let mut terms2: Vec<String> = doc2.fields[0]
            .terms
            .iter()
            .map(|t| String::from_utf8(t.term.clone()).unwrap())
            .collect();
        terms2.sort();
        assert_eq!(terms2, vec!["fox", "runs", "the"]);
    }

    #[test]
    fn commit_with_no_term_vector_field_configured_stays_stored_only() {
        // Backward compatibility: a writer that never calls
        // `set_term_vector_field` must produce exactly the same on-disk
        // shape as before this feature existed -- no `.tvd`/`.tvx`/`.tvm`
        // files at all.
        let tmp = tempdir("no-tv-field");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), tv_body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        writer.add_document(doc_with_body("a", "the quick fox"));
        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let files = dir.list_all().unwrap();
        for ext in ["tvd", "tvx", "tvm"] {
            assert!(!files.contains(&format!("{}.{ext}", sci.segment_name)));
        }
    }

    #[test]
    fn commit_with_term_vector_field_but_no_pending_docs_writes_no_term_vector_files() {
        let tmp = tempdir("tv-empty-commit");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), tv_body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_term_vector_field(Some("body")).unwrap();

        let sis = writer.commit().unwrap().clone();
        assert!(sis.segments.is_empty());
    }

    #[test]
    fn commit_with_term_vector_field_but_no_doc_has_that_fields_text_skips_term_vectors() {
        let tmp = tempdir("tv-no-text");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), tv_body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_term_vector_field(Some("body")).unwrap();

        writer.add_document(doc("a")); // only field_number 0 ("id"), no "body"
        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let files = dir.list_all().unwrap();
        assert!(!files.contains(&format!("{}.tvd", sci.segment_name)));
    }

    #[test]
    fn commit_with_term_vector_field_holding_a_non_string_value_skips_that_doc() {
        let tmp = tempdir("tv-non-string-value");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), tv_body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_term_vector_field(Some("body")).unwrap();

        writer.add_document(Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("a".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::Int(42), // not a String -- must be skipped
                },
            ],
        });
        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let files = dir.list_all().unwrap();
        assert!(!files.contains(&format!("{}.tvd", sci.segment_name)));
    }

    #[test]
    fn commit_with_term_vector_field_text_that_tokenizes_to_nothing_skips_term_vectors() {
        let tmp = tempdir("tv-empty-tokenization");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), tv_body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_term_vector_field(Some("body")).unwrap();

        writer.add_document(doc_with_body("a", "   "));
        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let files = dir.list_all().unwrap();
        assert!(!files.contains(&format!("{}.tvd", sci.segment_name)));
    }

    #[test]
    fn commit_with_term_vector_field_where_only_some_docs_have_text_still_writes_an_entry_per_doc()
    {
        // A doc with no indexable text for the opted-in field still needs a
        // `TermVectorsDocument` entry (empty `fields`) so doc IDs stay
        // aligned with the segment's real doc count --
        // `write_best_speed` derives `max_doc` from `docs.len()`.
        let tmp = tempdir("tv-partial-docs");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), tv_body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_term_vector_field(Some("body")).unwrap();

        writer.add_document(doc_with_body("a", "alpha"));
        writer.add_document(doc("b")); // no "body" field at all
        writer.add_document(doc_with_body("c", "gamma"));
        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let tvd = dir.open(&format!("{}.tvd", sci.segment_name)).unwrap();
        let tvx = dir.open(&format!("{}.tvx", sci.segment_name)).unwrap();
        let tvm = dir.open(&format!("{}.tvm", sci.segment_name)).unwrap();
        let reader =
            lucene_codecs::term_vectors::open(&tvd, &tvx, &tvm, &sci.segment_id, "").unwrap();
        assert_eq!(reader.max_doc(), 3);
        assert_eq!(reader.document(0).unwrap().unwrap().fields.len(), 1);
        // A doc contributing zero fields to this chunk decodes as `None`,
        // not `Some(fields: vec![])` -- see `TermVectorsReader::document`'s
        // own `doc_num_fields == 0 => Ok(None)` branch.
        assert!(reader.document(1).unwrap().is_none());
        assert_eq!(reader.document(2).unwrap().unwrap().fields.len(), 1);
    }

    #[test]
    fn setting_term_vector_field_back_to_none_restores_stored_only_behavior() {
        let tmp = tempdir("tv-field-reset");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), tv_body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_term_vector_field(Some("body")).unwrap();
        writer.set_term_vector_field(None).unwrap();

        writer.add_document(doc_with_body("a", "the quick fox"));
        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let files = dir.list_all().unwrap();
        assert!(!files.contains(&format!("{}.tvd", sci.segment_name)));
    }

    #[test]
    fn segments_with_term_vectors_are_never_automatically_merged_away() {
        // Same class of bug as `segments_with_postings_are_never_...`, for
        // term vectors instead of postings: enabling both
        // `set_term_vector_field` and `set_merge_policy` at once must not
        // let automatic merging silently drop a segment's term vectors --
        // `execute_merge` only knows how to merge stored fields, so
        // `segment_stats()` excludes any segment carrying `.tvd` from
        // `find_merges`' candidate pool entirely.
        let tmp = tempdir("tv-and-merge-policy");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), tv_body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_term_vector_field(Some("body")).unwrap();
        writer.set_merge_policy(Some(tight_merge_policy()));

        // tight_merge_policy's segments_per_tier is 2 -- three one-doc
        // commits, each producing a segment with real term vectors, must
        // cross that threshold and would normally trigger a merge.
        for id in ["a", "b", "c"] {
            writer.add_document(doc_with_body(id, "shared text"));
            writer.commit().unwrap();
        }

        let final_count = writer.segment_infos().segments.len();
        assert_eq!(
            final_count, 3,
            "segments carrying term vectors must never be automatically merged"
        );

        // Every segment's real term-vector files must still be present and
        // correctly listed in its own .si -- nothing was silently dropped.
        for sci in &writer.segment_infos().segments.clone() {
            let files = dir.list_all().unwrap();
            assert!(files.contains(&format!("{}.tvd", sci.segment_name)));
            let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
            let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
            assert!(si.files.iter().any(|f| f.ends_with(".tvd")));
        }
    }

    #[test]
    fn a_field_with_both_postings_and_term_vectors_configured_at_once_writes_both_correctly() {
        // Real Lucene's ordinary case: a field is both indexed (postings)
        // and has term vectors stored, in the same commit. Both write-side
        // passes must coexist: the segment's `.si` lists all seven files,
        // and both remain independently readable.
        let tmp = tempdir("postings-and-tv-together");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), tv_body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("body")).unwrap();
        writer.set_term_vector_field(Some("body")).unwrap();

        writer.add_document(doc_with_body("a", "the quick fox"));
        writer.add_document(doc_with_body("b", "the lazy fox"));
        let sis = writer.commit().unwrap().clone();
        assert_eq!(sis.segments.len(), 1);
        let sci = &sis.segments[0];

        let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
        let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
        for ext in ["doc", "tim", "tip", "tmd", "tvd", "tvx", "tvm"] {
            let name = format!("{}.{ext}", sci.segment_name);
            assert!(si.files.contains(&name), "missing {name} in .si files");
        }

        // Postings side still readable.
        let tim = dir.open(&format!("{}.tim", sci.segment_name)).unwrap();
        let tip = dir.open(&format!("{}.tip", sci.segment_name)).unwrap();
        let tmd = dir.open(&format!("{}.tmd", sci.segment_name)).unwrap();
        let field_infos = fi::FieldInfos {
            fields: vec![
                fi::FieldInfo {
                    index_options: IndexOptions::None,
                    ..stored_only_field("id", 0)
                },
                tv_body_field(1),
            ],
        };
        let block_fields = blocktree::open(&tim, &tip, &tmd, &field_infos, &sci.segment_id, "", 2)
            .expect("blocktree::open on IndexWriter-produced .tim/.tip/.tmd");
        let field = block_fields.field("body").unwrap();
        let doc_bytes = dir.open(&format!("{}.doc", sci.segment_name)).unwrap();
        let doc_in = DocInput::open(&doc_bytes, &sci.segment_id, "").expect("open .doc");
        let postings = field.postings(b"fox", Some(&doc_in)).unwrap().unwrap();
        assert_eq!(postings.docs, vec![0, 1]);

        // Term-vector side also readable.
        let tvd = dir.open(&format!("{}.tvd", sci.segment_name)).unwrap();
        let tvx = dir.open(&format!("{}.tvx", sci.segment_name)).unwrap();
        let tvm = dir.open(&format!("{}.tvm", sci.segment_name)).unwrap();
        let reader =
            lucene_codecs::term_vectors::open(&tvd, &tvx, &tvm, &sci.segment_id, "").unwrap();
        let doc0 = reader.document(0).unwrap().unwrap();
        let mut terms0: Vec<String> = doc0.fields[0]
            .terms
            .iter()
            .map(|t| String::from_utf8(t.term.clone()).unwrap())
            .collect();
        terms0.sort();
        assert_eq!(terms0, vec!["fox", "quick", "the"]);
    }

    fn numeric_field(name: &str, number: i32) -> FieldInfo {
        FieldInfo {
            doc_values_type: DocValuesType::Numeric,
            ..stored_only_field(name, number)
        }
    }

    fn doc_with_score(id: &str, score: i64) -> Document {
        Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String(id.to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::Long(score),
                },
            ],
        }
    }

    #[test]
    fn set_doc_values_field_rejects_an_unknown_field_name() {
        let tmp = tempdir("unknown-dv-field");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        let err = writer
            .set_doc_values_field(Some("nonexistent"))
            .unwrap_err();
        assert!(matches!(err, Error::UnknownDocValuesField(name) if name == "nonexistent"));
    }

    #[test]
    fn set_doc_values_field_rejects_a_field_with_no_doc_values_type() {
        let tmp = tempdir("untyped-dv-field");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        let err = writer.set_doc_values_field(Some("id")).unwrap_err();
        assert!(matches!(
            err,
            Error::UnsupportedDocValuesType(name, DocValuesType::None) if name == "id"
        ));
    }

    #[test]
    fn commit_with_doc_values_field_writes_readable_numeric_values_for_multiple_docs() {
        let tmp = tempdir("dv-commit");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), numeric_field("score", 1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_doc_values_field(Some("score")).unwrap();

        writer.add_document(doc_with_score("a", 5));
        writer.add_document(doc_with_score("b", 250));
        writer.add_document(doc_with_score("c", -7));
        let sis = writer.commit().unwrap().clone();
        assert_eq!(sis.segments.len(), 1);
        let sci = &sis.segments[0];

        // Stored fields are still intact (backward-compatible).
        assert_eq!(read_all_docs(&dir, &sis), vec!["a", "b", "c"]);

        // The doc-values files exist and are listed in `.si`.
        let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
        let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
        for ext in ["dvm", "dvd", "dvs"] {
            let name = format!("{}.{ext}", sci.segment_name);
            assert!(si.files.contains(&name), "missing {name} in .si files");
            assert!(
                dir.list_all().unwrap().contains(&name),
                "missing {name} on disk"
            );
        }

        // Readable via the existing, unmodified read side.
        let dvm = dir.open(&format!("{}.dvm", sci.segment_name)).unwrap();
        let dvd = dir.open(&format!("{}.dvd", sci.segment_name)).unwrap();
        let field_infos = fi::FieldInfos {
            fields: vec![stored_only_field("id", 0), numeric_field("score", 1)],
        };
        let (_, meta) =
            lucene_codecs::doc_values::parse_meta(&dvm, &sci.segment_id, "", &field_infos)
                .expect("parse_meta on IndexWriter-produced .dvm");
        let entry = meta.numeric_entry(1).unwrap();
        assert!(entry.is_dense());
        for (doc, want) in [(0, 5i64), (1, 250), (2, -7)] {
            assert_eq!(
                lucene_codecs::doc_values::numeric_value(&dvd, entry, doc).unwrap(),
                Some(want)
            );
        }
    }

    #[test]
    fn commit_with_doc_values_field_accepts_int_values_not_just_long() {
        // build_doc_values_output accepts both FieldValue::Int and
        // FieldValue::Long -- the multi-doc test above only exercises Long,
        // so this covers the Int arm's i64 sign-extension explicitly.
        let tmp = tempdir("dv-commit-int");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), numeric_field("score", 1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_doc_values_field(Some("score")).unwrap();

        writer.add_document(Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("a".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::Int(-42),
                },
            ],
        });
        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let dvm = dir.open(&format!("{}.dvm", sci.segment_name)).unwrap();
        let dvd = dir.open(&format!("{}.dvd", sci.segment_name)).unwrap();
        let field_infos = fi::FieldInfos {
            fields: vec![stored_only_field("id", 0), numeric_field("score", 1)],
        };
        let (_, meta) =
            lucene_codecs::doc_values::parse_meta(&dvm, &sci.segment_id, "", &field_infos)
                .expect("parse_meta on IndexWriter-produced .dvm");
        let entry = meta.numeric_entry(1).unwrap();
        assert_eq!(
            lucene_codecs::doc_values::numeric_value(&dvd, entry, 0).unwrap(),
            Some(-42i64)
        );
    }

    #[test]
    fn commit_with_no_doc_values_field_configured_stays_stored_only() {
        // Backward compatibility: a writer that never calls
        // `set_doc_values_field` must produce exactly the same on-disk shape
        // as before this feature existed -- no `.dvm`/`.dvd`/`.dvs` files at
        // all.
        let tmp = tempdir("no-dv-field");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), numeric_field("score", 1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        writer.add_document(doc_with_score("a", 5));
        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let files = dir.list_all().unwrap();
        for ext in ["dvm", "dvd", "dvs"] {
            assert!(!files.contains(&format!("{}.{ext}", sci.segment_name)));
        }
    }

    #[test]
    fn commit_with_doc_values_field_but_no_pending_docs_writes_no_doc_values_files() {
        let tmp = tempdir("dv-empty-commit");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), numeric_field("score", 1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_doc_values_field(Some("score")).unwrap();

        let sis = writer.commit().unwrap().clone();
        assert!(sis.segments.is_empty());
    }

    #[test]
    fn commit_rejects_and_leaves_state_unchanged_when_a_pending_doc_has_no_doc_values_field() {
        // Dense-only: a doc missing the opted-in field entirely must reject
        // the whole commit atomically, not silently skip that doc (there is
        // no sparse/missing-value doc-values shape to fall back to).
        let tmp = tempdir("dv-missing-value");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), numeric_field("score", 1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_doc_values_field(Some("score")).unwrap();

        writer.add_document(doc_with_score("a", 5));
        writer.add_document(doc("b")); // only field_number 0 ("id"), no "score"
        let before = writer.segment_infos().clone();
        let before_pending = writer.pending_doc_count();

        let err = writer.commit().unwrap_err();
        assert!(matches!(err, Error::MissingDenseDocValue(name, 1) if name == "score"));

        assert_eq!(writer.segment_infos(), &before);
        assert_eq!(writer.pending_doc_count(), before_pending);
        assert!(!tmp.join("segments_1").exists());
    }

    #[test]
    fn commit_rejects_when_a_pending_docs_value_is_not_numeric() {
        let tmp = tempdir("dv-non-numeric-value");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), numeric_field("score", 1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_doc_values_field(Some("score")).unwrap();

        writer.add_document(Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("a".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::String("not-a-number".to_string()), // wrong type
                },
            ],
        });

        let err = writer.commit().unwrap_err();
        assert!(matches!(
            err,
            Error::NonNumericDocValue(name, 0, "String") if name == "score"
        ));
    }

    #[test]
    fn setting_doc_values_field_back_to_none_restores_stored_only_behavior() {
        let tmp = tempdir("dv-field-reset");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), numeric_field("score", 1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_doc_values_field(Some("score")).unwrap();
        writer.set_doc_values_field(None).unwrap();

        writer.add_document(doc_with_score("a", 5));
        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let files = dir.list_all().unwrap();
        assert!(!files.contains(&format!("{}.dvm", sci.segment_name)));
    }

    #[test]
    fn segments_with_doc_values_are_never_automatically_merged_away() {
        // Same class of bug as `segments_with_postings_are_never_...`/
        // `segments_with_term_vectors_are_never_...`, for doc values:
        // enabling both `set_doc_values_field` and `set_merge_policy` at once
        // must not let automatic merging silently drop a segment's doc
        // values -- `execute_merge` only knows how to merge stored fields,
        // so `segment_stats()` excludes any segment carrying `.dvd` from
        // `find_merges`' candidate pool entirely.
        let tmp = tempdir("dv-and-merge-policy");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), numeric_field("score", 1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_doc_values_field(Some("score")).unwrap();
        writer.set_merge_policy(Some(tight_merge_policy()));

        for (id, score) in [("a", 1i64), ("b", 2), ("c", 3)] {
            writer.add_document(doc_with_score(id, score));
            writer.commit().unwrap();
        }

        let final_count = writer.segment_infos().segments.len();
        assert_eq!(
            final_count, 3,
            "segments carrying doc values must never be automatically merged"
        );

        for sci in &writer.segment_infos().segments.clone() {
            let files = dir.list_all().unwrap();
            assert!(files.contains(&format!("{}.dvd", sci.segment_name)));
            let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
            let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
            assert!(si.files.iter().any(|f| f.ends_with(".dvd")));
        }
    }

    #[test]
    fn postings_term_vectors_and_doc_values_configured_together_all_write_correctly() {
        // Three independent write-side passes (postings, term vectors, doc
        // values) each patching the same `.si` after the fact -- proves
        // there's no ordering bug where a later writer's read-modify-write
        // clobbers an earlier one's file-list additions.
        let tmp = tempdir("postings-tv-and-dv-together");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![
            stored_only_field("id", 0),
            tv_body_field(1),
            numeric_field("score", 2),
        ];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("body")).unwrap();
        writer.set_term_vector_field(Some("body")).unwrap();
        writer.set_doc_values_field(Some("score")).unwrap();

        writer.add_document(Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("a".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::String("the quick fox".to_string()),
                },
                StoredField {
                    field_number: 2,
                    value: FieldValue::Long(10),
                },
            ],
        });
        writer.add_document(Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("b".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::String("the lazy fox".to_string()),
                },
                StoredField {
                    field_number: 2,
                    value: FieldValue::Long(20),
                },
            ],
        });
        let sis = writer.commit().unwrap().clone();
        assert_eq!(sis.segments.len(), 1);
        let sci = &sis.segments[0];

        let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
        let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
        for ext in [
            "doc", "tim", "tip", "tmd", "tvd", "tvx", "tvm", "dvm", "dvd", "dvs",
        ] {
            let name = format!("{}.{ext}", sci.segment_name);
            assert!(si.files.contains(&name), "missing {name} in .si files");
        }

        // Doc-values side readable.
        let dvm = dir.open(&format!("{}.dvm", sci.segment_name)).unwrap();
        let dvd = dir.open(&format!("{}.dvd", sci.segment_name)).unwrap();
        let field_infos = fi::FieldInfos {
            fields: vec![
                stored_only_field("id", 0),
                tv_body_field(1),
                numeric_field("score", 2),
            ],
        };
        let (_, meta) =
            lucene_codecs::doc_values::parse_meta(&dvm, &sci.segment_id, "", &field_infos).unwrap();
        let entry = meta.numeric_entry(2).unwrap();
        assert_eq!(
            lucene_codecs::doc_values::numeric_value(&dvd, entry, 0).unwrap(),
            Some(10)
        );
        assert_eq!(
            lucene_codecs::doc_values::numeric_value(&dvd, entry, 1).unwrap(),
            Some(20)
        );

        // Term-vector side still readable too.
        let tvd = dir.open(&format!("{}.tvd", sci.segment_name)).unwrap();
        let tvx = dir.open(&format!("{}.tvx", sci.segment_name)).unwrap();
        let tvm = dir.open(&format!("{}.tvm", sci.segment_name)).unwrap();
        let reader =
            lucene_codecs::term_vectors::open(&tvd, &tvx, &tvm, &sci.segment_id, "").unwrap();
        assert!(reader.document(0).unwrap().is_some());
    }
}
