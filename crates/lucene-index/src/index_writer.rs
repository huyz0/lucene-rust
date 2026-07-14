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
//! # What this deliberately is not
//!
//! - **No automatic merge triggering.** This port's merge-policy decision
//!   function ([`crate::merge_policy::find_merges`]) and merge executor
//!   ([`crate::merge::merge_stored_only_segments`]) both already exist, but
//!   nothing wires "call `find_merges` after every commit and merge whatever
//!   it proposes" into this facade -- that automatic trigger is a genuinely
//!   separate, later task (see `PLAN.md`/`docs/parity.md`). [`IndexWriter`]
//!   exposes [`IndexWriter::segment_infos`] (its current committed segment
//!   list) precisely so a caller can still drive [`crate::merge_policy`]/
//!   [`crate::merge`] manually today.
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
use crate::merge;
use crate::segment_info::LuceneVersion;
use crate::segment_infos::{self, SegmentCommitInfo, SegmentInfos};
use crate::segment_writer::{self, flush_stored_only_segment};
use crate::term_delete;
use crate::update_document::{self, SegmentDeleteSource};

use lucene_codecs::field_infos::FieldInfo;
use lucene_codecs::stored_fields::Document;
use lucene_store::codec_util::ID_LENGTH;
use lucene_store::directory::Directory;

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
        })
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
    /// `DocumentsWriterPerThread.flush()`-worth of documents, minus the
    /// automatic merge-triggering real `commit()` also performs (see module
    /// doc comment: this port has no automatic merge trigger yet).
    ///
    /// A `commit()` with an empty pending-document buffer still writes the
    /// next `segments_N` generation (bumping `version`) with no new
    /// segment appended -- matches real Lucene's `commit()` being a valid,
    /// if unusual, no-op-content commit rather than a special "nothing to do"
    /// case that skips writing. Returns the new committed [`SegmentInfos`].
    pub fn commit(&mut self) -> Result<&SegmentInfos> {
        let mut new_segment_infos = self.segment_infos.clone();
        new_segment_infos.generation += 1;
        new_segment_infos.version += 1;

        if !self.pending_docs.is_empty() {
            let segment_name = self.next_segment_name();
            let segment_id = generate_segment_id(self.segment_infos.counter);
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
            new_segment_infos.segments.push(sci);
            new_segment_infos.counter += 1;
            self.pending_docs.clear();
        }

        segment_infos::write(&new_segment_infos, self.dir)?;
        self.segment_infos = new_segment_infos;
        Ok(&self.segment_infos)
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
}
