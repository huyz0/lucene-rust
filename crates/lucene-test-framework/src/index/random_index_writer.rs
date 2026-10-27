//! `RandomIndexWriter`: an `IndexWriter` that now and then changes its live
//! configuration, flushes, commits, force-merges, and chooses between the
//! near-real-time and the on-disk reader, all from a seeded stream.
//!
//! The draws follow Lucene 10.5.0's, call for call: the constructor's (the
//! writer's own stream, the soft-deletes ratio, `mockIndexWriter`'s two,
//! the first flush point, whether to force-merge at random), and every
//! operation's `maybeChangeLiveIndexWriterConfig` and its own choices. The
//! test's `random()` (what the live changes also draw from) and the class
//! environment's `liveIWCFlushMode` travel in a [`WriterEnv`].
//!
//! Differs:
//!
//! - Documents are this port's [`Document`]s, added through the document
//!   API (`add_fields_document`).
//! - Soft deletes: Java's near-real-time reader resolves them itself; this
//!   port's is wrapped in a `SoftDeletesDirectoryReaderWrapper`
//!   ([`DirectoryReader::soft_deletes_wrapper`]) to the same effect, as the
//!   reader opened from the directory is in both. Because explicit-documents
//!   mode is enabled only with soft deletes, a doc-values update of a field
//!   no document has yet creates it only when `useSoftDeletes` was drawn;
//!   tests should index the field first.
//! - `mockIndexWriter`'s test point (a `Thread.yield`) and its open from a
//!   reader's commit are drawn but do nothing: the writer is single-threaded
//!   and opens the latest commit, which is the reader's.
//! - `commit(true)` flushes before committing rather than beside it: the
//!   writer is single-threaded. `getDocWriterThreadPoolSize` is 1 while
//!   documents are buffered, else 0, as a single-threaded Java writer's.
//! - The codec is the default (`RandomCodec` is not drawn).
//! - A random partial `forceMerge`'s segment-count assert counts the
//!   committed segments: this writer's `force_merge` does not flush first,
//!   so segments flushed since the last commit stay outside the merge.

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use lucene_analysis::Analyzer;
use lucene_codecs::field_infos::{DocValuesSkipIndexType, DocValuesType, FieldInfo};
use lucene_index::buffered_updates::{DeleteQuery, DocValuesUpdate, SeqNo, Term};
use lucene_index::document::Document;
use lucene_index::index_writer::IndexWriter;
use lucene_index::nrt::NrtSource;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::{DirectoryReader, Result};
use lucene_store::Directory;
use lucene_util::java_random::JavaRandom;

use super::index_writer_config::{
    maybe_change_live_index_writer_config, new_index_writer_config, Draws, LiveFlushMode,
    RandomIndexWriterConfig,
};
use crate::analysis::mock_analyzer::MockAnalyzer;
use crate::util::test_util;

/// The soft-deletes field Java's `RandomIndexWriter` configures.
pub const SOFT_DELETES_FIELD: &str = "___soft_deletes";

/// What a `RandomIndexWriter` reads off the test and its class
/// environment: the test's `random()`, `liveIWCFlushMode`, and the time
/// zone `AlcoholicMergePolicy` is drawn in.
pub struct WriterEnv {
    /// The test's `random()`.
    pub random: JavaRandom,
    /// `TestRuleSetupAndRestoreClassEnv`'s `liveIWCFlushMode`.
    pub flush_mode: LiveFlushMode,
    /// The test's time zone offset, in milliseconds.
    pub utc_offset_millis: i64,
}

impl WriterEnv {
    /// `random()` seeded with `seed`, the flush mode drawn from it first (as
    /// the class rule draws it), in UTC.
    pub fn new(seed: i64) -> Self {
        let mut random = JavaRandom::new(seed);
        let flush_mode = LiveFlushMode::random(&mut random);
        Self {
            random,
            flush_mode,
            utc_offset_millis: 0,
        }
    }
}

/// `org.apache.lucene.tests.index.RandomIndexWriter`.
pub struct RandomIndexWriter<'d> {
    w: Mutex<IndexWriter<'d>>,
    dir: &'d dyn Directory,
    r: JavaRandom,
    env: WriterEnv,
    config: RandomIndexWriterConfig,
    doc_count: i32,
    flush_at: i32,
    flush_at_factor: f64,
    get_reader_called: bool,
    soft_deletes: bool,
    soft_deletes_ratio: f64,
    do_random_force_merge: bool,
    do_random_force_merge_assert: bool,
}

impl<'d> RandomIndexWriter<'d> {
    /// `new RandomIndexWriter(r, dir)`: a `MockAnalyzer` and a random
    /// configuration, both drawn from `r`.
    ///
    /// # Errors
    /// What opening the writer fails with.
    pub fn new(r: &mut JavaRandom, dir: &'d dyn Directory, env: WriterEnv) -> Result<Self> {
        let analyzer = Arc::new(MockAnalyzer::new(r).build());
        Self::with_analyzer(r, dir, analyzer, env)
    }

    /// `new RandomIndexWriter(r, dir, a)`.
    ///
    /// # Errors
    /// What opening the writer fails with.
    pub fn with_analyzer(
        r: &mut JavaRandom,
        dir: &'d dyn Directory,
        analyzer: Arc<Analyzer>,
        mut env: WriterEnv,
    ) -> Result<Self> {
        let config = new_index_writer_config(
            &mut Draws::with_context(r, &mut env.random),
            analyzer,
            env.utc_offset_millis,
        );
        Self::with_config(r, dir, config, env)
    }

    /// `new RandomIndexWriter(r, dir, c)`.
    ///
    /// # Errors
    /// What opening the writer fails with.
    pub fn with_config(
        r: &mut JavaRandom,
        dir: &'d dyn Directory,
        config: RandomIndexWriterConfig,
        env: WriterEnv,
    ) -> Result<Self> {
        let use_soft_deletes = r.next_boolean();
        Self::with_soft_deletes(r, dir, config, use_soft_deletes, env)
    }

    /// `new RandomIndexWriter(r, dir, c, useSoftDeletes)`.
    ///
    /// # Errors
    /// What opening the writer fails with.
    pub fn with_soft_deletes(
        r: &mut JavaRandom,
        dir: &'d dyn Directory,
        config: RandomIndexWriterConfig,
        use_soft_deletes: bool,
        env: WriterEnv,
    ) -> Result<Self> {
        let own = JavaRandom::new(r.next_long());
        // Java's `1.d / (double) 1 + r.nextInt(10)`: at least 1, so
        // `useSoftDeletes()` is then always true.
        let soft_deletes_ratio = if use_soft_deletes {
            1.0 + f64::from(r.next_int_bounded(10))
        } else {
            0.0
        };
        // mockIndexWriter: the test point's stream, then whether to open
        // from a reader's commit.
        let _test_point = r.next_long();
        let _open_from_reader = r.next_boolean();
        let mut w = IndexWriter::open(
            dir,
            Vec::new(),
            "Lucene104",
            LuceneVersion {
                major: 10,
                minor: 5,
                bugfix: 0,
            },
        )?;
        config.apply(&mut w)?;
        if use_soft_deletes {
            w.enable_explicit_documents()?;
            w.register_field(
                FieldInfo::new(SOFT_DELETES_FIELD, 0)
                    .with_doc_values(DocValuesType::Numeric, DocValuesSkipIndexType::None, -1)
                    .with_soft_deletes_field(true),
            )?;
        }
        let flush_at = test_util::next_int(r, 10, 1000);
        // Make sure we sometimes test indices that don't get any forced
        // merges.
        let do_random_force_merge = !config.is_no_merge_policy() && r.next_boolean();
        Ok(Self {
            w: Mutex::new(w),
            dir,
            r: own,
            env,
            config,
            doc_count: 0,
            flush_at,
            flush_at_factor: 1.0,
            get_reader_called: false,
            soft_deletes: use_soft_deletes,
            soft_deletes_ratio,
            do_random_force_merge,
            do_random_force_merge_assert: true,
        })
    }

    /// `w`: the writer itself.
    pub fn writer(&self) -> MutexGuard<'_, IndexWriter<'d>> {
        self.w.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The writer, as the source of a near-real-time reader.
    pub fn nrt_source(&self) -> &dyn NrtSource {
        &self.w
    }

    /// The configuration the writer was opened with, and changed since.
    pub fn config(&self) -> &RandomIndexWriterConfig {
        &self.config
    }

    /// `softDeletesRatio`: 0 unless the constructor chose soft deletes (see
    /// the module doc).
    pub fn soft_deletes_ratio(&self) -> f64 {
        self.soft_deletes_ratio
    }

    /// `flushAt`: the document count at which the next flush or commit is
    /// due.
    pub fn flush_at(&self) -> i32 {
        self.flush_at
    }

    /// `docCount`: documents added or updated so far.
    pub fn doc_count(&self) -> i32 {
        self.doc_count
    }

    /// The next long of the writer's own stream: where a differential test
    /// checks the writer drew what Java's drew.
    pub fn probe_stream(&mut self) -> i64 {
        self.r.next_long()
    }

    /// Whether `do_random_force_merge` is on (`setDoRandomForceMerge`).
    pub fn random_force_merge(&self) -> bool {
        self.do_random_force_merge
    }

    fn maybe_change(&mut self) -> Result<()> {
        let mut w = self.w.lock().unwrap_or_else(PoisonError::into_inner);
        let mut d = Draws::with_context(&mut self.r, &mut self.env.random);
        maybe_change_live_index_writer_config(
            &mut d,
            self.env.flush_mode,
            &mut self.config,
            &mut w,
        )?;
        Ok(())
    }

    /// `addDocument(doc)`: one time in five as a block of one.
    ///
    /// # Errors
    /// What the writer fails with.
    pub fn add_document(&mut self, doc: &Document) -> Result<SeqNo> {
        self.maybe_change()?;
        let seq_no = if self.r.next_int_bounded(5) == 3 {
            self.writer()
                .add_fields_documents(std::slice::from_ref(doc))?
        } else {
            self.writer().add_fields_document(doc)?
        };
        self.maybe_flush_or_commit()?;
        Ok(seq_no)
    }

    /// `addDocuments(docs)`.
    ///
    /// # Errors
    /// What the writer fails with.
    pub fn add_documents(&mut self, docs: &[Document]) -> Result<SeqNo> {
        self.maybe_change()?;
        let seq_no = self.writer().add_fields_documents(docs)?;
        self.maybe_flush_or_commit()?;
        Ok(seq_no)
    }

    fn maybe_flush_or_commit(&mut self) -> Result<()> {
        self.maybe_change()?;
        let at = self.doc_count;
        self.doc_count = self.doc_count.wrapping_add(1);
        if at == self.flush_at {
            if self.r.next_boolean() {
                self.flush_all_buffers_sequentially()?;
            } else if self.r.next_boolean() {
                self.writer().flush()?;
            } else {
                self.writer().commit()?;
            }
            let lo = (self.flush_at_factor * 10.0) as i32;
            let hi = (self.flush_at_factor * 1000.0) as i32;
            self.flush_at = self
                .flush_at
                .saturating_add(test_util::next_int(&mut self.r, lo, hi));
            if self.flush_at_factor < 2e6 {
                // gradually but exponentially increase time b/w flushes
                self.flush_at_factor *= 1.05;
            }
        }
        Ok(())
    }

    fn flush_all_buffers_sequentially(&mut self) -> Result<()> {
        let thread_pool_size = i32::from(self.writer().pending_doc_count() > 0);
        let num_flushes = 1.min(self.r.next_int_bounded(thread_pool_size + 1));
        for _ in 0..num_flushes {
            if !self.writer().flush_next_buffer()? {
                break; // stop once we didn't flush anything
            }
        }
        Ok(())
    }

    fn use_soft_deletes(&mut self) -> bool {
        self.r.next_double() < self.soft_deletes_ratio
    }

    /// `updateDocuments(delTerm, docs)`.
    ///
    /// # Errors
    /// What the writer fails with.
    pub fn update_documents(&mut self, del_term: Term, docs: &[Document]) -> Result<SeqNo> {
        self.maybe_change()?;
        let seq_no = if self.use_soft_deletes() {
            let soft = self.soft_delete(&del_term);
            self.writer()
                .soft_update_fields_documents(del_term, docs, &[soft])?
        } else if self.r.next_int_bounded(10) < 3 {
            // 30% chance
            self.writer()
                .update_fields_documents_by_query(DeleteQuery::Term(del_term), docs)?
        } else {
            self.writer().update_fields_documents(del_term, docs)?
        };
        self.maybe_flush_or_commit()?;
        Ok(seq_no)
    }

    /// The `NumericDocValuesField(softDeletesField, 1)` a soft update
    /// writes.
    fn soft_delete(&self, term: &Term) -> DocValuesUpdate {
        DocValuesUpdate::Numeric {
            term: term.clone(),
            field: SOFT_DELETES_FIELD.to_string(),
            value: Some(1),
        }
    }

    /// `updateDocument(t, doc)`: one time in five as a block of one.
    ///
    /// # Errors
    /// What the writer fails with.
    pub fn update_document(&mut self, t: Term, doc: &Document) -> Result<SeqNo> {
        self.maybe_change()?;
        let soft = self.use_soft_deletes();
        // Both spellings, a block of one or one document, are one call here.
        let _as_block = self.r.next_int_bounded(5) == 3;
        let docs = std::slice::from_ref(doc);
        let seq_no = if soft {
            let soft = self.soft_delete(&t);
            self.writer()
                .soft_update_fields_documents(t, docs, &[soft])?
        } else {
            self.writer().update_fields_documents(t, docs)?
        };
        self.maybe_flush_or_commit()?;
        Ok(seq_no)
    }

    /// `addIndexes(Directory...)`.
    ///
    /// # Errors
    /// What the writer fails with.
    pub fn add_indexes(&mut self, dirs: &[&dyn Directory]) -> Result<SeqNo> {
        self.maybe_change()?;
        Ok(self.writer().add_indexes(dirs)?)
    }

    /// `updateNumericDocValue(term, field, value)`.
    ///
    /// # Errors
    /// What the writer fails with.
    pub fn update_numeric_doc_value(
        &mut self,
        term: Term,
        field: &str,
        value: i64,
    ) -> Result<SeqNo> {
        self.maybe_change()?;
        Ok(self.writer().update_numeric_doc_value(term, field, value)?)
    }

    /// `updateBinaryDocValue(term, field, value)`.
    ///
    /// # Errors
    /// What the writer fails with.
    pub fn update_binary_doc_value(
        &mut self,
        term: Term,
        field: &str,
        value: &[u8],
    ) -> Result<SeqNo> {
        self.maybe_change()?;
        Ok(self.writer().update_binary_doc_value(term, field, value)?)
    }

    /// `updateDocValues(term, updates...)`.
    ///
    /// # Errors
    /// What the writer fails with.
    pub fn update_doc_values(&mut self, term: Term, updates: &[DocValuesUpdate]) -> Result<SeqNo> {
        self.maybe_change()?;
        Ok(self.writer().update_doc_values(term, updates)?)
    }

    /// `deleteDocuments(Term)`.
    ///
    /// # Errors
    /// What the writer fails with.
    pub fn delete_documents(&mut self, term: Term) -> Result<SeqNo> {
        self.maybe_change()?;
        Ok(self.writer().delete_documents_by_term(&[term])?)
    }

    /// `deleteDocuments(Query)`.
    ///
    /// # Errors
    /// What the writer fails with.
    pub fn delete_documents_by_query(&mut self, query: DeleteQuery) -> Result<SeqNo> {
        self.maybe_change()?;
        Ok(self.writer().delete_documents_by_query(&[query])?)
    }

    /// `commit()`: one time in ten flushing first.
    ///
    /// # Errors
    /// What the writer fails with.
    pub fn commit(&mut self) -> Result<()> {
        let flush_concurrently = self.r.next_int_bounded(10) == 0;
        self.commit_with(flush_concurrently)
    }

    /// `commit(flushConcurrently)`.
    ///
    /// # Errors
    /// What the writer fails with.
    pub fn commit_with(&mut self, flush_concurrently: bool) -> Result<()> {
        self.maybe_change()?;
        if flush_concurrently {
            self.flush_all_buffers_sequentially()?;
        }
        self.writer().commit()?;
        Ok(())
    }

    /// `deleteAll()`.
    ///
    /// # Errors
    /// What the writer fails with.
    pub fn delete_all(&mut self) -> Result<()> {
        Ok(self.writer().delete_all()?)
    }

    /// `getReader()`.
    ///
    /// # Errors
    /// What the writer or the reader fails with.
    pub fn get_reader(&mut self) -> Result<DirectoryReader> {
        self.maybe_change()?;
        self.get_reader_with(true, false)
    }

    /// `forceMergeDeletes()`.
    ///
    /// # Errors
    /// What the writer fails with.
    pub fn force_merge_deletes(&mut self) -> Result<()> {
        self.maybe_change()?;
        Ok(self.writer().force_merge_deletes()?)
    }

    /// `setDoRandomForceMerge`.
    pub fn set_do_random_force_merge(&mut self, v: bool) {
        self.do_random_force_merge = v;
    }

    /// `setDoRandomForceMergeAssert`: whether a partial forced merge is
    /// checked to have reached its limit.
    pub fn set_do_random_force_merge_assert(&mut self, v: bool) {
        self.do_random_force_merge_assert = v;
    }

    fn do_random_force_merge(&mut self) -> Result<()> {
        if !self.do_random_force_merge {
            return Ok(());
        }
        let seg_count = self.writer().segment_count();
        if self.r.next_boolean() || seg_count == 0 {
            // full forceMerge
            self.writer().force_merge(1)?;
        } else if self.r.next_boolean() {
            // partial forceMerge
            let limit = test_util::next_int(&mut self.r, 1, seg_count as i32);
            self.writer().force_merge(limit as usize)?;
            if limit == 1 || !self.config.is_tiered_merge_policy() {
                // Java flushes first, so `getSegmentCount()` is all merged.
                // This writer's `force_merge` reaches the committed
                // segments only: those flushed since stay outside it.
                let actual = self.writer().segment_infos().segments.len();
                assert!(
                    !self.do_random_force_merge_assert || actual <= limit as usize,
                    "limit={limit} actual={actual}"
                );
            }
        } else {
            self.writer().force_merge_deletes()?;
        }
        Ok(())
    }

    /// `getReader(applyDeletions, writeAllDeletes)`: now and then after a
    /// random forced merge, a near-real-time reader or, after a commit, one
    /// of either kind.
    ///
    /// # Errors
    /// What the writer or the reader fails with.
    pub fn get_reader_with(
        &mut self,
        apply_deletions: bool,
        write_all_deletes: bool,
    ) -> Result<DirectoryReader> {
        self.maybe_change()?;
        self.get_reader_called = true;
        if self.r.next_int_bounded(20) == 2 {
            self.do_random_force_merge()?;
        }
        if !apply_deletions || self.r.next_boolean() {
            if self.r.next_int_bounded(5) == 1 {
                self.writer().commit()?;
            }
            self.nrt_reader(apply_deletions, write_all_deletes)
        } else {
            self.writer().commit()?;
            if self.r.next_boolean() {
                let reader = DirectoryReader::open(self.dir)?;
                match self.soft_deletes_field() {
                    Some(field) => reader.soft_deletes_wrapper(field),
                    None => Ok(reader),
                }
            } else {
                self.nrt_reader(apply_deletions, write_all_deletes)
            }
        }
    }

    /// `config.getSoftDeletesField()`.
    pub fn soft_deletes_field(&self) -> Option<&'static str> {
        self.soft_deletes.then_some(SOFT_DELETES_FIELD)
    }

    /// `IndexWriter.getReader(applyDeletions, writeAllDeletes)`. Java's
    /// near-real-time reader resolves soft deletes itself (its live docs are
    /// `PendingSoftDeletes`'); this port's is wrapped to the same effect.
    fn nrt_reader(
        &self,
        apply_deletions: bool,
        write_all_deletes: bool,
    ) -> Result<DirectoryReader> {
        let reader = DirectoryReader::open_nrt(&self.w, apply_deletions, write_all_deletes)?;
        match self.soft_deletes_field() {
            Some(field) => reader.soft_deletes_wrapper(field),
            None => Ok(reader),
        }
    }

    /// `close()`: if no reader was taken, one time in eight force-merges at
    /// random first.
    ///
    /// # Errors
    /// What the writer fails with.
    pub fn close(mut self) -> Result<()> {
        self.maybe_change()?;
        // if someone isn't using getReader() API, we want to be sure to
        // forceMerge since presumably they might open a reader on the dir.
        if !self.get_reader_called && self.r.next_int_bounded(8) == 2 {
            self.do_random_force_merge()?;
            if !self.config.commit_on_close {
                // index may have changed, must commit the changes, or
                // otherwise they are discarded by the call to close()
                self.writer().commit()?;
            }
        }
        let w = self.w.into_inner().unwrap_or_else(PoisonError::into_inner);
        Ok(w.close()?)
    }

    /// `forceMerge(maxSegmentCount)`.
    ///
    /// # Errors
    /// What the writer fails with.
    pub fn force_merge(&mut self, max_segment_count: usize) -> Result<()> {
        self.maybe_change()?;
        Ok(self.writer().force_merge(max_segment_count)?)
    }

    /// `flush()`.
    ///
    /// # Errors
    /// What the writer fails with.
    pub fn flush(&mut self) -> Result<()> {
        Ok(self.writer().flush()?)
    }
}
