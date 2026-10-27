//! `ThreadedIndexingAndSearchingTestCase`: indexing threads adding,
//! updating and deleting `LineFileDocs` documents and blocks of them
//! ("packs") while search threads open near-real-time readers over the
//! writer and search them; then every delete, every pack's contiguity and
//! every surviving document is checked against a final reader, the writer's
//! document count after a commit, and `CheckIndex`.
//!
//! Differs:
//!
//! - The document API of the port's `IndexWriter` takes `&mut self`, so the
//!   indexing threads share it behind a `Mutex` (`NrtSource` is implemented
//!   for `Mutex<IndexWriter>`): their operations interleave one at a time
//!   rather than buffering into separate `DocumentsWriterPerThread`s. The
//!   searches run concurrently with them, over readers the writer handed
//!   out.
//! - Java's abstract `getCurrentSearcher`/`doSearching` (filled in by
//!   `TestNRTThreads`, `TestSearcherManager`, ...) are a [`ReaderMode`].
//!   [`ReaderMode::Nrt`] is `TestNRTThreads`: its `setUp` coin, the
//!   no-delete-open-file directory for NRT-only runs, the pooling reader of
//!   `doAfterWriter`, the main-thread loop that reopens (`openIfChanged`) or
//!   closes, commits and opens one reader, `runSearchThreads(100)` over it,
//!   the open-but-deleted-files check, and the final searcher that may
//!   commit and open the directory. Each search thread draws from its own
//!   stream, seeded off the test's. [`ReaderMode::SearcherManager`] is
//!   `TestSearcherManager`: the warming factory, the NRT or committed-
//!   directory coin, the reopen thread (commit, then a blocking or trying
//!   refresh and a prune), the 1-in-10 `maybeRefresh` and the
//!   `SearcherLifetimeManager` follow-on searches of `getCurrentSearcher`,
//!   and the final searcher. Its `SearcherManager` is this module's, with
//!   lucene-search's refresh rules (`maybeRefresh` gives way to a refresh
//!   in progress; a directory refresh retries while the commit generation
//!   advances): the library's needs a `'static` writer, and this one is
//!   borrowed. The `addDocument` hooks have no overrides to make.
//! - The port's writer merges on commit, not on flush, so the merged
//!   segments the searches check come from the commits `doSearching`
//!   makes: `TestNRTThreads` commits before half its reopens,
//!   `TestSearcherManager`'s reopen thread before each refresh.
//! - The threads run iteration counts, not `RUN_TIME_SEC`; a
//!   `SearcherManager` search thread whose searcher is still empty while
//!   documents are indexed does not use one up.
//! - The port has no `InfoStream`, so no `FailOnNonBulkMergesInfoStream`.
//! - The merged-segment warmer records the merged segment's name (the port's
//!   warmer is handed the directory and the segment, not a reader); the
//!   search threads check every `source=merge` segment they see was warmed,
//!   reading its diagnostics off its `.si`.
//! - A search thread walks each leaf's `body` terms in turn (no
//!   `MultiTerms`); the sorted searches sort on `title` (Java's `titleDV`
//!   is a field `LineFileDocs` no longer adds); the final checks count a
//!   `TermQuery`'s hits as its live postings
//!   ([`docs_with_term`]). Thread sleeps are kept; nothing is printed.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use lucene_index::buffered_updates::Term;
use lucene_index::document::{Document, Store, StoredValue, StringField, TextField};
use lucene_index::index_writer::{IndexWriter, MergedSegmentWarmer};
use lucene_index::segment_info;
use lucene_index::segment_infos::SegmentCommitInfo;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
use lucene_search::query::{BooleanQuery, Clause, PhraseQuery, TermQuery};
use lucene_search::reader::LeafReader as _;
use lucene_search::reference_manager::{PruneByAge, SearcherLifetimeManager};
use lucene_search::top_field::SortField;
use lucene_store::Directory;
use lucene_util::java_random::JavaRandom;
use lucene_util::test_support::TempDir;

use super::base_index_file_format::{docs_with_term, get, new_config, stored, writer_with};
use super::index_writer_config::ensure_sane_iwc_on_nightly;
use crate::analysis::mock_analyzer::MockAnalyzer;
use crate::store::base_directory_wrapper::BaseDirectoryWrapper;
use crate::store::mock_directory_wrapper::MockDirectoryWrapper;
use crate::util::line_file_docs::{LineDoc, LineFileDocs};
use crate::util::lucene_test_case::LuceneTestCase;
use crate::util::test_util;

/// `IndexWriter.MAX_TERM_LENGTH`.
const MAX_TERM_LENGTH: i32 = 32766;

/// Java's `getCurrentSearcher`: where a search thread's reader comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReaderMode {
    /// `TestNRTThreads`: one reader per `doSearching` pass, reopened or
    /// opened anew after a commit, searched by every search thread.
    Nrt,
    /// `TestSearcherManager`: a `SearcherManager` over the writer or the
    /// committed directory, refreshed by a reopen thread, with past
    /// searchers kept in a `SearcherLifetimeManager`.
    SearcherManager,
}

/// What a run did, for the caller's own checks (gate rule #10: a run that
/// deleted nothing, or never searched, proved nothing).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ThreadedRunStats {
    /// `addCount`.
    pub adds: i32,
    /// `delCount`.
    pub deletes: i32,
    /// Packs (`addDocuments`/`updateDocuments` blocks) indexed.
    pub packs: i32,
    /// Readers the search threads searched.
    pub searches: u64,
    /// `TermQuery`s over sampled `body` terms.
    pub term_queries: u64,
    /// Merged segments the search threads saw (each checked as warmed).
    pub merged_segments_seen: u64,
    /// Merged segments the warmer warmed.
    pub warmed: usize,
    /// Documents in the final index.
    pub num_docs: i32,
}

/// `SubDocs`: one pack's id and its documents' ids.
#[derive(Debug)]
struct SubDocs {
    pack_id: String,
    sub_ids: Vec<String>,
    deleted: bool,
}

/// The state Java keeps in the test's fields, shared by every thread.
struct Shared<'w, 'd> {
    writer: &'w Mutex<IndexWriter<'d>>,
    dir: &'d dyn Directory,
    docs: Mutex<LineFileDocs>,
    failed: AtomicBool,
    add_count: AtomicI32,
    del_count: AtomicI32,
    pack_count: AtomicI32,
    packs: AtomicI32,
    del_ids: Mutex<HashSet<String>>,
    del_pack_ids: Mutex<HashSet<String>>,
    all_sub_docs: Mutex<Vec<SubDocs>>,
    warmed: Arc<Mutex<HashSet<String>>>,
    mode: ReaderMode,
    current: Mutex<Option<Arc<DirectoryReader>>>,
    nightly: bool,
    max_iterations: i32,
    tot_term_count: AtomicI32,
    searches: AtomicU64,
    term_queries: AtomicU64,
    merged_seen: AtomicU64,
    /// `assertMergedSegmentsWarmed`.
    assert_merged_warmed: AtomicBool,
    /// `TestSearcherManager`'s state: `isNRT`, `warmCalled`, the refresh
    /// lock, `lifetimeMGR`, `pastSearchers` and the pruner.
    is_nrt: AtomicBool,
    warm_called: AtomicBool,
    refresh_lock: Mutex<()>,
    lifetime: SearcherLifetimeManager<DirectoryReader>,
    past_searchers: Mutex<Vec<i64>>,
    pruner: PruneByAge,
    /// Indexing threads still running.
    indexers: AtomicI32,
}

/// A failure as the message a thread reports.
fn text(e: impl std::fmt::Display) -> String {
    e.to_string()
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// `conf.setMergedSegmentWarmer(...)`: records each merged segment.
struct Warmer(Arc<Mutex<HashSet<String>>>);

impl MergedSegmentWarmer for Warmer {
    fn warm(
        &self,
        dir: &dyn Directory,
        segment: &SegmentCommitInfo,
    ) -> lucene_index::index_writer::Result<()> {
        // Java's warmer reads the segment; this one opens its `.si`.
        dir.open(&format!("{}.si", segment.segment_name))?;
        lock(&self.0).insert(segment.segment_name.clone());
        Ok(())
    }
}

impl<'d> Shared<'_, 'd> {
    fn next_line(&self) -> Result<LineDoc, String> {
        lock(&self.docs).next_doc().map_err(text)
    }

    /// One writer operation.
    fn write<R>(
        &self,
        op: impl FnOnce(&mut IndexWriter<'d>) -> lucene_index::index_writer::Result<R>,
    ) -> Result<(), String> {
        op(&mut lock(self.writer)).map(|_| ()).map_err(text)
    }

    /// `writer.commit()`, which also runs the merges the policy wants.
    fn commit(&self) -> Result<(), String> {
        self.write(|w| w.commit().map(|_| ()))
    }

    /// One indexing thread (`launchIndexingThreads`' `run`).
    fn index(&self, r: &mut JavaRandom) -> Result<(), String> {
        let mut to_delete_ids: Vec<String> = Vec::new();
        let mut to_delete_sub_docs: Vec<usize> = Vec::new();
        let mut iterations = 0;
        loop {
            iterations += 1;
            if iterations >= self.max_iterations || self.failed.load(Ordering::SeqCst) {
                break;
            }
            // Occasional longish pause if running nightly.
            if self.nightly && r.next_int_bounded(6) == 3 {
                let ms = test_util::next_int(r, 50, 500);
                std::thread::sleep(Duration::from_millis(ms as u64));
            }
            // Rate limit ingest rate.
            if r.next_int_bounded(7) == 5 {
                let ms = test_util::next_int(r, 1, 10);
                std::thread::sleep(Duration::from_millis(ms as u64));
            }
            let line = self.next_line()?;
            // Maybe add a randomly named field.
            let added_field = r
                .next_boolean()
                .then(|| format!("extra{}", r.next_int_bounded(40)));
            let build = |line: &LineDoc| {
                let mut doc = line.to_document();
                if let Some(name) = &added_field {
                    doc.add(TextField::new(name.as_str(), "a random field", Store::Yes));
                }
                doc
            };
            // Maybe advance the segment counter.
            if r.next_int_bounded(7) == 5 {
                self.write(|w| {
                    let counter = w.segment_infos_counter().saturating_add(100);
                    w.advance_segment_infos_counter(counter);
                    Ok(())
                })?;
            }
            if r.next_boolean() {
                if r.next_boolean() {
                    self.add_pack(r, line, &build, &mut to_delete_sub_docs)?;
                } else {
                    // Add a single document.
                    let id = line.id.to_string();
                    self.write(|w| w.add_fields_document(&build(&line)))?;
                    self.add_count.fetch_add(1, Ordering::SeqCst);
                    if r.next_int_bounded(5) == 3 {
                        to_delete_ids.push(id);
                    }
                }
            } else {
                // Update a single document; ids are never reused, so the
                // delete never happens.
                let id = line.id.to_string();
                self.write(|w| {
                    w.update_fields_documents(Term::new("docid", id.clone()), &[build(&line)])
                })?;
                self.add_count.fetch_add(1, Ordering::SeqCst);
                if r.next_int_bounded(5) == 3 {
                    to_delete_ids.push(id);
                }
            }
            if r.next_int_bounded(30) == 17 {
                self.apply_deletes(&mut to_delete_ids, &mut to_delete_sub_docs)?;
            }
        }
        Ok(())
    }

    /// Adds a pack, or updates one this thread buffered for deletion.
    fn add_pack(
        &self,
        r: &mut JavaRandom,
        first: LineDoc,
        build: &dyn Fn(&LineDoc) -> Document,
        to_delete_sub_docs: &mut Vec<usize>,
    ) -> Result<(), String> {
        let (pack_id, del_sub_docs) = if !to_delete_sub_docs.is_empty() && r.next_boolean() {
            let i = r.next_int_bounded(to_delete_sub_docs.len() as i32) as usize;
            let index = to_delete_sub_docs.remove(i);
            let all = lock(&self.all_sub_docs);
            assert!(!all[index].deleted);
            // Update the pack, replacing its documents.
            (all[index].pack_id.clone(), Some(index))
        } else {
            // A new pack, under a new id.
            let id = self.pack_count.fetch_add(1, Ordering::SeqCst);
            (id.to_string(), None)
        };
        let mut lines = vec![first];
        let max_doc_count = test_util::next_int(r, 1, 10) as usize;
        while lines.len() < max_doc_count {
            lines.push(self.next_line()?);
        }
        let docs: Vec<Document> = lines
            .iter()
            .map(|line| {
                let mut doc = build(line);
                doc.add(StringField::new("packID", pack_id.clone(), Store::Yes));
                doc
            })
            .collect();
        let sub_ids: Vec<String> = lines.iter().map(|l| l.id.to_string()).collect();
        let sub_index = {
            let mut all = lock(&self.all_sub_docs);
            all.push(SubDocs {
                pack_id: pack_id.clone(),
                sub_ids,
                deleted: false,
            });
            all.len() - 1
        };
        self.add_count
            .fetch_add(docs.len() as i32, Ordering::SeqCst);
        self.packs.fetch_add(1, Ordering::SeqCst);
        let pack_term = Term::new("packID", pack_id);
        match del_sub_docs {
            Some(index) => {
                self.delete_sub_docs(index);
                self.write(|w| w.update_fields_documents(pack_term, &docs))?;
            }
            None => {
                self.write(|w| w.add_fields_documents(&docs))?;
            }
        }
        if r.next_int_bounded(5) == 2 {
            to_delete_sub_docs.push(sub_index);
        }
        Ok(())
    }

    /// Marks a pack deleted, its documents' ids with it.
    fn delete_sub_docs(&self, index: usize) -> String {
        let mut all = lock(&self.all_sub_docs);
        let sub = &mut all[index];
        assert!(!sub.deleted);
        sub.deleted = true;
        lock(&self.del_ids).extend(sub.sub_ids.iter().cloned());
        self.del_count
            .fetch_add(sub.sub_ids.len() as i32, Ordering::SeqCst);
        sub.pack_id.clone()
    }

    /// Applies the deletes a thread buffered.
    fn apply_deletes(
        &self,
        to_delete_ids: &mut Vec<String>,
        to_delete_sub_docs: &mut Vec<usize>,
    ) -> Result<(), String> {
        for id in to_delete_ids.iter() {
            self.write(|w| w.delete_documents_by_term(&[Term::new("docid", id.clone())]))?;
        }
        self.del_count
            .fetch_add(to_delete_ids.len() as i32, Ordering::SeqCst);
        lock(&self.del_ids).extend(to_delete_ids.drain(..));
        for index in to_delete_sub_docs.drain(..) {
            let pack_id = self.delete_sub_docs(index);
            lock(&self.del_pack_ids).insert(pack_id.clone());
            self.write(|w| w.delete_documents_by_term(&[Term::new("packID", pack_id)]))?;
        }
        Ok(())
    }

    /// `getCurrentSearcher()`.
    fn current_reader(&self, r: &mut JavaRandom) -> Result<Arc<DirectoryReader>, String> {
        match self.mode {
            // `TestNRTThreads`: the reader `doSearching` fixed for this pass.
            ReaderMode::Nrt => lock(&self.current)
                .clone()
                .ok_or_else(|| "no fixed searcher".to_string()),
            ReaderMode::SearcherManager => self.manager_searcher(r),
        }
    }

    /// `TestSearcherManager.getCurrentSearcher`: now and then a refresh on
    /// the search thread; a quarter of the time a past searcher (a
    /// follow-on request), else the manager's, recorded for later.
    fn manager_searcher(&self, r: &mut JavaRandom) -> Result<Arc<DirectoryReader>, String> {
        if r.next_int_bounded(10) == 7 && self.maybe_refresh()? {
            self.lifetime.prune(&self.pruner);
        }
        {
            let mut past = lock(&self.past_searchers);
            while !past.is_empty() && r.next_double() < 0.25 {
                let token = past[r.next_int_bounded(past.len() as i32) as usize];
                match self.lifetime.acquire(token).map_err(text)? {
                    Some(s) => return Ok(s),
                    // Pruned.
                    None => {
                        if let Some(i) = past.iter().position(|&t| t == token) {
                            past.remove(i);
                        }
                    }
                }
            }
        }
        let s = self.acquire()?;
        if s.num_docs() != 0 {
            let token = self.lifetime.record(&s).map_err(text)?;
            let mut past = lock(&self.past_searchers);
            if !past.contains(&token) {
                past.push(token);
            }
        }
        Ok(s)
    }

    /// `SearcherManager.acquire()`.
    fn acquire(&self) -> Result<Arc<DirectoryReader>, String> {
        lock(&self.current)
            .clone()
            .ok_or_else(|| "this ReferenceManager is closed".to_string())
    }

    /// The warming `SearcherFactory`: searches `body:united` before the
    /// searcher is published.
    fn new_searcher(&self, reader: DirectoryReader) -> Result<Arc<DirectoryReader>, String> {
        self.warm_called.store(true, Ordering::SeqCst);
        with_searcher(&reader, |s| {
            s.search(&must(TermQuery::new("body", "united")), 10)
                .map_err(text)
        })?;
        Ok(Arc::new(reader))
    }

    /// `doMaybeRefresh`: `openIfChanged` from the writer or the directory,
    /// swapped in when something changed.
    fn refresh_if_needed(&self) -> Result<(), String> {
        let current = self.acquire()?;
        let next = if self.is_nrt.load(Ordering::SeqCst) {
            current.open_if_changed_nrt(self.writer).map_err(text)?
        } else {
            open_if_changed_retrying(&current, self.dir)?
        };
        drop(current);
        if let Some(reader) = next {
            let searcher = self.new_searcher(reader)?;
            *lock(&self.current) = Some(searcher);
        }
        Ok(())
    }

    /// `maybeRefresh()`: `false` when another thread is refreshing.
    fn maybe_refresh(&self) -> Result<bool, String> {
        let _guard = match self.refresh_lock.try_lock() {
            Ok(g) => g,
            Err(std::sync::TryLockError::Poisoned(p)) => p.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => return Ok(false),
        };
        self.refresh_if_needed()?;
        Ok(true)
    }

    /// `maybeRefreshBlocking()`.
    fn maybe_refresh_blocking(&self) -> Result<(), String> {
        let _guard = lock(&self.refresh_lock);
        self.refresh_if_needed()
    }

    /// `isSearcherCurrent()`.
    fn is_searcher_current(&self) -> Result<bool, String> {
        let s = self.acquire()?;
        if self.is_nrt.load(Ordering::SeqCst) {
            s.is_current_nrt(self.writer).map_err(text)
        } else {
            let latest = lucene_index::segment_infos::read_latest(self.dir).map_err(text)?;
            Ok(latest.generation == s.segment_infos.generation)
        }
    }

    /// `TestSearcherManager.doAfterWriter`: a manager over the writer, or
    /// over the directory after an empty commit (whose merged segments no
    /// pooled reader warms).
    fn manager_after_writer(&self, random: &mut JavaRandom) -> Result<(), String> {
        let first = if random.next_boolean() {
            self.is_nrt.store(true, Ordering::SeqCst);
            DirectoryReader::open_from_writer(self.writer).map_err(text)?
        } else {
            // SearcherManager needs to see an empty commit.
            self.commit()?;
            self.assert_merged_warmed.store(false, Ordering::SeqCst);
            DirectoryReader::open(self.dir).map_err(text)?
        };
        let searcher = self.new_searcher(first)?;
        *lock(&self.current) = Some(searcher);
        Ok(())
    }

    /// `TestSearcherManager.doSearching`: a reopen thread commits and
    /// refreshes (blocking or trying) while `runSearchThreads` searches.
    fn do_searching_manager(&self, random: &mut JavaRandom) -> Result<(), String> {
        let seed = random.next_long();
        std::thread::scope(|scope| {
            let reopen = scope.spawn(move || {
                let outcome = self.reopen_loop(&mut JavaRandom::new(seed));
                if outcome.is_err() {
                    self.failed.store(true, Ordering::SeqCst);
                }
                outcome
            });
            let searched = self.run_search_threads(random, self.max_iterations);
            let reopened = join(reopen).map_or(Ok(()), Err);
            searched.and(reopened)
        })
    }

    /// The reopen thread's `run`.
    fn reopen_loop(&self, r: &mut JavaRandom) -> Result<(), String> {
        let mut iterations = 0;
        loop {
            iterations += 1;
            if iterations >= self.max_iterations {
                return Ok(());
            }
            sleep_ms(test_util::next_int(r, 1, 100));
            self.commit()?;
            sleep_ms(test_util::next_int(r, 1, 5));
            if r.next_boolean() {
                self.maybe_refresh_blocking()?;
                self.lifetime.prune(&self.pruner);
            } else if self.maybe_refresh()? {
                self.lifetime.prune(&self.pruner);
            }
        }
    }

    /// `TestSearcherManager.getFinalSearcher`.
    fn manager_final_searcher(&self) -> Result<Arc<DirectoryReader>, String> {
        if !self.is_nrt.load(Ordering::SeqCst) {
            self.commit()?;
        }
        if !(self.maybe_refresh()? || self.is_searcher_current()?) {
            return Err("the final refresh neither ran nor found the searcher current".into());
        }
        self.acquire()
    }

    /// One search thread (`runSearchThreads`' `run`).
    fn search(&self, r: &mut JavaRandom, max_iterations: i32) -> Result<(), String> {
        let mut iterations = 0;
        loop {
            iterations += 1;
            if iterations >= max_iterations || self.failed.load(Ordering::SeqCst) {
                break;
            }
            let reader = self.current_reader(r)?;
            // See the module documentation: an empty searcher while
            // documents are indexed is not an iteration.
            if self.mode == ReaderMode::SearcherManager
                && reader.num_docs() == 0
                && self.indexers.load(Ordering::SeqCst) > 0
            {
                iterations -= 1;
                std::thread::yield_now();
                continue;
            }
            self.searches.fetch_add(1, Ordering::SeqCst);
            // Verify 1) the writer sets diagnostics, and 2) merged
            // segments are warmed.
            for leaf in reader.segment_readers() {
                let si_bytes = self
                    .dir
                    .open(&format!("{}.si", leaf.segment_name))
                    .map_err(text)?;
                let si = segment_info::parse(&si_bytes, &leaf.segment_id()).map_err(text)?;
                let source = si
                    .diagnostics
                    .iter()
                    .find(|(k, _)| k == "source")
                    .map(|(_, v)| v.as_str())
                    .ok_or_else(|| format!("{} has no source diagnostic", leaf.segment_name))?;
                if source == "merge" && self.assert_merged_warmed.load(Ordering::SeqCst) {
                    self.merged_seen.fetch_add(1, Ordering::SeqCst);
                    if !lock(&self.warmed).contains(&leaf.segment_name) {
                        return Err(format!(
                            "sub reader {} wasn't warmed: diagnostics={:?}",
                            leaf.segment_name, si.diagnostics
                        ));
                    }
                }
            }
            if reader.num_docs() > 0 {
                with_searcher(&reader, |s| smoke_test_searcher(s, &reader))?;
                self.search_body_terms(r, &reader, max_iterations)?;
            }
        }
        Ok(())
    }

    /// Samples about 30 of the `body` terms and searches each.
    fn search_body_terms(
        &self,
        r: &mut JavaRandom,
        reader: &DirectoryReader,
        max_iterations: i32,
    ) -> Result<(), String> {
        let tot = self.tot_term_count.load(Ordering::SeqCst);
        let (shift, trigger) = if tot < 30 {
            (0, 1)
        } else {
            let trigger = tot / 30;
            (r.next_int_bounded(trigger), trigger)
        };
        let mut terms = Vec::new();
        let mut seen_term_count = 0;
        let mut iters = 0;
        'leaves: for leaf in reader.segment_readers() {
            let Some(leaf_terms) = leaf.terms("body").map_err(text)? else {
                continue;
            };
            let mut te = leaf_terms.iterator().map_err(text)?;
            loop {
                iters += 1;
                if iters >= max_iterations {
                    break 'leaves;
                }
                let Some(term) = te.next().map_err(text)? else {
                    break;
                };
                seen_term_count += 1;
                if (seen_term_count + shift) % trigger == 0 {
                    terms.push(term.to_vec());
                }
            }
        }
        if iters < max_iterations {
            self.tot_term_count.store(seen_term_count, Ordering::SeqCst);
        }
        with_searcher(reader, |s| {
            for term in terms {
                self.term_queries.fetch_add(1, Ordering::SeqCst);
                run_query(s, reader, must(TermQuery::new("body", term)))?;
            }
            Ok(())
        })
    }
}

impl Shared<'_, '_> {
    /// `runSearchThreads(maxIterations)`: search threads over
    /// `getCurrentSearcher()`, joined before returning. Each draws from its
    /// own stream, seeded off `random`.
    fn run_search_threads(
        &self,
        random: &mut JavaRandom,
        max_iterations: i32,
    ) -> Result<(), String> {
        let num_threads = if self.nightly {
            test_util::next_int(random, 1, 5)
        } else {
            2
        };
        // Silly starting guess.
        self.tot_term_count.store(100, Ordering::SeqCst);
        let seeds: Vec<i64> = (0..num_threads).map(|_| random.next_long()).collect();
        let failures: Vec<String> = std::thread::scope(|scope| {
            let handles: Vec<_> = seeds
                .iter()
                .map(|&seed| {
                    scope.spawn(move || {
                        let outcome = self.search(&mut JavaRandom::new(seed), max_iterations);
                        if outcome.is_err() {
                            self.failed.store(true, Ordering::SeqCst);
                        }
                        outcome
                    })
                })
                .collect();
            handles.into_iter().filter_map(join).collect()
        });
        failures.into_iter().next().map_or(Ok(()), Err)
    }

    /// `TestNRTThreads.doSearching`: on the calling thread, while the
    /// indexing threads run, reopen the reader (`openIfChanged`) or close
    /// it, commit and open anew; then search the one reader with
    /// `runSearchThreads(100)`. No file deleted while open, after any
    /// commit or at the end.
    fn do_searching_nrt(
        &self,
        dir: &MockDirectoryWrapper,
        random: &mut JavaRandom,
    ) -> Result<(), String> {
        let mut any_open_del_files = false;
        let mut r = Arc::new(DirectoryReader::open_from_writer(self.writer).map_err(text)?);
        let mut iterations = 0;
        loop {
            iterations += 1;
            if iterations >= self.max_iterations || self.failed.load(Ordering::SeqCst) {
                break;
            }
            if random.next_boolean() {
                if let Some(r2) = r.open_if_changed_nrt(self.writer).map_err(text)? {
                    r = Arc::new(r2);
                }
            } else {
                // `r.close()`: the fixed searcher holds the last reference.
                *lock(&self.current) = None;
                drop(r);
                self.commit()?;
                any_open_del_files |= !dir.open_deleted_files().is_empty();
                r = Arc::new(DirectoryReader::open_from_writer(self.writer).map_err(text)?);
            }
            if r.num_docs() > 0 {
                *lock(&self.current) = Some(Arc::clone(&r));
                with_searcher(&r, |s| smoke_test_searcher(s, &r))?;
                self.searches.fetch_add(1, Ordering::SeqCst);
                self.run_search_threads(random, 100)?;
            }
        }
        *lock(&self.current) = None;
        drop(r);
        any_open_del_files |= !dir.open_deleted_files().is_empty();
        if any_open_del_files {
            return Err(format!(
                "saw non-zero open-but-deleted count: {:?}",
                dir.open_deleted_files()
            ));
        }
        Ok(())
    }
}

/// A thread's outcome as its failure, if any.
fn join(h: std::thread::ScopedJoinHandle<'_, Result<(), String>>) -> Option<String> {
    match h.join() {
        Ok(Ok(())) => None,
        Ok(Err(e)) => Some(e),
        Err(_) => Some("a thread panicked".to_string()),
    }
}

fn sleep_ms(ms: i32) {
    std::thread::sleep(Duration::from_millis(u64::try_from(ms).unwrap_or(0)));
}

/// `DirectoryReader.openIfChanged(reader)` under `FindSegmentsFile`'s
/// retry: a commit landing during the open may delete the files it is
/// reading, so a failure is retried while the commit generation advances.
fn open_if_changed_retrying(
    reader: &DirectoryReader,
    dir: &dyn Directory,
) -> Result<Option<DirectoryReader>, String> {
    let generation = || {
        dir.list_all()
            .and_then(|files| lucene_store::directory::last_commit_generation(&files))
            .map_err(text)
    };
    let mut last = generation()?;
    loop {
        match reader.open_if_changed(dir) {
            Ok(next) => return Ok(next),
            Err(e) => {
                let now = generation()?;
                if now <= last {
                    return Err(text(e));
                }
                last = now;
            }
        }
    }
}

/// A query of one required clause.
fn must(clause: impl Into<Clause>) -> BooleanQuery {
    BooleanQuery {
        must: vec![clause.into()],
        ..Default::default()
    }
}

/// A searcher over `reader`, with norms for the text fields.
fn with_searcher<R>(
    reader: &DirectoryReader,
    f: impl FnOnce(&IndexSearcher<'_, '_>) -> Result<R, String>,
) -> Result<R, String> {
    let opened = reader.open_segments().map_err(text)?;
    let segments = opened.as_open_segments();
    let fields = ["body".to_string(), "titleTokenized".to_string()];
    let owned = reader
        .field_norms_by_field(&fields)
        .map_err(|error| error.to_string())?;
    let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
    let searcher = IndexSearcher::new(&segments, &norms).map_err(text)?;
    f(&searcher)
}

/// `runQuery`: the query by score, then twice by `title`; the two sorted
/// searches' totals must agree.
fn run_query(
    s: &IndexSearcher<'_, '_>,
    reader: &DirectoryReader,
    q: BooleanQuery,
) -> Result<u64, String> {
    s.search(&q, 10).map_err(text)?;
    let sort = [SortField::string("title", false)];
    let sorted = |s: &IndexSearcher<'_, '_>| {
        s.search_sorted(reader.segment_readers(), &q, 10, &sort, None)
            .map(|td| td.total.value)
            .map_err(text)
    };
    let hit_count = sorted(s)?;
    let hit_count2 = sorted(s)?;
    if hit_count != hit_count2 {
        return Err(format!("sorted totals differ: {hit_count} vs {hit_count2}"));
    }
    Ok(hit_count)
}

/// `smokeTestSearcher`.
fn smoke_test_searcher(s: &IndexSearcher<'_, '_>, reader: &DirectoryReader) -> Result<(), String> {
    run_query(s, reader, must(TermQuery::new("body", "united")))?;
    run_query(s, reader, must(TermQuery::new("titleTokenized", "states")))?;
    let phrase = must(PhraseQuery::new("body", ["united", "states"]));
    run_query(s, reader, phrase)?;
    Ok(())
}

/// The final checks of `runTest`: every deleted id and pack is gone, each
/// live pack's documents are contiguous and in order with their `packID`
/// stored, and every id below `end_id` not deleted is there once. Returns
/// what did not hold.
fn verify_final(
    s: &DirectoryReader,
    del_ids: &HashSet<String>,
    del_pack_ids: &HashSet<String>,
    all_sub_docs: &[SubDocs],
    end_id: i32,
) -> Vec<String> {
    let mut problems = Vec::new();
    let hits = |field: &str, id: &str| docs_with_term(s, field, id.as_bytes());
    for id in del_ids {
        let n = hits("docid", id).len();
        if n != 0 {
            problems.push(format!(
                "doc id={id} is supposed to be deleted, but got {n} hits"
            ));
        }
    }
    for id in del_pack_ids {
        let n = hits("packID", id).len();
        if n != 0 {
            problems.push(format!(
                "packID={id} is supposed to be deleted, but got {n} matches"
            ));
        }
    }
    for sub in all_sub_docs {
        if sub.deleted {
            for id in sub
                .sub_ids
                .iter()
                .filter(|id| !hits("docid", id).is_empty())
            {
                problems.push(format!(
                    "doc id={id} of deleted pack {} is there",
                    sub.pack_id
                ));
            }
            continue;
        }
        let pack = hits("packID", &sub.pack_id);
        if pack.len() != sub.sub_ids.len() {
            let (want, got) = (sub.sub_ids.len(), pack.len());
            problems.push(format!(
                "packID={}: expected {want} hits but got {got}",
                sub.pack_id
            ));
            continue;
        }
        let stored_pack = StoredValue::String(sub.pack_id.clone());
        for (i, id) in sub.sub_ids.iter().enumerate() {
            let want = vec![pack[0] + i as i32];
            let got = hits("docid", id);
            if got != want || get(&stored(s, want[0]), "packID") != Some(&stored_pack) {
                problems.push(format!(
                    "pack {} doc {id}: want {want:?}, got {got:?}",
                    sub.pack_id
                ));
            }
        }
    }
    for id in (0..end_id).map(|id| id.to_string()) {
        let n = if del_ids.contains(&id) {
            1
        } else {
            hits("docid", &id).len()
        };
        if n != 1 {
            problems.push(format!(
                "doc id={id} is not supposed to be deleted, but got hitCount={n}"
            ));
        }
    }
    problems
}

/// `runTest(testName)`.
///
/// # Panics
/// When a thread failed or a check does not hold, with what differed.
pub fn run_test(t: &mut LuceneTestCase, test_name: &str, mode: ReaderMode) -> ThreadedRunStats {
    // `TestNRTThreads.setUp`.
    let use_non_nrt_readers = mode == ReaderMode::Nrt && t.random().next_boolean();
    // `TestSearcherManager.testSearcherManager`'s pruner.
    let prune_age = if mode == ReaderMode::SearcherManager && t.nightly() {
        test_util::next_int(t.random(), 1, 10)
    } else {
        1
    };
    let mut random = JavaRandom::new(t.random().next_long());
    let docs = LineFileDocs::new(&mut random).expect("LineFileDocs");
    let tmp = TempDir::new(test_name);
    let dir = t.new_mock_fs_directory(tmp.path());
    // `TestNRTThreads.getDirectory`.
    if mode == ReaderMode::Nrt && !use_non_nrt_readers {
        dir.set_assert_no_delete_open_file(true);
    }
    // Don't double-checkIndex; this test checks it itself.
    dir.set_check_index_on_close(false);
    let max_token_length = test_util::next_int(t.random(), 1, MAX_TERM_LENGTH);
    let analyzer = Arc::new(
        MockAnalyzer::new(t.random())
            .max_token_length(max_token_length)
            .build(),
    );
    let mut config = new_config(t, analyzer);
    config.commit_on_close = false;
    // `setInfoStream(new FailOnNonBulkMergesInfoStream())`: no InfoStream.
    ensure_sane_iwc_on_nightly(t.nightly(), &mut config);
    let warmed = Arc::new(Mutex::new(HashSet::new()));
    let mut writer = writer_with(&dir, &config);
    writer.set_merged_segment_warmer(Some(Arc::new(Warmer(Arc::clone(&warmed)))));
    test_util::reduce_open_files(&config.merge_policy, None);
    let writer = Mutex::new(writer);

    let nightly = t.nightly();
    let max_iterations = if nightly { 200 } else { 10 * t.multiplier() };
    let shared = Shared {
        writer: &writer,
        dir: &dir,
        docs: Mutex::new(docs),
        failed: AtomicBool::new(false),
        add_count: AtomicI32::new(0),
        del_count: AtomicI32::new(0),
        pack_count: AtomicI32::new(0),
        packs: AtomicI32::new(0),
        del_ids: Mutex::new(HashSet::new()),
        del_pack_ids: Mutex::new(HashSet::new()),
        all_sub_docs: Mutex::new(Vec::new()),
        warmed: Arc::clone(&warmed),
        mode,
        current: Mutex::new(None),
        nightly,
        max_iterations,
        tot_term_count: AtomicI32::new(100),
        searches: AtomicU64::new(0),
        term_queries: AtomicU64::new(0),
        merged_seen: AtomicU64::new(0),
        assert_merged_warmed: AtomicBool::new(true),
        is_nrt: AtomicBool::new(false),
        warm_called: AtomicBool::new(false),
        refresh_lock: Mutex::new(()),
        lifetime: SearcherLifetimeManager::new(),
        past_searchers: Mutex::new(Vec::new()),
        pruner: PruneByAge::new(f64::from(prune_age)).expect("a positive age"),
        indexers: AtomicI32::new(0),
    };
    // The `ExecutorService` coin: searches here run on their own threads.
    let _ = t.random().next_boolean();
    match mode {
        // `TestNRTThreads.doAfterWriter`: pool readers from the start.
        ReaderMode::Nrt => {
            drop(DirectoryReader::open_from_writer(&writer).expect("doAfterWriter"));
        }
        ReaderMode::SearcherManager => shared
            .manager_after_writer(t.random())
            .expect("doAfterWriter"),
    }
    let num_index_threads = test_util::next_int(t.random(), 2, 4);
    let index_seeds: Vec<i64> = (0..num_index_threads)
        .map(|_| t.random().next_long())
        .collect();
    shared.indexers.store(num_index_threads, Ordering::SeqCst);
    let failures: Vec<String> = std::thread::scope(|scope| {
        let shared = &shared;
        let handles: Vec<_> = index_seeds
            .iter()
            .map(|&seed| {
                scope.spawn(move || {
                    let outcome = shared.index(&mut JavaRandom::new(seed));
                    shared.indexers.fetch_sub(1, Ordering::SeqCst);
                    if outcome.is_err() {
                        shared.failed.store(true, Ordering::SeqCst);
                    }
                    outcome
                })
            })
            .collect();
        // Let the index build up a bit.
        std::thread::sleep(Duration::from_millis(100));
        let mut failures = Vec::new();
        let searched = match mode {
            ReaderMode::Nrt => shared.do_searching_nrt(&dir, t.random()),
            ReaderMode::SearcherManager => shared.do_searching_manager(t.random()),
        };
        if let Err(e) = searched {
            shared.failed.store(true, Ordering::SeqCst);
            failures.push(e);
        }
        failures.extend(handles.into_iter().filter_map(join));
        failures
    });
    assert!(failures.is_empty(), "{test_name}: {failures:?}");
    assert!(!shared.failed.load(Ordering::SeqCst));

    // `getFinalSearcher()`: `TestNRTThreads` may commit and open the
    // directory instead.
    let s = match mode {
        ReaderMode::Nrt => {
            *lock(&shared.current) = None;
            if use_non_nrt_readers && !t.random().next_boolean() {
                lock(&writer).commit().expect("commit");
                DirectoryReader::open(&dir)
            } else {
                DirectoryReader::open_from_writer(&writer)
            }
            .map(Arc::new)
            .expect("getFinalSearcher")
        }
        ReaderMode::SearcherManager => shared.manager_final_searcher().expect("getFinalSearcher"),
    };
    let del_ids = lock(&shared.del_ids).clone();
    let end_id = lock(&shared.docs).next_doc().expect("nextDoc").id;
    let problems = verify_final(
        &s,
        &del_ids,
        &lock(&shared.del_pack_ids),
        &lock(&shared.all_sub_docs),
        end_id,
    );
    assert!(problems.is_empty(), "{test_name}: {problems:#?}");
    let adds = shared.add_count.load(Ordering::SeqCst);
    let deletes = shared.del_count.load(Ordering::SeqCst);
    assert_eq!(
        adds - deletes,
        s.num_docs(),
        "addCount={adds} delCount={deletes}"
    );
    drop(s);
    if mode == ReaderMode::SearcherManager {
        // `TestSearcherManager.doClose`.
        assert!(shared.warm_called.load(Ordering::SeqCst), "warmCalled");
        *lock(&shared.current) = None;
        shared.lifetime.close();
    }
    let packs = shared.packs.load(Ordering::SeqCst);
    let searches = shared.searches.load(Ordering::SeqCst);
    let term_queries = shared.term_queries.load(Ordering::SeqCst);
    let merged_segments_seen = shared.merged_seen.load(Ordering::SeqCst);
    drop(shared);

    let mut writer = writer.into_inner().unwrap_or_else(|p| p.into_inner());
    writer.commit().expect("commit");
    // `writer.getDocStats().numDocs`, read off the commit.
    let committed = DirectoryReader::open(&dir).expect("open the commit");
    assert_eq!(adds - deletes, committed.num_docs(), "after commit");
    drop(committed);
    writer.commit().expect("commit");
    writer.close().expect("close");
    test_util::check_index(&dir).expect("CheckIndex");
    let warmed = lock(&warmed).len();
    ThreadedRunStats {
        adds,
        deletes,
        packs,
        searches,
        term_queries,
        merged_segments_seen,
        warmed,
        num_docs: adds - deletes,
    }
}

#[cfg(test)]
mod tests {
    use lucene_store::FsDirectory;

    use super::super::base_index_file_format::plain_writer;
    use super::*;

    fn doc(id: &str, pack: Option<&str>) -> Document {
        let mut d = Document::new();
        d.add(StringField::new("docid", id.to_string(), Store::Yes));
        if let Some(pack) = pack {
            d.add(StringField::new("packID", pack.to_string(), Store::Yes));
        }
        d
    }

    fn sub(pack: &str, ids: &[&str], deleted: bool) -> SubDocs {
        SubDocs {
            pack_id: pack.to_string(),
            sub_ids: ids.iter().map(|s| s.to_string()).collect(),
            deleted,
        }
    }

    fn set(ids: &[&str]) -> HashSet<String> {
        ids.iter().map(|s| s.to_string()).collect()
    }

    /// Gate rule #10: each final check fires on the state it guards
    /// against, and stays quiet on a consistent one.
    #[test]
    fn the_final_checks_report_what_does_not_hold() {
        let tmp = TempDir::new("threaded-verify-final");
        let dir = FsDirectory::open(tmp.path());
        let mut w = plain_writer(&dir);
        for d in [
            doc("0", Some("p0")),
            doc("1", Some("p0")),
            doc("2", None),
            doc("3", None),
        ] {
            w.add_fields_document(&d).unwrap();
        }
        w.commit().unwrap();
        let r = DirectoryReader::open(&dir).unwrap();
        let fine = [sub("p0", &["0", "1"], false), sub("gone", &["9"], true)];
        assert_eq!(
            verify_final(&r, &set(&["9"]), &set(&["gone"]), &fine, 4),
            Vec::<String>::new()
        );
        let wrong = [
            sub("p0", &["1", "0"], false),
            sub("p9", &["9"], false),
            sub("x", &["3"], true),
        ];
        let problems = verify_final(&r, &set(&["2"]), &set(&["p0"]), &wrong, 5);
        for want in [
            "doc id=2 is supposed to be deleted, but got 1 hits",
            "packID=p0 is supposed to be deleted, but got 2 matches",
            "pack p0 doc 1: want [0], got [1]",
            "packID=p9: expected 1 hits but got 0",
            "doc id=3 of deleted pack x is there",
            "doc id=4 is not supposed to be deleted, but got hitCount=0",
        ] {
            assert!(problems.iter().any(|p| p == want), "{want}: {problems:#?}");
        }
        assert_eq!(problems.len(), 7, "{problems:#?}");
        assert_eq!(text(std::io::Error::other("e")), "e");
    }
}
