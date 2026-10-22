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
//!   `TestNRTThreads`, `TestSearcherManager`, ...) are a [`ReaderMode`],
//!   simplified. Java's `TestNRTThreads` runs one loop on the main thread
//!   that reopens (`openIfChanged`) or commits and opens, then runs
//!   `runSearchThreads(100)` over that one searcher; here each search
//!   thread opens its own reader every iteration (after a commit half the
//!   time), never through `openIfChanged`, and walks at most
//!   `maxIterations` terms (Java's search threads, 100). Its
//!   open-but-deleted-files assertion is not ported. `TestSearcherManager`'s
//!   reopen thread, its 1-in-10 `maybeRefresh` in `getCurrentSearcher`,
//!   `SearcherLifetimeManager`, the non-NRT `SearcherManager(dir)` branch
//!   and its warming factory are not ported: [`ReaderMode::Reopen`] commits
//!   and reopens one shared reader before each search. Each thread draws
//!   from its own stream, seeded off the test's. `getDirectory`,
//!   `doAfterWriter` and the `addDocument` hooks have no overrides to make.
//! - The port's writer merges on commit, not on flush, so the merged
//!   segments the searches check come from the commits `doSearching`
//!   makes: `TestNRTThreads` commits before half its reopens,
//!   `TestSearcherManager`'s reopen thread before each refresh.
//! - `advanceSegmentInfosCounter` is not ported (its draw is kept), nor
//!   `FailOnNonBulkMergesInfoStream`, `ensureSaneIWCOnNightly` or
//!   `TestUtil.reduceOpenFiles`.
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
use lucene_search::top_field::SortField;
use lucene_store::Directory;
use lucene_util::java_random::JavaRandom;
use lucene_util::test_support::TempDir;

use super::base_index_file_format::{docs_with_term, get, new_config, stored, writer_with};
use crate::analysis::mock_analyzer::MockAnalyzer;
use crate::store::base_directory_wrapper::BaseDirectoryWrapper;
use crate::util::line_file_docs::{LineDoc, LineFileDocs};
use crate::util::lucene_test_case::LuceneTestCase;
use crate::util::test_util;

/// `IndexWriter.MAX_TERM_LENGTH`.
const MAX_TERM_LENGTH: i32 = 32766;

/// Java's `getCurrentSearcher`: where a search thread's reader comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReaderMode {
    /// `TestNRTThreads`: `DirectoryReader.open(writer)` for every search,
    /// after a commit half the time.
    Nrt,
    /// `TestSearcherManager`: one reader shared by every search thread,
    /// replaced by `openIfChanged(reader, writer)` after a commit before
    /// each search (`SearcherManager.maybeRefresh`).
    Reopen,
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
            // Maybe advance the segment counter: not ported, the draw kept.
            let _ = r.next_int_bounded(7) == 5;
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

    /// `getCurrentSearcher()`, after the commit `doSearching` makes.
    fn current_reader(&self, r: &mut JavaRandom) -> Result<Arc<DirectoryReader>, String> {
        match self.mode {
            ReaderMode::Nrt => {
                // `TestNRTThreads`: reopen, or commit and open anew.
                if !r.next_boolean() {
                    self.commit()?;
                }
                DirectoryReader::open_from_writer(self.writer)
                    .map(Arc::new)
                    .map_err(text)
            }
            ReaderMode::Reopen => {
                // `TestSearcherManager`'s reopen thread commits, then
                // refreshes.
                self.commit()?;
                let mut current = lock(&self.current);
                let next = match current.as_ref() {
                    None => Some(DirectoryReader::open_from_writer(self.writer)),
                    Some(r) => r.open_if_changed_nrt(self.writer).transpose(),
                };
                if let Some(next) = next {
                    *current = Some(Arc::new(next.map_err(text)?));
                }
                Ok(Arc::clone(current.as_ref().expect("a reader")))
            }
        }
    }

    /// One search thread (`runSearchThreads`' `run`).
    fn search(&self, r: &mut JavaRandom) -> Result<(), String> {
        let mut iterations = 0;
        loop {
            iterations += 1;
            if iterations >= self.max_iterations || self.failed.load(Ordering::SeqCst) {
                break;
            }
            let reader = self.current_reader(r)?;
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
                if source == "merge" {
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
                self.search_body_terms(r, &reader)?;
            }
        }
        Ok(())
    }

    /// Samples about 30 of the `body` terms and searches each.
    fn search_body_terms(
        &self,
        r: &mut JavaRandom,
        reader: &DirectoryReader,
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
                if iters >= self.max_iterations {
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
        if iters < self.max_iterations {
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
    let owned = reader.field_norms_by_field(&fields);
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
    let mut random = JavaRandom::new(t.random().next_long());
    let docs = LineFileDocs::new(&mut random).expect("LineFileDocs");
    let tmp = TempDir::new(test_name);
    let dir = t.new_mock_fs_directory(tmp.path());
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
    let warmed = Arc::new(Mutex::new(HashSet::new()));
    let mut writer = writer_with(&dir, &config);
    writer.set_merged_segment_warmer(Some(Arc::new(Warmer(Arc::clone(&warmed)))));
    let writer = Mutex::new(writer);

    let nightly = t.nightly();
    // The `ExecutorService` coin: searches here run on their own threads.
    let _ = t.random().next_boolean();
    let num_index_threads = test_util::next_int(t.random(), 2, 4);
    let max_iterations = if nightly { 200 } else { 10 * t.multiplier() };
    let num_search_threads = if nightly {
        test_util::next_int(t.random(), 1, 5)
    } else {
        2
    };
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
    };
    let index_seeds: Vec<i64> = (0..num_index_threads)
        .map(|_| t.random().next_long())
        .collect();
    let search_seeds: Vec<i64> = (0..num_search_threads)
        .map(|_| t.random().next_long())
        .collect();
    let failures: Vec<String> = std::thread::scope(|scope| {
        let shared = &shared;
        let run = |seed: i64, search: bool| {
            move || {
                let mut r = JavaRandom::new(seed);
                let outcome = if search {
                    shared.search(&mut r)
                } else {
                    shared.index(&mut r)
                };
                if outcome.is_err() {
                    shared.failed.store(true, Ordering::SeqCst);
                }
                outcome
            }
        };
        let mut handles: Vec<_> = index_seeds
            .iter()
            .map(|&seed| scope.spawn(run(seed, false)))
            .collect();
        // Let the index build up a bit.
        std::thread::sleep(Duration::from_millis(100));
        handles.extend(
            search_seeds
                .iter()
                .map(|&seed| scope.spawn(run(seed, true))),
        );
        handles
            .into_iter()
            .filter_map(|h| match h.join() {
                Ok(Ok(())) => None,
                Ok(Err(e)) => Some(e),
                Err(_) => Some("a thread panicked".to_string()),
            })
            .collect()
    });
    assert!(failures.is_empty(), "{test_name}: {failures:?}");
    assert!(!shared.failed.load(Ordering::SeqCst));
    *lock(&shared.current) = None;

    let s = DirectoryReader::open_from_writer(&writer).expect("getFinalSearcher");
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
