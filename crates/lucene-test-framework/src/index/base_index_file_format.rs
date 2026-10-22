//! `BaseIndexFileFormatTestCase`: what every format suite runs, as
//! functions over an [`IndexFileFormatSuite`] (Java's abstract class, whose
//! subclass supplies `addRandomFields`), plus the helpers the h5 and h6
//! suites share.
//!
//! Differs:
//!
//! - `getCodec()`, `Codec.setDefault` and `applyCreatedVersionMajor` have
//!   nothing to choose: the writer always writes `Lucene104` at 10.5.0, so a
//!   suite is its documents, not its codec.
//! - `testMultiClose` is not ported: there is no codec-level writer or
//!   reader object here whose `close` could be called twice.
//! - `testCheckIntegrityReadsAllBytes`: this port's directories hand out
//!   whole files, so no wrapper can see which bytes `checkIntegrity` read.
//!   It is checked from the other side: each file of the segment in turn is
//!   damaged at one random byte, and opening the reader plus
//!   `check_integrity` must refuse it -- a byte nothing verifies would pass.
//! - `forceMerge`: this port's `force_merge` merges the committed segments
//!   only (Java's flushes first), so [`force_merge`] commits before it.
//! - `TestUtil.addIndexesSlowly(w, reader)` is [`add_indexes_slowly`]:
//!   `addIndexes(CodecReader...)` over the source directory's latest commit
//!   (the port's `add_indexes_merged`), not over arbitrary readers.
//! - `testRandomExceptions` is weaker: every injected error drops the
//!   writer and the next document opens a new one, as `examples/io_faults.rs`
//!   does. Java keeps writing through a writer a non-tragic `IOException`
//!   left open and reopens only on `AlreadyClosedException`, so it also
//!   tests a writer carrying on after a failed operation; this does not.
//!   An error that is not an injected one fails the test. The flush
//!   trigger's near-real-time branch (`DirectoryReader.open(iw)` plus
//!   `checkReader`) is a `flush` and a commit: an NRT reader needs the
//!   writer behind a `Mutex`, which would hide the drop-and-reopen. A
//!   reopened writer keeps the first configuration rather than drawing a
//!   new one.
//! - `newField`, `newStringField` and `newTextField` do not randomize the
//!   field type (`LuceneTestCase.newField`'s random term vectors): they are
//!   the plain constructors.

use lucene_search::reader::IndexReader as _;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use lucene_analysis::Analyzer;
use lucene_index::buffered_updates::Term;
use lucene_index::document::{Document, Store, StoredValue, StringField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::{IndexSortField, LuceneVersion};
use lucene_search::directory_reader::{DirectoryReader, SegmentReader};
use lucene_search::reader::{LeafReader, PostingsFlags, NO_MORE_DOCS};
use lucene_store::{ByteBuffersDirectory, Directory};
use lucene_util::java_random::JavaRandom;

use super::index_writer_config::{
    new_index_writer_config, new_tiered_merge_policy, Draws, LiveMergePolicy, PolicyKind,
    RandomIndexWriterConfig,
};
use super::random_index_writer::{RandomIndexWriter, WriterEnv};
use crate::analysis::mock_analyzer::MockAnalyzer;
use crate::store::mock_directory_wrapper::{
    is_injected_io_exception, MockDirectoryWrapper, Throttling,
};
use crate::util::lucene_test_case::LuceneTestCase;
use crate::util::test_util;

/// The version every writer here writes.
pub const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

/// One entry of a suite's `TESTS`: Java's method name and its port.
pub type SuiteTest = (&'static str, fn(&mut LuceneTestCase));

/// `BaseIndexFileFormatTestCase`'s abstract and overridable parts.
pub trait IndexFileFormatSuite {
    /// `addRandomFields(doc)`: fields exercising this suite's format.
    fn add_random_fields(&self, t: &mut LuceneTestCase, doc: &mut Document);

    /// `mergeIsStable()`.
    fn merge_is_stable(&self) -> bool {
        true
    }

    /// `excludedExtensionsFromByteCounts()`: `si` (its diagnostics carry a
    /// timestamp) and `lock`.
    fn excluded_extensions_from_byte_counts(&self) -> BTreeSet<&'static str> {
        BTreeSet::from(["si", "lock"])
    }
}

/// `new IndexWriter(dir, new IndexWriterConfig())`: the writer's defaults.
///
/// # Panics
/// When the writer cannot open.
pub fn plain_writer(dir: &dyn Directory) -> IndexWriter<'_> {
    IndexWriter::open(dir, Vec::new(), "Lucene104", VERSION).expect("open an IndexWriter")
}

/// `new MockAnalyzer(random())`.
pub fn mock_analyzer(t: &mut LuceneTestCase) -> Arc<Analyzer> {
    Arc::new(MockAnalyzer::new(t.random()).build())
}

/// `newIndexWriterConfig(analyzer)` on the test's stream.
pub fn new_config(t: &mut LuceneTestCase, analyzer: Arc<Analyzer>) -> RandomIndexWriterConfig {
    let (nightly, multiplier) = (t.nightly(), t.multiplier());
    new_index_writer_config(
        &mut Draws::new(t.random()).knobs(nightly, multiplier),
        analyzer,
        0,
    )
}

/// `new IndexWriter(dir, config)`.
///
/// # Panics
/// When the writer cannot open or refuses the configuration.
pub fn writer_with<'d>(
    dir: &'d dyn Directory,
    config: &RandomIndexWriterConfig,
) -> IndexWriter<'d> {
    let mut w = plain_writer(dir);
    config.apply(&mut w).expect("apply the configuration");
    w
}

/// `new IndexWriter(dir, newIndexWriterConfig(new MockAnalyzer(random())))`.
pub fn new_writer<'d>(t: &mut LuceneTestCase, dir: &'d dyn Directory) -> IndexWriter<'d> {
    let analyzer = mock_analyzer(t);
    let config = new_config(t, analyzer);
    writer_with(dir, &config)
}

/// The environment a `RandomIndexWriter` of this test draws its live
/// changes from: a stream split off the test's.
pub fn writer_env(t: &mut LuceneTestCase) -> WriterEnv {
    WriterEnv::new(t.random().next_long())
}

/// `new RandomIndexWriter(random(), dir)`.
///
/// # Panics
/// When the writer cannot open.
pub fn new_riw<'d>(t: &mut LuceneTestCase, dir: &'d dyn Directory) -> RandomIndexWriter<'d> {
    let env = writer_env(t);
    RandomIndexWriter::new(t.random(), dir, env).expect("open a RandomIndexWriter")
}

/// `new RandomIndexWriter(random(), dir, config)`.
///
/// # Panics
/// When the writer cannot open.
pub fn riw_with<'d>(
    t: &mut LuceneTestCase,
    dir: &'d dyn Directory,
    config: RandomIndexWriterConfig,
) -> RandomIndexWriter<'d> {
    let env = writer_env(t);
    RandomIndexWriter::with_config(t.random(), dir, config, env).expect("open a RandomIndexWriter")
}

/// `w.forceMerge(n)`: committed first, since this port's `force_merge`
/// reaches the committed segments only.
///
/// # Panics
/// When the writer fails.
pub fn force_merge(w: &mut IndexWriter<'_>, n: usize) {
    w.commit().expect("commit before forceMerge");
    w.force_merge(n).expect("forceMerge");
}

/// [`force_merge`] through a `RandomIndexWriter`.
///
/// # Panics
/// When the writer fails.
pub fn riw_force_merge(w: &mut RandomIndexWriter<'_>, n: usize) {
    w.commit().expect("commit before forceMerge");
    w.force_merge(n).expect("forceMerge");
}

/// `IndexWriterConfig.setIndexSort(sort)`: Java's sort may name fields no
/// document has yet, which this port's writer accepts once it takes
/// `Document`s (explicit documents, as every suite's writer does), so that
/// is switched on first.
///
/// # Panics
/// When the writer refuses the sort.
pub fn set_index_sort(w: &mut IndexWriter<'_>, sort: &[IndexSortField]) {
    w.enable_explicit_documents().expect("explicit documents");
    w.set_index_sort(Some(sort)).expect("setIndexSort");
}

/// `TestUtil.addIndexesSlowly(w, DirectoryReader.open(source))`: every live
/// document of the source's latest commit merged into one new segment.
///
/// # Panics
/// When the writer fails.
pub fn add_indexes_slowly(w: &mut IndexWriter<'_>, sources: &[&dyn Directory]) {
    w.add_indexes_merged(sources).expect("addIndexes");
}

/// `getOnlyLeafReader(reader)`.
///
/// # Panics
/// When the reader has more or fewer than one segment.
pub fn only_leaf(reader: &DirectoryReader) -> &SegmentReader {
    let leaves = reader.segment_readers();
    assert_eq!(leaves.len(), 1, "expected one segment");
    &leaves[0]
}

/// The segment holding global document `doc`, and its id there.
///
/// # Panics
/// When `doc` is outside the reader.
pub fn leaf_of(reader: &DirectoryReader, doc: i32) -> (&SegmentReader, i32) {
    let mut base = 0;
    for leaf in reader.segment_readers() {
        let max = leaf.max_doc();
        if doc < base + max {
            return (leaf, doc - base);
        }
        base += max;
    }
    panic!("doc {doc} is outside the reader (maxDoc={base})");
}

/// Whether global document `doc` is live.
pub fn is_live(reader: &DirectoryReader, doc: i32) -> bool {
    let (leaf, local) = leaf_of(reader, doc);
    leaf.live_docs().is_none_or(|l| l.get(local as usize))
}

/// The live documents holding `term` in `field`, by global id: what
/// `searcher.search(new TermQuery(new Term(field, term)), n)` finds.
///
/// # Panics
/// When the postings fail to read.
pub fn docs_with_term(reader: &DirectoryReader, field: &str, term: &[u8]) -> Vec<i32> {
    let mut hits = Vec::new();
    let mut base = 0;
    for leaf in reader.segment_readers() {
        if let Some(mut p) = leaf
            .postings(field, term, PostingsFlags::None)
            .expect("postings")
        {
            loop {
                let d = p.next_doc().expect("nextDoc");
                if d == NO_MORE_DOCS {
                    break;
                }
                if leaf.live_docs().is_none_or(|l| l.get(d as usize)) {
                    hits.push(base + d);
                }
            }
        }
        base += leaf.max_doc();
    }
    hits
}

/// `docID(reader, id)`: the one live document whose `id` is `id`.
///
/// # Panics
/// When there is not exactly one.
pub fn doc_with_id(reader: &DirectoryReader, id: &str) -> i32 {
    let hits = docs_with_term(reader, "id", id.as_bytes());
    assert_eq!(hits.len(), 1, "id {id}: {hits:?}");
    hits[0]
}

/// `storedFields.document(doc)`: global document `doc`'s stored fields, by
/// name, in stored order.
///
/// # Panics
/// When the stored fields fail to read.
pub fn stored(reader: &DirectoryReader, doc: i32) -> Vec<(String, StoredValue)> {
    let (leaf, local) = leaf_of(reader, doc);
    leaf_stored(leaf, local)
}

/// [`stored`] within one segment.
///
/// # Panics
/// When the stored fields fail to read.
pub fn leaf_stored(leaf: &SegmentReader, doc: i32) -> Vec<(String, StoredValue)> {
    let Some(d) = leaf.stored_document(doc).expect("stored fields") else {
        return Vec::new();
    };
    d.fields
        .into_iter()
        .map(|f| {
            let name = leaf
                .field_infos()
                .field_by_number(f.field_number)
                .map_or_else(|| format!("#{}", f.field_number), |i| i.name.clone());
            (name, f.value)
        })
        .collect()
}

/// `doc.get(name)`: the first stored string value named `name`.
pub fn get<'a>(fields: &'a [(String, StoredValue)], name: &str) -> Option<&'a StoredValue> {
    fields.iter().find(|(n, _)| n == name).map(|(_, v)| v)
}

/// Every stored value named `name` (`doc.getFields(name)`).
pub fn get_all<'a>(fields: &'a [(String, StoredValue)], name: &str) -> Vec<&'a StoredValue> {
    fields
        .iter()
        .filter(|(n, _)| n == name)
        .map(|(_, v)| v)
        .collect()
}

/// `IndexFileNames.CODEC_FILE_PATTERN`: `_[a-z0-9]+(_.*)?\..*`.
fn is_codec_file(name: &str) -> bool {
    let Some(rest) = name.strip_prefix('_') else {
        return false;
    };
    let gen_end = rest
        .find(|c: char| !(c.is_ascii_lowercase() || c.is_ascii_digit()))
        .unwrap_or(rest.len());
    gen_end > 0
        && rest[gen_end..].contains('.')
        && matches!(rest[gen_end..].chars().next(), Some('.' | '_'))
}

/// `bytesUsedByExtension(d)`.
fn bytes_used_by_extension(
    dir: &dyn Directory,
    excluded: &BTreeSet<&'static str>,
) -> BTreeMap<String, u64> {
    let mut used = BTreeMap::new();
    for file in dir.list_all().expect("listAll") {
        if !is_codec_file(&file) {
            continue;
        }
        let ext = file.rsplit_once('.').map_or("", |(_, e)| e).to_string();
        let len = dir.file_length(&file).expect("fileLength");
        *used.entry(ext).or_insert(0) += len;
    }
    used.retain(|ext, _| !excluded.contains(ext.as_str()));
    used
}

/// `newTieredMergePolicy()` with `setNoCFSRatio(0)`, the merge policy
/// `testMergeStability` installs.
fn tiered_no_cfs(t: &mut LuceneTestCase) -> Arc<LiveMergePolicy> {
    let (nightly, multiplier) = (t.nightly(), t.multiplier());
    let mut tmp = new_tiered_merge_policy(&mut Draws::new(t.random()).knobs(nightly, multiplier));
    use lucene_index::merge_policy::api::MergePolicy as _;
    tmp.compound_file_settings_mut()
        .set_no_cfs_ratio(0.0)
        .expect("noCFSRatio");
    Arc::new(LiveMergePolicy::new(PolicyKind::Tiered(tmp)))
}

/// `testMergeStability`: a force-merged index and its copy through
/// `addIndexes` hold the same bytes per extension.
pub fn test_merge_stability(suite: &dyn IndexFileFormatSuite, t: &mut LuceneTestCase) {
    if !suite.merge_is_stable() {
        return;
    }
    let dir = ByteBuffersDirectory::new();
    {
        let mut w = plain_writer(&dir);
        w.set_analyzer(Some(mock_analyzer(t)));
        w.set_use_compound_file(false);
        w.set_pluggable_merge_policy(Some(tiered_no_cfs(t)));
        let num_docs = t.at_least(500);
        for _ in 0..num_docs {
            let mut d = Document::new();
            suite.add_random_fields(t, &mut d);
            w.add_fields_document(&d).expect("addDocument");
        }
        force_merge(&mut w, 1);
        w.commit().expect("commit");
        w.close().expect("close");
    }
    let dir2 = ByteBuffersDirectory::new();
    {
        let mut w = plain_writer(&dir2);
        w.set_analyzer(Some(mock_analyzer(t)));
        w.set_use_compound_file(false);
        w.set_pluggable_merge_policy(Some(tiered_no_cfs(t)));
        add_indexes_slowly(&mut w, &[&dir]);
        w.commit().expect("commit");
        w.close().expect("close");
    }
    let excluded = suite.excluded_extensions_from_byte_counts();
    assert_eq!(
        bytes_used_by_extension(&dir, &excluded),
        bytes_used_by_extension(&dir2, &excluded)
    );
}

/// An error from `testRandomExceptions`' stream: an injected one drops the
/// writer; anything else fails the test.
fn expect_injected(e: &dyn std::fmt::Display, log: &mut Vec<String>) {
    let text = e.to_string();
    assert!(
        is_injected_io_exception(&text),
        "Unexpected exception: {text}\nfake-exception-log: {log:?}"
    );
    log.push(text);
}

/// `new IndexWriterConfig(analyzer)` with a `SerialMergeScheduler`, as
/// `testRandomExceptions` opens (and reopens) its writer; `None` when the
/// open hit an injected error.
fn exceptions_writer<'d>(
    dir: &'d MockDirectoryWrapper,
    config: &RandomIndexWriterConfig,
    log: &mut Vec<String>,
) -> Option<IndexWriter<'d>> {
    let rate = dir.random_io_exception_rate_on_open();
    dir.set_random_io_exception_rate_on_open(0.0);
    let opened = IndexWriter::open(dir, Vec::new(), "Lucene104", VERSION)
        .and_then(|mut w| config.apply(&mut w).map(|()| w));
    dir.set_random_io_exception_rate_on_open(rate);
    match opened {
        Ok(w) => Some(w),
        Err(e) => {
            expect_injected(&e, log);
            None
        }
    }
}

/// `testRandomExceptions`: documents through a writer whose directory fails
/// one write in a thousand and one open in fifty; the codec must neither
/// corrupt the index nor fail with anything but the injected errors.
/// Returns how many errors were injected (the run's evidence).
pub fn test_random_exceptions(suite: &dyn IndexFileFormatSuite, t: &mut LuceneTestCase) -> usize {
    let dir = MockDirectoryWrapper::new(t.random(), ByteBuffersDirectory::new());
    dir.set_throttling(Throttling::Never);
    dir.set_use_slow_open_closers(false);
    dir.set_random_io_exception_rate(0.001);
    let analyzer = mock_analyzer(t);
    let config = new_config(t, analyzer);
    let num_docs = t.at_least(500);
    let mut log = Vec::new();
    let mut w = exceptions_writer(&dir, &config, &mut log);
    for i in 0..num_docs {
        dir.set_random_io_exception_rate_on_open(0.02);
        let mut doc = Document::new();
        doc.add(StringField::new("id", i.to_string(), Store::No));
        suite.add_random_fields(t, &mut doc);
        let Some(iw) = w.as_mut() else {
            w = exceptions_writer(&dir, &config, &mut log);
            continue;
        };
        let added = iw.add_fields_document(&doc).and_then(|_| {
            iw.delete_documents_by_term(&[Term::new("id", i.to_string())])
                .map(drop)
        });
        if let Err(e) = added {
            expect_injected(&e, &mut log);
            w = None;
            continue;
        }
        if t.random().next_int_bounded(10) == 0 {
            let flushed = if t.random().next_boolean() {
                iw.flush().map_err(|e| e.to_string()).and_then(|()| {
                    dir.set_random_io_exception_rate_on_open(0.0);
                    iw.commit().map(drop).map_err(|e| e.to_string())
                })
            } else {
                dir.set_random_io_exception_rate_on_open(0.0);
                iw.commit().map(drop).map_err(|e| e.to_string())
            };
            match flushed {
                Ok(()) => {
                    let rate = dir.random_io_exception_rate();
                    dir.set_random_io_exception_rate(0.0);
                    if crate::store::base_directory_wrapper::index_exists(&dir)
                        .expect("indexExists")
                    {
                        test_util::check_index(&dir).expect("CheckIndex");
                    }
                    dir.set_random_io_exception_rate(rate);
                }
                Err(e) => {
                    expect_injected(&e, &mut log);
                    w = None;
                }
            }
        }
    }
    dir.set_random_io_exception_rate_on_open(0.0);
    if let Some(iw) = w.take() {
        if let Err(e) = iw.close() {
            expect_injected(&e, &mut log);
        }
    }
    dir.set_random_io_exception_rate(0.0);
    if crate::store::base_directory_wrapper::index_exists(&dir).expect("indexExists") {
        test_util::check_index(&dir).expect("CheckIndex after the run");
    }
    log.len()
}

/// `testCheckIntegrityReadsAllBytes`, from the other side (see the module
/// documentation): a force-merged index whose files, damaged one at a time
/// at a random byte, each make opening the reader or `checkIntegrity` fail.
pub fn test_check_integrity_reads_all_bytes(
    suite: &dyn IndexFileFormatSuite,
    t: &mut LuceneTestCase,
) {
    let dir = ByteBuffersDirectory::new();
    {
        let mut w = plain_writer(&dir);
        w.set_analyzer(Some(mock_analyzer(t)));
        let num_docs = t.at_least(100);
        for _ in 0..num_docs {
            let mut d = Document::new();
            suite.add_random_fields(t, &mut d);
            w.add_fields_document(&d).expect("addDocument");
        }
        force_merge(&mut w, 1);
        w.commit().expect("commit");
        w.close().expect("close");
    }
    let reader = DirectoryReader::open(&dir).expect("open");
    only_leaf(&reader)
        .check_integrity()
        .expect("checkIntegrity");
    let files = dir.list_all().expect("listAll");
    let mut damaged = 0;
    for victim in files.iter().filter(|f| is_codec_file(f)) {
        let copy = ByteBuffersDirectory::new();
        for file in &files {
            let mut bytes = read_all(&dir, file);
            if file == victim && !bytes.is_empty() {
                let at = t.random().next_int_bounded(bytes.len() as i32) as usize;
                bytes[at] ^= 1 << t.random().next_int_bounded(8);
            }
            write_all(&copy, file, &bytes);
        }
        let refused = DirectoryReader::open(&copy)
            .map_err(|e| e.to_string())
            .and_then(|r| {
                r.segment_readers()
                    .iter()
                    .try_for_each(|l| l.check_integrity().map_err(|e| e.to_string()))
            })
            .is_err();
        assert!(refused, "a damaged byte of {victim} went unnoticed");
        damaged += 1;
    }
    assert!(damaged > 0, "no codec file to damage: {files:?}");
}

fn read_all(dir: &dyn Directory, name: &str) -> Vec<u8> {
    let input = dir.open(name).expect("open");
    input.as_ref().to_vec()
}

fn write_all(dir: &dyn Directory, name: &str, bytes: &[u8]) {
    write_file(dir, name, bytes).expect("write");
}

/// A codec format's `write(dir, ...)` of `bytes` as `name`:
/// `createOutput`, the bytes, `close` -- each a point a directory's
/// [`crate::store::mock_directory_wrapper::Failure`] may fail.
///
/// # Errors
/// What the directory refuses.
pub fn write_file(dir: &dyn Directory, name: &str, bytes: &[u8]) -> lucene_store::Result<()> {
    use lucene_store::DataOutput as _;
    let mut out = dir.create_output(name)?;
    out.write_bytes(bytes);
    out.close().map(|_| ())
}

/// `randomByteArray(length, max)`: `length` bytes below `max`.
pub fn random_byte_array(r: &mut JavaRandom, length: usize, max: i32) -> Vec<u8> {
    (0..length).map(|_| r.next_int_bounded(max) as u8).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codec_file_pattern_is_javas() {
        for yes in ["_0.cfs", "_a1_Lucene104_0.doc", "_3_1.liv", "_z.si"] {
            assert!(is_codec_file(yes), "{yes}");
        }
        for no in ["segments_1", "write.lock", "_.si", "_A.si", "_0", "x_0.si"] {
            assert!(!is_codec_file(no), "{no}");
        }
    }
}
