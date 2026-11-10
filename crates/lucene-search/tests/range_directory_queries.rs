//! Real DirectoryReader/IndexSearcher range queries over native-written files.
//! The source is an immutable file snapshot, not a provider/network claim.
#![allow(clippy::arithmetic_side_effects)]
use lucene_codecs::stored_fields::FieldValue;
use lucene_index::buffered_updates::Term;
use lucene_index::document::{
    Document, Field, NumericDocValuesField, SortedDocValuesField, Store, StoredValue, StringField,
};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::index_searcher::IndexSearcher;
use lucene_search::query::{BooleanQuery, TermQuery};
use lucene_search::top_field::{SortField, SortType};
use lucene_store::range_input::{RangeDirectory, RangeInput, RangeLimits, ReadAt};
use lucene_store::{Directory, EstimatedWrites, MmapDirectory};
use lucene_util::test_support::TempDir;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug)]
struct Event {
    phase: &'static str,
    file: String,
    offset: usize,
    requested: usize,
    received: usize,
    failed: bool,
    elapsed_micros: u128,
}
struct Trace {
    phase: &'static str,
    events: Vec<Event>,
    requested: usize,
    fault: Option<(&'static str, bool)>,
}
impl Trace {
    fn new() -> Self {
        Self {
            phase: "prepare",
            events: Vec::new(),
            requested: 0,
            fault: None,
        }
    }
}
struct Source {
    file: Mutex<File>,
    name: String,
    length: usize,
    trace: Arc<Mutex<Trace>>,
}
impl ReadAt for Source {
    fn len(&self) -> usize {
        self.length
    }
    fn read_at(&self, at: usize, out: &mut [u8]) -> lucene_store::Result<usize> {
        let mut trace = self.trace.lock().unwrap();
        if trace.events.len() >= 32768
            || trace
                .requested
                .checked_add(out.len())
                .is_none_or(|n| n > 16 * 1024 * 1024)
        {
            return Err(std::io::Error::other("physical trace/byte envelope exhausted").into());
        }
        trace.requested += out.len();
        let fault = trace
            .fault
            .filter(|(suffix, _)| self.name.ends_with(suffix));
        let started = std::time::Instant::now();
        let mut received = 0;
        let result = if let Some((_, short)) = fault {
            if short {
                Ok(0)
            } else {
                Err(std::io::Error::other("injected range error"))
            }
        } else {
            let mut file = self.file.lock().unwrap();
            file.seek(SeekFrom::Start(at as u64))
                .and_then(|_| file.read(out))
        };
        if let Ok(n) = result {
            received = n;
        }
        let phase = trace.phase;
        trace.events.push(Event {
            phase,
            file: self.name.clone(),
            offset: at,
            requested: out.len(),
            received,
            failed: result.is_err() || received != out.len(),
            elapsed_micros: started.elapsed().as_micros(),
        });
        Ok(result?)
    }
}
fn inputs(root: &Path, names: &[String], trace: &Arc<Mutex<Trace>>) -> Vec<(String, RangeInput)> {
    names
        .iter()
        .map(|name| {
            let file = File::open(root.join(name)).unwrap();
            let length = usize::try_from(file.metadata().unwrap().len()).unwrap();
            let source = Arc::new(Source {
                file: Mutex::new(file),
                name: name.clone(),
                length,
                trace: trace.clone(),
            });
            let input = RangeInput::new(
                source,
                RangeLimits {
                    block_bytes: 256,
                    cache_blocks: 2,
                    attempts: 32768,
                    transferred_bytes: 16 * 1024 * 1024,
                },
            )
            .unwrap();
            (name.clone(), input)
        })
        .collect()
}
fn corpus(root: &Path) -> Vec<String> {
    corpus_profile(root, false, 384, false)
}
fn corpus_profile(root: &Path, compound: bool, docs: i32, sorted_values: bool) -> Vec<String> {
    let dir = MmapDirectory::with_read_threshold(root, 0);
    let mut writer = IndexWriter::open(
        &dir,
        Vec::new(),
        "Lucene104",
        LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
    )
    .unwrap();
    writer.set_use_compound_file(compound);
    writer.set_max_buffered_docs(8192).unwrap();
    writer.set_ram_buffer_size_mb(32.0).unwrap();
    for i in 0..docs {
        let mut d = Document::new();
        d.add(StringField::new("id", format!("id{i:04}"), Store::Yes));
        d.add(StringField::new("body", "needle", Store::No));
        d.add(StringField::new("unique", format!("u{i:04}"), Store::No));
        d.add(StringField::new(
            "tag",
            if i % 2 == 0 { "even" } else { "odd" },
            Store::No,
        ));
        if sorted_values {
            d.add(SortedDocValuesField::new("unsupported", b"value".to_vec()));
        }
        if i % 7 != 0 {
            d.add(NumericDocValuesField::new("rank", i64::from(383 - i)));
        }
        let payload: String = (0..512)
            .map(|j| char::from(b'a' + ((i * 37 + j * 17 + j * j) % 26) as u8))
            .collect();
        d.add(Field::stored("payload", StoredValue::String(payload)));
        writer.add_fields_document(&d).unwrap();
    }
    writer.commit().unwrap();
    writer
        .delete_documents_by_term(&[Term::new("id", b"id0007".to_vec())])
        .unwrap();
    writer.commit().unwrap();
    writer.set_commit_on_close(false);
    writer.close().unwrap();
    let names = dir.list_all().unwrap();
    assert!(
        names.iter().any(|n| n.ends_with(".liv")),
        "deletions must touch real live-docs format"
    );
    names
}
fn query(filtered: bool) -> BooleanQuery {
    let q = BooleanQuery::default().with_must([TermQuery::new("body", "needle")]);
    if filtered {
        q.with_filter([TermQuery::new("tag", "even")])
    } else {
        q
    }
}
fn expected(filtered: bool) -> Vec<(i32, i64)> {
    let mut docs: Vec<_> = (0..384)
        .filter(|i| *i != 7 && (!filtered || i % 2 == 0))
        .map(|i| {
            (
                i,
                if i % 7 == 0 {
                    i64::MAX
                } else {
                    i64::from(383 - i)
                },
            )
        })
        .collect();
    docs.sort_by_key(|&(doc, value)| (value, doc));
    docs.truncate(32);
    docs
}
fn execute(
    dir: &dyn Directory,
    filtered: bool,
    hook: impl FnOnce(),
    projection: bool,
) -> lucene_search::Result<Vec<(i32, i64)>> {
    let reader = DirectoryReader::open(dir).map_err(lucene_search::Error::from)?;
    assert_eq!(reader.segment_readers().len(), 1);
    let mut opened = reader.open_segments().map_err(lucene_search::Error::from)?;
    opened.open_points().map_err(lucene_search::Error::from)?;
    let segments = opened.as_open_segments();
    let norms = vec![None; segments.len()];
    let searcher = IndexSearcher::new(&segments, &norms)?;
    hook();
    let mut sort = SortField::numeric("rank", SortType::Long, false);
    sort.missing = i64::MAX;
    let result = searcher.search_sorted(
        reader.segment_readers(),
        &query(filtered),
        32,
        &[sort],
        None,
    )?;
    assert_eq!(result.total.value, if filtered { 192 } else { 383 });
    if projection {
        for hit in &result.hits {
            let document = reader.segment_readers()[0]
                .stored_document(hit.doc)
                .map_err(lucene_search::Error::from)?
                .unwrap();
            let strings: Vec<_> = document
                .fields
                .iter()
                .filter_map(|f| match &f.value {
                    FieldValue::String(s) => Some(s.as_str()),
                    _ => None,
                })
                .collect();
            assert!(strings.contains(&format!("id{:04}", hit.doc).as_str()));
            assert!(strings.iter().any(|s| s.len() == 512));
        }
    }
    Ok(result
        .hits
        .into_iter()
        .map(|hit| (hit.doc, hit.values[0]))
        .collect())
}
fn clear(files: &[(String, RangeInput)]) {
    for (_, input) in files {
        input.clear_cache().unwrap();
    }
}

#[test]
fn directory_reader_queries_are_range_native_cold_bounded_and_fail_closed() {
    let source_root = TempDir::new("range-query-source");
    let names = corpus(source_root.path());
    let trace = Arc::new(Mutex::new(Trace::new()));
    let files = inputs(source_root.path(), &names, &trace);
    let data_bytes: usize = files
        .iter()
        .filter(|(n, _)| {
            [".tim", ".doc", ".dvd", ".fdt"]
                .iter()
                .any(|e| n.ends_with(e))
        })
        .map(|(_, i)| i.len())
        .sum();
    let data_cache = files
        .iter()
        .filter(|(n, _)| {
            [".tim", ".doc", ".dvd", ".fdt"]
                .iter()
                .any(|e| n.ends_with(e))
        })
        .count()
        * 512;
    assert!(
        data_bytes > data_cache * 2,
        "source data inventory must exceed cache"
    );
    let scratch = TempDir::new("range-query-metadata");
    let dir = RangeDirectory::new(scratch.path(), files.clone()).unwrap();
    let concrete = dir;
    let wrapped: Arc<&dyn Directory> = Arc::new(&concrete);
    let dir = EstimatedWrites::new(&wrapped, 0);
    assert!(dir.is_range_directory());
    assert!(dir
        .open_range(
            files
                .iter()
                .find(|(n, _)| n.ends_with(".doc"))
                .unwrap()
                .0
                .as_str()
        )
        .unwrap()
        .is_some());
    for filtered in [false, true] {
        clear(&files);
        trace.lock().unwrap().phase = "cold-query";
        assert_eq!(
            execute(&dir, filtered, || clear(&files), true).unwrap(),
            expected(filtered)
        );
    }
    // Faults happen after native reader/query preparation; each selected data
    // family must propagate real source errors and short replies as failure.
    for suffix in [".tim", ".doc", ".dvd", ".fdt"] {
        for short in [false, true] {
            trace.lock().unwrap().fault = None;
            clear(&files);
            trace.lock().unwrap().phase = "fault-query";
            let result = execute(
                &dir,
                true,
                || {
                    clear(&files);
                    trace.lock().unwrap().fault = Some((suffix, short));
                },
                true,
            );
            assert!(
                result.is_err(),
                "{suffix}, short={short} must not become successful/empty query"
            );
            trace.lock().unwrap().fault = None;
        }
    }
    clear(&files);
    trace.lock().unwrap().phase = "integrity";
    let reader = DirectoryReader::open(&dir).unwrap();
    for segment in reader.segment_readers() {
        segment.check_integrity().unwrap();
    }
    // A separate full streaming materialization, measured independently.
    let baseline_trace = Arc::new(Mutex::new(Trace::new()));
    baseline_trace.lock().unwrap().phase = "materialized-baseline";
    let baseline_files = inputs(source_root.path(), &names, &baseline_trace);
    let baseline_root = TempDir::new("range-query-baseline");
    let mut total = 0usize;
    for (name, input) in &baseline_files {
        let mut out = File::create(baseline_root.path().join(name)).unwrap();
        let mut buffer = [0u8; 4096];
        let mut at = 0;
        while at < input.len() {
            let count = (input.len() - at).min(buffer.len());
            input.read_exact_at(at, &mut buffer[..count]).unwrap();
            out.write_all(&buffer[..count]).unwrap();
            at += count;
        }
        total += input.len();
    }
    let baseline = MmapDirectory::with_read_threshold(baseline_root.path(), 0);
    for filtered in [false, true] {
        assert_eq!(
            execute(&baseline, filtered, || {}, true).unwrap(),
            expected(filtered)
        );
    }
    let baseline_actual: usize = baseline_trace
        .lock()
        .unwrap()
        .events
        .iter()
        .map(|e| e.received)
        .sum();
    assert_eq!(baseline_actual, total);
    let events = &trace.lock().unwrap().events;
    // Retain a complete finite file/offset/request/response/outcome trace.
    let mut report =
        File::create(std::env::temp_dir().join("range-query-physical-trace.tsv")).unwrap();
    let baseline_guard = baseline_trace.lock().unwrap();
    for e in events.iter().chain(baseline_guard.events.iter()) {
        writeln!(
            report,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}",
            e.phase, e.file, e.offset, e.requested, e.received, e.failed, e.elapsed_micros
        )
        .unwrap();
    }
    let touched: BTreeSet<_> = events.iter().map(|e| e.file.clone()).collect();
    for name in &touched {
        assert!(
            name.starts_with("segments_")
                || [
                    ".si", ".fnm", ".tim", ".tip", ".tmd", ".doc", ".dvm", ".dvd", ".fdm", ".fdx",
                    ".fdt", ".liv"
                ]
                .iter()
                .any(|suffix| name.ends_with(suffix)),
            "unexpected touched reader format {name}"
        );
    }
    for phase in ["prepare", "cold-query", "fault-query", "integrity"] {
        let selected: Vec<_> = events.iter().filter(|e| e.phase == phase).collect();
        let requested: usize = selected.iter().map(|e| e.requested).sum();
        let received: usize = selected.iter().map(|e| e.received).sum();
        println!(
            "RANGE_PHASE phase={phase} attempts={} requested={requested} received={received}",
            selected.len()
        );
    }
    for suffix in [
        ".si", ".fnm", ".tim", ".tip", ".tmd", ".doc", ".dvm", ".dvd", ".fdm", ".fdx", ".fdt",
        ".liv", ".psm",
    ] {
        assert!(
            touched.iter().any(|n| n.ends_with(suffix)),
            "missing native reader trace {suffix}"
        );
    }
    let mut pages = BTreeMap::<String, BTreeSet<usize>>::new();
    for e in events.iter().filter(|e| e.phase == "cold-query") {
        pages.entry(e.file.clone()).or_default().insert(e.offset);
    }
    assert!(pages
        .iter()
        .any(|(name, pages)| name.ends_with(".fdt") && pages.len() > 2));
    let mut working_set = BTreeMap::new();
    for e in events.iter().filter(|e| {
        e.phase == "cold-query"
            && [".tim", ".doc", ".dvd", ".fdt"]
                .iter()
                .any(|suffix| e.file.ends_with(suffix))
    }) {
        working_set.insert((e.file.as_str(), e.offset), e.requested);
    }
    let accessed_data_bytes: usize = working_set.values().sum();
    assert!(
        accessed_data_bytes > data_cache,
        "actually accessed unique data blocks must exceed aggregate cache"
    );
    let sampled_cache_bytes: usize = files
        .iter()
        .map(|(_, i)| i.usage().unwrap().cache_bytes)
        .sum();
    assert!(events.iter().any(|e| e.failed));
    assert!(files
        .iter()
        .all(|(_, i)| i.usage().unwrap().cache_bytes <= 512));
    assert!(concrete.metadata_usage().unwrap().reserved_bytes <= 16 * 1024 * 1024);
    let physical: usize = events.iter().map(|e| e.requested).sum();
    println!("RANGE_QUERY data_bytes={data_bytes} aggregate_data_cache={data_cache} accessed_data_bytes={accessed_data_bytes} sampled_all_file_cache_bytes={sampled_cache_bytes} physical_range_bytes={physical} materialized_transfer_bytes={baseline_actual} baseline_scratch_bytes={total} trace_events={} metadata_scratch_bytes={}",events.len(),concrete.metadata_usage().unwrap().reserved_bytes);
}

#[test]
fn unadvertised_range_profiles_refuse_before_query_results() {
    for (compound, docs, sorted_values, diagnostic) in [
        (true, 384, false, "non-compound"),
        (false, 4097, false, "bounded"),
        (false, 384, true, "single-valued numeric"),
    ] {
        let source_root = TempDir::new("range-profile-source");
        let names = corpus_profile(source_root.path(), compound, docs, sorted_values);
        // These are valid materialized indexes. Refusal belongs specifically
        // to the declared range profile, not to corrupt test bytes.
        let resident = MmapDirectory::with_read_threshold(source_root.path(), 0);
        let reader = DirectoryReader::open(&resident).unwrap();
        assert_eq!(reader.segment_readers().len(), 1);
        drop(reader);
        let trace = Arc::new(Mutex::new(Trace::new()));
        let files = inputs(source_root.path(), &names, &trace);
        let scratch = TempDir::new("range-profile-metadata");
        let dir = RangeDirectory::new(scratch.path(), files).unwrap();
        let error = match DirectoryReader::open(&dir) {
            Err(error) => error.to_string(),
            Ok(_) => panic!("unadvertised range profile opened successfully"),
        };
        assert!(error.contains(diagnostic), "{error}");
        assert!(
            !trace
                .lock()
                .unwrap()
                .events
                .iter()
                .any(|e| e.file.ends_with(".cfs")),
            "compound refusal must not materialize/read the archive"
        );
    }
}
