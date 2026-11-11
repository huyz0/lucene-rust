//! Cold queries over an object store read by range through the page cache:
//! how many requests each query sends and how many dependent round trips
//! (waves) it waits for.
//!
//! The index is written to an in-memory blob store, then read through a
//! [`CachedBlobDirectory`] over a store that sleeps a fixed latency on every
//! ranged `GET`, plus the transfer time at a per-request throughput. Each query runs against a fresh, empty cache after the
//! reader has opened (the pinned hot files already local), so the numbers
//! are the cost of a first query on a newly opened shard. Waves are elapsed
//! time over the injected latency: requests sent in parallel overlap and
//! count once.
//!
//! ```text
//! cargo run --release -p lucene-search --example cold_range_reads -- [docs] [latency_ms] [MB/s per request] [--no-prefetch] [--no-listed-lengths]
//! ```

#![allow(clippy::arithmetic_side_effects)]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use lucene_index::document::{Document, Store, StringField, TextField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
use lucene_search::query::{BooleanQuery, Clause, PhraseQuery, TermQuery};
use lucene_store::{
    BlobDirectory, BlobStore, CachedBlobDirectory, InMemoryBlobStore, Input, PageCache,
    PageCacheConfig, Result,
};
use lucene_util::test_support::TempDir;

const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

/// A blob store that sleeps on every ranged read for its latency plus the
/// transfer at a per-request throughput, and on every length request and
/// listing for its latency, and counts the requests by file extension
/// (`HEAD` and `LIST` for those two).
#[derive(Clone)]
struct SlowStore {
    inner: InMemoryBlobStore,
    latency: Duration,
    bytes_per_sec: f64,
    /// Whether the listing carries lengths (`--no-listed-lengths` turns it
    /// off, as for a store whose listing does not).
    lengths_listed: bool,
    requests: Arc<Mutex<BTreeMap<String, (u64, u64)>>>,
}

impl SlowStore {
    fn take_counts(&self) -> BTreeMap<String, (u64, u64)> {
        std::mem::take(&mut *self.requests.lock().unwrap())
    }

    fn count(&self, kind: String, bytes: usize) {
        let mut counts = self.requests.lock().unwrap();
        let entry = counts.entry(kind).or_default();
        entry.0 += 1;
        entry.1 += bytes as u64;
    }
}

impl BlobStore for SlowStore {
    fn list(&self) -> Result<Vec<String>> {
        self.count("LIST".into(), 0);
        std::thread::sleep(self.latency);
        self.inner.list()
    }
    fn get(&self, name: &str) -> Result<Input> {
        self.inner.get(name)
    }
    fn get_range(&self, name: &str, offset: u64, len: usize) -> Result<Vec<u8>> {
        let ext = name.rsplit('.').next().unwrap_or(name).to_string();
        let bytes = self.inner.get_range(name, offset, len)?;
        self.count(ext, bytes.len());
        let transfer = Duration::from_secs_f64(bytes.len() as f64 / self.bytes_per_sec);
        std::thread::sleep(self.latency + transfer);
        Ok(bytes)
    }
    fn size(&self, name: &str) -> Result<u64> {
        self.count("HEAD".into(), 0);
        std::thread::sleep(self.latency);
        self.inner.size(name)
    }
    fn list_with_sizes(&self) -> Result<Option<Vec<(String, u64)>>> {
        if self.lengths_listed {
            self.count("LIST".into(), 0);
            std::thread::sleep(self.latency);
            return self.inner.list_with_sizes();
        }
        Ok(None)
    }
    fn put(&self, name: &str, bytes: Vec<u8>) -> Result<()> {
        self.inner.put(name, bytes)
    }
    fn delete(&self, name: &str) -> Result<()> {
        self.inner.delete(name)
    }
}

/// A deterministic Zipf-ish word: rank `r` of a 20k-word vocabulary.
fn word(state: &mut u64) -> String {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    let u = (*state >> 11) as f64 / (1u64 << 53) as f64;
    let rank = (20_000f64.powf(u)) as u64;
    format!("w{rank}")
}

fn build(store: &InMemoryBlobStore, docs: usize) {
    let dir = BlobDirectory::new(store.clone());
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.set_ram_buffer_size_mb(64.0).unwrap();
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    for i in 0..docs {
        let mut d = Document::new();
        d.add(StringField::new("id", format!("doc-{i:07}"), Store::Yes));
        let body: Vec<String> = (0..40).map(|_| word(&mut state)).collect();
        d.add(TextField::new("body", body.join(" "), Store::Yes));
        w.add_fields_document(&d).unwrap();
    }
    w.commit().unwrap();
    w.force_merge(1).unwrap();
    w.commit().unwrap();
}

fn term(t: &str) -> Clause {
    Clause::Term(TermQuery::new("body", t.as_bytes().to_vec()))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let docs: usize = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(200_000);
    let latency = Duration::from_millis(args.get(2).and_then(|a| a.parse().ok()).unwrap_or(30));
    let mb_per_sec: f64 = args.get(3).and_then(|a| a.parse().ok()).unwrap_or(100.0);
    // `--no-prefetch`: fetch the hits one by one, as without the hint.
    let prefetch = !args.iter().any(|a| a == "--no-prefetch");

    let inner = InMemoryBlobStore::new();
    let t = Instant::now();
    build(&inner, docs);
    let mut sizes: Vec<(String, u64)> = inner
        .list()
        .unwrap()
        .into_iter()
        .map(|n| {
            let len = inner.size(&n).unwrap();
            (n, len)
        })
        .collect();
    sizes.sort();
    eprintln!("indexed {docs} docs in {:.1?}", t.elapsed());
    for (n, len) in &sizes {
        eprintln!("  {n:<24} {len:>12}");
    }

    let store = SlowStore {
        inner,
        latency,
        bytes_per_sec: mb_per_sec * 1e6,
        lengths_listed: !args.iter().any(|a| a == "--no-listed-lengths"),
        requests: Arc::default(),
    };
    let queries: Vec<(&str, BooleanQuery)> = vec![
        (
            "term (common)",
            BooleanQuery {
                must: vec![term("w3")],
                ..BooleanQuery::default()
            },
        ),
        (
            "term (rare)",
            BooleanQuery {
                must: vec![term("w4000")],
                ..BooleanQuery::default()
            },
        ),
        (
            "and of 2",
            BooleanQuery {
                must: vec![term("w20"), term("w50")],
                ..BooleanQuery::default()
            },
        ),
        (
            "or of 3",
            BooleanQuery {
                should: vec![term("w100"), term("w700"), term("w3000")],
                ..BooleanQuery::default()
            },
        ),
        (
            "phrase",
            BooleanQuery {
                must: vec![Clause::Phrase(PhraseQuery::new("body", ["w1", "w2"]))],
                ..BooleanQuery::default()
            },
        ),
    ];

    println!(
        "latency {latency:?}, {mb_per_sec} MB/s per request; per phase: requests, MiB fetched, \
         elapsed in round trips (elapsed / latency)"
    );
    println!(
        "{:<14} {:>6} | {:>5} {:>6} {:>5} | {:>5} {:>6} {:>5} | {:>5} {:>6} {:>5}",
        "query", "hits", "open", "MiB", "rt", "srch", "MiB", "rt", "fetch", "MiB", "rt"
    );
    for (label, query) in &queries {
        let tmp = TempDir::new("cold-range-reads");
        let cache = Arc::new(
            PageCache::new(PageCacheConfig {
                disk_dir: Some(tmp.path().join("nvme")),
                ..PageCacheConfig::default()
            })
            .unwrap(),
        );
        let dir = CachedBlobDirectory::new(store.clone(), cache, "idx");
        let t = Instant::now();
        let reader = DirectoryReader::open(&dir).unwrap();
        let opened = reader.open_segments().unwrap();
        let segments = opened.as_open_segments();
        let owned = reader.field_norms_by_field(&["body".to_string()]).unwrap();
        let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
        let searcher = IndexSearcher::new(&segments, &norms).unwrap();
        let open_time = t.elapsed();
        let open_counts = store.take_counts();

        let t = Instant::now();
        let top = searcher.search(query, 10).unwrap();
        let search_time = t.elapsed();
        let search_counts = store.take_counts();

        let t = Instant::now();
        let leaves = reader.segment_readers();
        if prefetch {
            let mut base = 0;
            for seg in leaves {
                let docs: Vec<i32> = top
                    .score_docs
                    .iter()
                    .filter(|h| h.doc >= base && h.doc - base < seg.max_doc)
                    .map(|h| h.doc - base)
                    .collect();
                seg.prefetch_stored_documents(&docs).unwrap();
                base += seg.max_doc;
            }
        }
        for hit in &top.score_docs {
            let mut doc = hit.doc;
            for seg in leaves {
                if doc < seg.max_doc {
                    seg.stored_document(doc).unwrap();
                    break;
                }
                doc -= seg.max_doc;
            }
        }
        let fetch_time = t.elapsed();
        let fetch_counts = store.take_counts();

        let phase = |c: &BTreeMap<String, (u64, u64)>, d: Duration| {
            let (reqs, bytes) = c
                .values()
                .fold((0, 0), |(r, b), &(cr, cb)| (r + cr, b + cb));
            format!(
                "{:>5} {:>6.1} {:>5.1}",
                reqs,
                bytes as f64 / (1 << 20) as f64,
                d.as_secs_f64() / latency.as_secs_f64()
            )
        };
        let by_file: Vec<String> = search_counts
            .iter()
            .chain(fetch_counts.iter())
            .map(|(k, (r, _))| format!("{k}:{r}"))
            .collect();
        println!(
            "{:<14} {:>6} | {} | {} | {}  {}",
            label,
            top.total_hits.value,
            phase(&open_counts, open_time),
            phase(&search_counts, search_time),
            phase(&fetch_counts, fetch_time),
            by_file.join(" "),
        );
    }
}
