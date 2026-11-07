//! Memory and latency of a local index read through `mmap`
//! ([`MmapDirectory`]) against the same index read through the demand pager
//! ([`PagedFsDirectory`]), for full-text queries with a stored-fields fetch.
//!
//! ```text
//! cargo run --release -p lucene-search --example local_store -- build <dir> [docs]
//! cargo run --release -p lucene-search --example local_store -- query <dir> mmap|paged \
//!     [--budget-mb N] [--page-kb N] [--queries N] [--passes N] [--threads N] [--buffered] [--readahead-kb N] [--fixed]
//!     [--tenants N --hot N --hot-share F]
//! cargo run --release -p lucene-search --example local_store -- build-tenants <dir> [docs] [tenants]
//! ```
//!
//! With `--tenants`, the queries are a tenant filter plus body terms over an
//! index built by `build-tenants` (each tenant one doc-id run); `hot`
//! tenants take `hot-share` of them, and the latency is also split by hot
//! and cold tenants.
//!
//! `query` opens the index, runs the same query list `passes` times and
//! prints, per pass, latency percentiles and the memory the process holds:
//! anonymous and file-backed RSS, the pager's resident bytes, bytes read
//! from the device, and how much of the index the kernel's page cache holds
//! (`mincore`). Drop the page cache before a cold run
//! (`echo 3 > /proc/sys/vm/drop_caches`); run it in a memory cgroup to
//! compare both directories under the same limit.

#![allow(clippy::arithmetic_side_effects)]

use std::time::{Duration, Instant};

use lucene_index::document::{Document, Store, StringField, TextField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
use lucene_search::query::{BooleanQuery, Clause, PhraseQuery, TermQuery};
use lucene_store::{Directory, FsDirectory, MmapDirectory, PagedFsConfig, PagedFsDirectory};

const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

fn next(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 11) as f64 / (1u64 << 53) as f64
}

/// A Zipf-ish word of a 200k-word vocabulary.
fn word(state: &mut u64) -> String {
    let rank = (200_000f64.powf(next(state))) as u64;
    format!("w{rank}")
}

fn build(path: &str, docs: usize) {
    let dir = FsDirectory::open(path);
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.set_ram_buffer_size_mb(256.0).unwrap();
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let t = Instant::now();
    for i in 0..docs {
        let mut d = Document::new();
        d.add(StringField::new("id", format!("doc-{i:08}"), Store::Yes));
        let body: Vec<String> = (0..60).map(|_| word(&mut state)).collect();
        d.add(TextField::new("body", body.join(" "), Store::Yes));
        w.add_fields_document(&d).unwrap();
        if i % 500_000 == 0 {
            eprintln!("{i} docs, {:.0?}", t.elapsed());
        }
    }
    w.commit().unwrap();
    w.force_merge(1).unwrap();
    w.commit().unwrap();
    let mut total = 0u64;
    for name in dir.list_all().unwrap() {
        let len = dir.file_length(&name).unwrap();
        total += len;
        eprintln!("  {name:<24} {:>10.1} MiB", len as f64 / (1 << 20) as f64);
    }
    eprintln!(
        "indexed {docs} docs in {:.1?}, {:.0} MiB",
        t.elapsed(),
        total as f64 / (1 << 20) as f64
    );
}

/// Tenants' sizes: Zipf over `tenants`, `docs` in all.
fn tenant_sizes(tenants: usize, docs: usize) -> Vec<usize> {
    let weights: Vec<f64> = (1..=tenants).map(|r| 1.0 / (r as f64).powf(1.1)).collect();
    let sum: f64 = weights.iter().sum();
    weights
        .iter()
        .map(|w| ((w / sum) * docs as f64).ceil() as usize)
        .collect()
}

/// An index of `docs` documents of `tenants` tenants added tenant by tenant,
/// so each tenant's documents are one doc-id run, as an index sorted by
/// tenant has them.
fn build_tenants(path: &str, docs: usize, tenants: usize) {
    let dir = FsDirectory::open(path);
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.set_ram_buffer_size_mb(256.0).unwrap();
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let t = Instant::now();
    let mut i = 0usize;
    for (tenant, size) in tenant_sizes(tenants, docs).into_iter().enumerate() {
        for _ in 0..size {
            let mut d = Document::new();
            d.add(StringField::new("id", format!("doc-{i:08}"), Store::Yes));
            d.add(StringField::new("tenant", format!("t{tenant}"), Store::No));
            let body: Vec<String> = (0..60).map(|_| word(&mut state)).collect();
            d.add(TextField::new("body", body.join(" "), Store::Yes));
            w.add_fields_document(&d).unwrap();
            i += 1;
        }
    }
    w.commit().unwrap();
    w.force_merge(1).unwrap();
    w.commit().unwrap();
    eprintln!(
        "indexed {i} docs of {tenants} tenants in {:.1?}",
        t.elapsed()
    );
}

/// Queries of tenants picked with a skew: `hot` tenants (spread over every
/// size class) take `hot_share` of the queries, the rest go uniformly to
/// the others. Each is a tenant filter and one or two body terms.
fn tenant_queries(
    n: usize,
    tenants: usize,
    hot: usize,
    hot_share: f64,
) -> Vec<(bool, BooleanQuery)> {
    let mut state = 0x5151_2323_7777_abcdu64;
    let stride = (tenants / hot.max(1)).max(1);
    (0..n)
        .map(|i| {
            let is_hot = next(&mut state) < hot_share;
            let tenant = if is_hot {
                ((next(&mut state) * hot as f64) as usize % hot) * stride
            } else {
                loop {
                    let t = (next(&mut state) * tenants as f64) as usize % tenants;
                    if !t.is_multiple_of(stride) || t / stride >= hot {
                        break t;
                    }
                }
            };
            let rank = 10.0 + 5000.0 * next(&mut state).powi(2);
            let mut must = vec![term(&format!("w{}", rank as u64))];
            if i % 2 == 0 {
                must.push(term(&format!("w{}", 1 + i % 50)));
            }
            (
                is_hot,
                BooleanQuery {
                    must,
                    filter: vec![Clause::Term(TermQuery::new(
                        "tenant",
                        format!("t{tenant}").into_bytes(),
                    ))],
                    ..BooleanQuery::default()
                },
            )
        })
        .collect()
}

fn term(t: &str) -> Clause {
    Clause::Term(TermQuery::new("body", t.as_bytes().to_vec()))
}

/// A mix of term, conjunction, disjunction and phrase queries over words of
/// every frequency, the same list for every run.
fn queries(n: usize) -> Vec<BooleanQuery> {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    // Query words skew rarer than the text: most of a real query log.
    let qword = |s: &mut u64| {
        let rank = 10.0 + (20_000f64 - 10.0) * next(s).powi(2);
        format!("w{}", rank as u64)
    };
    (0..n)
        .map(|i| match i % 5 {
            0 => BooleanQuery {
                must: vec![term(&qword(&mut state))],
                ..BooleanQuery::default()
            },
            1 => BooleanQuery {
                must: vec![term(&qword(&mut state)), term(&qword(&mut state))],
                ..BooleanQuery::default()
            },
            2 => BooleanQuery {
                should: (0..3).map(|_| term(&qword(&mut state))).collect(),
                ..BooleanQuery::default()
            },
            3 => {
                let r = 1 + (next(&mut state) * 30.0) as u64;
                BooleanQuery {
                    must: vec![Clause::Phrase(PhraseQuery::new(
                        "body",
                        [format!("w{r}"), format!("w{}", r + 1)],
                    ))],
                    ..BooleanQuery::default()
                }
            }
            _ => BooleanQuery {
                should: vec![term(&qword(&mut state)), term(&format!("w{}", i % 7 + 1))],
                ..BooleanQuery::default()
            },
        })
        .collect()
}

#[derive(Default)]
struct Mem {
    rss_anon: u64,
    rss_file: u64,
    read_bytes: u64,
    cgroup: u64,
}

fn mem() -> Mem {
    let mut m = Mem::default();
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in status.lines() {
        let kb = || {
            line.split_whitespace()
                .nth(1)
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(0)
                * 1024
        };
        if line.starts_with("RssAnon:") {
            m.rss_anon = kb();
        } else if line.starts_with("RssFile:") {
            m.rss_file = kb();
        }
    }
    let io = std::fs::read_to_string("/proc/self/io").unwrap_or_default();
    for line in io.lines() {
        if let Some(v) = line.strip_prefix("read_bytes: ") {
            m.read_bytes = v.trim().parse().unwrap_or(0);
        }
    }
    let cg = std::fs::read_to_string("/proc/self/cgroup").unwrap_or_default();
    for line in cg.lines() {
        let parts: Vec<&str> = line.splitn(3, ':').collect();
        if parts.len() == 3 && parts[1] == "memory" {
            let p = format!("/sys/fs/cgroup/memory{}/memory.usage_in_bytes", parts[2]);
            m.cgroup = std::fs::read_to_string(p)
                .ok()
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(0);
        }
    }
    m
}

/// Bytes of the files under `path` that the kernel's page cache holds.
fn page_cache_bytes(path: &str) -> u64 {
    let mut total = 0u64;
    let page = 4096usize;
    for entry in std::fs::read_dir(path).unwrap() {
        let p = entry.unwrap().path();
        let Ok(f) = std::fs::File::open(&p) else {
            continue;
        };
        let len = f.metadata().map(|m| m.len()).unwrap_or(0) as usize;
        if len == 0 {
            continue;
        }
        // SAFETY: a read-only shared mapping only inspected with mincore.
        let map = unsafe { memmap2::Mmap::map(&f) }.unwrap();
        let pages = len.div_ceil(page);
        let mut vec = vec![0u8; pages];
        // SAFETY: `vec` holds one byte per page of the mapping.
        unsafe {
            libc::mincore(map.as_ptr() as *mut libc::c_void, len, vec.as_mut_ptr());
        }
        total += vec.iter().filter(|&&b| b & 1 == 1).count() as u64 * page as u64;
    }
    total
}

fn mib(b: u64) -> f64 {
    b as f64 / (1 << 20) as f64
}

fn percentile(sorted: &[Duration], p: f64) -> f64 {
    let i = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[i].as_secs_f64() * 1e6
}

fn opt<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn query(args: &[String]) {
    let path = &args[2];
    let mode = args[3].as_str();
    let budget_mb: u64 = opt(args, "--budget-mb").map_or(1024, |v| v.parse().unwrap());
    let page_kb: usize = opt(args, "--page-kb").map_or(64, |v| v.parse().unwrap());
    let n: usize = opt(args, "--queries").map_or(2000, |v| v.parse().unwrap());
    let passes: usize = opt(args, "--passes").map_or(3, |v| v.parse().unwrap());
    let threads: usize = opt(args, "--threads").map_or(1, |v| v.parse().unwrap());
    let buffered = args.iter().any(|a| a == "--buffered");
    let tenants: usize = opt(args, "--tenants").map_or(0, |v| v.parse().unwrap());
    let hot: usize = opt(args, "--hot").map_or(20, |v| v.parse().unwrap());
    let hot_share: f64 = opt(args, "--hot-share").map_or(0.9, |v| v.parse().unwrap());

    let base = mem();
    let t = Instant::now();
    let dir: Box<dyn Directory> = match mode {
        "mmap" => Box::new(MmapDirectory::open(path)),
        "paged" => {
            let d = PagedFsDirectory::with_config(
                path,
                PagedFsConfig {
                    page_size: page_kb << 10,
                    direct_io: !buffered,
                    adaptive: !args.iter().any(|a| a == "--fixed"),
                    readahead: opt(args, "--readahead-kb")
                        .map_or(lucene_store::paged_fs_directory::DEFAULT_READAHEAD, |v| {
                            v.parse::<usize>().unwrap() << 10
                        }),
                    ..PagedFsConfig::default()
                },
            );
            assert!(d.is_paged(), "userfaultfd is unavailable");
            lucene_store::paged_input::Pager::global()
                .unwrap()
                .set_budget(budget_mb << 20);
            Box::new(d)
        }
        m => panic!("unknown mode {m}"),
    };
    let reader = DirectoryReader::open(dir.as_ref()).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let owned = reader.field_norms_by_field(&["body".to_string()]).unwrap();
    let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let open = t.elapsed();
    let tagged: Vec<(bool, BooleanQuery)> = if tenants > 0 {
        tenant_queries(n, tenants, hot, hot_share)
    } else {
        queries(n).into_iter().map(|q| (true, q)).collect()
    };
    let qs: Vec<BooleanQuery> = tagged.iter().map(|(_, q)| q.clone()).collect();
    let leaves = reader.segment_readers();
    println!(
        "mode={mode} budget={budget_mb}MiB page={page_kb}KiB queries={n} threads={threads} open={:.1}ms",
        open.as_secs_f64() * 1e3
    );
    println!(
        "pass,p50_us,p90_us,p99_us,mean_us,qps,rss_anon_mib,rss_file_mib,pager_mib,read_mib,page_cache_mib,cgroup_mib,hot_p50_us,hot_p99_us,cold_p50_us,cold_p99_us,faults,refaults,fault_ms,evictions"
    );
    let run_one = |q: &BooleanQuery| {
        let t = Instant::now();
        let top = searcher.search(q, 10).unwrap();
        let mut b = 0;
        for seg in leaves {
            let docs: Vec<i32> = top
                .score_docs
                .iter()
                .filter(|h| h.doc >= b && h.doc - b < seg.max_doc)
                .map(|h| h.doc - b)
                .collect();
            seg.prefetch_stored_documents(&docs).unwrap();
            b += seg.max_doc;
        }
        for hit in &top.score_docs {
            let mut doc = hit.doc;
            for seg in leaves {
                if doc < seg.max_doc {
                    std::hint::black_box(seg.stored_document(doc).unwrap());
                    break;
                }
                doc -= seg.max_doc;
            }
        }
        t.elapsed()
    };
    for pass in 0..passes {
        let wall = Instant::now();
        let mut lat: Vec<Duration> = if threads <= 1 {
            qs.iter().map(run_one).collect()
        } else {
            let chunk = qs.len().div_ceil(threads);
            std::thread::scope(|s| {
                let hs: Vec<_> = qs
                    .chunks(chunk)
                    .map(|c| s.spawn(|| c.iter().map(run_one).collect::<Vec<_>>()))
                    .collect();
                hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
            })
        };
        let wall = wall.elapsed();
        let split = |want: bool| {
            let mut v: Vec<Duration> = lat
                .iter()
                .zip(&tagged)
                .filter(|(_, (h, _))| *h == want)
                .map(|(d, _)| *d)
                .collect();
            v.sort();
            if v.is_empty() {
                (0.0, 0.0)
            } else {
                (percentile(&v, 0.5), percentile(&v, 0.99))
            }
        };
        let (hot50, hot99) = split(true);
        let (cold50, cold99) = split(false);
        lat.sort();
        let mean = lat.iter().sum::<Duration>().as_secs_f64() * 1e6 / lat.len() as f64;
        let m = mem();
        let pager = lucene_store::paged_input::Pager::global()
            .filter(|_| mode == "paged")
            .map_or(0, |p| p.resident_bytes());
        let st = lucene_store::paged_input::Pager::global()
            .filter(|_| mode == "paged")
            .map(|p| p.stats())
            .unwrap_or_default();
        println!(
            "{pass},{:.0},{:.0},{:.0},{:.0},{:.0},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{hot50:.0},{hot99:.0},{cold50:.0},{cold99:.0},{},{},{},{}",
            percentile(&lat, 0.5),
            percentile(&lat, 0.9),
            percentile(&lat, 0.99),
            mean,
            lat.len() as f64 / wall.as_secs_f64(),
            mib(m.rss_anon.saturating_sub(base.rss_anon)),
            mib(m.rss_file),
            mib(pager),
            mib(m.read_bytes.saturating_sub(base.read_bytes)),
            mib(page_cache_bytes(path)),
            mib(m.cgroup),
            st.faults,
            st.refaults,
            st.fault_nanos / 1_000_000,
            st.evictions,
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("build") => build(
            &args[2],
            args.get(3)
                .and_then(|a| a.parse().ok())
                .unwrap_or(2_000_000),
        ),
        Some("build-tenants") => build_tenants(
            &args[2],
            args.get(3)
                .and_then(|a| a.parse().ok())
                .unwrap_or(3_000_000),
            args.get(4).and_then(|a| a.parse().ok()).unwrap_or(2000),
        ),
        Some("query") => query(&args),
        _ => eprintln!("usage: local_store build <dir> [docs] | query <dir> mmap|paged [options]"),
    }
}
