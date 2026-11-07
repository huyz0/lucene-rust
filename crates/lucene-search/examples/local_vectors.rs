//! Memory and latency of vector search over a local index read through
//! `mmap` ([`MmapDirectory`]) against the same files read through the demand
//! pager ([`PagedFsDirectory`]): an IVF index (`lucene-ivf`: centroids in
//! `.ivm`, cluster codes in `.ivp`, rerank vectors in `.ivv`) and an HNSW
//! field written by `IndexWriter` (graph in `.vex`, vectors in `.vec`).
//!
//! ```text
//! local_vectors build-ivf <dir> [n] [dim]
//! local_vectors build-hnsw <dir> [n] [dim]
//! local_vectors gen-queries <dir> [nq]
//! local_vectors query-ivf|query-hnsw <dir> mmap|paged [--budget-mb N] [--page-kb N]
//!     [--pin ext,ext] [--page-sizes ext:kb,..] [--map ext,ext] [--fixed] [--queries N] [--passes N] [--nprobe N] [--rerank N] [--ef N]
//! ```
//!
//! The vectors are synthetic: Gaussian blobs around 2,000 random centres,
//! and the queries perturbed base vectors. `--pin` adds extensions the pager
//! reads whole at open (outside the budget). Each pass prints latency
//! percentiles and memory as `local_store` does.

#![allow(clippy::arithmetic_side_effects)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use lucene_codecs::field_infos::{self, VectorSimilarityFunction};
use lucene_codecs::hnsw_vectors::HnswVectorsReader;
use lucene_codecs::vectors::FlatVectorsReader;
use lucene_index::document::{Document, KnnFloatVectorField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_ivf::{FieldVectors, IvfParams, IvfReader, IvfWriter, Probes, Rerank, SearchParams};
use lucene_search::vector_query::{
    search_knn_float_vector_query, KnnFloatVectorQuery, VectorsInput,
};
use lucene_store::{Directory, FsDirectory, Input, MmapDirectory, PagedFsConfig, PagedFsDirectory};
use lucene_vector_quant::osq::OsqFactory;
use lucene_vector_quant::QuantizerRegistry;

const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};
const ID: [u8; 16] = [7; 16];

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f32 {
        let (u, v) = (self.next().max(1e-12), self.next());
        ((-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()) as f32
    }
}

fn data(n: usize, dim: usize) -> Vec<f32> {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let centres: Vec<f32> = (0..2000 * dim).map(|_| rng.gauss() * 4.0).collect();
    let mut out = Vec::with_capacity(n * dim);
    for _ in 0..n {
        let c = (rng.next() * 2000.0) as usize % 2000;
        for d in 0..dim {
            out.push(centres[c * dim + d] + rng.gauss());
        }
    }
    out
}

fn queries(base: &[f32], dim: usize, nq: usize) -> Vec<f32> {
    let n = base.len() / dim;
    let mut rng = Rng(0xdead_beef_cafe_f00d);
    let mut out = Vec::with_capacity(nq * dim);
    for _ in 0..nq {
        let i = (rng.next() * n as f64) as usize % n;
        for d in 0..dim {
            out.push(base[i * dim + d] + rng.gauss() * 0.3);
        }
    }
    out
}

fn write(dir: &dyn Directory, name: &str, bytes: &[u8]) {
    use lucene_store::DataOutput;
    let mut out = dir.create_output(name).unwrap();
    out.write_bytes(bytes);
    out.close().unwrap();
}

fn write_meta(path: &str, n: usize, dim: usize) {
    std::fs::write(format!("{path}/bench.meta"), format!("{n} {dim}")).unwrap();
}

fn read_meta(path: &str) -> (usize, usize) {
    let s = std::fs::read_to_string(format!("{path}/bench.meta")).unwrap();
    let v: Vec<usize> = s.split_whitespace().map(|x| x.parse().unwrap()).collect();
    (v[0], v[1])
}

/// Writes `nq` queries for the index at `path` to `queries.f32`, so a
/// query run does not hold the base vectors in its own memory.
fn gen_queries(path: &str, nq: usize) {
    let (n, dim) = read_meta(path);
    let base = data(n, dim);
    let qs = queries(&base, dim, nq);
    let bytes: Vec<u8> = qs.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(format!("{path}/queries.f32"), bytes).unwrap();
}

fn load_queries(path: &str, dim: usize, nq: usize) -> Vec<f32> {
    let bytes = std::fs::read(format!("{path}/queries.f32")).expect("run gen-queries first");
    let qs: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert!(qs.len() >= nq * dim, "fewer queries written than asked for");
    qs
}

fn build_ivf(path: &str, n: usize, dim: usize) {
    let t = Instant::now();
    let base = data(n, dim);
    let docs: Vec<i32> = (0..n as i32).collect();
    let mut w = IvfWriter::new(&ID, "");
    w.add_field(
        &FieldVectors {
            number: 0,
            similarity: lucene_ivf::Similarity::Euclidean,
            dimension: dim,
            docs: &docs,
            vectors: &base,
        },
        &OsqFactory::bbq(),
        &IvfParams::default(),
    )
    .unwrap();
    let files = w.finish();
    std::fs::create_dir_all(path).unwrap();
    let dir = FsDirectory::open(path);
    write(&dir, "_v.ivm", &files.meta);
    write(&dir, "_v.ivp", &files.postings);
    write(&dir, "_v.ivv", &files.vectors);
    write_meta(path, n, dim);
    eprintln!(
        "ivf {n} x {dim} in {:.1?}: ivm {:.1} MiB, ivp {:.1} MiB, ivv {:.1} MiB",
        t.elapsed(),
        mib(files.meta.len() as u64),
        mib(files.postings.len() as u64),
        mib(files.vectors.len() as u64)
    );
}

fn build_hnsw(path: &str, n: usize, dim: usize) {
    let t = Instant::now();
    let base = data(n, dim);
    std::fs::create_dir_all(path).unwrap();
    let dir = FsDirectory::open(path);
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.set_ram_buffer_size_mb(1024.0).unwrap();
    for i in 0..n {
        let mut d = Document::new();
        d.add(
            KnnFloatVectorField::new(
                "v",
                base[i * dim..(i + 1) * dim].to_vec(),
                VectorSimilarityFunction::Euclidean,
            )
            .unwrap(),
        );
        w.add_fields_document(&d).unwrap();
    }
    w.commit().unwrap();
    w.force_merge(1).unwrap();
    w.commit().unwrap();
    write_meta(path, n, dim);
    for name in dir.list_all().unwrap() {
        eprintln!(
            "  {name:<40} {:>8.1} MiB",
            mib(dir.file_length(&name).unwrap())
        );
    }
    eprintln!("hnsw {n} x {dim} in {:.1?}", t.elapsed());
}

fn mib(b: u64) -> f64 {
    b as f64 / (1 << 20) as f64
}

fn opt<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn status_bytes(key: &str) -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .unwrap_or_default()
        .lines()
        .find(|l| l.starts_with(key))
        .and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
        .unwrap_or(0)
        * 1024
}

fn read_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/io")
        .unwrap_or_default()
        .lines()
        .find_map(|l| l.strip_prefix("read_bytes: ")?.trim().parse().ok())
        .unwrap_or(0)
}

fn cgroup_bytes() -> u64 {
    let cg = std::fs::read_to_string("/proc/self/cgroup").unwrap_or_default();
    cg.lines()
        .find_map(|line| {
            let parts: Vec<&str> = line.splitn(3, ':').collect();
            (parts.len() == 3 && parts[1] == "memory").then(|| {
                std::fs::read_to_string(format!(
                    "/sys/fs/cgroup/memory{}/memory.usage_in_bytes",
                    parts[2]
                ))
                .ok()
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(0)
            })
        })
        .unwrap_or(0)
}

fn open_dir(args: &[String], path: &str) -> (String, Box<dyn Directory>) {
    let mode = args[3].clone();
    let budget_mb: u64 = opt(args, "--budget-mb").map_or(1024, |v| v.parse().unwrap());
    let page_kb: usize = opt(args, "--page-kb").map_or(64, |v| v.parse().unwrap());
    let dir: Box<dyn Directory> = match mode.as_str() {
        "mmap" => Box::new(MmapDirectory::open(path)),
        "paged" => {
            let mut config = PagedFsConfig {
                page_size: page_kb << 10,
                adaptive: !args.iter().any(|a| a == "--fixed"),
                readahead: opt(args, "--readahead-kb")
                    .map_or(lucene_store::paged_fs_directory::DEFAULT_READAHEAD, |v| {
                        v.parse::<usize>().unwrap() << 10
                    }),
                ..PagedFsConfig::default()
            };
            if let Some(spec) = opt(args, "--page-sizes") {
                for item in spec.split(',') {
                    let (ext, kb) = item.split_once(':').expect("--page-sizes ext:kb,...");
                    config
                        .page_sizes
                        .push((ext.to_string(), kb.parse::<usize>().unwrap() << 10));
                }
            }
            if let Some(maps) = opt(args, "--map") {
                config
                    .mapped_extensions
                    .extend(maps.split(',').map(str::to_string));
            }
            if let Some(pins) = opt(args, "--pin") {
                config
                    .pinned_extensions
                    .extend(pins.split(',').map(str::to_string));
            }
            let d = PagedFsDirectory::with_config(path, config);
            assert!(d.is_paged(), "userfaultfd is unavailable");
            lucene_store::paged_input::Pager::global()
                .unwrap()
                .set_budget(budget_mb << 20);
            Box::new(d)
        }
        m => panic!("unknown mode {m}"),
    };
    println!(
        "{} mode={mode} budget={budget_mb}MiB page={page_kb}KiB pin={} page_sizes={} map={}",
        args[1],
        opt(args, "--pin").unwrap_or("-"),
        opt(args, "--page-sizes").unwrap_or("-"),
        opt(args, "--map").unwrap_or("-")
    );
    (mode, dir)
}

fn report(
    pass: usize,
    mode: &str,
    mut lat: Vec<Duration>,
    wall: Duration,
    base_anon: u64,
    base_read: u64,
    extra: &str,
) {
    lat.sort();
    let pct = |p: f64| lat[((lat.len() as f64 - 1.0) * p).round() as usize].as_secs_f64() * 1e6;
    let mean = lat.iter().sum::<Duration>().as_secs_f64() * 1e6 / lat.len() as f64;
    let pager = lucene_store::paged_input::Pager::global()
        .filter(|_| mode == "paged")
        .map_or(0, |p| p.resident_bytes());
    println!(
        "{pass},{:.0},{:.0},{:.0},{:.0},{:.0},{:.1},{:.1},{:.1},{:.1},{:.1},{extra}",
        pct(0.5),
        pct(0.9),
        pct(0.99),
        mean,
        lat.len() as f64 / wall.as_secs_f64(),
        mib(status_bytes("RssAnon:").saturating_sub(base_anon)),
        mib(status_bytes("RssFile:")),
        mib(pager),
        mib(read_bytes().saturating_sub(base_read)),
        mib(cgroup_bytes()),
    );
}

const HEADER: &str =
    "pass,p50_us,p90_us,p99_us,mean_us,qps,rss_anon_mib,rss_file_mib,pager_mib,read_mib,cgroup_mib,recall_vs_first";

fn query_ivf(args: &[String]) {
    let path = &args[2];
    let (_, dim) = read_meta(path);
    let nq: usize = opt(args, "--queries").map_or(2000, |v| v.parse().unwrap());
    let passes: usize = opt(args, "--passes").map_or(3, |v| v.parse().unwrap());
    let nprobe: usize = opt(args, "--nprobe").map_or(16, |v| v.parse().unwrap());
    let rerank: usize = opt(args, "--rerank").map_or(50, |v| v.parse().unwrap());
    let qs = load_queries(path, dim, nq);
    let base_anon = status_bytes("RssAnon:");
    let base_read = read_bytes();
    let (mode, dir) = open_dir(args, path);
    let t = Instant::now();
    let meta = dir.open("_v.ivm").unwrap();
    let postings: Arc<Input> = Arc::new(dir.open("_v.ivp").unwrap());
    let vectors: Arc<Input> = Arc::new(dir.open("_v.ivv").unwrap());
    let registry = QuantizerRegistry::with_builtins();
    let r = IvfReader::open(&meta, postings, vectors, &ID, "", &registry).unwrap();
    println!(
        "open={:.1}ms nprobe={nprobe} rerank={rerank}",
        t.elapsed().as_secs_f64() * 1e3
    );
    println!("{HEADER}");
    let params = SearchParams {
        k: 10,
        nprobe: Probes::Fixed(nprobe),
        rerank: Rerank::Fixed(rerank),
    };
    let mut first: Vec<Vec<i32>> = Vec::new();
    for pass in 0..passes {
        let wall = Instant::now();
        let mut lat = Vec::with_capacity(nq);
        let mut same = 0usize;
        for q in 0..nq {
            let t = Instant::now();
            let out = r
                .search_with(0, &qs[q * dim..(q + 1) * dim], &params, None, None)
                .unwrap();
            lat.push(t.elapsed());
            let docs: Vec<i32> = out.hits.iter().map(|h| h.doc).collect();
            if pass == 0 {
                first.push(docs);
            } else {
                same += docs.iter().filter(|d| first[q].contains(d)).count();
            }
        }
        let agree = if pass == 0 {
            1.0
        } else {
            same as f64 / (nq * 10) as f64
        };
        report(
            pass,
            &mode,
            lat,
            wall.elapsed(),
            base_anon,
            base_read,
            &format!("{agree:.3}"),
        );
    }
}

fn query_hnsw(args: &[String]) {
    let path = &args[2];
    let (n, dim) = read_meta(path);
    let nq: usize = opt(args, "--queries").map_or(2000, |v| v.parse().unwrap());
    let passes: usize = opt(args, "--passes").map_or(3, |v| v.parse().unwrap());
    let ef: usize = opt(args, "--ef").map_or(100, |v| v.parse().unwrap());
    let qs = load_queries(path, dim, nq);
    let base_anon = status_bytes("RssAnon:");
    let base_read = read_bytes();
    let (mode, dir) = open_dir(args, path);
    let t = Instant::now();
    let infos = lucene_index::segment_infos::read_latest(dir.as_ref()).unwrap();
    let seg = &infos.segments[0];
    let names = dir.list_all().unwrap();
    let find = |ext: &str| {
        names
            .iter()
            .find(|f| f.starts_with(&format!("{}_", seg.segment_name)) && f.ends_with(ext))
            .unwrap_or_else(|| panic!("no {ext} file"))
            .clone()
    };
    let vec_name = find(".vec");
    let suffix = vec_name
        .strip_prefix(&format!("{}_", seg.segment_name))
        .unwrap()
        .strip_suffix(".vec")
        .unwrap()
        .to_string();
    let fnm = dir.open(&format!("{}.fnm", seg.segment_name)).unwrap();
    let vemf = dir.open(&find(".vemf")).unwrap();
    let vec = dir.open(&vec_name).unwrap();
    let vem = dir.open(&find(".vem")).unwrap();
    let vex = dir.open(&find(".vex")).unwrap();
    let infos_f = field_infos::parse(&fnm, &seg.segment_id, "").unwrap();
    let input = VectorsInput {
        flat: FlatVectorsReader::open(&vemf, &vec, &seg.segment_id, &suffix).unwrap(),
        hnsw: Some(
            HnswVectorsReader::open(&vem, &vex, &seg.segment_id, &suffix)
                .unwrap()
                .into(),
        ),
        field_infos: &infos_f,
        live_docs: None,
        filter: None,
        max_doc: n as i32,
    };
    println!("open={:.1}ms ef={ef}", t.elapsed().as_secs_f64() * 1e3);
    println!("{HEADER}");
    let mut first: Vec<Vec<i32>> = Vec::new();
    for pass in 0..passes {
        let wall = Instant::now();
        let mut lat = Vec::with_capacity(nq);
        let mut same = 0usize;
        for q in 0..nq {
            let query = KnnFloatVectorQuery::new("v", qs[q * dim..(q + 1) * dim].to_vec(), 10)
                .unwrap()
                .with_ef_search(ef);
            let t = Instant::now();
            let hits = search_knn_float_vector_query(&input, &query).unwrap();
            lat.push(t.elapsed());
            let docs: Vec<i32> = hits.iter().map(|h| h.doc_id).collect();
            if pass == 0 {
                first.push(docs);
            } else {
                same += docs.iter().filter(|d| first[q].contains(d)).count();
            }
        }
        let agree = if pass == 0 {
            1.0
        } else {
            same as f64 / (nq * 10) as f64
        };
        report(
            pass,
            &mode,
            lat,
            wall.elapsed(),
            base_anon,
            base_read,
            &format!("{agree:.3}"),
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let num = |i: usize, d: usize| args.get(i).and_then(|a| a.parse().ok()).unwrap_or(d);
    match args.get(1).map(String::as_str) {
        Some("build-ivf") => build_ivf(&args[2], num(3, 2_000_000), num(4, 128)),
        Some("build-hnsw") => build_hnsw(&args[2], num(3, 500_000), num(4, 128)),
        Some("gen-queries") => gen_queries(&args[2], num(3, 5000)),
        Some("query-ivf") => query_ivf(&args),
        Some("query-hnsw") => query_hnsw(&args),
        _ => eprintln!("see the module docs for usage"),
    }
}
