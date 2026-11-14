//! Would a block-statistics selectivity estimate help the filtered k-NN
//! planner? Splits a filtered k-NN query into its filter materialisation and
//! the search, and scores a per-1024-document min/max estimate against the
//! exact count.
//!
//! ```text
//! knn_filter_stats <dir> [docs] [dim] [clustered|shuffled] [skip]
//! ```
//!
//! One segment, a dense `ts` field written `Adaptive90` (zones), either in
//! document order (`clustered`) or shuffled, an HNSW `v` field. `skip` adds
//! the doc-values skip index to `ts`.

#![allow(clippy::arithmetic_side_effects)]

use std::time::Instant;

use lucene_codecs::doc_values::Lucene90DocValuesFormat;
use lucene_codecs::field_infos::VectorSimilarityFunction;
use lucene_index::document::{Document, KnnFloatVectorField, NumericDocValuesField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::extended_query::{ExtendedQuery, NumericDocValuesRangeQuery};
use lucene_search::knn_search::{knn_top_hits, KnnTarget};
use lucene_search::vector_query::filter_bitsets;
use lucene_search::Clause;
use lucene_store::FsDirectory;

const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    }
}

fn build(path: &str, n: usize, dim: usize, clustered: bool, skip: bool) {
    std::fs::create_dir_all(path).unwrap();
    let dir = FsDirectory::open(path);
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.set_ram_buffer_size_mb(2048.0).unwrap();
    if std::env::var_os("STOCK_DV").is_none() {
        w.set_default_doc_values_format(Lucene90DocValuesFormat::rust_adaptive());
    }
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut perm: Vec<i64> = (0..n as i64).collect();
    if !clustered {
        for i in (1..n).rev() {
            perm.swap(i, (rng.next() % (i as u64 + 1)) as usize);
        }
    }
    for (i, ts) in perm.iter().enumerate() {
        let mut d = Document::new();
        let v: Vec<f32> = (0..dim).map(|_| rng.unit()).collect();
        d.add(KnnFloatVectorField::new("v", v, VectorSimilarityFunction::Euclidean).unwrap());
        let value = 1_700_000_000_000 + ts * 37;
        d.add(if skip {
            NumericDocValuesField::indexed_field("ts", value)
        } else {
            NumericDocValuesField::new("ts", value)
        });
        let _ = i;
        w.add_fields_document(&d).unwrap();
    }
    w.commit().unwrap();
    w.force_merge(1).unwrap();
    w.commit().unwrap();
}

fn clause(lo: i64, hi: i64) -> Clause {
    Clause::Extended(Box::new(ExtendedQuery::NumericDocValuesRange(
        NumericDocValuesRangeQuery::new(
            "ts",
            1_700_000_000_000 + lo * 37,
            1_700_000_000_000 + hi * 37,
        ),
    )))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let n: usize = args.get(2).map_or(500_000, |s| s.parse().unwrap());
    let dim: usize = args.get(3).map_or(32, |s| s.parse().unwrap());
    let clustered = args.get(4).is_none_or(|s| s == "clustered");
    let skip = args.get(5).is_some_and(|s| s == "skip");
    if !std::path::Path::new(path).join("segments_1").exists()
        && std::fs::read_dir(path).map_or(true, |mut d| d.next().is_none())
    {
        let t = Instant::now();
        build(path, n, dim, clustered, skip);
        eprintln!("built {n} x {dim} in {:.1?}", t.elapsed());
    }
    let dir = FsDirectory::open(path);
    let reader = DirectoryReader::open(&dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let mut rng = Rng(42);
    let queries: Vec<Vec<f32>> = (0..20)
        .map(|_| (0..dim).map(|_| rng.unit()).collect())
        .collect();
    println!(
        "layout={} skip={skip} docs={n} dim={dim}",
        if clustered { "clustered" } else { "shuffled" }
    );
    println!("share      exact_n   filter_us  total_us  search_us");
    for share in [0.0005f64, 0.005, 0.02, 0.1, 0.3, 0.6, 0.9] {
        let width = (n as f64 * share) as i64;
        // A window in the middle of the time span.
        let lo = n as i64 / 3;
        let c = clause(lo, lo + width - 1);
        let mut exact = 0;
        let rounds = 30;
        let t = Instant::now();
        for _ in 0..rounds {
            let b = filter_bitsets(&segments, &c).unwrap();
            exact = b[0].cardinality();
        }
        let filter_us = t.elapsed().as_secs_f64() * 1e6 / rounds as f64;
        let t = Instant::now();
        for r in 0..rounds {
            let q = KnnTarget::Float(queries[r % queries.len()].clone());
            let _ = knn_top_hits(&segments, "v", &q, 10, 100, Some(&c)).unwrap();
        }
        let total_us = t.elapsed().as_secs_f64() * 1e6 / rounds as f64;
        println!(
            "{share:<9} {exact:>8} {filter_us:>10.0} {total_us:>9.0} {:>10.0}",
            total_us - filter_us
        );
    }
}
