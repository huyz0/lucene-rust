//! Time `force_merge(1)` over an existing Lucene index directory.
//! cargo run --release -p lucene-search --example merge_existing -- <dir>
use std::time::Instant;
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_store::FsDirectory;

fn cpu_s() -> f64 {
    let s = std::fs::read_to_string("/proc/self/stat").unwrap();
    let p: Vec<&str> = s.rsplit(')').next().unwrap().split_whitespace().collect();
    (p[11].parse::<f64>().unwrap() + p[12].parse::<f64>().unwrap()) / 100.0
}

fn main() {
    let dir = std::env::args().nth(1).expect("dir");
    let d = FsDirectory::open(&dir);
    let mut w = IndexWriter::open(&d, Vec::new(), "Lucene104", LuceneVersion { major: 10, minor: 5, bugfix: 0 }).unwrap();
    let (t, c) = (Instant::now(), cpu_s());
    w.force_merge(1).unwrap();
    w.commit().unwrap();
    println!("ours force_merge(1): wall {:.1}s cpu {:.1}s", t.elapsed().as_secs_f64(), cpu_s() - c);
}
