//! `bp_force_merge <src> <dst> <bp|plain> [ram_mb]`: copies the index at
//! `src` to `dst` and force-merges it to one segment, with
//! `BPReorderingMergePolicy` (`bp`) or `TieredMergePolicy` alone (`plain`).
//! Prints the merge time and the merged index's size by file extension.
//!
//! `bp_force_merge <src> - docmap [ram_mb]` only computes the BP doc map over
//! `src`'s merged view and prints how long it took.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use lucene_index::index_writer::IndexWriter;
use lucene_index::merge_policy::{MergePolicy, TieredMergePolicy};
use lucene_index::segment_info::LuceneVersion;
use lucene_misc::index::{BpIndexReorderer, BpReorderingMergePolicy};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::reader::slow_codec::SlowCompositeCodecReaderWrapper;
use lucene_search::reader::CodecReader;
use lucene_store::FsDirectory;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        eprintln!("usage: bp_force_merge <src> <dst> <bp|plain|docmap> [ram_mb]");
        std::process::exit(2);
    }
    let (src, dst, mode) = (Path::new(&a[1]), Path::new(&a[2]), a[3].as_str());
    let ram_mb: f64 = a.get(4).map_or(1024.0, |v| v.parse().expect("ram_mb"));
    let mut reorderer = BpIndexReorderer::new();
    reorderer.set_ram_budget_mb(ram_mb);

    if mode == "docmap" {
        let dir = FsDirectory::open(src);
        let reader = DirectoryReader::open(&dir).expect("open");
        let leaves: Vec<Arc<dyn CodecReader>> = reader
            .segment_readers()
            .iter()
            .map(|r| Arc::new(r.clone()) as Arc<dyn CodecReader>)
            .collect();
        let merged = SlowCompositeCodecReaderWrapper::wrap(leaves).expect("merged view");
        let t = Instant::now();
        let map = reorderer
            .compute_doc_map(merged.as_ref())
            .unwrap_or_else(|e| panic!("doc map: {e}"));
        println!(
            "docmap\t{} docs\t{:.2}s",
            map.size(),
            t.elapsed().as_secs_f64()
        );
        return;
    }

    std::fs::create_dir_all(dst).expect("create dst");
    for entry in std::fs::read_dir(src).expect("read src") {
        let entry = entry.expect("entry");
        if entry.file_name() != "write.lock" {
            std::fs::copy(entry.path(), dst.join(entry.file_name())).expect("copy");
        }
    }
    let dir = FsDirectory::open(dst);
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 2,
    };
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", version).expect("writer");
    w.set_use_compound_file(false);
    let mut tiered = TieredMergePolicy::default();
    tiered
        .compound_file_settings_mut()
        .set_no_cfs_ratio(0.0)
        .expect("no cfs");
    let policy: Arc<dyn MergePolicy> = match mode {
        "bp" => Arc::new(BpReorderingMergePolicy::new(Box::new(tiered), reorderer)),
        "plain" => Arc::new(tiered),
        other => panic!("unknown mode {other}"),
    };
    w.set_pluggable_merge_policy(Some(policy));
    let t = Instant::now();
    w.force_merge(1).expect("force merge");
    w.close().expect("close");
    let secs = t.elapsed().as_secs_f64();

    let mut by_ext: BTreeMap<String, u64> = BTreeMap::new();
    let mut total = 0u64;
    for entry in std::fs::read_dir(dst).expect("read dst") {
        let entry = entry.expect("entry");
        let name = entry.file_name().to_string_lossy().into_owned();
        let len = entry.metadata().expect("metadata").len();
        let ext = name.rsplit_once('.').map_or("", |(_, e)| e).to_string();
        *by_ext.entry(ext).or_default() += len;
        total += len;
    }
    println!("{mode}\tmerge\t{secs:.2}s");
    for (ext, len) in &by_ext {
        println!("{mode}\t{ext}\t{len}");
    }
    println!("{mode}\ttotal\t{total}");
}
