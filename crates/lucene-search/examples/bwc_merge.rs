//! M8 T8.4: merge an index an older Lucene wrote into one `Lucene104`
//! segment with this port's `IndexWriter`, for `scripts/verify-bwc-merge.sh`
//! to hand to real Lucene 10.5.0.
//!
//! ```text
//! bwc_merge <src-index> <dst-dir> [force|policy]
//! ```
//!
//! Copies `<src-index>` into `<dst-dir>` (which must not exist), opens an
//! `IndexWriter` over the copy with no declared fields -- so every source
//! field keeps its own `FieldInfo`, as `SegmentMerger` keeps the sources' --
//! and merges every segment into one: `force` through
//! `IndexWriter::force_merge(1)`, `policy` through an ordinary commit under a
//! merge policy that wants a single segment. Then commits and prints the
//! resulting segments.

use std::path::Path;

use lucene_index::index_writer::{IndexWriter, MergePolicyConfig};
use lucene_index::segment_info::LuceneVersion;
use lucene_store::directory::FsDirectory;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: bwc_merge <src-index> <dst-dir> [force|policy]");
        std::process::exit(2);
    }
    let (src, dst) = (Path::new(&args[1]), Path::new(&args[2]));
    let mode = args.get(3).map(String::as_str).unwrap_or("force");
    std::fs::create_dir(dst).expect("dst must not exist");
    for entry in std::fs::read_dir(src).expect("src") {
        let entry = entry.expect("entry");
        let name = entry.file_name();
        let name = name.to_string_lossy();
        // The fixture's own sidecars are not index files.
        if name.ends_with(".txt") || name == "write.lock" {
            continue;
        }
        std::fs::copy(entry.path(), dst.join(&*name)).expect("copy");
    }
    let dir = FsDirectory::open(dst);
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
    .expect("open writer");
    let before = writer.segment_infos().segments.len();
    match mode {
        "force" => writer.force_merge(1).expect("force merge"),
        // An ordinary merge at commit, under a `TieredMergePolicy` that wants
        // few segments and tolerates few deletes (Java's own minimum, 5%):
        // the fixtures' old segments carry ~6%, so reclaiming them makes the
        // policy pick old segments too, not only the small new ones.
        "policy" => writer.set_merge_policy(Some(MergePolicyConfig {
            max_merge_at_once: 100,
            segments_per_tier: 2,
            max_merged_segment_size: u64::MAX / 4,
            floor_segment_size: 1 << 40,
            deletes_pct_allowed: 5.0,
            ..MergePolicyConfig::default()
        })),
        other => panic!("unknown mode {other}"),
    }
    writer.commit().expect("commit");
    let infos = writer.segment_infos();
    for s in &infos.segments {
        println!(
            "bwc_merge: {} codec={} docs-deleted={}",
            s.segment_name, s.codec_name, s.del_count
        );
    }
    println!(
        "bwc_merge: {before} segments -> {} ({mode})",
        infos.segments.len()
    );
}
