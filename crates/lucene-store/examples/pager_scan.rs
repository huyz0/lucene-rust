//! Whether a scan through the demand pager (a merge reading segments in
//! order) pushes out a hot working set (an HNSW graph walked by queries):
//! one thread reads random 4 KiB blocks of a small hot file, announcing
//! each read as the vector readers do (`prefetch`) or not, while another
//! reads a file larger than the budget from start to end, twice.
//!
//! `pager_scan <dir> [hot_mb] [scan_mb] [budget_mb] [announce:0|1]`
#![allow(clippy::arithmetic_side_effects)]
use lucene_store::{Directory, PagedFsConfig, PagedFsDirectory};
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

fn rng(x: &mut u64) -> u64 {
    *x ^= *x << 13;
    *x ^= *x >> 7;
    *x ^= *x << 17;
    *x
}

fn file(dir: &std::path::Path, name: &str, mb: usize) {
    let path = dir.join(name);
    if std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0) == (mb << 20) as u64 {
        return;
    }
    let mut f = std::fs::File::create(&path).unwrap();
    let mut x = 7u64;
    let mut buf = vec![0u8; 1 << 20];
    for _ in 0..mb {
        for c in buf.chunks_mut(8) {
            c.copy_from_slice(&rng(&mut x).to_le_bytes());
        }
        f.write_all(&buf).unwrap();
    }
    f.sync_all().unwrap();
}

fn main() {
    let mut a = std::env::args().skip(1);
    let dir = std::path::PathBuf::from(a.next().unwrap());
    let hot_mb: usize = a.next().map_or(32, |s| s.parse().unwrap());
    let scan_mb: usize = a.next().map_or(1024, |s| s.parse().unwrap());
    let budget_mb: u64 = a.next().map_or(128, |s| s.parse().unwrap());
    let announce = a.next().is_none_or(|s| s == "1");
    std::fs::create_dir_all(&dir).unwrap();
    file(&dir, "hot.vex", hot_mb);
    file(&dir, "scan.doc", scan_mb);
    let pager = lucene_store::paged_input::Pager::global().expect("userfaultfd");
    pager.set_budget(budget_mb << 20);
    let d = PagedFsDirectory::with_config(
        &dir,
        PagedFsConfig {
            page_size: 64 << 10,
            adaptive: false,
            pinned_extensions: Vec::new(),
            ..PagedFsConfig::default()
        },
    );
    let hot = d.open("hot.vex").unwrap();
    let scan = d.open("scan.doc").unwrap();
    // Warm the hot set.
    hot.will_need(0, hot.len()).unwrap();
    let done = AtomicBool::new(false);
    let (reads, misses, scan_s) = std::thread::scope(|s| {
        let h = s.spawn(|| {
            let mut x = 42u64;
            let (mut reads, mut misses, mut sum) = (0u64, 0u64, 0u64);
            while !done.load(Ordering::Relaxed) {
                let at = (rng(&mut x) as usize) % (hot.len() - 4096);
                let block = &hot[at..at + 4096];
                if !lucene_store::is_resident(block) {
                    misses += 1;
                }
                if announce {
                    lucene_store::prefetch(&[block]).unwrap();
                }
                sum = sum.wrapping_add(block[0] as u64 + block[4095] as u64);
                reads += 1;
            }
            std::hint::black_box(sum);
            (reads, misses)
        });
        let t = Instant::now();
        let mut sum = 0u64;
        for _ in 0..2 {
            for at in (0..scan.len()).step_by(4096) {
                sum = sum.wrapping_add(scan[at] as u64);
            }
        }
        std::hint::black_box(sum);
        let scan_s = t.elapsed().as_secs_f64();
        done.store(true, Ordering::Relaxed);
        let (r, m) = h.join().unwrap();
        (r, m, scan_s)
    });
    println!(
        "hot {hot_mb} MiB, scan {scan_mb} MiB x2, budget {budget_mb} MiB, announce {announce}: \
         scan {scan_s:.2}s, hot reads {reads}, hot misses {misses} ({:.3}%)",
        misses as f64 * 100.0 / reads.max(1) as f64
    );
}
