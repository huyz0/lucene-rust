//! Page size and read-ahead of the demand pager against one access pattern,
//! cold, under a budget smaller than the file: a sequential scan, random
//! point reads, or a scan of runs at random places (stored fields read in
//! docid order).
//!
//! `pager_pattern <dir> <file_mb> <budget_mb> <page_kb> <readahead_kb> seq|seq4|rand|runs [reads]`
#![allow(clippy::arithmetic_side_effects)]
use lucene_store::{Directory, PagedFsConfig, PagedFsDirectory};
use std::io::Write;
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
    let a: Vec<String> = std::env::args().skip(1).collect();
    let dir = std::path::PathBuf::from(&a[0]);
    let mb: usize = a[1].parse().unwrap();
    let budget: u64 = a[2].parse().unwrap();
    let page_kb: usize = a[3].parse().unwrap();
    let ra_kb: usize = a[4].parse().unwrap();
    let pattern = a[5].as_str();
    let reads: usize = a.get(6).map_or(20_000, |s| s.parse().unwrap());
    std::fs::create_dir_all(&dir).unwrap();
    file(&dir, "p.dat", mb);
    let pager = lucene_store::paged_input::Pager::global().expect("userfaultfd");
    pager.set_budget(budget << 20);
    let d = PagedFsDirectory::with_config(
        &dir,
        PagedFsConfig {
            page_size: page_kb << 10,
            adaptive: false,
            pinned_extensions: Vec::new(),
            readahead: ra_kb << 10,
            ..PagedFsConfig::default()
        },
    );
    if std::env::var("PAGER_DROP").is_ok() {
        let _ = std::process::Command::new("sh")
            .args(["-c", "sync; echo 3 > /proc/sys/vm/drop_caches"])
            .status();
    }
    let f = d.open("p.dat").unwrap();
    let t = Instant::now();
    let mut sum = 0u64;
    let mut x = 42u64;
    let n = f.len();
    match pattern {
        "seq" => {
            for at in (0..n).step_by(4096) {
                sum = sum.wrapping_add(f[at] as u64);
            }
        }
        "rand" => {
            for _ in 0..reads {
                let at = (rng(&mut x) as usize) % (n - 4096);
                sum = sum.wrapping_add(f[at] as u64 + f[at + 1023] as u64);
            }
        }
        "seq4" => {
            // Four threads each scanning a quarter of the file in order,
            // as concurrent merges or searches reading one file.
            let q = n / 4;
            let total = std::sync::atomic::AtomicU64::new(0);
            std::thread::scope(|sc| {
                for i in 0..4 {
                    let (f, total) = (&f, &total);
                    sc.spawn(move || {
                        let mut local = 0u64;
                        for at in (i * q..(i + 1) * q).step_by(4096) {
                            local = local.wrapping_add(f[at] as u64);
                        }
                        total.fetch_add(local, std::sync::atomic::Ordering::Relaxed);
                    });
                }
            });
            sum = total.load(std::sync::atomic::Ordering::Relaxed);
        }
        "runs" => {
            // 4 MiB runs at random places, read in order.
            for _ in 0..reads.max(1) {
                let at = (rng(&mut x) as usize) % (n - (4 << 20));
                let at = at & !4095;
                for p in (at..at + (4 << 20)).step_by(4096) {
                    sum = sum.wrapping_add(f[p] as u64);
                }
            }
        }
        _ => panic!("pattern"),
    }
    std::hint::black_box(sum);
    let s = pager.stats();
    println!(
        "{pattern} page={page_kb}K ra={ra_kb}K: {:.2}s faults={} refaults={} evictions={}",
        t.elapsed().as_secs_f64(),
        s.faults,
        s.refaults,
        s.evictions
    );
}
