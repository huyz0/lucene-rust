//! Cost of one demand fault: touches random pages of a file through
//! `PagedFsDirectory` (pager), `MmapDirectory` (kernel faults) or plain
//! `pread`, one at a time, and prints per-touch latency.
//!
//! `fault_micro <dir> <file> paged|mmap|pread <page_kb> <touches> [readahead_kb]`
// Bench-harness arithmetic -- see `docs/arithmetic-gate.md`'s "Test code".
#![allow(clippy::arithmetic_side_effects)]
use std::time::Instant;

use lucene_store::{Directory, MmapDirectory, PagedFsConfig, PagedFsDirectory};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (dir, file, mode) = (&a[1], &a[2], a[3].as_str());
    let page: usize = a[4].parse::<usize>().unwrap() << 10;
    let n: usize = a[5].parse().unwrap();
    let readahead: usize = a.get(6).map_or(0, |s| s.parse::<usize>().unwrap() << 10);
    let path = std::path::Path::new(dir).join(file);
    let len = std::fs::metadata(&path).unwrap().len() as usize;
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    let pages: Vec<usize> = (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x as usize % (len / page)) * page
        })
        .collect();
    let mut lat = Vec::with_capacity(n);
    let mut sum = 0u64;
    let started = Instant::now();
    match mode {
        "pread" => {
            use std::os::unix::fs::FileExt;
            let f = std::fs::File::open(&path).unwrap();
            let mut buf = vec![0u8; page];
            for &p in &pages {
                let t = Instant::now();
                f.read_at(&mut buf, p as u64).unwrap();
                sum += buf[0] as u64;
                lat.push(t.elapsed());
            }
        }
        _ => {
            let input = if mode == "paged" {
                let pager = lucene_store::paged_input::Pager::global().unwrap();
                pager.set_budget(u64::MAX / 2);
                let cfg = PagedFsConfig {
                    page_size: page,
                    adaptive: false,
                    readahead,
                    pinned_extensions: Vec::new(),
                    ..PagedFsConfig::default()
                };
                PagedFsDirectory::with_config(dir, cfg).open(file).unwrap()
            } else {
                MmapDirectory::open(dir).open(file).unwrap()
            };
            let bytes: &[u8] = &input;
            for &p in &pages {
                let t = Instant::now();
                sum += std::hint::black_box(bytes[p]) as u64;
                lat.push(t.elapsed());
            }
        }
    }
    let wall = started.elapsed();
    lat.sort();
    let us = |q: f64| lat[((lat.len() - 1) as f64 * q) as usize].as_secs_f64() * 1e6;
    let mean = lat.iter().map(|d| d.as_secs_f64()).sum::<f64>() / n as f64 * 1e6;
    let st =
        (mode == "paged").then(|| lucene_store::paged_input::Pager::global().map(|p| p.stats()));
    println!(
        "{mode} page={}K n={n} p50={:.0}us p90={:.0}us p99={:.0}us mean={mean:.0}us wall={:.0}ms sum={sum} {st:?}",
        page >> 10,
        us(0.5),
        us(0.9),
        us(0.99),
        wall.as_secs_f64() * 1e3
    );
}
