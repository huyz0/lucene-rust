//! Costs of the remote page cache's open paths, for the PR that added them.
//!
//! ```text
//! cargo run --release -p lucene-store --example remote_cache_bench -- spill <heap|spill> [MiB]
//! cargo run --release -p lucene-store --example remote_cache_bench -- window <tuned|profile|fixed> [runs]
//! cargo run --release -p lucene-store --example remote_cache_bench -- version [opens]
//! ```
//!
//! `spill` opens one file with demand paging off and reports peak heap and
//! resident memory (run it once per mode: the high-water marks are per
//! process) and scan and random-read time. `window` reports the bytes a
//! cold open fetches, run to run, against a store whose latency jitters.
//! `version` times opens with and without a store version.

#![allow(clippy::arithmetic_side_effects)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use lucene_store::{
    BlobStore, CachedBlobDirectory, Directory, InMemoryBlobStore, Input, PageCache,
    PageCacheConfig, Result, StoreProfile,
};
use lucene_util::test_support::TempDir;

fn status(field: &str) -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with(field))
                .and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse().ok()))
        })
        .unwrap_or(0)
}

#[derive(Clone)]
struct Jittery {
    inner: InMemoryBlobStore,
    latency: Duration,
    bytes_per_sec: f64,
    jitter: bool,
    version: Option<String>,
    rng: Arc<std::sync::atomic::AtomicU64>,
}

impl Jittery {
    fn new(jitter: bool) -> Self {
        Self {
            inner: InMemoryBlobStore::default(),
            latency: Duration::from_millis(8),
            bytes_per_sec: 200e6,
            jitter,
            version: None,
            rng: Arc::new(std::sync::atomic::AtomicU64::new(0x9e37_79b9_7f4a_7c15)),
        }
    }
    fn noise(&self) -> f64 {
        use std::sync::atomic::Ordering::Relaxed;
        let mut x = self.rng.load(Relaxed);
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng.store(x, Relaxed);
        // Seeded from the clock below so runs differ, like real timings.
        0.25 + (x % 1000) as f64 / 1000.0 * 2.5
    }
}

impl BlobStore for Jittery {
    fn list(&self) -> Result<Vec<String>> {
        self.inner.list()
    }
    fn get(&self, name: &str) -> Result<Input> {
        self.inner.get(name)
    }
    fn get_range(&self, name: &str, offset: u64, len: usize) -> Result<Vec<u8>> {
        let k = if self.jitter { self.noise() } else { 1.0 };
        std::thread::sleep(
            self.latency.mul_f64(k) + Duration::from_secs_f64(len as f64 / self.bytes_per_sec),
        );
        self.inner.get_range(name, offset, len)
    }
    fn size(&self, name: &str) -> Result<u64> {
        self.inner.size(name)
    }
    fn version(&self, _name: &str) -> Result<Option<String>> {
        Ok(self.version.clone())
    }
    fn put(&self, name: &str, bytes: Vec<u8>) -> Result<()> {
        self.inner.put(name, bytes)
    }
    fn delete(&self, name: &str) -> Result<()> {
        self.inner.delete(name)
    }
}

fn bytes(len: usize) -> Vec<u8> {
    let mut x = 0x2545_f491_4f6c_dd1du64;
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x as u8
        })
        .collect()
}

fn spill(mode: &str, mib: usize) {
    let tmp = TempDir::new("remote-cache-bench-spill");
    let store = Jittery::new(false);
    let len = mib << 20;
    store.put("_0.doc", bytes(len)).unwrap();
    let cache = Arc::new(
        PageCache::new(PageCacheConfig {
            page_size: 256 << 10,
            demand_paging: false,
            spill_min_bytes: (mode == "spill").then_some(1 << 20),
            disk_dir: Some(tmp.path().join("cache")),
            ..PageCacheConfig::default()
        })
        .unwrap(),
    );
    let mut fast = store.clone();
    fast.latency = Duration::ZERO;
    fast.bytes_per_sec = 1e12;
    let dir = CachedBlobDirectory::new(fast, cache, "idx");
    let (rss0, data0) = (status("VmRSS:"), status("VmData:"));
    let t = Instant::now();
    let input = dir.open("_0.doc").unwrap();
    let open = t.elapsed();
    let t = Instant::now();
    let mut sum = 0u64;
    for chunk in input[..].chunks(4096) {
        sum = sum.wrapping_add(chunk[0] as u64);
    }
    let scan = t.elapsed();
    let t = Instant::now();
    let mut x = 12345u64;
    for _ in 0..2_000_000 {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        sum = sum.wrapping_add(input[(x as usize) % len] as u64);
    }
    let random = t.elapsed();
    println!(
        "{mode:5} {mib} MiB: open {:.0} ms, 4K-stride scan {:.1} ms, 2M random byte reads {:.1} ms; \
         VmHWM {} MiB, VmData grew {} MiB, VmRSS grew {} MiB (sum {sum})",
        open.as_secs_f64() * 1e3,
        scan.as_secs_f64() * 1e3,
        random.as_secs_f64() * 1e3,
        status("VmHWM:") >> 10,
        (status("VmData:") - data0.min(status("VmData:"))) >> 10,
        (status("VmRSS:") - rss0.min(status("VmRSS:"))) >> 10,
    );
}

fn window(mode: &str, runs: usize) {
    let mut opened = Vec::new();
    let mut windows = Vec::new();
    for run in 0..runs {
        let tmp = TempDir::new("remote-cache-bench-window");
        let mut store = Jittery::new(true);
        // A different noise stream per run, like different timings.
        store.rng = Arc::new(std::sync::atomic::AtomicU64::new(
            0x9e37_79b9_7f4a_7c15 ^ ((run as u64 + 1) * 0x1000_0000_01b3),
        ));
        for i in 0..6 {
            store.put(&format!("_{i}.doc"), bytes(6 << 20)).unwrap();
        }
        // Files whose last window is two pages: the shard's earlier reads,
        // which give the tuner small and large requests to measure.
        for i in 0..4 {
            store
                .put(&format!("_w{i}.doc"), bytes((6 << 20) + (100 << 10)))
                .unwrap();
        }
        let cache = Arc::new(
            PageCache::new(PageCacheConfig {
                page_size: 64 << 10,
                disk_dir: Some(tmp.path().join("cache")),
                download_window: (mode == "fixed").then_some(1 << 20),
                store_profile: (mode == "profile").then_some(StoreProfile {
                    latency: Duration::from_millis(8),
                    bytes_per_sec: 200e6,
                }),
                ..PageCacheConfig::default()
            })
            .unwrap(),
        );
        let dir = CachedBlobDirectory::new(store, Arc::clone(&cache), "idx");
        for i in 0..4 {
            let w = dir.open(&format!("_w{i}.doc")).unwrap();
            let _ = (w[3 << 20], w[w.len() - 1]);
        }
        let before = cache.stats().fetched_bytes;
        // A shard opens six files: header and footer of each are read.
        let names: Vec<String> = (0..6).map(|i| format!("_{i}.doc")).collect();
        dir.prefetch_files(&names).unwrap();
        for n in &names {
            let input = dir.open(n).unwrap();
            let _ = (input[0], input[input.len() - 1]);
        }
        opened.push(cache.stats().fetched_bytes - before);
        windows.push(cache.download_window());
    }
    let kb = |b: u64| b as f64 / 1024.0;
    let min = *opened.iter().min().unwrap();
    let max = *opened.iter().max().unwrap();
    let mean = opened.iter().sum::<u64>() as f64 / opened.len() as f64;
    windows.sort_unstable();
    windows.dedup();
    println!(
        "        windows seen (KiB): {:?}",
        windows.iter().map(|w| w >> 10).collect::<Vec<_>>()
    );
    println!(
        "{mode:7}: cold open of 6 files fetched min {:.0} KiB, max {:.0} KiB, mean {:.0} KiB over {runs} runs (distinct values: {})",
        kb(min),
        kb(max),
        mean / 1024.0,
        {
            let mut d = opened.clone();
            d.sort_unstable();
            d.dedup();
            d.len()
        }
    );
}

fn version(opens: usize) {
    for versioned in [false, true] {
        let tmp = TempDir::new("remote-cache-bench-version");
        let mut store = Jittery::new(false);
        store.latency = Duration::ZERO;
        store.bytes_per_sec = 1e12;
        store.version = versioned.then(|| "etag-0123456789abcdef".to_string());
        for i in 0..64 {
            store.put(&format!("_{i}.doc"), bytes(1 << 20)).unwrap();
        }
        let cache = Arc::new(
            PageCache::new(PageCacheConfig {
                page_size: 256 << 10,
                disk_dir: Some(tmp.path().join("cache")),
                ..PageCacheConfig::default()
            })
            .unwrap(),
        );
        let dir = CachedBlobDirectory::new(store, cache, "idx");
        let names: Vec<String> = (0..64).map(|i| format!("_{i}.doc")).collect();
        let held: Vec<_> = names.iter().map(|n| dir.open(n).unwrap()).collect();
        let t = Instant::now();
        for i in 0..opens {
            let _ = dir.open(&names[i % 64]).unwrap();
        }
        println!(
            "version {versioned:5}: {:.0} ns per shared open ({opens} opens, {} held)",
            t.elapsed().as_nanos() as f64 / opens as f64,
            held.len()
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let n = |i: usize, d: usize| args.get(i).and_then(|v| v.parse().ok()).unwrap_or(d);
    match args.first().map(String::as_str) {
        Some("spill") => spill(args.get(1).map_or("spill", String::as_str), n(2, 256)),
        Some("window") => window(args.get(1).map_or("tuned", String::as_str), n(2, 20)),
        Some("version") => version(n(1, 200_000)),
        _ => eprintln!("usage: remote_cache_bench spill|window|version ..."),
    }
}
