//! Fault throughput of the demand pager under concurrent random reads, with
//! a budget below the file so every thread keeps faulting and evicting,
//! against `pread` (`O_DIRECT`) of the same pages from as many threads.
//!
//! `pager_contention <dir> [file_mb] [budget_mb] [secs]`
#![allow(clippy::arithmetic_side_effects)]
use lucene_store::{Directory, PagedFsConfig, PagedFsDirectory};
use std::io::Write;
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::time::{Duration, Instant};

fn rng(x: &mut u64) -> u64 {
    *x ^= *x << 13;
    *x ^= *x >> 7;
    *x ^= *x << 17;
    *x
}

fn main() {
    let mut a = std::env::args().skip(1);
    let dir = std::path::PathBuf::from(a.next().unwrap());
    let file_mb: usize = a.next().map_or(1024, |s| s.parse().unwrap());
    let budget_mb: u64 = a.next().map_or(128, |s| s.parse().unwrap());
    let secs: u64 = a.next().map_or(5, |s| s.parse().unwrap());
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("big.vec");
    if std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0) != (file_mb << 20) as u64 {
        let mut f = std::fs::File::create(&path).unwrap();
        let mut x = 99u64;
        let mut buf = vec![0u8; 1 << 20];
        for _ in 0..file_mb {
            for c in buf.chunks_mut(8) {
                c.copy_from_slice(&rng(&mut x).to_le_bytes());
            }
            f.write_all(&buf).unwrap();
        }
        f.sync_all().unwrap();
    }
    let pager = lucene_store::paged_input::Pager::global().expect("userfaultfd");
    pager.set_budget(budget_mb << 20);
    let d = PagedFsDirectory::with_config(
        &dir,
        PagedFsConfig {
            page_size: 4096,
            adaptive: false,
            readahead: 0,
            pinned_extensions: Vec::new(),
            ..PagedFsConfig::default()
        },
    );
    let input = d.open("big.vec").unwrap();
    let bytes: &[u8] = &input;
    let len = bytes.len();
    println!("file {file_mb} MiB, budget {budget_mb} MiB, 4 KiB pages, {secs}s per run");
    for threads in [1usize, 2, 4, 8, 16, 32] {
        let before = pager.stats();
        let t = Instant::now();
        let stop = t + Duration::from_secs(secs);
        let reads: u64 = std::thread::scope(|s| {
            (0..threads)
                .map(|i| {
                    s.spawn(move || {
                        let mut x = 0x9E37_79B9 + i as u64 * 7919;
                        let mut n = 0u64;
                        let mut sum = 0u64;
                        while Instant::now() < stop {
                            for _ in 0..64 {
                                let o = (rng(&mut x) as usize) % (len - 8);
                                sum = sum.wrapping_add(bytes[o] as u64);
                                n += 1;
                            }
                        }
                        std::hint::black_box(sum);
                        n
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|h| h.join().unwrap())
                .sum()
        });
        let el = t.elapsed().as_secs_f64();
        let st = pager.stats();
        let faults = st.faults - before.faults;
        let fault_us = (st.fault_nanos - before.fault_nanos) as f64 / 1e3 / faults.max(1) as f64;
        // pread baseline: the same number of threads reading random 4 KiB pages.
        let f = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(&path)
            .unwrap();
        let pages = len / 4096;
        let t2 = Instant::now();
        let stop2 = t2 + Duration::from_secs(secs);
        let preads: u64 = std::thread::scope(|s| {
            (0..threads)
                .map(|i| {
                    let f = &f;
                    s.spawn(move || {
                        let layout = std::alloc::Layout::from_size_align(4096, 4096).unwrap();
                        let p = unsafe { std::alloc::alloc(layout) };
                        let buf = unsafe { std::slice::from_raw_parts_mut(p, 4096) };
                        let mut x = 0x1234 + i as u64 * 31;
                        let mut n = 0u64;
                        while Instant::now() < stop2 {
                            let pg = (rng(&mut x) as usize) % pages;
                            f.read_at(buf, (pg * 4096) as u64).unwrap();
                            n += 1;
                        }
                        unsafe { std::alloc::dealloc(p, layout) };
                        n
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|h| h.join().unwrap())
                .sum()
        });
        let el2 = t2.elapsed().as_secs_f64();
        println!(
            "threads {threads:>2}: pager {:>8.0} faults/s ({:>6.0} reads/s, handler {fault_us:>5.0} us/fault, evictions {}), pread O_DIRECT {:>8.0} reads/s",
            faults as f64 / el,
            reads as f64 / el,
            st.evictions - before.evictions,
            preads as f64 / el2
        );
    }
}
