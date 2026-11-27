#![cfg(test)]
// Test fixtures' own arithmetic -- see `docs/arithmetic-gate.md`'s "Test code".
#![allow(clippy::arithmetic_side_effects)]
use super::*;
use std::sync::atomic::AtomicUsize;

fn bytes(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i * 31 % 251) as u8).collect()
}

fn reader(data: Vec<u8>, calls: Arc<AtomicUsize>) -> impl Fn(u64, usize) -> Result<Vec<u8>> + Sync {
    move |off, len| {
        calls.fetch_add(1, Ordering::Relaxed);
        let s = (off as usize).min(data.len());
        let e = (s + len).min(data.len());
        Ok(data[s..e].to_vec())
    }
}

#[test]
fn plain_geometry_cuts_whole_pages() {
    let g = Geometry::plain(Arc::from("k"), 2500, 1000);
    assert_eq!(g.pages, 3);
    assert_eq!(g.stored_range(0), (0, 1000));
    assert_eq!(g.stored_range(2), (2000, 2500));
    assert_eq!(g.plain_range(2), (2000, 2500));
    assert_eq!(g.stored_range(9), (2500, 2500));
}

#[test]
fn unit_geometry_aligns_pages_to_whole_units_after_the_origin() {
    let layout = UnitLayout {
        origin: 10,
        stored_unit: 128,
        plain_unit: 100,
        plain_len: 450,
    };
    // 5 units: 4 full of 128 stored bytes and one of 50 + 28.
    let stored = 10 + 4 * 128 + 78;
    let g = Geometry::units(Arc::from("k"), stored, layout, 250, "f").unwrap();
    assert_eq!(g.units_per_page, 2);
    assert_eq!(g.pages, 3);
    assert_eq!(g.stored_range(0), (10, 266));
    assert_eq!(g.stored_range(2), (522, stored));
    assert_eq!(g.plain_range(2), (400, 450));
    // A page smaller than a unit still holds one whole unit.
    let g = Geometry::units(Arc::from("k"), stored, layout, 1, "f").unwrap();
    assert_eq!(g.units_per_page, 1);
    let bad = UnitLayout {
        stored_unit: 0,
        ..layout
    };
    assert!(Geometry::units(Arc::from("k"), stored, bad, 250, "f").is_err());
    let bad = UnitLayout {
        origin: stored + 1,
        ..layout
    };
    assert!(Geometry::units(Arc::from("k"), stored, bad, 250, "f").is_err());
    let huge = UnitLayout {
        stored_unit: u64::MAX,
        ..layout
    };
    assert!(Geometry::units(Arc::from("k"), stored, huge, 250, "f").is_err());
}

#[test]
fn adjacent_misses_merge_into_one_request_and_disk_hits_skip_the_store() {
    let tmp = lucene_util::test_support::TempDir::new("page-cache-merge");
    let cache = PageCache::new(PageCacheConfig {
        page_size: 100,
        max_request: 300,
        download_window: Some(100),
        disk_dir: Some(tmp.path().join("cache")),
        ..PageCacheConfig::default()
    })
    .unwrap();
    let data = bytes(1000);
    let calls = Arc::new(AtomicUsize::new(0));
    let fetch = reader(data.clone(), Arc::clone(&calls));
    let g = Geometry::plain(Arc::from("ns\0f"), 1000, 100);
    // Pages 0..=4 and 7: runs [0,1,2] [3,4] (max 300 bytes) and [7].
    let got = cache.stored_pages(&g, &[0, 1, 2, 3, 4, 7], &fetch).unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 3);
    assert_eq!(&got[5][..], &data[700..800]);
    assert_eq!(&got[3][..], &data[300..400]);
    let again = cache.stored_pages(&g, &[2, 7], &fetch).unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 3, "served from disk");
    assert_eq!(&again[0][..], &data[200..300]);
    let s = cache.stats();
    assert_eq!((s.requests, s.fetched_pages, s.disk_hits), (3, 6, 2));
    assert_eq!(s.fetched_bytes, 600);
    assert_eq!(cache.disk_bytes(), 600);
    cache.forget_prefix("ns\0");
    assert_eq!(cache.disk_bytes(), 0);
    cache.stored_pages(&g, &[2], &fetch).unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 4);
}

#[test]
fn the_disk_tier_evicts_unpinned_pages_past_its_budget() {
    let tmp = lucene_util::test_support::TempDir::new("page-cache-evict");
    let cache = PageCache::new(PageCacheConfig {
        page_size: 100,
        download_window: Some(100),
        disk_dir: Some(tmp.path().join("cache")),
        // Three slots (a 100-byte page takes the smallest, 4 KiB).
        disk_budget: 3 * 4096,
        ..PageCacheConfig::default()
    })
    .unwrap();
    let data = bytes(1000);
    let calls = Arc::new(AtomicUsize::new(0));
    let fetch = reader(data, Arc::clone(&calls));
    let pinned = Geometry::plain(Arc::from("p"), 1000, 100);
    let other = Geometry::plain(Arc::from("o"), 1000, 100);
    cache.stored_pages(&pinned, &[0], &fetch).unwrap();
    cache.pin("p");
    cache.stored_pages(&pinned, &[1], &fetch).unwrap();
    for page in 0..5 {
        cache.stored_pages(&other, &[page], &fetch).unwrap();
    }
    // Budget 300: the two pinned pages and the newest other page stay.
    assert_eq!(cache.disk_bytes(), 300);
    let before = calls.load(Ordering::Relaxed);
    cache.stored_pages(&pinned, &[0, 1], &fetch).unwrap();
    cache.stored_pages(&other, &[4], &fetch).unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), before);
    cache.stored_pages(&other, &[0], &fetch).unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), before + 1);
}

#[test]
fn the_in_flight_limits_must_admit_one_request() {
    for config in [
        PageCacheConfig {
            max_inflight_requests: 0,
            ..PageCacheConfig::default()
        },
        PageCacheConfig {
            max_request: 1000,
            max_inflight_bytes: 999,
            ..PageCacheConfig::default()
        },
    ] {
        assert!(matches!(
            PageCache::new(config),
            Err(Error::IllegalArgument(_))
        ));
    }
    assert!(PageCache::new(PageCacheConfig {
        max_request: 1000,
        max_inflight_bytes: 1000,
        max_inflight_requests: 1,
        ..PageCacheConfig::default()
    })
    .is_ok());
}

/// A fetch that records how many requests and bytes are in flight at
/// once, holding each for `nap`.
struct InFlight {
    data: Vec<u8>,
    nap: std::time::Duration,
    now: AtomicUsize,
    bytes_now: AtomicUsize,
    peak: AtomicUsize,
    peak_bytes: AtomicUsize,
}

impl InFlight {
    fn new(data: Vec<u8>, nap_ms: u64) -> InFlight {
        InFlight {
            data,
            nap: std::time::Duration::from_millis(nap_ms),
            now: AtomicUsize::new(0),
            bytes_now: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            peak_bytes: AtomicUsize::new(0),
        }
    }

    fn fetch(&self, off: u64, len: usize) -> Result<Vec<u8>> {
        let n = self.now.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(n, Ordering::SeqCst);
        let b = self.bytes_now.fetch_add(len, Ordering::SeqCst) + len;
        self.peak_bytes.fetch_max(b, Ordering::SeqCst);
        std::thread::sleep(self.nap);
        self.bytes_now.fetch_sub(len, Ordering::SeqCst);
        self.now.fetch_sub(1, Ordering::SeqCst);
        Ok(self.data[off as usize..off as usize + len].to_vec())
    }
}

#[test]
fn requests_in_flight_are_bounded_over_every_reader_of_the_cache() {
    // Eight readers, each missing four scattered pages of its own file:
    // without the bound 32 requests would be in flight at once.
    let cache = PageCache::new(PageCacheConfig {
        page_size: 100,
        max_inflight_requests: 3,
        ..PageCacheConfig::default()
    })
    .unwrap();
    let store = InFlight::new(bytes(1000), 20);
    let fetch = |off: u64, len: usize| store.fetch(off, len);
    std::thread::scope(|s| {
        for r in 0..8 {
            let cache = &cache;
            let fetch = &fetch;
            s.spawn(move || {
                let g = Geometry::plain(Arc::from(format!("f{r}")), 1000, 100);
                let got = cache.stored_pages(&g, &[0, 2, 4, 6], fetch).unwrap();
                assert_eq!(&got[1][..], &bytes(1000)[200..300]);
            });
        }
    });
    assert_eq!(store.peak.load(Ordering::SeqCst), 3);
    assert_eq!(cache.stats().requests, 32);
    assert!(cache.stats().fetch_waits > 0);
}

#[test]
fn bytes_in_flight_are_bounded_over_every_reader_of_the_cache() {
    // Requests of 300 bytes (three adjacent pages) under a 600-byte
    // bound: two at once, whatever the request bound.
    let cache = PageCache::new(PageCacheConfig {
        page_size: 100,
        max_request: 300,
        max_inflight_bytes: 600,
        ..PageCacheConfig::default()
    })
    .unwrap();
    let store = InFlight::new(bytes(1200), 20);
    let fetch = |off: u64, len: usize| store.fetch(off, len);
    std::thread::scope(|s| {
        for r in 0..4 {
            let cache = &cache;
            let fetch = &fetch;
            s.spawn(move || {
                let g = Geometry::plain(Arc::from(format!("f{r}")), 1200, 100);
                let pages: Vec<u64> = (0..12).collect();
                assert_eq!(cache.stored_pages(&g, &pages, fetch).unwrap().len(), 12);
            });
        }
    });
    assert_eq!(store.peak_bytes.load(Ordering::SeqCst), 600);
    assert_eq!(store.peak.load(Ordering::SeqCst), 2);
}

#[test]
fn concurrent_misses_on_one_page_send_one_request() {
    let cache = PageCache::new(PageCacheConfig {
        page_size: 100,
        ..PageCacheConfig::default()
    })
    .unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let data = bytes(1000);
    let gate = Arc::new(std::sync::Barrier::new(2));
    let g = Geometry::plain(Arc::from("f"), 1000, 100);
    let slow = {
        let calls = Arc::clone(&calls);
        let gate = Arc::clone(&gate);
        move |off: u64, len: usize| {
            if calls.fetch_add(1, Ordering::Relaxed) == 0 {
                gate.wait();
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            Ok(data[off as usize..off as usize + len].to_vec())
        }
    };
    std::thread::scope(|s| {
        let a = s.spawn(|| cache.stored_pages(&g, &[3], &slow).unwrap());
        gate.wait();
        let b = cache.stored_pages(&g, &[3], &slow).unwrap();
        assert_eq!(a.join().unwrap()[0], b[0]);
    });
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(cache.stats().shared_fetches, 1);
}

#[test]
fn failed_and_short_fetches_are_errors_for_every_waiter() {
    let cache = PageCache::new(PageCacheConfig {
        page_size: 100,
        max_request: 100,
        ..PageCacheConfig::default()
    })
    .unwrap();
    let g = Geometry::plain(Arc::from("f"), 1000, 100);
    let failing = |_: u64, _: usize| -> Result<Vec<u8>> {
        Err(Error::Io(std::io::Error::other("store down")))
    };
    let e = cache.stored_pages(&g, &[0, 5], &failing).unwrap_err();
    assert!(e.to_string().contains("store down"), "{e}");
    let short = |_: u64, len: usize| -> Result<Vec<u8>> { Ok(vec![0; len / 2]) };
    let e = cache.stored_pages(&g, &[1], &short).unwrap_err();
    assert!(e.to_string().contains("short read"), "{e}");
    // Nothing is left claimed: a later good fetch succeeds.
    let ok = |_: u64, len: usize| -> Result<Vec<u8>> { Ok(vec![7; len]) };
    assert_eq!(cache.stored_pages(&g, &[0, 1], &ok).unwrap().len(), 2);
}

#[test]
fn errors_keep_their_kind_and_a_panicking_fetch_releases_its_pages() {
    let cache = PageCache::new(PageCacheConfig {
        page_size: 100,
        ..PageCacheConfig::default()
    })
    .unwrap();
    let g = Geometry::plain(Arc::from("f"), 1000, 100);
    let gone = |_: u64, _: usize| -> Result<Vec<u8>> { Err(Error::no_such_file("f")) };
    assert!(cache
        .stored_pages(&g, &[0], &gone)
        .unwrap_err()
        .is_no_such_file());
    let short = |_: u64, _: usize| -> Result<Vec<u8>> { Ok(Vec::new()) };
    let e = cache.stored_pages(&g, &[0], &short).unwrap_err();
    assert!(matches!(e, Error::Corrupted(_)), "{e}");
    let other = |_: u64, _: usize| -> Result<Vec<u8>> { Err(Error::IllegalArgument("x".into())) };
    assert!(cache.stored_pages(&g, &[0], &other).is_err());
    let panics = |_: u64, _: usize| -> Result<Vec<u8>> { panic!("sdk bug") };
    assert!(cache.stored_pages(&g, &[0], &panics).is_err());
    assert!(cache.stored_pages(&g, &[1, 3], &panics).is_err());
    let ok = |_: u64, len: usize| -> Result<Vec<u8>> { Ok(vec![1; len]) };
    assert_eq!(cache.stored_pages(&g, &[0, 1, 3], &ok).unwrap().len(), 3);
}

#[test]
fn config_is_checked_and_pinned_names_follow_extensions() {
    assert!(PageCache::new(PageCacheConfig {
        page_size: 0,
        ..PageCacheConfig::default()
    })
    .is_err());
    let cache = PageCache::new(PageCacheConfig::default()).unwrap();
    assert!(cache.is_pinned_name("_0.tip"));
    assert!(cache.is_pinned_name("segments_3"));
    assert!(!cache.is_pinned_name("_0.doc"));
    assert!(!cache.is_pinned_name("noext"));
    assert_eq!(cache.disk_bytes(), 0);
    cache.pin("x");
    cache.forget_prefix("x");
    assert!(format!("{cache:?}").contains("PageCache"));
}

#[test]
fn a_miss_reads_ahead_its_download_window_in_one_request() {
    let tmp = lucene_util::test_support::TempDir::new("page-cache-ahead");
    let cache = PageCache::new(PageCacheConfig {
        page_size: 100,
        download_window: Some(400),
        disk_dir: Some(tmp.path().join("cache")),
        ..PageCacheConfig::default()
    })
    .unwrap();
    let data = bytes(1000);
    let calls = Arc::new(AtomicUsize::new(0));
    let fetch = reader(data.clone(), Arc::clone(&calls));
    let g = Geometry::plain(Arc::from("f"), 1000, 100);
    // Page 5 is in window 4..8: one request for all four pages.
    let got = cache.stored_pages(&g, &[5], &fetch).unwrap();
    assert_eq!(&got[0][..], &data[500..600]);
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(cache.stats().read_ahead_pages, 3);
    assert_eq!(cache.stats().fetched_bytes, 400);
    let again = cache.stored_pages(&g, &[4, 6, 7], &fetch).unwrap();
    assert_eq!(&again[2][..], &data[700..800]);
    assert_eq!(calls.load(Ordering::Relaxed), 1, "read ahead");
    // The last window is cut at the file's end; cached pages are skipped.
    cache.stored_pages(&g, &[9], &fetch).unwrap();
    assert_eq!(cache.stats().fetched_bytes, 600);
    assert_eq!(cache.stats().download_window, 400);
    // Without a disk tier there is no read-ahead.
    let bare = PageCache::new(PageCacheConfig {
        page_size: 100,
        download_window: Some(400),
        ..PageCacheConfig::default()
    })
    .unwrap();
    bare.stored_pages(&g, &[5], &fetch).unwrap();
    assert_eq!(bare.stats().fetched_bytes, 100);
}

#[test]
fn the_tuned_window_is_latency_times_throughput() {
    let t = Tuner::default();
    let page = 256 << 10;
    let max = 8 << 20;
    // Unmeasured: the default.
    assert_eq!(t.window(page, max), DEFAULT_DOWNLOAD_WINDOW);
    // 30 ms latency, 100 MB/s: 3 MB, so 4 MiB.
    let ms = |m: f64| std::time::Duration::from_secs_f64(m / 1000.0);
    for _ in 0..20 {
        t.record(page, ms(30.0 + 2.62), page as u64);
        t.record(8 * page, ms(30.0 + 20.97), page as u64);
        t.record(3 * page, ms(1.0), page as u64);
    }
    assert_eq!(t.window(page, max), 4 << 20);
    // 0.5 ms latency, 1 GB/s (local disk): 0.5 MB, so 512 KiB.
    let t = Tuner::default();
    for _ in 0..20 {
        t.record(page, ms(0.5 + 0.262), page as u64);
        t.record(8 * page, ms(0.5 + 2.097), page as u64);
    }
    assert_eq!(t.window(page, max), 512 << 10);
    // Large requests no slower: the largest window.
    let t = Tuner::default();
    t.record(page, ms(5.0), page as u64);
    t.record(8 * page, ms(5.0), page as u64);
    assert_eq!(t.window(page, max), max);
    // Clamped to a page at least.
    let t = Tuner::default();
    t.record(page, ms(0.001), page as u64);
    t.record(8 * page, ms(100.0), page as u64);
    assert_eq!(t.window(page, max), page);
}

#[test]
fn a_store_profile_fixes_the_window_whatever_the_requests_take() {
    let ms = std::time::Duration::from_millis;
    // 30 ms at 100 MB/s is 3 MB: 4 MiB, between a page and 8 MiB.
    let slow = StoreProfile {
        latency: ms(30),
        bytes_per_sec: 100e6,
    };
    assert_eq!(slow.window(256 << 10, 8 << 20), 4 << 20);
    // Clamped to the page and to the largest request, and to the ends
    // for values that are not numbers.
    let local = StoreProfile {
        latency: ms(0),
        bytes_per_sec: 1e9,
    };
    assert_eq!(local.window(256 << 10, 8 << 20), 256 << 10);
    let far = StoreProfile {
        latency: ms(500),
        bytes_per_sec: 1e12,
    };
    assert_eq!(far.window(256 << 10, 8 << 20), 8 << 20);
    let inf = StoreProfile {
        latency: ms(5),
        bytes_per_sec: f64::INFINITY,
    };
    assert_eq!(inf.window(256 << 10, 8 << 20), 8 << 20);
    let nan = StoreProfile {
        latency: ms(0),
        bytes_per_sec: f64::INFINITY,
    };
    assert_eq!(nan.window(256 << 10, 8 << 20), 256 << 10);

    let tmp = lucene_util::test_support::TempDir::new("page-cache-profile");
    let config = |store_profile, download_window| PageCacheConfig {
        page_size: 100,
        max_request: 1 << 20,
        download_window,
        store_profile,
        disk_dir: Some(tmp.path().join("cache")),
        ..PageCacheConfig::default()
    };
    let profile = StoreProfile {
        latency: ms(1),
        bytes_per_sec: 400_000.0,
    };
    let cache = PageCache::new(config(Some(profile), None)).unwrap();
    assert_eq!(cache.download_window(), 512);
    // Requests that take wildly different times (which would move the
    // live tuner) leave it alone.
    let data = bytes(100_000);
    let g = Geometry::plain(Arc::from("f"), 100_000, 100);
    for (i, nap) in [0u64, 20, 0, 40, 5].into_iter().enumerate() {
        let slow_fetch = |off: u64, len: usize| {
            std::thread::sleep(std::time::Duration::from_millis(nap));
            let a = off as usize;
            Ok(data[a..a + len].to_vec())
        };
        cache
            .stored_pages(
                &g,
                &[
                    (i as u64) * 200,
                    (i as u64) * 200 + 1,
                    (i as u64) * 200 + 2,
                    (i as u64) * 200 + 3,
                    (i as u64) * 200 + 4,
                    (i as u64) * 200 + 5,
                ],
                &slow_fetch,
            )
            .unwrap();
        assert_eq!(cache.download_window(), 512);
    }
    // An explicit window wins over a profile.
    assert_eq!(
        PageCache::new(config(Some(profile), Some(300)))
            .unwrap()
            .download_window(),
        300
    );
}

#[test]
fn a_cache_lists_what_it_holds_and_takes_a_page_it_was_handed() {
    let tmp = lucene_util::test_support::TempDir::new("cache-resident");
    let config = PageCacheConfig {
        disk_dir: Some(tmp.path().join("d")),
        ..PageCacheConfig::default()
    };
    let cache = PageCache::new(config).unwrap();
    assert!(cache.resident_pages(&["f"], 10)[0].is_empty());
    assert!(cache.peek_page("f", 0).is_none());
    cache.insert_page("f/1", 0, &[7; 100]);
    cache.insert_page("f/1", 1, &[8; 50]);
    assert_eq!(&cache.peek_page("f/1", 1).unwrap()[..], &[8; 50][..]);
    let held = cache.resident_pages(&["f/"], 10).remove(0);
    assert_eq!(held.len(), 2);
    assert!(held.iter().all(|p| p.hits == 0), "neither was read");
    assert_eq!(cache.stats().disk_hits, 0, "a peek is not a hit");
    let bare = PageCache::new(PageCacheConfig::default()).unwrap();
    bare.insert_page("f/1", 0, &[7; 100]);
    assert!(bare.peek_page("f/1", 0).is_none() && bare.resident_pages(&["f"], 10)[0].is_empty());
}
