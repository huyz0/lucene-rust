#![cfg(test)]
// Test fixtures' own arithmetic -- see `docs/arithmetic-gate.md`'s "Test code".
#![allow(clippy::arithmetic_side_effects)]
use super::*;
use std::sync::atomic::AtomicUsize;

struct Pattern {
    len: usize,
    page: usize,
    fills: AtomicUsize,
    calls: AtomicUsize,
    fail_first: AtomicUsize,
    background: AtomicBool,
}

fn byte(i: usize) -> u8 {
    (i.wrapping_mul(2_654_435_761) >> 13) as u8
}

impl PageFiller for Pattern {
    fn fill(&self, pages: &[u64]) -> Result<Vec<Arc<[u8]>>> {
        if self.fail_first.load(Ordering::Relaxed) > 0 {
            self.fail_first.fetch_sub(1, Ordering::Relaxed);
            return Err(Error::Io(std::io::Error::other("transient")));
        }
        self.fills.fetch_add(pages.len(), Ordering::Relaxed);
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(pages
            .iter()
            .map(|&p| {
                let s = p as usize * self.page;
                let e = (s + self.page).min(self.len);
                (s..e).map(byte).collect::<Vec<u8>>().into()
            })
            .collect())
    }

    fn fill_background(&self, pages: Vec<u64>, sink: PageSink, done: BackgroundDone) -> bool {
        if !self.background.load(Ordering::Relaxed) {
            return false;
        }
        let me = Pattern {
            len: self.len,
            page: self.page,
            fills: AtomicUsize::new(0),
            calls: AtomicUsize::new(0),
            fail_first: AtomicUsize::new(0),
            background: AtomicBool::new(false),
        };
        self.fills.fetch_add(pages.len(), Ordering::Relaxed);
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(2));
            done(me.fill_into(&pages, &sink));
        });
        true
    }
}

fn pattern(len: usize, page: usize) -> Arc<Pattern> {
    Arc::new(Pattern {
        len,
        page,
        fills: AtomicUsize::new(0),
        calls: AtomicUsize::new(0),
        fail_first: AtomicUsize::new(0),
        background: AtomicBool::new(false),
    })
}

#[test]
fn a_file_is_wholly_resident_once_every_page_is_read_within_budget() {
    let Some(pager) = Pager::spawn(false) else {
        return;
    };
    let page = 16 * 1024;
    let len = 8 * page;
    let input = pager.open(len, page, pattern(len, page), "whole").unwrap();
    assert!(!input.file.all_resident());
    let bytes = input.bytes();
    for i in 0..len / page {
        assert_eq!(bytes[i * page], byte(i * page));
    }
    // A handler marks a page live just after waking its reader.
    let waited = std::time::Instant::now();
    while !input.file.all_resident() {
        assert!(waited.elapsed() < Duration::from_secs(5), "never resident");
        std::thread::yield_now();
    }
    // Over budget, residency can change at any fault: not settled.
    pager.set_budget(4 * page as u64);
    assert!(!input.file.all_resident() || pager.resident_bytes() <= pager.budget());
}

#[test]
fn a_resident_page_with_a_hole_is_filled_again() {
    let Some(pager) = Pager::spawn(false) else {
        return;
    };
    let page = 16 * 1024;
    let len = 4 * page;
    let p = pattern(len, page);
    let os = pager.os_page;
    if os >= page {
        // No room for a hole inside one page.
        return;
    }
    let input = pager.open(len, page, p.clone(), "hole").unwrap();
    let bytes = input.bytes();
    assert_eq!(bytes[5], byte(5));
    // The handler marks the page after the copy that woke this thread.
    let waited = std::time::Instant::now();
    while !input.file.live[0].load(Ordering::Acquire) {
        assert!(
            waited.elapsed() < Duration::from_secs(5),
            "page 0 never marked"
        );
        std::thread::yield_now();
    }
    // What an eviction racing an install can leave: the page marked
    // resident with one of its OS pages gone.
    // SAFETY: drops one OS page inside the file's live mapping.
    unsafe {
        libc::madvise(
            bytes.as_ptr().add(os) as *mut libc::c_void,
            os,
            libc::MADV_DONTNEED,
        )
    };
    assert_eq!(bytes[os + 7], byte(os + 7));
    assert!(bytes.iter().enumerate().all(|(i, b)| *b == byte(i)));
}

#[test]
fn faults_and_will_need_fill_the_same_bytes_under_a_tiny_budget() {
    // A pager of its own: the budget below would starve other tests.
    let Some(pager) = Pager::spawn(false) else {
        eprintln!("userfaultfd unavailable; skipping");
        return;
    };
    let page = 16 * 1024;
    let lens = [0usize, 1, page - 1, page, page + 1, 300_000, 1 << 20];
    let files: Vec<_> = lens
        .iter()
        .map(|&len| {
            let p = pattern(len, page);
            let input = pager.open(len, page, p.clone(), "f").unwrap();
            (len, p, input)
        })
        .collect();
    pager.set_budget(64 * 1024);
    std::thread::scope(|s| {
        for t in 0..6usize {
            let files = &files;
            s.spawn(move || {
                for (len, _, input) in files {
                    let bytes = input.bytes();
                    assert_eq!(bytes.len(), *len);
                    if t % 2 == 0 {
                        input.will_need(0, *len).unwrap();
                    }
                    let step = 4096 * (1 + t % 5) + 17;
                    let mut i = (t * 977) % len.max(&1);
                    while i < *len {
                        assert_eq!(bytes[i], byte(i), "[{i}]");
                        i += step;
                    }
                    assert!(bytes.iter().enumerate().all(|(i, b)| *b == byte(i)));
                }
            });
        }
    });
    pager.set_budget(DEFAULT_BUDGET);
    assert!(files
        .iter()
        .any(|(_, p, _)| p.fills.load(Ordering::Relaxed) > 0));
    assert!(pager.faults() > 0 || files.iter().all(|(l, _, _)| *l == 0));
}

#[test]
fn sequential_faults_read_ahead_and_random_ones_do_not() {
    let Some(pager) = Pager::global() else {
        return;
    };
    let page = 16 * 1024;
    let len = 64 * page;
    let seq = pattern(len, page);
    let input = pager
        .open_with_readahead(len, page, seq.clone(), "seq", 8)
        .unwrap();
    let bytes = input.bytes();
    // A scan in order: windows of 1, 2, 4, 8, 8, ... pages.
    assert!(bytes.iter().enumerate().all(|(i, b)| *b == byte(i)));
    let calls = seq.calls.load(Ordering::Relaxed);
    assert!(calls <= 12, "{calls} fills for 64 pages read in order");
    assert_eq!(seq.fills.load(Ordering::Relaxed), 64);
    // Touches far apart never start a run: one page each.
    let rnd = pattern(len, page);
    let input = pager
        .open_with_readahead(len, page, rnd.clone(), "rnd", 8)
        .unwrap();
    let bytes = input.bytes();
    for p in [40usize, 3, 61, 17, 29] {
        assert_eq!(bytes[p * page + 5], byte(p * page + 5));
    }
    assert_eq!(rnd.fills.load(Ordering::Relaxed), 5);
}

#[test]
fn interleaved_scans_of_one_file_each_read_ahead() {
    let Some(pager) = Pager::global() else {
        return;
    };
    let page = 16 * 1024;
    let len = 256 * page;
    let seq = pattern(len, page);
    let input = pager
        .open_with_readahead(len, page, seq.clone(), "interleaved", 8)
        .unwrap();
    let bytes = input.bytes();
    // Four readers each scanning a quarter of the file in order, one page
    // at a time in turn: each keeps its own run.
    let quarter = len / 4;
    for step in 0..quarter / page {
        for q in 0..4 {
            let at = q * quarter + step * page;
            assert_eq!(bytes[at], byte(at));
        }
    }
    let calls = seq.calls.load(Ordering::Relaxed);
    assert!(calls <= 60, "{calls} fills for 256 pages in four runs");
    assert_eq!(seq.fills.load(Ordering::Relaxed), 256);
}

#[test]
fn a_scan_keeps_its_window_among_readers_at_random_places() {
    let Some(pager) = Pager::global() else {
        return;
    };
    let page = 16 * 1024;
    let len = 4096 * page;
    // A scan of the first 256 pages and, after every four of its pages, twelve reads at
    // places far apart in the rest of the file; the same random reads alone
    // on a second file say what they cost by themselves.
    let run = |label: &str, scan: bool| {
        let p = pattern(len, page);
        let input = pager
            .open_with_readahead(len, page, p.clone(), label, 8)
            .unwrap();
        let bytes = input.bytes();
        let mut x = 12_345usize;
        for burst in 0..64 {
            if scan {
                for page_no in burst * 4..burst * 4 + 4 {
                    assert_eq!(bytes[page_no * page], byte(page_no * page));
                }
            }
            for _ in 0..12 {
                x = x
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let q = 256 + (x >> 33) % 3840;
                assert_eq!(bytes[q * page + 9], byte(q * page + 9));
            }
        }
        p.calls.load(Ordering::Relaxed)
    };
    let alone = run("random-alone", false);
    let with_scan = run("scan-among-random", true);
    // The scan's own windows add a few dozen fills, not one per page.
    assert!(
        with_scan <= alone + 40,
        "{with_scan} fills with a 256-page scan, {alone} for the random reads alone"
    );
}

#[test]
fn background_read_ahead_reads_each_page_once() {
    let Some(pager) = Pager::global() else {
        return;
    };
    let page = 16 * 1024;
    let len = 64 * page;
    let seq = pattern(len, page);
    seq.background.store(true, Ordering::Relaxed);
    let input = pager
        .open_with_readahead(len, page, seq.clone(), "bg", 8)
        .unwrap();
    let bytes = input.bytes();
    assert!(bytes.iter().enumerate().all(|(i, b)| *b == byte(i)));
    assert_eq!(seq.fills.load(Ordering::Relaxed), 64);
}

#[test]
fn a_page_announced_again_survives_eviction() {
    let Some(pager) = Pager::spawn(false) else {
        return;
    };
    let page = 16 * 1024;
    let len = 16 * page;
    let p = pattern(len, page);
    let input = pager.open(len, page, p.clone(), "clock").unwrap();
    pager.set_budget(4 * page as u64);
    // Pages 0..4 resident, page 0 the oldest; announcing it again marks it.
    for i in 0..4 {
        input.will_need(i * page, 1).unwrap();
    }
    input.will_need(0, 1).unwrap();
    // Page 4 needs room: page 1, not the announced page 0, goes.
    input.will_need(4 * page, 1).unwrap();
    let before = p.fills.load(Ordering::Relaxed);
    input.will_need(0, 1).unwrap();
    assert_eq!(
        p.fills.load(Ordering::Relaxed),
        before,
        "page 0 was evicted"
    );
    input.will_need(page, 1).unwrap();
    assert_eq!(p.fills.load(Ordering::Relaxed), before + 1, "page 1 stayed");
    assert!(input.bytes().iter().enumerate().all(|(i, b)| *b == byte(i)));
}

#[test]
fn a_page_back_soon_after_eviction_survives_a_scan() {
    let Some(pager) = Pager::spawn(false) else {
        return;
    };
    let page = 16 * 1024;
    let len = 64 * page;
    let p = pattern(len, page);
    let input = pager.open(len, page, p.clone(), "scan").unwrap();
    pager.set_budget(8 * page as u64);
    let bytes = input.bytes();
    // Page 0 is read, pushed out by a scan, and read again: in use.
    assert_eq!(bytes[5], byte(5));
    for i in 1..12 {
        assert_eq!(bytes[i * page], byte(i * page));
    }
    // Eviction runs on the handler threads after each install.
    let waited = std::time::Instant::now();
    while input.file.live[0].load(Ordering::Acquire) || pager.resident_bytes() > pager.budget() {
        assert!(
            waited.elapsed() < Duration::from_secs(5),
            "page 0 never evicted"
        );
        std::thread::yield_now();
    }
    assert_eq!(bytes[5], byte(5));
    // A longer scan of pages read once does not push it out again.
    let before = p.fills.load(Ordering::Relaxed);
    for i in 12..64 {
        assert_eq!(bytes[i * page], byte(i * page));
    }
    let scanned = p.fills.load(Ordering::Relaxed);
    assert_eq!(bytes[5], byte(5));
    assert_eq!(
        p.fills.load(Ordering::Relaxed),
        scanned,
        "page 0 was evicted"
    );
    assert!(scanned - before >= 52);
}

#[test]
fn will_need_reports_a_fetch_error_and_a_fault_retries_it() {
    let Some(pager) = Pager::global() else {
        return;
    };
    let page = 16 * 1024;
    let p = pattern(100_000, page);
    let input = pager.open(100_000, page, p.clone(), "f").unwrap();
    p.fail_first.store(1, Ordering::Relaxed);
    assert!(input.will_need(0, 10).is_err());
    // The next fill fails once more; the fault handler retries it.
    p.fail_first.store(1, Ordering::Relaxed);
    assert_eq!(input.bytes()[50_000], byte(50_000));
    // Resident pages are not fetched again.
    input.will_need(0, 100).unwrap();
    let before = p.fills.load(Ordering::Relaxed);
    input.will_need(0, 100).unwrap();
    assert_eq!(p.fills.load(Ordering::Relaxed), before);
    assert!(input.will_need(200_000, 10).is_ok());
    assert!(input.will_need(5, 0).is_ok());
}

#[test]
fn a_page_that_cannot_be_read_fails_its_reads_then_recovers() {
    let Some(pager) = Pager::spawn(false) else {
        return;
    };
    let page = 16 * 1024;
    let p = pattern(100_000, page);
    let name = format!(
        "{}unreadable{}f.vec",
        std::path::MAIN_SEPARATOR,
        std::path::MAIN_SEPARATOR
    );
    let root = format!("{}unreadable", std::path::MAIN_SEPARATOR);
    let input = pager.open(100_000, page, p.clone(), &name).unwrap();
    let started = crate::read_failures::epoch();
    // Every attempt fails: a checked call is served zeros, not an abort.
    p.fail_first.store(usize::MAX, Ordering::Relaxed);
    crate::read_failures::set_checked(true);
    assert_eq!(input.bytes()[20_000], 0);
    crate::read_failures::set_checked(false);
    assert_eq!(pager.stats().read_errors, 1);
    let recorded = crate::read_failures::failure_under(&root, started).expect("recorded");
    assert!(recorded.contains("could not be read"), "{recorded}");
    // An announced read of the page fails directly.
    assert!(input.will_need(20_000, 1).is_err());
    // The disk recovers: once the zeros are dropped, the page reads.
    p.fail_first.store(0, Ordering::Relaxed);
    pager.drop_unreadable(Duration::ZERO);
    assert_eq!(input.bytes()[20_000], byte(20_000));
    input.will_need(0, 100_000).unwrap();
    // A call that started during the failure still sees it; a new one
    // does not.
    assert!(crate::read_failures::failure_under(&root, started).is_some());
    assert_eq!(
        crate::read_failures::failure_under(&root, crate::read_failures::epoch()),
        None
    );
}

#[test]
fn a_thread_outside_a_checked_call_waits_for_the_disk() {
    let Some(pager) = Pager::spawn(false) else {
        return;
    };
    let page = 16 * 1024;
    let p = pattern(100_000, page);
    let input = pager
        .open(100_000, page, p.clone(), "/waits/f.doc")
        .unwrap();
    p.fail_first.store(usize::MAX, Ordering::Relaxed);
    let healer = {
        let p = p.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(3000));
            p.fail_first.store(0, Ordering::Relaxed);
        })
    };
    // Past the retries the read keeps waiting, then gets the real bytes.
    assert_eq!(input.bytes()[40_000], byte(40_000));
    healer.join().unwrap();
    assert!(pager.stats().read_errors >= 1);
    assert_eq!(
        crate::read_failures::failure_under("/waits", 0),
        None,
        "nothing was served as zeros"
    );
}

#[test]
fn prefetch_by_address_fetches_each_file_s_ranges_in_one_batch() {
    let Some(pager) = Pager::global() else {
        return;
    };
    let page = 16 * 1024;
    let (a, b) = (pattern(200_000, page), pattern(200_000, page));
    let fa = pager.open(200_000, page, a.clone(), "a").unwrap();
    let fb = pager.open(200_000, page, b.clone(), "b").unwrap();
    let (x, y) = (fa.bytes(), fb.bytes());
    let heap = vec![1u8; 10];
    assert!(is_paged(x) && is_paged(&y[100..]));
    assert!(!is_paged(&heap) && !is_paged(&[]));
    prefetch(&[&x[..1], &x[100_000..150_000], &y[190_000..], &heap, &[]]).unwrap();
    // Pages 0 and 6..=9 of `a`, 11..=12 of `b`: one fill call each.
    assert_eq!(a.fills.load(Ordering::Relaxed), 5);
    assert_eq!(a.calls.load(Ordering::Relaxed), 1);
    assert_eq!(b.fills.load(Ordering::Relaxed), 2);
    assert_eq!(b.calls.load(Ordering::Relaxed), 1);
    assert_eq!(x[120_000], byte(120_000));
    assert_eq!(y[199_999], byte(199_999));
    // A fetch error comes back to the caller.
    b.fail_first.store(1, Ordering::Relaxed);
    assert!(prefetch(&[&y[50_000..50_001]]).is_err());
    assert_eq!(y[50_000], byte(50_000));
}

#[test]
fn pages_must_be_whole_os_pages() {
    let Some(pager) = Pager::global() else {
        return;
    };
    assert!(!pager.supports(100));
    assert!(pager.open(10, 100, pattern(10, 100), "f").is_err());
    assert!(pager.resident_bytes() <= DEFAULT_BUDGET);
}

#[test]
fn faults_are_served_on_the_faulting_thread_or_by_handler_threads() {
    for faults in [Faults::OnThread, Faults::Handlers] {
        let Some(pager) = Pager::spawn_serving(false, faults) else {
            return;
        };
        assert_eq!(pager.serves_on_thread(), faults == Faults::OnThread);
        let page = 16 * 1024;
        let len = 64 * page;
        let p = pattern(len, page);
        let input = pager.open(len, page, p.clone(), "modes").unwrap();
        // A budget below the file: threads fault, evict and refault.
        pager.set_budget(8 * page as u64);
        let bytes = input.bytes();
        std::thread::scope(|s| {
            for t in 0..4usize {
                s.spawn(move || {
                    for round in 0..3usize {
                        for i in (0..len).step_by(4096) {
                            let at = (i + t * 7919 + round * 104_729) % len;
                            assert_eq!(bytes[at], byte(at), "{faults:?} at {at}");
                        }
                    }
                });
            }
        });
        let stats = pager.stats();
        assert!(
            stats.faults > 0 && stats.evictions > 0,
            "{faults:?} {stats:?}"
        );
        assert!(stats.fault_nanos > 0, "{faults:?} {stats:?}");
        // A page that cannot be read is served as zeros to a checked
        // call, however its fault is served.
        pager.set_budget(DEFAULT_BUDGET);
        let q = pattern(100_000, page);
        let name = format!("/modes-{faults:?}/f.doc");
        let unreadable = pager.open(100_000, page, q.clone(), &name).unwrap();
        q.fail_first.store(usize::MAX, Ordering::Relaxed);
        crate::read_failures::set_checked(true);
        assert_eq!(unreadable.bytes()[20_000], 0);
        crate::read_failures::set_checked(false);
        assert!(crate::read_failures::failure_under(&format!("/modes-{faults:?}"), 0).is_some());
        q.fail_first.store(0, Ordering::Relaxed);
        pager.drop_unreadable(Duration::ZERO);
        assert_eq!(unreadable.bytes()[20_000], byte(20_000));
    }
}

#[test]
fn a_user_mode_only_pager_serves_user_reads() {
    let Some(pager) = Pager::spawn(true) else {
        return;
    };
    let page = 16 * 1024;
    let p = pattern(200_000, page);
    let input = pager.open(200_000, page, p, "f").unwrap();
    assert!(input.bytes().iter().enumerate().all(|(i, b)| *b == byte(i)));
}
