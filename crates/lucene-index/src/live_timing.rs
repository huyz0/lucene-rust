//! Opt-in timing of realtime search phases: with `LUCENE_RUST_LIVE_TIMING`
//! set, each phase's count, total and worst time are summed and printed to
//! standard error every five seconds.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// A timed phase.
#[derive(Debug, Clone, Copy)]
pub enum Phase {
    /// Opening a view, slot locks included.
    ViewOpen = 0,
    /// Waiting for every slot's lock to open a view.
    ViewOpenLocks,
    /// Reading the requested terms of a view's buffers.
    Snapshot,
    /// Writing a chunk segment under its slot's lock.
    ChunkWrite,
    /// Building a view's reader for aggregations and sorts.
    ViewReader,
    /// A plain scored search over a view.
    PlainSearch,
    /// Building the postings overlays of a view's chunks.
    Overlay,
    /// Waiting for the lock of the buffers being flushed in a view or a
    /// snapshot.
    CoreWait,
    /// A view reader's chunks, written or found.
    ReaderChunks,
    /// A view reader's read of the buffers.
    ReaderSnapshot,
    /// A view reader's chunks given their postings.
    ReaderOverlay,
    /// A view reader's chunks appended to the reader.
    ReaderAppend,
    /// A view reader's live documents.
    ReaderLiveDocs,
}

const NAMES: [&str; 13] = [
    "view_open",
    "view_open_locks",
    "snapshot",
    "chunk_write",
    "view_reader",
    "plain_search",
    "overlay",
    "core_wait",
    "r_chunks",
    "r_snapshot",
    "r_overlay",
    "r_append",
    "r_live_docs",
];

struct Sums {
    count: [AtomicU64; 13],
    nanos: [AtomicU64; 13],
    max: [AtomicU64; 13],
    printing: AtomicBool,
    start: Instant,
    printed: AtomicU64,
}

fn sums() -> Option<&'static Sums> {
    static S: OnceLock<Option<Sums>> = OnceLock::new();
    S.get_or_init(|| {
        std::env::var_os("LUCENE_RUST_LIVE_TIMING").map(|_| Sums {
            count: Default::default(),
            nanos: Default::default(),
            max: Default::default(),
            printing: AtomicBool::new(false),
            start: Instant::now(),
            printed: AtomicU64::new(0),
        })
    })
    .as_ref()
}

/// Whether timing is on.
pub fn enabled() -> bool {
    sums().is_some()
}

/// Counts `elapsed` against `phase`.
pub fn record(phase: Phase, elapsed: Duration) {
    let Some(s) = sums() else {
        return;
    };
    let i = phase as usize;
    let n = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
    s.count[i].fetch_add(1, Ordering::Relaxed);
    s.nanos[i].fetch_add(n, Ordering::Relaxed);
    s.max[i].fetch_max(n, Ordering::Relaxed);
    let secs = s.start.elapsed().as_secs();
    let last = s.printed.load(Ordering::Relaxed);
    if secs >= last.saturating_add(5)
        && s.printed
            .compare_exchange(last, secs, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        && !s.printing.swap(true, Ordering::Acquire)
    {
        let mut line = format!("live-timing t={secs}s");
        for (k, name) in NAMES.iter().enumerate() {
            let c = s.count[k].swap(0, Ordering::Relaxed);
            let t = s.nanos[k].swap(0, Ordering::Relaxed);
            let m = s.max[k].swap(0, Ordering::Relaxed);
            if c > 0 {
                let avg = t.checked_div(c).unwrap_or(0);
                line.push_str(&format!(
                    " {name}: n={c} avg={:.2}ms max={:.2}ms;",
                    avg as f64 / 1e6,
                    m as f64 / 1e6
                ));
            }
        }
        eprintln!("{line}");
        s.printing.store(false, Ordering::Release);
    }
}

/// Runs `f`, counting its time against `phase` when timing is on.
pub fn timed<T>(phase: Phase, f: impl FnOnce() -> T) -> T {
    if !enabled() {
        return f();
    }
    let t = Instant::now();
    let out = f();
    record(phase, t.elapsed());
    out
}

/// Counts the time until it is dropped against its phase.
#[derive(Debug)]
pub struct Timer(Option<(Phase, Instant)>);

impl Drop for Timer {
    fn drop(&mut self) {
        if let Some((phase, t)) = self.0 {
            record(phase, t.elapsed());
        }
    }
}

/// A [`Timer`] for `phase`, doing nothing when timing is off.
pub fn start(phase: Phase) -> Timer {
    Timer(enabled().then(|| (phase, Instant::now())))
}
