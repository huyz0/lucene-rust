//! Time and peak memory of a force merge to one segment over a copy of a real
//! index: `MERGE_INDEX=/path/to/index cargo test --release -p lucene-index
//! --test force_merge_peak_timing -- --ignored --nocapture`.

use lucene_index::index_writer::IndexWriter;
use lucene_index::merge_policy::upgrade::LATEST;
use lucene_store::MmapDirectory;

fn cpu_seconds() -> f64 {
    let s = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let rest = s.rsplit(')').next().unwrap_or("");
    let f: Vec<&str> = rest.split_whitespace().collect();
    let ticks = |i: usize| f.get(i).and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
    (ticks(11) + ticks(12)) / 100.0
}

fn status_mb(key: &str) -> u64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    s.lines()
        .find(|l| l.starts_with(key))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse::<u64>().ok())
        .map_or(0, |kb| kb / 1024)
}

#[test]
#[ignore = "timing harness; needs MERGE_INDEX"]
fn force_merge_of_a_real_index() {
    let Ok(src) = std::env::var("MERGE_INDEX") else {
        return;
    };
    let tmp = std::env::temp_dir().join(format!("force-merge-peak-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    for e in std::fs::read_dir(&src).unwrap() {
        let e = e.unwrap();
        if e.file_name() != "write.lock" {
            std::fs::copy(e.path(), tmp.join(e.file_name())).unwrap();
        }
    }
    let dir = MmapDirectory::open(&tmp);
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", LATEST).unwrap();
    let _ = std::fs::write("/proc/self/clear_refs", "5");
    let before = status_mb("VmRSS");
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let sampler = {
        let stop = std::sync::Arc::clone(&stop);
        std::thread::spawn(move || {
            let mut peak = 0;
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                peak = peak.max(status_mb("RssAnon"));
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            peak
        })
    };
    let cpu = cpu_seconds();
    let t = std::time::Instant::now();
    w.force_merge(1).unwrap();
    w.commit().unwrap();
    let wall = t.elapsed();
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    let peak_anon = sampler.join().unwrap();
    eprintln!(
        "force merge {:.2} s, {:.2} CPU-s, RSS before {} MB, peak RSS {} MB, peak anon {} MB",
        wall.as_secs_f64(),
        cpu_seconds() - cpu,
        before,
        status_mb("VmHWM"),
        peak_anon
    );
    drop(w);
    let _ = std::fs::remove_dir_all(&tmp);
}
