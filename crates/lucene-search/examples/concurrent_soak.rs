//! The concurrent writer's endurance run (M4's T4.3 and its soak criteria):
//! indexing threads update and delete their own ids at random through one
//! `ConcurrentIndexWriter`, a merge thread merges, and the main thread commits
//! at random intervals. Every commit is checked **exactly**: `commit()`
//! returns the sequence number of the last operation it holds, and the
//! committed index must equal the model built from every operation numbered
//! at or below it -- each id's last version, or no document if its last
//! operation was a delete. Every few commits `CheckIndex` runs too, and the
//! process's resident memory and open file descriptors are reported, so a
//! leak shows as growth over the run.
//!
//! Usage: `concurrent_soak --dir PATH [--duration SECS] [--threads N]
//! [--seed S] [--ids N] [--report SECS]`. Exits non-zero on the first
//! mismatch, naming the id, the expected and the committed state.
// Example code: see `docs/arithmetic-gate.md`'s "Test code" section.
#![allow(clippy::arithmetic_side_effects)]

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use lucene_codecs::field_infos::{FieldInfo, IndexOptions};
use lucene_codecs::stored_fields::{self, Document, FieldValue, StoredField};
use lucene_index::buffered_updates::Term;
use lucene_index::concurrent_writer::ConcurrentIndexWriter;
use lucene_index::index_writer::{IndexWriter, DISABLE_AUTO_FLUSH_MB};
use lucene_index::merge_policy::MergePolicyConfig;
use lucene_index::segment_info::{self, LuceneVersion};
use lucene_index::{check_index, deletes, segment_infos};
use lucene_store::{Directory, FsDirectory};

/// splitmix64, seeded per thread.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// One operation as the thread that issued it saw it: its sequence number,
/// the id, and the version it left (`None`: deleted).
type Op = (i64, String, Option<u32>);

fn fields() -> Vec<FieldInfo> {
    vec![
        FieldInfo {
            index_options: IndexOptions::Docs,
            omit_norms: true,
            ..FieldInfo::new("id", 0)
        },
        FieldInfo {
            index_options: IndexOptions::DocsAndFreqs,
            ..FieldInfo::new("body", 1)
        },
    ]
}

fn doc(id: &str, version: u32) -> Document {
    Document {
        fields: vec![
            StoredField {
                field_number: 0,
                value: FieldValue::String(id.to_string()),
            },
            StoredField {
                field_number: 1,
                value: FieldValue::String(format!("v{version} common")),
            },
        ],
    }
}

/// id -> version of every live document of the latest commit.
fn committed(dir: &FsDirectory) -> Result<(BTreeMap<String, u32>, usize), String> {
    let infos = segment_infos::read_latest(dir).map_err(|e| e.to_string())?;
    let mut out = BTreeMap::new();
    for sci in &infos.segments {
        let name = &sci.segment_name;
        let open = |ext: &str| {
            dir.open(&format!("{name}.{ext}"))
                .map_err(|e| e.to_string())
        };
        let si = segment_info::parse(&open("si")?, &sci.segment_id).map_err(|e| e.to_string())?;
        let (fdt, fdx, fdm) = (open("fdt")?, open("fdx")?, open("fdm")?);
        let reader = stored_fields::open(&fdt, &fdx, &fdm, &sci.segment_id, "")
            .map_err(|e| e.to_string())?;
        let live = if sci.del_gen >= 0 {
            let liv = dir
                .open(&deletes::liv_file_name(name, sci.del_gen))
                .map_err(|e| e.to_string())?;
            Some(
                lucene_codecs::live_docs::parse(
                    &liv,
                    &sci.segment_id,
                    sci.del_gen,
                    si.doc_count as usize,
                    sci.del_count as usize,
                )
                .map_err(|e| e.to_string())?,
            )
        } else {
            None
        };
        for d in 0..si.doc_count {
            if live.as_ref().is_some_and(|l| !l.get(d as usize)) {
                continue;
            }
            let stored = reader.document(d).map_err(|e| e.to_string())?;
            let text = |n: i32| match stored.fields.iter().find(|f| f.field_number == n) {
                Some(StoredField {
                    value: FieldValue::String(s),
                    ..
                }) => Ok(s.clone()),
                other => Err(format!("{name} doc {d}: field {n} is {other:?}")),
            };
            let id = text(0)?;
            let version: u32 = text(1)?
                .trim_start_matches('v')
                .split(' ')
                .next()
                .and_then(|v| v.parse().ok())
                .ok_or_else(|| format!("{name} doc {d}: bad body"))?;
            if out.insert(id.clone(), version).is_some() {
                return Err(format!("id {id} is live twice (segment {name}, doc {d})"));
            }
        }
    }
    Ok((out, infos.segments.len()))
}

/// `VmRSS` in KiB and the number of open file descriptors of this process.
fn resources() -> (u64, usize) {
    let rss = std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(0);
    let fds = std::fs::read_dir("/proc/self/fd").map_or(0, |d| d.count());
    (rss, fds)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let value = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let path = value("--dir").expect("--dir PATH");
    let duration =
        Duration::from_secs(value("--duration").map_or(60, |v| v.parse().expect("--duration")));
    let threads: usize = value("--threads").map_or(4, |v| v.parse().expect("--threads"));
    let seed: u64 = value("--seed").map_or(1, |v| v.parse().expect("--seed"));
    let ids: u64 = value("--ids").map_or(500, |v| v.parse().expect("--ids"));
    let report =
        Duration::from_secs(value("--report").map_or(60, |v| v.parse().expect("--report")));

    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).expect("create dir");
    let dir = FsDirectory::open(&path);
    let mut single = IndexWriter::open(
        &dir,
        fields(),
        "Lucene104",
        LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
    )
    .expect("open writer");
    single.set_postings_field(Some("id")).expect("id");
    single.add_postings_field("body").expect("body");
    single.set_max_buffered_docs(64).expect("max buffered docs");
    single
        .set_ram_buffer_size_mb(DISABLE_AUTO_FLUSH_MB)
        .expect("ram buffer");
    single.set_merge_policy(Some(MergePolicyConfig {
        max_merge_at_once: 4,
        segments_per_tier: 4,
        floor_segment_size: 1 << 30,
        ..MergePolicyConfig::default()
    }));
    let w = ConcurrentIndexWriter::new(single, threads).expect("concurrent writer");

    let logs: Vec<Mutex<Vec<Op>>> = (0..threads).map(|_| Mutex::new(Vec::new())).collect();
    let ops_done = AtomicU64::new(0);
    let stop = AtomicBool::new(false);
    let start = Instant::now();
    let (rss0, fds0) = resources();
    println!(
        "concurrent_soak: {threads} threads x {ids} ids, seed {seed}, {}s; start rss {rss0} KiB, {fds0} fds",
        duration.as_secs()
    );

    let outcome: Result<(u64, u64, u64, usize, usize), String> = std::thread::scope(|scope| {
        let merger = scope.spawn(|| w.run_merges(&stop));
        for t in 0..threads {
            let (w, logs, ops_done, stop) = (&w, &logs, &ops_done, &stop);
            scope.spawn(move || {
                let mut rng = Rng(seed ^ ((t as u64 + 1) << 32));
                let mut versions = vec![0u32; ids as usize];
                while !stop.load(Ordering::Acquire) {
                    let k = rng.below(ids) as usize;
                    let id = format!("t{t}x{k}");
                    let term = Term::new("id", id.clone().into_bytes());
                    let op = if rng.below(10) < 8 {
                        versions[k] += 1;
                        let seq = w
                            .update_document(term, doc(&id, versions[k]))
                            .expect("update");
                        (seq, id, Some(versions[k]))
                    } else {
                        let seq = w.delete_documents_by_term(&[term]).expect("delete");
                        (seq, id, None)
                    };
                    logs[t].lock().unwrap().push(op);
                    ops_done.fetch_add(1, Ordering::Relaxed);
                }
            });
        }

        // The committer: commit, then check the commit against the model.
        let _stop = StopOnDrop(&stop);
        let mut rng = Rng(seed);
        let mut model: BTreeMap<String, u32> = BTreeMap::new();
        let (mut commits, mut checks) = (0u64, 0u64);
        let (mut rss_max, mut fds_max) = (rss0, fds0);
        let mut next_report = start + report;
        let mut last_segments = 0;
        while start.elapsed() < duration {
            std::thread::sleep(Duration::from_millis(20 + rng.below(180)));
            let last = w.commit().map_err(|e| format!("commit: {e}"))?;
            commits += 1;
            for log in &logs {
                let mut log = log.lock().unwrap();
                // Each id belongs to one thread, so its log is in seq order.
                let keep = log.partition_point(|&(seq, _, _)| seq <= last);
                for (_, id, version) in log.drain(..keep) {
                    match version {
                        Some(v) => model.insert(id, v),
                        None => model.remove(&id),
                    };
                }
            }
            let (got, segments) = committed(&dir)?;
            last_segments = segments;
            if got != model {
                let bad = model
                    .iter()
                    .map(|(k, v)| (k, Some(*v), got.get(k).copied()))
                    .chain(
                        got.iter()
                            .filter(|(k, _)| !model.contains_key(*k))
                            .map(|(k, v)| (k, None, Some(*v))),
                    )
                    .find(|(_, want, have)| want != have);
                return Err(format!(
                    "commit {commits} (last seq {last}): {} live, model {}; first difference {bad:?}",
                    got.len(),
                    model.len()
                ));
            }
            if commits % 25 == 0 {
                for result in check_index::check_directory(&dir).map_err(|e| e.to_string())? {
                    if !result.all_passed() {
                        return Err(format!(
                            "CheckIndex after commit {commits}: {:?}",
                            result.failures()
                        ));
                    }
                }
                checks += 1;
            }
            let (rss, fds) = resources();
            rss_max = rss_max.max(rss);
            fds_max = fds_max.max(fds);
            if Instant::now() >= next_report {
                next_report += report;
                println!(
                    "concurrent_soak: {:>6}s {commits} commits ok, {} ops, {} live, {segments} segments, rss {rss} KiB, {fds} fds",
                    start.elapsed().as_secs(),
                    ops_done.load(Ordering::Relaxed),
                    model.len()
                );
            }
        }
        stop.store(true, Ordering::Release);
        let merges = merger
            .join()
            .map_err(|_| "merge thread panicked".to_string())?
            .map_err(|e| format!("merge thread: {e}"))?;
        let _ = last_segments;
        Ok((commits, checks, merges as u64, rss_max as usize, fds_max))
    });

    match outcome {
        Ok((commits, checks, merges, rss_max, fds_max)) => {
            let (rss, fds) = resources();
            println!(
                "concurrent_soak: ok -- {}s, {commits} commits checked exactly, {checks} CheckIndex runs, {merges} merges, {} ops; rss {rss0} -> {rss} KiB (max {rss_max}), fds {fds0} -> {fds} (max {fds_max})",
                start.elapsed().as_secs(),
                ops_done.load(Ordering::Relaxed)
            );
        }
        Err(e) => {
            println!("concurrent_soak: FAIL -- {e}");
            std::process::exit(1);
        }
    }
}

struct StopOnDrop<'a>(&'a AtomicBool);

impl Drop for StopOnDrop<'_> {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}
