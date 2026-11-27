//! Update workloads against one `ConcurrentIndexWriter`, with and without the
//! hot tier: a counter updated at a fixed rate or flat out on a hot set of
//! rows, a bulk update of a share of the corpus, and a trickle of updates.
//!
//! ```text
//! cargo build -p lucene-index --release --example hot_tier_bench
//! hot_tier_bench --docs 1000000 --tier off|BYTES --workload stream|bulk|trickle \
//!     [--rate R] [--seconds S] [--hot-set N] [--hot-share 0.8] [--share 0.1] \
//!     [--threads 4] [--refresh-ms 1000] [--live]
//! ```
//!
//! The documents are what the engine writes: an `_id` term, a body of about 70
//! terms with positions and a norm, three numeric doc values (the counter,
//! `_seq_no`, `_version`) and a 600-byte stored `_source`. An update is
//! `soft_update_explicit_documents`: the new version is added and every
//! earlier one gets `__soft_deletes`. A refresher thread commits every
//! `--refresh-ms` (the plugin's refresh is a commit, which drains the tier) and a merge
//! thread merges all the while. Ids are owned by one thread each, as the
//! engine's per-id lock makes them.
//!
//! Output: one JSON object per run.
#![allow(clippy::arithmetic_side_effects)] // A measurement harness's own sizes.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use lucene_codecs::field_infos::{DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions};
use lucene_codecs::stored_fields::{FieldValue, StoredField};
use lucene_index::buffered_updates::{DocValuesUpdate, Term};
use lucene_index::concurrent_writer::ConcurrentIndexWriter;
use lucene_index::hot_tier::HotTierConfig;
use lucene_index::index_writer::{
    ExplicitDocument, ExplicitFields, IndexWriter, InvertedField, InvertedTerm,
};
use lucene_index::segment_info::LuceneVersion;
use lucene_index::segment_infos;
use lucene_store::FsDirectory;

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
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[derive(Clone, Copy)]
struct Fields {
    id: i32,
    body: i32,
    num: i32,
    seq: i32,
    ver: i32,
    source: i32,
}

fn arg<T: std::str::FromStr>(args: &[String], flag: &str, default: T) -> T {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn cpu_seconds() -> f64 {
    let s = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let rest = s.rsplit(')').next().unwrap_or("");
    let f: Vec<&str> = rest.split_whitespace().collect();
    let ticks = |i: usize| f.get(i).and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
    (ticks(11) + ticks(12)) / 100.0
}

fn status_kb(key: &str) -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with(key))
                .and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse().ok()))
        })
        .unwrap_or(0)
}

/// 2000 body templates of 20 to 120 Zipf-drawn terms, so building a document
/// costs a clone as decoding the engine's blob does.
fn templates(n: usize) -> Vec<(Vec<InvertedTerm>, i64)> {
    let mut rng = Rng(42);
    let vocab = 200_000u64;
    (0..n)
        .map(|_| {
            let len = 20 + rng.below(100) as usize;
            let mut map: std::collections::BTreeMap<u64, Vec<i32>> = Default::default();
            for pos in 0..len {
                // Zipf-ish: rank = vocab^u
                let rank = (vocab as f64).powf(rng.unit()) as u64;
                map.entry(rank).or_default().push(pos as i32);
            }
            let terms = map
                .into_iter()
                .map(|(rank, positions)| InvertedTerm {
                    term: format!("t{rank}").into_bytes(),
                    freq: positions.len() as i32,
                    positions,
                    offsets: Vec::new(),
                    payloads: Vec::new(),
                })
                .collect();
            (terms, len as i64)
        })
        .collect()
}

fn doc(
    f: &Fields,
    tpl: &[(Vec<InvertedTerm>, i64)],
    i: u64,
    counter: i64,
    seq: i64,
) -> ExplicitDocument {
    let (terms, len) = &tpl[(i % tpl.len() as u64) as usize];
    ExplicitDocument {
        stored: vec![StoredField {
            field_number: f.source,
            value: FieldValue::Binary(vec![b'x'; 600]),
        }],
        fields: ExplicitFields {
            inverted: vec![
                InvertedField {
                    field_number: f.id,
                    terms: vec![InvertedTerm {
                        term: format!("d{i}").into_bytes(),
                        freq: 1,
                        positions: Vec::new(),
                        offsets: Vec::new(),
                        payloads: Vec::new(),
                    }]
                    .into(),
                    norm: None,
                },
                InvertedField {
                    field_number: f.body,
                    terms: terms.clone().into(),
                    norm: Some(*len),
                },
            ],
            doc_values: vec![
                StoredField {
                    field_number: f.num,
                    value: FieldValue::Long(counter),
                },
                StoredField {
                    field_number: f.seq,
                    value: FieldValue::Long(seq),
                },
                StoredField {
                    field_number: f.ver,
                    value: FieldValue::Long(1),
                },
            ],
            ..ExplicitFields::default()
        },
    }
}

fn id_term(i: u64) -> Term {
    Term::new("_id", format!("d{i}").into_bytes())
}

fn soft(i: u64) -> DocValuesUpdate {
    DocValuesUpdate::Numeric {
        term: id_term(i),
        field: "__soft_deletes".to_string(),
        value: Some(1),
    }
}

fn json(pairs: &[(&str, String)]) -> String {
    let body: Vec<String> = pairs.iter().map(|(k, v)| format!("\"{k}\":{v}")).collect();
    format!("{{{}}}", body.join(","))
}

struct Run<'a> {
    w: &'a ConcurrentIndexWriter<'a>,
    f: Fields,
    tpl: &'a [(Vec<InvertedTerm>, i64)],
    docs: u64,
    threads: u64,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let docs: u64 = arg(&args, "--docs", 200_000);
    let tier: String = arg(&args, "--tier", "off".to_string());
    let workload: String = arg(&args, "--workload", "stream".to_string());
    let rate: f64 = arg(&args, "--rate", 0.0);
    let seconds: f64 = arg(&args, "--seconds", 20.0);
    let hot_set: u64 = arg(&args, "--hot-set", 1000);
    let hot_share: f64 = arg(&args, "--hot-share", 0.8);
    let share: f64 = arg(&args, "--share", 0.1);
    let threads: u64 = arg(&args, "--threads", 4);
    let refresh_ms: u64 = arg(&args, "--refresh-ms", 1000);
    let ram_mb: f64 = arg(&args, "--ram-mb", 64.0);
    let min_rate: f64 = arg(
        &args,
        "--min-rate",
        HotTierConfig::DEFAULT_MIN_UPDATES_PER_SECOND,
    );
    let live = args.iter().any(|a| a == "--live");
    let tier_on_load = args.iter().any(|a| a == "--tier-on-load");
    let view_ms: u64 = arg(&args, "--view-ms", 0);
    let views = AtomicU64::new(0);
    let view_us_sum = AtomicU64::new(0);
    let view_us_max = AtomicU64::new(0);
    let label: String = arg(&args, "--label", String::new());

    let tmp = std::env::temp_dir().join(format!("hot-tier-bench-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    let dir = FsDirectory::open(&tmp);
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut iw = IndexWriter::open(&dir, Vec::new(), "Lucene104", version).unwrap();
    iw.set_ram_buffer_size_mb(ram_mb).unwrap();
    iw.enable_explicit_documents().unwrap();
    iw.set_merge_policy(Some(
        lucene_index::merge_policy::MergePolicyConfig::default(),
    ));
    let num_dv = |name: &str| {
        FieldInfo::new(name, 0).with_doc_values(
            DocValuesType::Numeric,
            DocValuesSkipIndexType::None,
            -1,
        )
    };
    let f = Fields {
        id: iw
            .register_field(
                FieldInfo::new("_id", 0)
                    .with_index_options(IndexOptions::Docs)
                    .with_omit_norms(true),
            )
            .unwrap(),
        body: iw
            .register_field(
                FieldInfo::new("body", 0)
                    .with_index_options(IndexOptions::DocsAndFreqsAndPositions),
            )
            .unwrap(),
        num: iw.register_field(num_dv("num")).unwrap(),
        seq: iw.register_field(num_dv("_seq_no")).unwrap(),
        ver: iw.register_field(num_dv("_version")).unwrap(),
        source: iw.register_field(FieldInfo::new("_source", 0)).unwrap(),
    };
    iw.register_field(num_dv("__soft_deletes").with_soft_deletes_field(true))
        .unwrap();
    let w = ConcurrentIndexWriter::new(iw, threads as usize).unwrap();
    if live {
        w.enable_live_search().unwrap();
    }
    let tpl = templates(2000);
    let run = Run {
        w: &w,
        f,
        tpl: &tpl,
        docs,
        threads,
    };

    let stop = AtomicBool::new(false);
    let merges_done = AtomicU64::new(0);
    let flushes = AtomicU64::new(0);
    let flush_ms_sum = AtomicU64::new(0);
    let flush_ms_max = AtomicU64::new(0);
    let bg_stop = AtomicBool::new(false);
    let mut out: Vec<(&str, String)> = vec![
        ("label", format!("\"{label}\"")),
        ("tier", format!("\"{tier}\"")),
        ("workload", format!("\"{workload}\"")),
        ("docs", docs.to_string()),
        ("live", live.to_string()),
    ];

    std::thread::scope(|s| {
        // Merges run all the while, from the load on.
        s.spawn(|| {
            while !bg_stop.load(Ordering::Acquire) {
                let n = w.maybe_merge().unwrap_or(0);
                merges_done.fetch_add(n as u64, Ordering::Relaxed);
                std::thread::sleep(Duration::from_millis(20));
            }
        });

        let enable_tier = || {
            if tier != "off" {
                let bytes: usize = tier.parse().expect("--tier off|BYTES");
                let cfg = HotTierConfig::with_budget(bytes)
                    .with_min_updates_per_second(min_rate)
                    .unwrap();
                w.enable_hot_tier(cfg);
            }
        };
        if tier_on_load {
            enable_tier();
        }
        // Load.
        let t0 = Instant::now();
        let next = AtomicU64::new(0);
        std::thread::scope(|ls| {
            for _ in 0..threads {
                ls.spawn(|| loop {
                    let at = next.fetch_add(256, Ordering::Relaxed);
                    if at >= docs {
                        break;
                    }
                    let docs_vec: Vec<ExplicitDocument> = (at..(at + 256).min(docs))
                        .map(|i| doc(&f, &tpl, i, i as i64, 1))
                        .collect();
                    w.add_explicit_documents(docs_vec).unwrap();
                });
            }
        });
        w.flush().unwrap();
        let load_s = t0.elapsed().as_secs_f64();
        eprintln!("loaded {docs} in {load_s:.1}s");
        // Let the load's merges finish before the workload starts.
        let quiet = Instant::now();
        while w.has_pending_merges() && quiet.elapsed() < Duration::from_secs(120) {
            std::thread::sleep(Duration::from_millis(200));
        }
        out.push(("load_seconds", format!("{load_s:.1}")));

        if !tier_on_load {
            enable_tier();
        }

        // The refresher.
        let refresh_stop = AtomicBool::new(false);
        std::thread::scope(|ws| {
            ws.spawn(|| {
                while !refresh_stop.load(Ordering::Acquire) {
                    std::thread::sleep(Duration::from_millis(refresh_ms));
                    let t = Instant::now();
                    if live {
                        if let Ok(v) = w.open_live_view(None) {
                            w.close_live_view(&v);
                        }
                    }
                    w.commit().unwrap();
                    let ms = t.elapsed().as_millis() as u64;
                    flushes.fetch_add(1, Ordering::Relaxed);
                    flush_ms_sum.fetch_add(ms, Ordering::Relaxed);
                    flush_ms_max.fetch_max(ms, Ordering::Relaxed);
                }
            });

            if live && view_ms > 0 {
                ws.spawn(|| {
                    while !refresh_stop.load(Ordering::Acquire) {
                        std::thread::sleep(Duration::from_millis(view_ms));
                        let t = Instant::now();
                        if let Ok(v) = w.open_live_view(None) {
                            w.close_live_view(&v);
                        }
                        let us = t.elapsed().as_micros() as u64;
                        views.fetch_add(1, Ordering::Relaxed);
                        view_us_sum.fetch_add(us, Ordering::Relaxed);
                        view_us_max.fetch_max(us, Ordering::Relaxed);
                    }
                });
            }
            let merges0 = merges_done.load(Ordering::Relaxed);
            let cpu0 = cpu_seconds();
            let rss_before = status_kb("VmRSS:");
            let t0 = Instant::now();
            let (updates, lat) = match workload.as_str() {
                "stream" => stream(&run, rate, seconds, hot_set, hot_share),
                "bulk" => bulk(&run, share),
                "trickle" => stream_uniform(&run, rate, seconds),
                "dv" => stream_dv(&run, rate, seconds, hot_set, hot_share),
                other => panic!("unknown workload {other}"),
            };
            let wall = t0.elapsed().as_secs_f64();
            // Everything acknowledged is applied before the numbers are taken.
            w.drain_hot_tier().unwrap();
            w.flush().unwrap();
            let cpu = cpu_seconds() - cpu0;
            let merges = merges_done.load(Ordering::Relaxed) - merges0;
            refresh_stop.store(true, Ordering::Release);

            let (mut p50, mut p99, mut pmax) = (0.0, 0.0, 0.0);
            if !lat.is_empty() {
                let mut l = lat.clone();
                l.sort_by(|a, b| a.partial_cmp(b).unwrap());
                p50 = l[l.len() / 2];
                p99 = l[(l.len() * 99 / 100).min(l.len() - 1)];
                pmax = l[l.len() - 1];
            }
            out.push(("updates", updates.to_string()));
            out.push(("seconds", format!("{wall:.2}")));
            out.push(("updates_per_s", format!("{:.0}", updates as f64 / wall)));
            out.push(("cpu_seconds", format!("{cpu:.2}")));
            out.push((
                "cpu_us_per_update",
                format!("{:.1}", cpu * 1e6 / updates.max(1) as f64),
            ));
            out.push(("op_ms_p50", format!("{p50:.3}")));
            out.push(("op_ms_p99", format!("{p99:.3}")));
            out.push(("op_ms_max", format!("{pmax:.3}")));
            out.push(("merges", merges.to_string()));
            out.push(("flushes", flushes.load(Ordering::Relaxed).to_string()));
            let fl = flushes.load(Ordering::Relaxed).max(1);
            out.push((
                "flush_ms_mean",
                format!(
                    "{:.1}",
                    flush_ms_sum.load(Ordering::Relaxed) as f64 / fl as f64
                ),
            ));
            out.push((
                "flush_ms_max",
                flush_ms_max.load(Ordering::Relaxed).to_string(),
            ));
            if view_ms > 0 {
                let n = views.load(Ordering::Relaxed).max(1);
                out.push(("views", n.to_string()));
                out.push((
                    "view_us_mean",
                    (view_us_sum.load(Ordering::Relaxed) / n).to_string(),
                ));
                out.push((
                    "view_us_max",
                    view_us_max.load(Ordering::Relaxed).to_string(),
                ));
            }
            out.push(("rss_mb_before", (rss_before / 1024).to_string()));
            out.push(("rss_mb_hwm", (status_kb("VmHWM:") / 1024).to_string()));
            out.push(("ram_bytes", w.ram_bytes_used().to_string()));
            if let Some(st) = w.hot_tier_stats() {
                out.push((
                    "hot",
                    json(&[
                        ("offered", st.offered.to_string()),
                        ("through", st.written_through.to_string()),
                        ("admitted", st.admitted.to_string()),
                        ("coalesced", st.coalesced.to_string()),
                        ("spilled", st.spilled.to_string()),
                        ("drained", st.drained.to_string()),
                        ("refused_full", st.refused_full.to_string()),
                        ("peak_bytes", st.peak_bytes.to_string()),
                        ("sketch_bytes", st.sketch_bytes.to_string()),
                    ]),
                ));
            }
        });
        bg_stop.store(true, Ordering::Release);
        stop.store(true, Ordering::Release);
    });

    w.commit().unwrap();
    let infos = segment_infos::read_latest(&dir).unwrap();
    let max_doc: i64 = infos.segments.iter().map(|s| i64::from(s.del_count)).sum();
    out.push(("segments_at_end", infos.segments.len().to_string()));
    out.push(("hard_deleted_at_end", max_doc.to_string()));
    println!("{}", json(&out));
    let _ = std::fs::remove_dir_all(&tmp);
}

/// Updates on a hot set (`hot_share` of them) and the rest uniform, from
/// `threads` threads, paced to `rate` a second in all (0: flat out) for
/// `seconds`. Returns the updates made and each one's latency in ms.
fn stream(run: &Run, rate: f64, seconds: f64, hot_set: u64, hot_share: f64) -> (u64, Vec<f64>) {
    drive(run, rate, seconds, move |rng, owned| {
        if rng.unit() < hot_share {
            rng.below(hot_set.min(owned))
        } else {
            rng.below(owned)
        }
    })
}

/// The same stream as numeric doc-values updates of the counter, in place,
/// instead of new versions of the rows.
fn stream_dv(run: &Run, rate: f64, seconds: f64, hot_set: u64, hot_share: f64) -> (u64, Vec<f64>) {
    DV.store(true, Ordering::Relaxed);
    let r = stream(run, rate, seconds, hot_set, hot_share);
    DV.store(false, Ordering::Relaxed);
    r
}

static DV: AtomicBool = AtomicBool::new(false);

fn stream_uniform(run: &Run, rate: f64, seconds: f64) -> (u64, Vec<f64>) {
    drive(run, rate, seconds, |rng, owned| rng.below(owned))
}

/// Each thread's `owned` rows are the ids congruent to its number; `pick`
/// chooses an index below `owned`.
fn drive(
    run: &Run,
    rate: f64,
    seconds: f64,
    pick: impl Fn(&mut Rng, u64) -> u64 + Sync,
) -> (u64, Vec<f64>) {
    let count = AtomicU64::new(0);
    let lats = std::sync::Mutex::new(Vec::new());
    let deadline = Instant::now() + Duration::from_secs_f64(seconds);
    let per_thread = if rate > 0.0 {
        rate / run.threads as f64
    } else {
        0.0
    };
    std::thread::scope(|s| {
        for t in 0..run.threads {
            let (count, lats, pick) = (&count, &lats, &pick);
            s.spawn(move || {
                let mut rng = Rng(7 + t * 1000);
                let owned = run.docs / run.threads;
                let mut mine = Vec::new();
                let start = Instant::now();
                let mut n = 0u64;
                let mut seq = 2i64;
                while Instant::now() < deadline {
                    if per_thread > 0.0 {
                        let due = start + Duration::from_secs_f64(n as f64 / per_thread);
                        let now = Instant::now();
                        if due > now {
                            std::thread::sleep(due - now);
                        }
                    }
                    let id = pick(&mut rng, owned) * run.threads + t;
                    let counter = rng.below(1_000_000_000) as i64;
                    seq += 1;
                    let t0 = Instant::now();
                    if DV.load(Ordering::Relaxed) {
                        run.w
                            .update_numeric_doc_value(id_term(id), "num", counter)
                            .unwrap();
                    } else {
                        let d = doc(&run.f, run.tpl, id, counter, seq);
                        run.w
                            .soft_update_explicit_documents(id_term(id), vec![d], &[soft(id)])
                            .unwrap();
                    }
                    if n.is_multiple_of(8) || (rate > 0.0 && rate < 1000.0) {
                        mine.push(t0.elapsed().as_secs_f64() * 1000.0);
                    }
                    n += 1;
                }
                count.fetch_add(n, Ordering::Relaxed);
                lats.lock().unwrap().extend(mine);
            });
        }
    });
    (count.load(Ordering::Relaxed), lats.into_inner().unwrap())
}

/// Updates `share` of the rows once each, in random order, flat out.
fn bulk(run: &Run, share: f64) -> (u64, Vec<f64>) {
    let want = (run.docs as f64 * share) as u64;
    let next = AtomicU64::new(0);
    // A multiplicative permutation of 0..docs: each id once, scattered.
    let stride = {
        let mut s = 0x9E37_79B9u64 % run.docs.max(2);
        while gcd(s, run.docs) != 1 {
            s += 1;
        }
        s
    };
    std::thread::scope(|s| {
        for _ in 0..run.threads {
            let next = &next;
            s.spawn(move || {
                let mut seq = 2i64;
                loop {
                    let k = next.fetch_add(1, Ordering::Relaxed);
                    if k >= want {
                        break;
                    }
                    let id = (k * stride) % run.docs;
                    seq += 1;
                    let d = doc(&run.f, run.tpl, id, 7, seq);
                    run.w
                        .soft_update_explicit_documents(id_term(id), vec![d], &[soft(id)])
                        .unwrap();
                }
            });
        }
    });
    (want, Vec::new())
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}
