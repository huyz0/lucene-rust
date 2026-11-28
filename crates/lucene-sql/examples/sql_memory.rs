//! Peak resident memory of SQL statements over an index `sql_bench` built.
//!
//! `cargo run --release -p lucene-sql --example sql_memory -- <dir> [memory_limit_mb]`
//! runs each statement once on a fresh engine and prints the rows returned,
//! the time, and how far the process's peak resident size rose above what it
//! held before the statement.

use std::sync::Arc;
use std::time::Instant;

use lucene_search::directory_reader::DirectoryReader;
use lucene_sql::{Snapshot, SqlConfig, SqlEngine, TableSpec};
use lucene_store::FsDirectory;

fn status_kb(key: &str) -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .unwrap()
        .lines()
        .find_map(|l| l.strip_prefix(key))
        .and_then(|v| v.trim().trim_end_matches("kB").trim().parse().ok())
        .unwrap()
}

fn reset_peak() {
    std::fs::write("/proc/self/clear_refs", "5").unwrap();
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let path = &a[1];
    let limit_mb: usize = a.get(2).map_or(512, |s| s.parse().unwrap());
    let statements = [
        "SELECT cat, count(*) c, avg(price) a FROM t GROUP BY cat ORDER BY c DESC LIMIT 5",
        "SELECT id, count(*) c FROM t GROUP BY id ORDER BY c DESC, id LIMIT 10",
        "SELECT cat, price, count(*) c FROM t GROUP BY cat, price ORDER BY c DESC, cat, price LIMIT 10",
        "SELECT id, price FROM t ORDER BY price DESC, id LIMIT 100",
        "SELECT id FROM t ORDER BY id DESC LIMIT 10",
        "SELECT count(DISTINCT id) FROM t",
        "SELECT cat, count(DISTINCT price) FROM t GROUP BY cat ORDER BY cat LIMIT 5",
        "SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY price) FROM t",
        "SELECT id, price, cat FROM t WHERE price > 5 ORDER BY id LIMIT 10000",
    ];
    let spec = || {
        TableSpec::new("t")
            .key("id")
            .long("price")
            .keyword("cat")
            .text("body")
            .vector("vec", 32)
    };
    let cfg = SqlConfig {
        memory_limit: limit_mb << 20,
        ..SqlConfig::default()
    };
    for sql in statements {
        let snap = Arc::new(
            Snapshot::new(
                DirectoryReader::open(&FsDirectory::open(path.clone())).unwrap(),
                spec(),
            )
            .unwrap(),
        );
        let engine = SqlEngine::new(snap, &cfg).unwrap();
        let before = status_kb("VmRSS:");
        reset_peak();
        let t = Instant::now();
        let out = engine.query_blocking(sql);
        let took = t.elapsed();
        let peak = status_kb("VmHWM:");
        let rows = out
            .as_ref()
            .map_or(0, |b| b.iter().map(|x| x.num_rows()).sum());
        let result = out.map_or_else(
            |e| format!("ERR {}", e.to_string().chars().take(70).collect::<String>()),
            |_| "ok".into(),
        );
        println!(
            "{:>8} MiB peak over {:>6} MiB base  {:>8.0?}  rows={rows:<6} {result}  {}",
            (peak.saturating_sub(before)) / 1024,
            before / 1024,
            took,
            &sql[..sql.len().min(80)]
        );
    }
}
