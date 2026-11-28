//! Which statements of a realistic workload the router answers, and how fast
//! each path is.
//!
//! `cargo run --release -p lucene-sql --example sql_coverage -- <dir> [secs]`
//! runs over the index `sql_bench` built. For every statement of the corpus it
//! prints whether the router took it and the p50 latency of the engine (router
//! when it takes the statement) and of DataFusion from a cached plan.

use std::sync::Arc;
use std::time::Instant;

use lucene_search::directory_reader::DirectoryReader;
use lucene_sql::{Snapshot, SqlConfig, SqlEngine, TableSpec};
use lucene_store::FsDirectory;

const CORPUS: &[(&str, &str)] = &[
    ("point", "SELECT id, price, cat FROM t WHERE id = 'doc-00012345'"),
    ("point", "SELECT * FROM t WHERE id = 'doc-00012345'"),
    ("point", "SELECT id FROM t WHERE id IN ('doc-00000001', 'doc-00000002', 'doc-00000003')"),
    ("filter", "SELECT id, price FROM t WHERE cat = 'c007' LIMIT 20"),
    ("filter", "SELECT id, price FROM t WHERE cat = 'c007' AND price > 9000 LIMIT 20"),
    ("filter", "SELECT id, price FROM t WHERE price BETWEEN 100 AND 200 LIMIT 20"),
    ("filter", "SELECT id, price FROM t WHERE cat = 'c007' OR cat = 'c008' LIMIT 20"),
    ("filter", "SELECT id, price FROM t WHERE cat IN ('c001', 'c002', 'c003') AND price < 50 LIMIT 20"),
    ("filter", "SELECT id, price FROM t WHERE NOT cat = 'c007' AND price < 20 LIMIT 20"),
    ("filter", "SELECT id, price FROM t WHERE cat <> 'c007' AND price < 20 LIMIT 20"),
    ("filter", "SELECT id, price FROM t WHERE cat LIKE 'c00%' AND price < 20 LIMIT 20"),
    ("filter", "SELECT id, price FROM t WHERE price IS NOT NULL AND price < 20 LIMIT 20"),
    ("filter", "SELECT id, price FROM t WHERE cat IS NULL LIMIT 20"),
    ("sort", "SELECT id, price FROM t ORDER BY price DESC LIMIT 10"),
    ("sort", "SELECT id, price FROM t ORDER BY price ASC, id LIMIT 10"),
    ("sort", "SELECT id, price FROM t WHERE cat = 'c007' ORDER BY price DESC LIMIT 10"),
    ("sort", "SELECT id, price FROM t WHERE ts BETWEEN 1000 AND 20000 ORDER BY price DESC LIMIT 10 OFFSET 20"),
    ("sort", "SELECT id, ts FROM t WHERE cat = 'c007' ORDER BY ts DESC LIMIT 5"),
    ("sort", "SELECT id FROM t ORDER BY id LIMIT 10"),
    ("agg", "SELECT count(*) FROM t"),
    ("agg", "SELECT count(*) FROM t WHERE ts BETWEEN 1000 AND 20000"),
    ("agg", "SELECT count(*) FROM t WHERE cat = 'c007' AND ts > 500000"),
    ("agg", "SELECT count(*) FROM t WHERE cat = 'c007' OR cat = 'c008'"),
    ("agg", "SELECT sum(price), avg(price), min(price), max(price) FROM t"),
    ("agg", "SELECT sum(price), avg(price) FROM t WHERE ts BETWEEN 1000 AND 20000"),
    ("agg", "SELECT count(*), max(price) FROM t WHERE cat = 'c007'"),
    ("agg", "SELECT count(DISTINCT cat) FROM t"),
    ("agg", "SELECT count(DISTINCT cat) FROM t WHERE ts BETWEEN 1000 AND 200000"),
    ("agg", "SELECT approx_percentile_cont(price, 0.5) FROM t"),
    ("group", "SELECT cat, count(*) FROM t GROUP BY cat"),
    ("group", "SELECT cat, count(*), sum(price) FROM t WHERE ts BETWEEN 1000 AND 20000 GROUP BY cat"),
    ("group", "SELECT cat, count(*) AS n FROM t GROUP BY cat ORDER BY n DESC LIMIT 5"),
    ("group", "SELECT cat, avg(price) FROM t GROUP BY cat ORDER BY cat"),
    ("group", "SELECT cat, count(*) FROM t GROUP BY cat HAVING count(*) > 9800"),
    ("group", "SELECT cat, count(DISTINCT price) FROM t GROUP BY cat"),
    ("group", "SELECT cat, ts / 100000 AS b, count(*) FROM t GROUP BY cat, ts / 100000"),
    ("group", "SELECT price / 1000 AS b, count(*) FROM t GROUP BY price / 1000 ORDER BY b"),
    ("search", "SELECT id, _score FROM t WHERE match_text(body, 'w10 w20') ORDER BY _score DESC LIMIT 10"),
    ("search", "SELECT id, price, _score FROM t WHERE match_text(body, 'w10 w20') AND price > 5000 ORDER BY _score DESC LIMIT 10"),
    ("search", "SELECT id, _score FROM t WHERE match_text(body, 'w10') AND cat = 'c007' ORDER BY _score DESC LIMIT 10 OFFSET 10"),
    ("search", "SELECT id, _score FROM t WHERE match_text(body, 'w10 w20') ORDER BY _score DESC, id LIMIT 10"),
    ("search", "SELECT count(*) FROM t WHERE match_text(body, 'w10 w20')"),
    ("search", "SELECT cat, count(*) FROM t WHERE match_text(body, 'w10 w20') GROUP BY cat"),
    ("search", "SELECT id, price FROM t WHERE match_text(body, 'w10 w20') AND price > 5000 LIMIT 10"),
    ("hybrid", "SELECT id, _score FROM t WHERE match_text(body, 'w10') AND knn(vec, '0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1') ORDER BY _score DESC LIMIT 10"),
    ("misc", "SELECT upper(cat), price FROM t WHERE id = 'doc-00012345'"),
    ("misc", "SELECT id, price * 2 AS p2 FROM t WHERE cat = 'c007' LIMIT 10"),
    ("misc", "SELECT DISTINCT cat FROM t"),
];

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let path = &a[1];
    let secs: f64 = a.get(2).map_or(1.0, |s| s.parse().unwrap());
    let spec = TableSpec::new("t")
        .key("id")
        .long("price")
        .long("ts")
        .keyword("cat")
        .text("body")
        .vector("vec", 32);
    let snap = Arc::new(
        Snapshot::new(
            DirectoryReader::open(&FsDirectory::open(path.clone())).unwrap(),
            spec,
        )
        .unwrap(),
    );
    let cfg = SqlConfig {
        worker_threads: 0,
        target_partitions: 1,
        ..SqlConfig::default()
    };
    let engine = SqlEngine::new(snap, &cfg).unwrap();
    let mut routed = 0;
    println!(
        "{:<7}{:>8}{:>12}{:>12}{:>8}  statement",
        "kind", "path", "engine us", "DF us", "x"
    );
    for (kind, sql) in CORPUS {
        // The router took it when it recognised the statement and answered.
        let path = matches!(engine.router().run(sql), Ok(Some(_)));
        routed += usize::from(path);
        // Each path runs twice, in the order A B B A, so that neither gains
        // from running first or second.
        let sample = |f: &dyn Fn() -> bool| -> Option<Vec<std::time::Duration>> {
            if !f() {
                return None;
            }
            let mut lat = Vec::new();
            let start = Instant::now();
            while start.elapsed().as_secs_f64() < secs / 2.0 || lat.len() < 3 {
                let t = Instant::now();
                f();
                lat.push(t.elapsed());
                if lat.len() >= 1000 {
                    break;
                }
            }
            Some(lat)
        };
        let median = |mut lat: Vec<std::time::Duration>| {
            lat.sort();
            lat[lat.len() / 2].as_secs_f64() * 1e6
        };
        let run_engine = || engine.query_blocking(sql).is_ok();
        let run_df = || engine.standard_blocking(sql).is_ok();
        let (e1, d1, d2, e2) = (
            sample(&run_engine),
            sample(&run_df),
            sample(&run_df),
            sample(&run_engine),
        );
        let join = |a: Option<Vec<_>>, b: Option<Vec<_>>| {
            a.zip(b).map(|(mut a, b)| {
                a.extend(b);
                median(a)
            })
        };
        let (eng, df) = (join(e1, e2), join(d1, d2));
        let fmt = |v: Option<f64>| v.map_or("error".to_string(), |v| format!("{v:.1}"));
        let ratio = match (eng, df) {
            (Some(e), Some(d)) => format!("{:.1}", d / e),
            _ => "-".into(),
        };
        println!(
            "{kind:<7}{:>8}{:>12}{:>12}{ratio:>8}  {}",
            if path { "router" } else { "df" },
            fmt(eng),
            fmt(df),
            &sql[..sql.len().min(110)]
        );
    }
    println!("\nrouter coverage: {routed}/{}", CORPUS.len());
}
