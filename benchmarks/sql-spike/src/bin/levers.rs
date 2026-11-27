//! (A) What the router recognises, near-misses, and recognition cost.
//! (B) Per-query DataFusion cost by phase under different configurations.
use std::sync::atomic::Ordering::Relaxed;
use std::sync::Arc;
use std::time::Instant;

use datafusion::execution::SessionStateBuilder;
use datafusion::optimizer::analyzer::type_coercion::TypeCoercion;
use datafusion::optimizer::optimize_projections::OptimizeProjections;
use datafusion::optimizer::push_down_filter::PushDownFilter;
use datafusion::optimizer::push_down_limit::PushDownLimit;
use datafusion::prelude::{SessionConfig, SessionContext};
use sql_spike::data::{id_of, next, unit};
use sql_spike::index::SpikeIndex;
use sql_spike::provider::{knn_udf, match_text_udf, LuceneTable, SEARCH_NANOS, SETUP_NANOS};
use sql_spike::router::{recognise, templatize, Router};
use sql_spike::rule::ScoreTopKRule;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn make(
    ix: &'static SpikeIndex,
    parts: usize,
    opt: bool,
    phys: bool,
    analyzer: bool,
) -> SessionContext {
    let cfg = SessionConfig::new().with_target_partitions(parts);
    let mut b = SessionStateBuilder::new()
        .with_default_features()
        .with_config(cfg);
    if opt {
        b = b.with_optimizer_rules(vec![
            Arc::new(PushDownFilter::new()),
            Arc::new(PushDownLimit::new()),
            Arc::new(OptimizeProjections::new()),
            Arc::new(ScoreTopKRule),
        ]);
    } else {
        b = b.with_optimizer_rule(Arc::new(ScoreTopKRule));
    }
    if phys {
        b = b.with_physical_optimizer_rules(vec![]);
    }
    if analyzer {
        b = b.with_analyzer_rules(vec![Arc::new(TypeCoercion::new())]);
    }
    let c = SessionContext::new_with_state(b.build());
    c.register_udf(match_text_udf());
    c.register_udf(knn_udf());
    c.register_table("t", Arc::new(LuceneTable::new(ix, parts)))
        .unwrap();
    c
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let ix = SpikeIndex::open(&a[1]);
    let n = ix.max_doc() as u64;

    println!("(A) router recognition");
    let cases = [
        "SELECT id, price FROM t WHERE id = 'doc-00000042'",
        "select id,price from t where id='doc-00000042'",
        "SELECT id, _score FROM t WHERE match_text(body, 'w5 w9') ORDER BY _score DESC LIMIT 10",
        "SELECT id, _score FROM t WHERE match_text(body, 'w5 w9') ORDER BY _score DESC LIMIT 100",
        "SELECT id, _score FROM t WHERE knn(vec, '0.1,0.2') ORDER BY _score DESC LIMIT 10",
        "SELECT cat, count(*), sum(price) FROM t GROUP BY cat",
        // near misses
        "SELECT id, price FROM t WHERE id = 'x' AND price > 10",
        "SELECT price, id FROM t WHERE id = 'x'",
        "SELECT id, price, cat FROM t WHERE id = 'x'",
        "SELECT id, price FROM t WHERE id IN ('x', 'y')",
        "SELECT id, _score FROM t WHERE match_text(body, 'w5') AND cat = 'c001' ORDER BY _score DESC LIMIT 10",
        "SELECT id, _score FROM t WHERE match_text(body, 'w5') ORDER BY _score DESC",
        "SELECT cat, count(*) FROM t GROUP BY cat",
        "SELECT cat, count(*), sum(price) FROM t WHERE price > 5 GROUP BY cat",
        "SELECT a.id FROM t a JOIN t b ON a.id = b.id",
    ];
    for c in cases {
        let r = recognise(c);
        println!("  {:<6} {}", if r.is_some() { "FAST" } else { "-> DF" }, c);
    }
    let reps = 200_000;
    let sql =
        "SELECT id, _score FROM t WHERE match_text(body, 'w5 w9') ORDER BY _score DESC LIMIT 10";
    let t = Instant::now();
    for _ in 0..reps / 10 {
        std::hint::black_box(recognise(sql));
    }
    let parse = t.elapsed().as_secs_f64() * 1e6 / (reps / 10) as f64;
    let t = Instant::now();
    for _ in 0..reps {
        std::hint::black_box(templatize(sql));
    }
    let tpl = t.elapsed().as_secs_f64() * 1e6 / reps as f64;
    let router = Router::default();
    router.fast_plan(sql);
    let t = Instant::now();
    for _ in 0..reps {
        std::hint::black_box(router.fast_plan(sql));
    }
    let hit = t.elapsed().as_secs_f64() * 1e6 / reps as f64;
    let miss_sql = "SELECT a.id FROM t a JOIN t b ON a.id = b.id";
    router.fast_plan(miss_sql);
    let t = Instant::now();
    for _ in 0..reps {
        std::hint::black_box(router.fast_plan(miss_sql));
    }
    let miss = t.elapsed().as_secs_f64() * 1e6 / reps as f64;
    let t = Instant::now();
    for _ in 0..reps / 10 {
        std::hint::black_box(recognise(miss_sql));
    }
    let cold_miss = t.elapsed().as_secs_f64() * 1e6 / (reps / 10) as f64;
    println!("  cost: full parse+match {parse:.1}us | literal cut only {tpl:.2}us | cached shape hit {hit:.2}us | cached near-miss (fall to DataFusion) {miss:.2}us | first sight of a miss {cold_miss:.1}us\n");

    println!("(B) DataFusion per-query cost by phase, us, 1 partition, caller thread");
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let configs: [(&str, bool, bool, bool); 5] = [
        ("default rules", false, false, false),
        ("trimmed logical optimiser (4 rules)", true, false, false),
        ("no physical optimiser rules", false, true, false),
        ("trimmed analyzer (type coercion only)", false, false, true),
        ("all three trimmed", true, true, true),
    ];
    println!(
        "{:<40} {:<7} {:>8} {:>8} {:>8} | {:>7} {:>7} {:>7} | {:>8}",
        "config", "shape", "parse+lp", "opt+phys", "execute", "setup", "search", "rest", "total"
    );
    for (label, opt, phys, an) in configs {
        let ctx = make(ix, 1, opt, phys, an);
        for shape in ["point", "match", "group"] {
            let mut st = 0x1357_9bdf_2468u64;
            let iters = if shape == "group" { 60 } else { 1200 };
            let warm = iters / 5;
            let mut t = [0f64; 3];
            SEARCH_NANOS.store(0, Relaxed);
            SETUP_NANOS.store(0, Relaxed);
            let mut ok = true;
            for i in 0..iters + warm {
                if i == warm {
                    t = [0.0; 3];
                    SEARCH_NANOS.store(0, Relaxed);
                    SETUP_NANOS.store(0, Relaxed);
                }
                let sql = match shape {
                    "point" => format!("SELECT id, price FROM t WHERE id = '{}'", id_of((next(&mut st) % n) as usize)),
                    "match" => format!("SELECT id, _score FROM t WHERE match_text(body, 'w{} w{}') ORDER BY _score DESC LIMIT 10", 5 + (3000.0 * unit(&mut st).powi(2)) as u64, 5 + (3000.0 * unit(&mut st).powi(2)) as u64),
                    _ => "SELECT cat, count(*), sum(price) FROM t GROUP BY cat".to_string(),
                };
                let r: Result<(), String> = rt.block_on(async {
                    let t0 = Instant::now();
                    let df = ctx.sql(&sql).await.map_err(|e| e.to_string())?;
                    let t1 = Instant::now();
                    let plan = df.create_physical_plan().await.map_err(|e| e.to_string())?;
                    let t2 = Instant::now();
                    let out = datafusion::physical_plan::collect(plan, ctx.task_ctx())
                        .await
                        .map_err(|e| e.to_string())?;
                    let t3 = Instant::now();
                    std::hint::black_box(out);
                    t[0] += (t1 - t0).as_secs_f64();
                    t[1] += (t2 - t1).as_secs_f64();
                    t[2] += (t3 - t2).as_secs_f64();
                    Ok(())
                });
                if let Err(e) = r {
                    println!(
                        "{label:<40} {shape:<7} failed: {}",
                        e.chars().take(90).collect::<String>()
                    );
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }
            let us = |x: f64| x * 1e6 / iters as f64;
            let setup = SETUP_NANOS.load(Relaxed) as f64 / 1e3 / iters as f64;
            let search = SEARCH_NANOS.load(Relaxed) as f64 / 1e3 / iters as f64;
            let exec = us(t[2]);
            println!(
                "{label:<40} {shape:<7} {:>8.1} {:>8.1} {:>8.1} | {:>7.1} {:>7.1} {:>7.1} | {:>8.1}",
                us(t[0]), us(t[1]), exec, setup, search, exec - setup - search, us(t[0] + t[1] + t[2])
            );
        }
    }
}
