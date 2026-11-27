//! Where DataFusion's time goes for one query: SQL to logical plan, optimise
//! and physical plan, execute (of which the native search is a part).
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::{SessionConfig, SessionContext};
use sql_spike::data::{id_of, next, random_vec, unit};
use sql_spike::index::SpikeIndex;
use sql_spike::provider::{knn_udf, match_text_udf, LuceneTable, SEARCH_NANOS};
use sql_spike::rule::ScoreTopKRule;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let ix = SpikeIndex::open(&a[1]);
    let n = ix.max_doc() as u64;
    let cfg = SessionConfig::new().with_target_partitions(1);
    let b = SessionStateBuilder::new()
        .with_default_features()
        .with_config(cfg)
        .with_optimizer_rule(Arc::new(ScoreTopKRule));
    let ctx = SessionContext::new_with_state(b.build());
    ctx.register_udf(match_text_udf());
    ctx.register_udf(knn_udf());
    ctx.register_table("t", Arc::new(LuceneTable::new(ix, 1)))
        .unwrap();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut st = 0xabcdef12345u64;
    let iters = 1500;
    for shape in ["point", "match", "knn"] {
        let mut t = [0f64; 3];
        let mut search = 0f64;
        for i in 0..iters + 200 {
            if i == 200 {
                t = [0.0; 3];
                SEARCH_NANOS.store(0, Ordering::Relaxed);
            }
            let sql = match shape {
                "point" => format!("SELECT id, price FROM t WHERE id = '{}'", id_of((next(&mut st) % n) as usize)),
                "match" => format!(
                    "SELECT id, _score FROM t WHERE match_text(body, 'w{} w{}') ORDER BY _score DESC LIMIT 10",
                    5 + (3000.0 * unit(&mut st).powi(2)) as u64,
                    5 + (3000.0 * unit(&mut st).powi(2)) as u64
                ),
                _ => {
                    let v = random_vec(&mut st);
                    format!(
                        "SELECT id, _score FROM t WHERE knn(vec, '{}') ORDER BY _score DESC LIMIT 10",
                        v.iter().map(|x| format!("{x:.5}")).collect::<Vec<_>>().join(",")
                    )
                }
            };
            rt.block_on(async {
                let t0 = Instant::now();
                let df = ctx.sql(&sql).await.unwrap();
                let t1 = Instant::now();
                let plan = df.create_physical_plan().await.unwrap();
                let t2 = Instant::now();
                let out = datafusion::physical_plan::collect(plan, ctx.task_ctx())
                    .await
                    .unwrap();
                let t3 = Instant::now();
                std::hint::black_box(out);
                t[0] += (t1 - t0).as_secs_f64();
                t[1] += (t2 - t1).as_secs_f64();
                t[2] += (t3 - t2).as_secs_f64();
            });
        }
        search = search + SEARCH_NANOS.load(Ordering::Relaxed) as f64 / 1e3 / iters as f64;
        let us = |x: f64| x * 1e6 / iters as f64;
        println!(
            "{shape:<6} sql->logical {:>7.1}us | optimise+physical {:>7.1}us | execute {:>7.1}us (native search inside: {:>6.1}us) | total {:>7.1}us",
            us(t[0]), us(t[1]), us(t[2]), search, us(t[0] + t[1] + t[2])
        );
    }
}
