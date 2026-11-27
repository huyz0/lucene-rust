//! MATCH top-10 over rare, mid and very common terms: is DataFusion's gap a
//! fixed cost (shrinks as the query gets heavier) or a ratio?
use std::sync::Arc;
use std::time::{Duration, Instant};

use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::{SessionConfig, SessionContext};
use sql_spike::direct;
use sql_spike::index::SpikeIndex;
use sql_spike::provider::{knn_udf, match_text_udf, LuceneTable};
use sql_spike::router::Router;
use sql_spike::rule::ScoreTopKRule;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn p50(mut v: Vec<Duration>) -> f64 {
    v.sort_unstable();
    v[v.len() / 2].as_secs_f64() * 1e6
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let ix = SpikeIndex::open(&a[1]);
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
    let router = Router::default();
    println!("MATCH top-10, p50 us, 1 thread");
    println!(
        "{:<22} {:>9} {:>9} {:>9} {:>8} {:>10}",
        "query terms", "direct", "router", "DataFusion", "DF/direct", "DF-direct"
    );
    for (label, words) in [
        ("rare (w30000 w40000)", "w30000 w40000"),
        ("mid (w300 w700)", "w300 w700"),
        ("common (w20 w50)", "w20 w50"),
        ("very common (w1 w2)", "w1 w2"),
        ("very common x3", "w1 w2 w3"),
    ] {
        let sql = format!("SELECT id, _score FROM t WHERE match_text(body, '{words}') ORDER BY _score DESC LIMIT 10");
        let ws: Vec<&str> = words.split(' ').collect();
        let mut run = |f: &mut dyn FnMut()| {
            let t = Instant::now();
            while t.elapsed().as_secs_f64() < 1.0 {
                f();
            }
            let mut v = Vec::new();
            let t = Instant::now();
            while t.elapsed().as_secs_f64() < 4.0 {
                let s = Instant::now();
                f();
                v.push(s.elapsed());
            }
            p50(v)
        };
        let d = run(&mut || {
            std::hint::black_box(direct::match_topk(ix, &ws, 10));
        });
        let r = run(&mut || {
            std::hint::black_box(router.fast_plan(&sql).unwrap().execute(ix));
        });
        let f = run(&mut || {
            rt.block_on(async {
                std::hint::black_box(ctx.sql(&sql).await.unwrap().collect().await.unwrap());
            });
        });
        println!(
            "{label:<22} {d:>9.1} {r:>9.1} {f:>10.1} {:>8.2}x {:>9.1}us",
            f / d,
            f - d
        );
    }
}
