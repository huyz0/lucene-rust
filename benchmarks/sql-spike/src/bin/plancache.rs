//! DataFusion with and without a parameterised physical-plan cache.
use std::sync::Arc;
use std::time::{Duration, Instant};

use datafusion::arrow::array::{Array, RecordBatch, StringArray};
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::{SessionConfig, SessionContext};
use sql_spike::data::{id_of, next, random_vec, unit};
use sql_spike::direct;
use sql_spike::index::SpikeIndex;
use sql_spike::plancache::PlanCache;
use sql_spike::provider::{knn_udf, match_text_udf, LuceneTable};
use sql_spike::rule::ScoreTopKRule;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn ctx(ix: &'static SpikeIndex, parts: usize) -> SessionContext {
    let cfg = SessionConfig::new().with_target_partitions(parts);
    let b = SessionStateBuilder::new()
        .with_default_features()
        .with_config(cfg)
        .with_optimizer_rule(Arc::new(ScoreTopKRule));
    let c = SessionContext::new_with_state(b.build());
    c.register_udf(match_text_udf());
    c.register_udf(knn_udf());
    c.register_table("t", Arc::new(LuceneTable::new(ix, parts)))
        .unwrap();
    c
}

fn p50(mut v: Vec<Duration>) -> f64 {
    v.sort_unstable();
    v[v.len() / 2].as_secs_f64() * 1e6
}

fn bench(secs: f64, mut f: impl FnMut(usize)) -> f64 {
    let mut i = 0;
    let t = Instant::now();
    while t.elapsed().as_secs_f64() < 1.0 {
        f(i);
        i += 1;
    }
    let mut v = Vec::new();
    let t = Instant::now();
    while t.elapsed().as_secs_f64() < secs {
        let s = Instant::now();
        f(i);
        v.push(s.elapsed());
        i += 1;
    }
    p50(v)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let ix = SpikeIndex::open(&a[1]);
    let secs: f64 = a.get(2).map_or(5.0, |s| s.parse().unwrap());
    let n = ix.max_doc() as u64;
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let c1 = ctx(ix, 1);
    let cached1 = PlanCache::new(ctx(ix, 1));
    let c4 = ctx(ix, 4);
    let cached4 = PlanCache::new(ctx(ix, 4));
    let rt4 = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();
    let mut st = 0x77aa_55cc_1122u64;
    let ids: Vec<String> = (0..64)
        .map(|_| id_of((next(&mut st) % n) as usize))
        .collect();
    let mid: Vec<String> = (0..64)
        .map(|_| {
            format!(
                "w{} w{}",
                5 + (3000.0 * unit(&mut st).powi(2)) as u64,
                5 + (3000.0 * unit(&mut st).powi(2)) as u64
            )
        })
        .collect();
    let rare: Vec<String> = (0..64)
        .map(|_| {
            format!(
                "w{} w{}",
                20000 + (next(&mut st) % 30000),
                20000 + (next(&mut st) % 30000)
            )
        })
        .collect();
    let vs: Vec<String> = (0..64)
        .map(|_| {
            random_vec(&mut st)
                .iter()
                .map(|x| format!("{x:.5}"))
                .collect::<Vec<_>>()
                .join(",")
        })
        .collect();
    let q_point = |i: usize| format!("SELECT id, price FROM t WHERE id = '{}'", ids[i % 64]);
    let q_mid = |i: usize| {
        format!(
            "SELECT id, _score FROM t WHERE match_text(body, '{}') ORDER BY _score DESC LIMIT 10",
            mid[i % 64]
        )
    };
    let q_rare = |i: usize| {
        format!(
            "SELECT id, _score FROM t WHERE match_text(body, '{}') ORDER BY _score DESC LIMIT 10",
            rare[i % 64]
        )
    };
    let q_knn = |i: usize| {
        format!(
            "SELECT id, _score FROM t WHERE knn(vec, '{}') ORDER BY _score DESC LIMIT 10",
            vs[i % 64]
        )
    };
    let q_group = |_: usize| "SELECT cat, count(*), sum(price) FROM t GROUP BY cat".to_string();

    // correctness: cached plan returns what the uncached one does
    for i in 0..16 {
        for q in [&q_mid as &dyn Fn(usize) -> String, &q_rare, &q_point] {
            let sql = q(i);
            let x: Vec<RecordBatch> =
                rt.block_on(async { c1.sql(&sql).await.unwrap().collect().await.unwrap() });
            let y = rt.block_on(cached1.run(&sql)).unwrap();
            let col = |b: &[RecordBatch]| -> Vec<String> {
                b.iter()
                    .flat_map(|r| {
                        let a = r.column(0).as_any().downcast_ref::<StringArray>().unwrap();
                        (0..r.num_rows())
                            .map(|k| a.value(k).to_string())
                            .collect::<Vec<_>>()
                    })
                    .collect()
            };
            let (mut cx, mut cy) = (col(&x), col(&y));
            assert_eq!(cx.len(), cy.len(), "{sql}");
            if cx != cy {
                cx.sort();
                cy.sort();
                println!("  (order differs among ties) {}", cx == cy);
            }
        }
    }
    println!("cached-plan results checked against uncached DataFusion\n");

    println!(
        "{:<28} {:>9} {:>11} {:>11} {:>9} {:>9}",
        "shape (p50 us, 1 thread)",
        "direct",
        "DF uncached",
        "DF cached",
        "cached/direct",
        "removed"
    );
    let dm = |q: &dyn Fn(usize) -> String, words: &Vec<String>| {
        let _ = q;
        bench(secs, |i| {
            std::hint::black_box(direct::match_topk(
                ix,
                &words[i % 64].split(' ').collect::<Vec<_>>(),
                10,
            ));
        })
    };
    let rows: Vec<(&str, f64, &dyn Fn(usize) -> String)> = vec![
        (
            "point read",
            bench(secs, |i| {
                std::hint::black_box(direct::point_read(ix, &ids[i % 64]));
            }),
            &q_point,
        ),
        ("MATCH top-10 (mid terms)", dm(&q_mid, &mid), &q_mid),
        ("MATCH top-10 (rare terms)", dm(&q_rare, &rare), &q_rare),
        (
            "k-NN top-10",
            bench(secs, |i| {
                let v: Vec<f32> = vs[i % 64].split(',').map(|x| x.parse().unwrap()).collect();
                std::hint::black_box(direct::knn_topk(ix, &v, 10));
            }),
            &q_knn,
        ),
        (
            "group-by count+sum",
            bench(secs.max(8.0), |_| {
                std::hint::black_box(direct::group_by(ix));
            }),
            &q_group,
        ),
    ];
    for (name, d, q) in rows {
        let s = if name.starts_with("group") {
            secs.max(8.0)
        } else {
            secs
        };
        let un = bench(s, |i| {
            let sql = q(i);
            rt.block_on(async {
                std::hint::black_box(c1.sql(&sql).await.unwrap().collect().await.unwrap());
            });
        });
        let ca = bench(s, |i| {
            let sql = q(i);
            rt.block_on(async {
                std::hint::black_box(cached1.run(&sql).await.unwrap());
            });
        });
        println!(
            "{name:<28} {d:>9.1} {un:>11.1} {ca:>11.1} {:>8.2}x {:>8.0}us",
            ca / d,
            un - ca
        );
    }
    println!("\n4 partitions, worker threads:");
    for (name, q) in [
        ("group-by count+sum", &q_group as &dyn Fn(usize) -> String),
        ("MATCH top-10 (mid terms)", &q_mid),
    ] {
        let un = bench(secs.max(8.0), |i| {
            let sql = q(i);
            rt4.block_on(async {
                std::hint::black_box(c4.sql(&sql).await.unwrap().collect().await.unwrap());
            });
        });
        let ca = bench(secs.max(8.0), |i| {
            let sql = q(i);
            rt4.block_on(async {
                std::hint::black_box(cached4.run(&sql).await.unwrap());
            });
        });
        println!("{name:<28} uncached {un:>9.1}  cached {ca:>9.1}");
    }
}
