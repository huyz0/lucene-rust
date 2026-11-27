use std::sync::Arc;
use std::time::{Duration, Instant};

use datafusion::arrow::array::{Array, Float32Array, Int64Array, RecordBatch, StringArray};
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion::scalar::ScalarValue;
use sql_spike::data::{id_of, next, random_vec, unit};
use sql_spike::index::SpikeIndex;
use sql_spike::provider::{knn_udf, match_text_udf, LuceneTable};
use sql_spike::router::{recognise, Fast, Router};
use sql_spike::rule::ScoreTopKRule;
use sql_spike::{direct, VEC_DIM};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn make_ctx(ix: &'static SpikeIndex, partitions: usize, rule: bool) -> SessionContext {
    let cfg = SessionConfig::new().with_target_partitions(partitions);
    let mut b = SessionStateBuilder::new()
        .with_default_features()
        .with_config(cfg);
    if rule {
        b = b.with_optimizer_rule(Arc::new(ScoreTopKRule));
    }
    let ctx = SessionContext::new_with_state(b.build());
    ctx.register_udf(match_text_udf());
    ctx.register_udf(knn_udf());
    ctx.register_table("t", Arc::new(LuceneTable::new(ix, partitions)))
        .unwrap();
    ctx
}

struct Stats(Vec<Duration>);
impl Stats {
    fn pct(&self, p: f64) -> f64 {
        let mut v = self.0.clone();
        v.sort_unstable();
        v[((v.len() as f64 - 1.0) * p).round() as usize].as_secs_f64() * 1e6
    }
}

fn run(label: &str, secs: f64, mut f: impl FnMut(usize) -> usize) -> f64 {
    let warm = Instant::now();
    let mut i = 0;
    while warm.elapsed().as_secs_f64() < (secs / 4.0).min(1.5) {
        std::hint::black_box(f(i));
        i += 1;
    }
    let mut lat = Vec::new();
    let t = Instant::now();
    while t.elapsed().as_secs_f64() < secs {
        let s = Instant::now();
        std::hint::black_box(f(i));
        lat.push(s.elapsed());
        i += 1;
    }
    let st = Stats(lat);
    let p50 = st.pct(0.5);
    println!(
        "  {label:<34} n={:<7} p50={:>9.1}us p99={:>9.1}us",
        st.0.len(),
        p50,
        st.pct(0.99)
    );
    p50
}

fn rows(b: &[RecordBatch]) -> usize {
    b.iter().map(RecordBatch::num_rows).sum()
}

/// (id, score-or-price) rows of a result, for the cross-path check.
fn ids_scores(b: &[RecordBatch]) -> Vec<(String, f32)> {
    let mut out = Vec::new();
    for batch in b {
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let sc = batch
            .column(1)
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        for i in 0..batch.num_rows() {
            out.push((ids.value(i).to_string(), sc.value(i)));
        }
    }
    out
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let ix = SpikeIndex::open(&a[1]);
    let secs: f64 = a.get(2).map_or(6.0, |s| s.parse().unwrap());
    let n = ix.max_doc() as u64;
    println!(
        "index: {} docs, {} segments; cpus={}",
        n,
        ix.segments.len(),
        std::thread::available_parallelism().unwrap()
    );
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();
    let rt1 = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let df1 = make_ctx(ix, 1, true);
    let df4 = make_ctx(ix, 4, true);
    let df1_norule = make_ctx(ix, 1, false);
    let df4_norule = make_ctx(ix, 4, false);

    let mut st = 0x1234_5678_9abc_def1u64;
    let ids: Vec<String> = (0..64)
        .map(|_| id_of((next(&mut st) % n) as usize))
        .collect();
    let words: Vec<String> = (0..64)
        .map(|_| {
            let a = 5.0 + 3000.0 * unit(&mut st).powi(2);
            let b = 5.0 + 3000.0 * unit(&mut st).powi(2);
            format!("w{} w{}", a as u64, b as u64)
        })
        .collect();
    let vecs: Vec<Vec<f32>> = (0..64).map(|_| random_vec(&mut st)).collect();
    let vstr: Vec<String> = vecs
        .iter()
        .map(|v| {
            v.iter()
                .map(|x| format!("{x:.5}"))
                .collect::<Vec<_>>()
                .join(",")
        })
        .collect();
    let vq: Vec<Vec<f32>> = vstr
        .iter()
        .map(|s| s.split(',').map(|x| x.parse().unwrap()).collect())
        .collect();
    assert_eq!(vq[0].len(), VEC_DIM);

    let sql_point = |i: usize| format!("SELECT id, price FROM t WHERE id = '{}'", ids[i % 64]);
    let sql_match = |i: usize| {
        format!(
            "SELECT id, _score FROM t WHERE match_text(body, '{}') ORDER BY _score DESC LIMIT 10",
            words[i % 64]
        )
    };
    let sql_knn = |i: usize| {
        format!(
            "SELECT id, _score FROM t WHERE knn(vec, '{}') ORDER BY _score DESC LIMIT 10",
            vstr[i % 64]
        )
    };
    let sql_group = "SELECT cat, count(*), sum(price) FROM t GROUP BY cat";

    let df = |ctx: &SessionContext, sql: String| -> Vec<RecordBatch> {
        rt.block_on(async { ctx.sql(&sql).await.unwrap().collect().await.unwrap() })
    };
    // One partition on the calling thread: no task hand-off, no worker wake-up.
    let dft = |ctx: &SessionContext, sql: String| -> Vec<RecordBatch> {
        rt1.block_on(async { ctx.sql(&sql).await.unwrap().collect().await.unwrap() })
    };

    // ---- plans (shows the rule at work)
    for (label, ctx, sql) in [
        ("match, rule on", &df4, sql_match(0)),
        ("match, rule off", &df4_norule, sql_match(0)),
    ] {
        let plan = rt.block_on(async {
            ctx.sql(&format!("EXPLAIN {sql}"))
                .await
                .unwrap()
                .collect()
                .await
                .unwrap()
        });
        println!("\nEXPLAIN ({label}):");
        for b in &plan {
            let kinds = b.column(0).as_any().downcast_ref::<StringArray>().unwrap();
            let text = b.column(1).as_any().downcast_ref::<StringArray>().unwrap();
            for i in 0..b.num_rows() {
                println!("  [{}]", kinds.value(i));
                for l in text.value(i).lines() {
                    println!("      {l}");
                }
            }
        }
    }

    // ---- result equality across paths (the double-run check)
    println!("\ncross-path result check");
    for i in 0..8 {
        let d = direct::match_topk(ix, &words[i].split(' ').collect::<Vec<_>>(), 10);
        let r = ids_scores(&[recognise(&sql_match(i)).unwrap().execute(ix)]);
        for (name, got) in [
            ("df1", ids_scores(&df(&df1, sql_match(i)))),
            ("df4", ids_scores(&df(&df4, sql_match(i)))),
            ("df4-norule", ids_scores(&df(&df4_norule, sql_match(i)))),
            ("router", r.clone()),
        ] {
            // Same scores in the same order; ids may differ only among equal scores.
            let kth = d.last().map_or(0.0, |x| x.1);
            let same = got.len() == d.len()
                && got.iter().zip(&d).all(|(x, y)| {
                    (x.1 - y.1).abs() <= 1e-4 * y.1.abs().max(1.0)
                        && (x.0 == y.0 || (y.1 - kth).abs() < 1e-6 || {
                            // ids may swap within a run of equal scores
                            d.iter()
                                .filter(|z| (z.1 - y.1).abs() < 1e-6)
                                .any(|z| z.0 == x.0)
                        })
                });
            if !same {
                println!(
                    "  MATCH MISMATCH q{i} {name}: {:?} vs direct {:?}",
                    &got[..got.len().min(3)],
                    &d[..d.len().min(3)]
                );
            }
        }
        let dk = direct::knn_topk(ix, &vq[i], 10);
        for (name, got) in [
            ("df1", ids_scores(&df(&df1, sql_knn(i)))),
            ("df4", ids_scores(&df(&df4, sql_knn(i)))),
            (
                "router",
                ids_scores(&[recognise(&sql_knn(i)).unwrap().execute(ix)]),
            ),
        ] {
            let set_d: std::collections::HashSet<_> = dk.iter().map(|x| &x.0).collect();
            let overlap = got.iter().filter(|x| set_d.contains(&x.0)).count();
            if overlap != dk.len() || got.len() != dk.len() {
                println!("  KNN differs q{i} {name}: overlap {overlap}/{}", dk.len());
            }
        }
        let p = direct::point_read(ix, &ids[i]).unwrap();
        let got = df(&df1, sql_point(i));
        let pid = got[0]
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(0)
            .to_string();
        let pp = got[0]
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0);
        if pid != p.0 || Some(pp) != p.1 {
            println!("  POINT MISMATCH {i}");
        }
    }
    let g = direct::group_by(ix);
    let gd = df(&df1, sql_group.to_string());
    let mut dfg: Vec<(String, i64, i64)> = Vec::new();
    for b in &gd {
        let c = b.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let n = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        let s = b.column(2).as_any().downcast_ref::<Int64Array>().unwrap();
        for i in 0..b.num_rows() {
            dfg.push((c.value(i).to_string(), n.value(i), s.value(i)));
        }
    }
    dfg.sort();
    let same = dfg.len() == g.len()
        && dfg
            .iter()
            .zip(&g)
            .all(|(x, y)| x.0 == y.0 && x.1 as u64 == y.1 && x.2 as f64 == y.2);
    println!(
        "  group-by df vs kernels: {}",
        if same { "identical" } else { "MISMATCH" }
    );
    println!("  (done; mismatches, if any, are printed above)");

    let router = Router::default();
    let fast = |sql: &str| router.fast_plan(sql).unwrap();
    let mut summary: Vec<(&str, f64, f64, f64, f64, f64)> = Vec::new();

    // ---- 1. point read
    println!("\n== point read: SELECT id, price FROM t WHERE id = ?");
    let d = run("direct API", secs, |i| {
        direct::point_read(ix, &ids[i % 64]).is_some() as usize
    });
    let parse_only = run("sqlparser parse only (no cache)", secs, |i| {
        recognise(&sql_point(i)).is_some() as usize
    });
    let rt_text = run("router (sql text)", secs, |i| {
        rows(&[fast(&sql_point(i)).execute(ix)])
    });
    let prepared = Fast::Point { id: String::new() };
    let rt_prep = run("router (prepared, bound id)", secs, |i| {
        let Fast::Point { .. } = &prepared else {
            unreachable!()
        };
        rows(&[Fast::Point {
            id: ids[i % 64].clone(),
        }
        .execute(ix)])
    });
    let _f1m = run("DataFusion 1 part., worker thread", secs, |i| {
        rows(&df(&df1, sql_point(i)))
    });
    let f1 = run("DataFusion 1 part., caller thread", secs, |i| {
        rows(&dft(&df1, sql_point(i)))
    });
    let f4 = run("DataFusion sql() 4 partitions", secs, |i| {
        rows(&df(&df4, sql_point(i)))
    });
    // prepared DataFusion plan: parse and logical planning once, params bound.
    let dfp = rt.block_on(async {
        df1.sql("SELECT id, price FROM t WHERE id = $1")
            .await
            .unwrap()
    });
    let f1p = run("DataFusion prepared ($1) 1 part.", secs, |i| {
        rt.block_on(async {
            rows(
                &dfp.clone()
                    .with_param_values(vec![("1", ScalarValue::from(ids[i % 64].as_str()))])
                    .unwrap()
                    .collect()
                    .await
                    .unwrap(),
            )
        })
    });
    summary.push(("point read", d, rt_text, rt_prep, f1, f4));
    let _ = (parse_only, f1p);

    // ---- 2. match top-k
    println!("\n== MATCH top-10: ... WHERE match_text(body, ?) ORDER BY _score DESC LIMIT 10");
    let d = run("direct API", secs, |i| {
        direct::match_topk(ix, &words[i % 64].split(' ').collect::<Vec<_>>(), 10).len()
    });
    let rt_text = run("router (sql text)", secs, |i| {
        rows(&[fast(&sql_match(i)).execute(ix)])
    });
    let rt_prep = run("router (prepared)", secs, |i| {
        rows(&[Fast::Match {
            words: words[i % 64].split(' ').map(String::from).collect(),
            k: 10,
        }
        .execute(ix)])
    });
    let _f1m = run("DF 1 part. + rule, worker thread", secs, |i| {
        rows(&df(&df1, sql_match(i)))
    });
    let f1 = run("DF 1 part. + rule, caller thread", secs, |i| {
        rows(&dft(&df1, sql_match(i)))
    });
    let f4 = run("DataFusion 4 partitions + rule", secs, |i| {
        rows(&df(&df4, sql_match(i)))
    });
    let f4n = run("DataFusion 4 partitions, NO rule", secs.min(4.0), |i| {
        rows(&df(&df4_norule, sql_match(i)))
    });
    let _ = df1_norule;
    summary.push(("match top-10", d, rt_text, rt_prep, f1, f4));
    println!("  (no-rule / rule-on, 4 partitions: {:.1}x)", f4n / f4);

    // ---- 3. knn top-k
    println!("\n== k-NN top-10: ... WHERE knn(vec, ?) ORDER BY _score DESC LIMIT 10");
    let d = run("direct API", secs, |i| {
        direct::knn_topk(ix, &vq[i % 64], 10).len()
    });
    let rt_text = run("router (sql text)", secs, |i| {
        rows(&[fast(&sql_knn(i)).execute(ix)])
    });
    let rt_prep = run("router (prepared)", secs, |i| {
        rows(&[Fast::Knn {
            vector: vq[i % 64].clone(),
            k: 10,
        }
        .execute(ix)])
    });
    let _f1m = run("DF 1 part. + rule, worker thread", secs, |i| {
        rows(&df(&df1, sql_knn(i)))
    });
    let f1 = run("DF 1 part. + rule, caller thread", secs, |i| {
        rows(&dft(&df1, sql_knn(i)))
    });
    let f4 = run("DataFusion 4 partitions + rule", secs, |i| {
        rows(&df(&df4, sql_knn(i)))
    });
    let f4n = run("DataFusion 4 partitions, NO rule", secs.min(4.0), |i| {
        rows(&df(&df4_norule, sql_knn(i)))
    });
    summary.push(("k-NN top-10", d, rt_text, rt_prep, f1, f4));
    println!("  (no-rule / rule-on, 4 partitions: {:.1}x)", f4n / f4);

    // ---- 4. group by
    println!("\n== group-by: SELECT cat, count(*), sum(price) FROM t GROUP BY cat");
    let gsecs = secs.max(8.0);
    let d = run("aggregation kernels (direct)", gsecs, |_| {
        direct::group_by(ix).len()
    });
    let rt_text = run("router (sql text)", gsecs, |_| {
        rows(&[fast(sql_group).execute(ix)])
    });
    let f1 = run("DataFusion 1 part., caller thread", gsecs, |_| {
        rows(&dft(&df1, sql_group.to_string()))
    });
    let f4 = run("DataFusion 4 partitions", gsecs, |_| {
        rows(&df(&df4, sql_group.to_string()))
    });
    summary.push(("group-by", d, rt_text, rt_text, f1, f4));

    println!("\nSUMMARY (p50 us; ratio to direct in brackets)");
    println!(
        "{:<14} {:>10} {:>16} {:>16} {:>16} {:>16}",
        "shape", "direct", "router text", "router prepared", "DF 1 part/thread", "DF 4 part"
    );
    for (name, d, rt, rp, f1, f4) in summary {
        println!(
            "{name:<14} {d:>10.1} {rt:>9.1} [{:>4.2}x] {rp:>9.1} [{:>4.2}x] {f1:>9.1} [{:>4.2}x] {f4:>9.1} [{:>4.2}x]",
            rt / d, rp / d, f1 / d, f4 / d
        );
    }
}
