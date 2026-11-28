//! Latency of the SQL paths against the native API called directly.
//!
//! `cargo run --release -p lucene-sql --example sql_bench -- <dir> [docs] [secs]`
//! builds a 4-segment index of `docs` documents (default 1M) in `<dir>` when it
//! is empty, then times each shape on the router, on the cached DataFusion
//! plan, on DataFusion planned from the text, and on the direct API.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use lucene_aggs::bucket_aggs::{aggregate_tree, AggNode, AggResult, Globals, TermsOrder};
use lucene_aggs::metric_aggs::{Source, ValueKind, NEED_SUM};
use lucene_codecs::field_infos::VectorSimilarityFunction;
use lucene_index::document::{
    Document, KeywordField, KnnFloatVectorField, LongField, NumericDocValuesField, Store,
    StringField, TextField,
};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
use lucene_search::knn_search::{knn_top_hits, KnnTarget};
use lucene_search::query::{BooleanQuery, Clause, MatchAllDocsQuery, TermQuery};
use lucene_sql::{Snapshot, SqlConfig, SqlEngine, TableSpec};
use lucene_store::{FsDirectory, MmapDirectory};

const DIM: usize = 32;

fn next(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}
fn unit(s: &mut u64) -> f64 {
    (next(s) >> 11) as f64 / (1u64 << 53) as f64
}
fn id_of(i: usize) -> String {
    format!("doc-{i:08}")
}
fn rvec(s: &mut u64) -> Vec<f32> {
    (0..DIM)
        .map(|_| (unit(s) + unit(s) + unit(s) - 1.5) as f32)
        .collect()
}

fn build(path: &str, docs: usize, segments: usize) {
    let dir = FsDirectory::open(path);
    let v = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", v).unwrap();
    w.set_ram_buffer_size_mb(512.0).unwrap();
    let mut st = 0x9e37_79b9_7f4a_7c15u64;
    let per = docs.div_ceil(segments);
    for i in 0..docs {
        let mut d = Document::new();
        d.add(StringField::new("id", id_of(i), Store::Yes));
        let body: Vec<String> = (0..40)
            .map(|_| format!("w{}", 50_000f64.powf(unit(&mut st)) as u64))
            .collect();
        d.add(TextField::new("body", body.join(" "), Store::No));
        d.add(NumericDocValuesField::new(
            "price",
            (next(&mut st) % 10_000) as i64,
        ));
        d.add(LongField::new(
            "ts",
            (next(&mut st) % 1_000_000) as i64,
            Store::No,
        ));
        d.add(KeywordField::new(
            "cat",
            format!("c{:03}", next(&mut st) % 100),
            Store::No,
        ));
        d.add(
            KnnFloatVectorField::new("vec", rvec(&mut st), VectorSimilarityFunction::Euclidean)
                .unwrap(),
        );
        w.add_fields_document(&d).unwrap();
        if (i + 1) % per == 0 {
            w.commit().unwrap();
        }
    }
    w.commit().unwrap();
}

fn pct(v: &mut [Duration], p: f64) -> f64 {
    v.sort_unstable();
    v[((v.len() as f64 - 1.0) * p).round() as usize].as_secs_f64() * 1e6
}

fn time(label: &str, secs: f64, mut f: impl FnMut(usize)) -> f64 {
    let w = Instant::now();
    let mut i = 0;
    while w.elapsed().as_secs_f64() < (secs / 4.0).min(1.5) {
        f(i);
        i += 1;
    }
    let mut lat = Vec::new();
    let t = Instant::now();
    while t.elapsed().as_secs_f64() < secs {
        let s = Instant::now();
        f(i);
        lat.push(s.elapsed());
        i += 1;
    }
    let p50 = pct(&mut lat, 0.5);
    println!(
        "  {label:<32} n={:<7} p50={p50:>9.1}us p99={:>9.1}us",
        lat.len(),
        pct(&mut lat, 0.99)
    );
    p50
}

fn term_q(field: &str, t: &str) -> BooleanQuery {
    BooleanQuery {
        must: vec![Clause::Term(TermQuery::new(field, t.as_bytes().to_vec()))],
        ..BooleanQuery::default()
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let path = &a[1];
    let docs: usize = a.get(2).map_or(1_000_000, |s| s.parse().unwrap());
    let secs: f64 = a.get(3).map_or(5.0, |s| s.parse().unwrap());
    if !std::path::Path::new(path).join("segments_1").exists()
        && std::fs::read_dir(path).map_or(true, |mut d| d.next().is_none())
    {
        std::fs::create_dir_all(path).unwrap();
        let t = Instant::now();
        build(path, docs, 4);
        eprintln!("built {docs} docs in {:.0?}", t.elapsed());
    }

    // The direct API.
    let dir: &'static MmapDirectory = Box::leak(Box::new(MmapDirectory::open(path.clone())));
    let reader: &'static DirectoryReader = Box::leak(Box::new(DirectoryReader::open(dir).unwrap()));
    let opened = Box::leak(Box::new({
        let mut o = reader.open_segments().unwrap();
        o.open_points().unwrap();
        o
    }));
    let segments = Box::leak(opened.as_open_segments().into_boxed_slice());
    let owned = Box::leak(Box::new(
        reader.field_norms_by_field(&["body".to_string()]).unwrap(),
    ));
    let norms: &'static [SegmentNorms<'static, 'static>] = Box::leak(
        owned
            .iter()
            .map(Some)
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    );
    let searcher = IndexSearcher::new(segments, norms).unwrap();
    let field_no = |seg: usize, n: &str| {
        reader.segment_readers()[seg]
            .field_infos()
            .field_by_name(n)
            .unwrap()
            .number
    };
    let locate = |doc: i32| {
        let s = searcher.segment_of(doc).unwrap();
        (s, doc - segments[s].doc_base)
    };
    let stored_id = |seg: usize, local: i32| {
        let mut v =
            lucene_codecs::stored_fields::DocumentVisitor::for_fields(&[field_no(seg, "id")]);
        let _ = reader.segment_readers()[seg]
            .visit_stored_document(local, &mut v)
            .unwrap();
        match &v.document().fields[0].value {
            lucene_codecs::stored_fields::FieldValue::String(s) => s.clone(),
            _ => unreachable!(),
        }
    };
    let price = |seg: usize, local: i32| {
        let f = field_no(seg, "price");
        let (meta, data) = reader.segment_readers()[seg].doc_values_for_field(f)?;
        let e = meta.numeric_entry(f)?;
        lucene_codecs::doc_values::NumericReader::new(data, e)
            .value(local)
            .unwrap()
    };
    let cat_global = reader.global_ords("cat").unwrap();
    let n = reader.max_doc() as u64;

    // The SQL engine, over its own reader.
    let spec = TableSpec::new("t")
        .key("id")
        .long("price")
        .long("ts")
        .keyword("cat")
        .text("body")
        .vector("vec", DIM);
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
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    println!(
        "index: {} docs, {} segments; cpus={}",
        n,
        segments.len(),
        std::thread::available_parallelism().unwrap()
    );

    let mut st = 0x1234_5678_9abc_def1u64;
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
                20_000 + (next(&mut st) % 20_000),
                20_000 + (next(&mut st) % 20_000)
            )
        })
        .collect();
    let vq: Vec<Vec<f32>> = (0..64)
        .map(|_| {
            rvec(&mut st)
                .iter()
                .map(|x| (x * 1e5).round() / 1e5)
                .collect()
        })
        .collect();
    let vs: Vec<String> = vq
        .iter()
        .map(|v| {
            v.iter()
                .map(|x| format!("{x:.5}"))
                .collect::<Vec<_>>()
                .join(",")
        })
        .collect();

    let words_q = |q: &str| BooleanQuery {
        should: q
            .split(' ')
            .map(|w| Clause::Term(TermQuery::new("body", w.as_bytes().to_vec())))
            .collect(),
        minimum_should_match: 1,
        ..BooleanQuery::default()
    };
    use std::hint::black_box as sink;
    let both = |sql: String| {
        sink(engine.query_blocking(&sql).unwrap());
    };
    let df_cached = |sql: String| {
        sink(engine.standard_blocking(&sql).unwrap());
    };
    let df_text = |sql: String| {
        sink(rt.block_on(engine.uncached(&sql)).unwrap());
    };

    let mut sum: Vec<(&str, f64, f64, f64, f64)> = Vec::new();

    println!("\n== point read");
    let sql = |i: usize| format!("SELECT id, price FROM t WHERE id = '{}'", ids[i % 64]);
    let d = time("direct API", secs, |i| {
        let top = searcher.search(&term_q("id", &ids[i % 64]), 1).unwrap();
        let h = top.score_docs.first().unwrap();
        let (s, l) = locate(h.doc);
        sink((stored_id(s, l), price(s, l)));
    });
    let r = time("router (SQL text)", secs, |i| both(sql(i)));
    let c = time("DataFusion, cached plan", secs, |i| df_cached(sql(i)));
    let u = time("DataFusion, planned from text", secs, |i| df_text(sql(i)));
    sum.push(("point read", d, r, c, u));

    for (name, qs) in [("MATCH top-10 mid", &mid), ("MATCH top-10 rare", &rare)] {
        println!("\n== {name}");
        let sql = |i: usize| {
            format!(
                "SELECT id, _score FROM t WHERE match_text(body, '{}') ORDER BY _score DESC LIMIT 10",
                qs[i % 64]
            )
        };
        let d = time("direct API", secs, |i| {
            let top = searcher.search(&words_q(&qs[i % 64]), 10).unwrap();
            for h in &top.score_docs {
                let (s, l) = locate(h.doc);
                sink((stored_id(s, l), h.score));
            }
        });
        let r = time("router (SQL text)", secs, |i| both(sql(i)));
        let c = time("DataFusion, cached plan", secs, |i| df_cached(sql(i)));
        let u = time("DataFusion, planned from text", secs, |i| df_text(sql(i)));
        sum.push((name, d, r, c, u));
    }

    println!("\n== k-NN top-10");
    let sql = |i: usize| {
        format!(
            "SELECT id, _score FROM t WHERE knn(vec, '{}') ORDER BY _score DESC LIMIT 10",
            vs[i % 64]
        )
    };
    let d = time("direct API", secs, |i| {
        let hits = knn_top_hits(
            segments,
            "vec",
            &KnnTarget::Float(vq[i % 64].clone()),
            10,
            50,
            None,
        )
        .unwrap();
        for h in &hits {
            let (s, l) = locate(h.doc_id);
            sink((stored_id(s, l), h.score));
        }
    });
    let r = time("router (SQL text)", secs, |i| both(sql(i)));
    let c = time("DataFusion, cached plan", secs, |i| df_cached(sql(i)));
    let u = time("DataFusion, planned from text", secs, |i| df_text(sql(i)));
    sum.push(("k-NN top-10", d, r, c, u));

    println!("\n== group-by");
    let g = "SELECT cat, count(*), sum(price) FROM t GROUP BY cat";
    let g2 =
        "SELECT cat AS c, count(*) AS n, sum(price) AS s FROM t GROUP BY cat ORDER BY n DESC, c";
    let d = time("direct API (kernels)", secs.min(4.0), |_| {
        let node = AggNode::Terms {
            field: "cat".into(),
            missing: None,
            accepted: None,
            zero_buckets: false,
            shard_size: 1000,
            order: TermsOrder::KeyAsc,
            subs: vec![AggNode::Metric {
                field: "price".into(),
                missing: None,
                kind: ValueKind::Long,
                source: Source::DocValues,
                needs: NEED_SUM,
                script: None,
            }],
        };
        let q = BooleanQuery {
            must: vec![Clause::MatchAllDocs(MatchAllDocsQuery::new(
                reader.max_doc(),
            ))],
            ..BooleanQuery::default()
        };
        let ords: HashMap<String, _> = HashMap::from([("cat".to_string(), cat_global.clone())]);
        let slices = vec![(0..segments.len()).collect::<Vec<_>>()];
        let out = aggregate_tree(
            segments,
            reader.segment_readers(),
            &q,
            &[node],
            &Globals::new(&ords),
            &slices,
            None,
        )
        .unwrap();
        sink(matches!(out[0][0], AggResult::Terms { .. }));
    });
    let r = time("router (SQL text)", secs.min(4.0), |_| both(g.into()));
    let c = time("DataFusion + group rule (aliased)", secs.min(4.0), |_| {
        df_cached(g2.into())
    });
    let u = time("DataFusion, hash aggregate", secs.min(4.0), |_| {
        sink(rt.block_on(async {
            engine
                .plain_context()
                .sql(g2)
                .await
                .unwrap()
                .collect()
                .await
                .unwrap()
        }));
    });
    sum.push(("group-by", d, r, c, u));

    let range_q = |lo: i64, hi: i64| BooleanQuery {
        filter: vec![Clause::from(
            lucene_search::document::long_point::new_range_query("ts", lo, hi).unwrap(),
        )],
        ..BooleanQuery::default()
    };
    let bounds: Vec<(i64, i64)> = (0..64)
        .map(|_| {
            let lo = (next(&mut st) % 900_000) as i64;
            (lo, lo + 10_000)
        })
        .collect();
    let wide: Vec<(i64, i64)> = (0..64)
        .map(|_| {
            let lo = (next(&mut st) % 400_000) as i64;
            (lo, lo + 500_000)
        })
        .collect();

    for (name, bs) in [("range count 1%", &bounds), ("range count 50%", &wide)] {
        println!("\n== {name}");
        let sql = |i: usize| {
            let (lo, hi) = bs[i % 64];
            format!("SELECT count(*) FROM t WHERE ts BETWEEN {lo} AND {hi}")
        };
        let d = time("direct API (count)", secs, |i| {
            let (lo, hi) = bs[i % 64];
            sink(searcher.count(&range_q(lo, hi)).unwrap());
        });
        let r = time("engine (SQL text)", secs, |i| both(sql(i)));
        let c = time("DataFusion, cached plan", secs, |i| df_cached(sql(i)));
        let u = time("DataFusion, planned from text", secs, |i| df_text(sql(i)));
        sum.push((name, d, r, c, u));
    }

    println!("\n== filter count");
    let cats: Vec<String> = (0..64)
        .map(|_| format!("c{:03}", next(&mut st) % 100))
        .collect();
    let sql = |i: usize| format!("SELECT count(*) FROM t WHERE cat = '{}'", cats[i % 64]);
    let d = time("direct API (count)", secs, |i| {
        sink(searcher.count(&term_q("cat", &cats[i % 64])).unwrap());
    });
    let r = time("engine (SQL text)", secs, |i| both(sql(i)));
    let c = time("DataFusion, cached plan", secs, |i| df_cached(sql(i)));
    let u = time("DataFusion, planned from text", secs, |i| df_text(sql(i)));
    sum.push(("filter count", d, r, c, u));

    println!("\n== filtered group-by (ts 1%)");
    let gsql = |i: usize| {
        let (lo, hi) = bounds[i % 64];
        format!(
            "SELECT cat, count(*), sum(price) FROM t WHERE ts BETWEEN {lo} AND {hi} GROUP BY cat"
        )
    };
    let d = time("direct API (kernels)", secs.min(4.0), |i| {
        let node = AggNode::Terms {
            field: "cat".into(),
            missing: None,
            accepted: None,
            zero_buckets: false,
            shard_size: 1000,
            order: TermsOrder::KeyAsc,
            subs: vec![AggNode::Metric {
                field: "price".into(),
                missing: None,
                kind: ValueKind::Long,
                source: Source::DocValues,
                needs: NEED_SUM,
                script: None,
            }],
        };
        let (lo, hi) = bounds[i % 64];
        let ords: HashMap<String, _> = HashMap::from([("cat".to_string(), cat_global.clone())]);
        let slices = vec![(0..segments.len()).collect::<Vec<_>>()];
        let out = aggregate_tree(
            segments,
            reader.segment_readers(),
            &range_q(lo, hi),
            &[node],
            &Globals::new(&ords),
            &slices,
            None,
        )
        .unwrap();
        sink(matches!(out[0][0], AggResult::Terms { .. }));
    });
    let r = time("engine (SQL text)", secs.min(4.0), |i| both(gsql(i)));
    let c = time("DataFusion, cached plan", secs.min(4.0), |i| {
        df_cached(gsql(i))
    });
    let u = time("DataFusion, planned from text", secs.min(4.0), |i| {
        df_text(gsql(i))
    });
    sum.push(("filtered group-by", d, r, c, u));

    println!(
        "\n{:<20}{:>11}{:>11}{:>9}{:>11}{:>9}{:>13}{:>9}",
        "shape (p50 us)", "direct", "router", "x", "DF cached", "x", "DF no cache", "x"
    );
    for (n, d, r, c, u) in sum {
        println!(
            "{n:<20}{d:>11.1}{r:>11.1}{:>8.2}x{c:>11.1}{:>8.1}x{u:>13.1}{:>8.1}x",
            r / d,
            c / d,
            u / d
        );
    }
}
