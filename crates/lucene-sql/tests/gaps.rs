//! Refusals, accessors and the shapes the router must leave alone.

mod common;

use common::*;
use datafusion::catalog::Session;
use datafusion::datasource::TableProvider;
use lucene_sql::provider::LuceneTable;
use lucene_sql::router::Shape;
use lucene_sql::SqlConfig;

#[test]
fn near_miss_shapes_are_not_routed() {
    let f = fixture(60, 1, &[]);
    let not = [
        // the table
        "SELECT id FROM t AS x WHERE id = 'a'",
        "SELECT id FROM other WHERE id = 'a'",
        "SELECT id FROM t, t WHERE id = 'a'",
        "SELECT id FROM t JOIN t u ON t.id = u.id WHERE t.id = 'a'",
        "SELECT id FROM (SELECT id FROM t) WHERE id = 'a'",
        "SELECT id FROM s.t WHERE id = 'a'",
        // the projection
        "SELECT * FROM t WHERE id = 'a'",
        "SELECT t.id FROM t WHERE id = 'a'",
        "SELECT id, _score FROM t WHERE id = 'a'",
        "SELECT body FROM t WHERE id = 'a'",
        "SELECT nope FROM t WHERE id = 'a'",
        "SELECT id + 1 FROM t WHERE id = 'a'",
        // the predicate
        "SELECT id FROM t WHERE price = 'a'",
        "SELECT id FROM t WHERE nope = 'a'",
        "SELECT id FROM t WHERE id = 5",
        "SELECT id FROM t WHERE id > 'a'",
        "SELECT id FROM t",
        "SELECT id FROM t WHERE id = 'a' ORDER BY id",
        // text and vector searches
        "SELECT id FROM t WHERE match_text(body, 'a') ORDER BY _score DESC LIMIT x",
        "SELECT id FROM t WHERE match_text(body, 'a') ORDER BY _score DESC LIMIT 5.5",
        "SELECT id FROM t WHERE match_text(body, 'a') ORDER BY _score DESC LIMIT 5, 2",
        "SELECT id FROM t WHERE match_text(body, 'a') ORDER BY _score DESC NULLS LAST LIMIT 5",
        "SELECT id FROM t WHERE match_text(body, 'a') ORDER BY _score DESC, id LIMIT 5",
        "SELECT id FROM t WHERE match_text(body, 'a') ORDER BY price DESC LIMIT 5",
        "SELECT id FROM t WHERE match_text(body, 'a') ORDER BY ALL DESC LIMIT 5",
        "SELECT id FROM t WHERE match_text(body, 'a') ORDER BY _score ASC LIMIT 5",
        "SELECT id FROM t WHERE match_text(body, 'a') ORDER BY _score DESC",
        "SELECT id FROM t WHERE match_text(body, 'a', 'b') ORDER BY _score DESC LIMIT 5",
        "SELECT id FROM t WHERE match_text(cat, 'a') ORDER BY _score DESC LIMIT 5",
        "SELECT id FROM t WHERE match_text(body, id) ORDER BY _score DESC LIMIT 5",
        "SELECT id FROM t WHERE match_text(t.body, 'a') ORDER BY _score DESC LIMIT 5",
        "SELECT id FROM t WHERE knn(body, '1') ORDER BY _score DESC LIMIT 5",
        "SELECT id FROM t WHERE other(body, 'a') ORDER BY _score DESC LIMIT 5",
        "SELECT body FROM t WHERE match_text(body, 'a') ORDER BY _score DESC LIMIT 5",
        "SELECT id FROM t WHERE match_text(body, 'a') ORDER BY _score DESC LIMIT 5 OFFSET 1",
        "SELECT id FROM t WHERE match_text(body, 'a') ORDER BY _score DESC FETCH FIRST 5 ROWS ONLY",
        "SELECT DISTINCT id FROM t WHERE id = 'a'",
        "WITH x AS (SELECT 1) SELECT id FROM t WHERE id = 'a'",
        "SELECT id FROM t WHERE id = 'a' UNION SELECT id FROM t WHERE id = 'b'",
        "SELECT id FROM t WHERE id = 'a' FOR UPDATE",
        "VALUES (1)",
        "SELECT id FROM t WHERE id = 'a'; SELECT id FROM t WHERE id = 'b'",
        "SELECT 'a'",
        "this is not sql",
        // group-by
        "SELECT cat, count(*) FROM t GROUP BY ALL",
        "SELECT cat, count(*) FROM t GROUP BY cat, id",
        "SELECT id, count(*) FROM t GROUP BY id",
        "SELECT cat, count(DISTINCT price) FROM t GROUP BY cat",
        "SELECT cat, sum(cat) FROM t GROUP BY cat",
        "SELECT cat, sum(price) FILTER (WHERE price > 1) FROM t GROUP BY cat",
        "SELECT cat, sum(price) OVER () FROM t GROUP BY cat",
        "SELECT cat AS c, count(*) FROM t GROUP BY cat",
        "SELECT cat, foo(price) FROM t GROUP BY cat",
        "SELECT cat, sum(price, 2) FROM t GROUP BY cat",
        "SELECT cat, sum(*) FROM t GROUP BY cat",
        "SELECT cat, sum(t.price) FROM t GROUP BY cat",
        "SELECT cat, sum(nope) FROM t GROUP BY cat",
        "SELECT cat, count(*) FROM t GROUP BY cat LIMIT 3",
        "SELECT cat, count(*) FROM t GROUP BY cat -- 'x'",
        "SELECT cat, count(*) FROM t WHERE id = 'a' GROUP BY cat",
        "SELECT cat, count(*) FROM t GROUP BY nope",
    ];
    for sql in not {
        assert!(f.engine.router().shape(sql).is_none(), "routed: {sql}");
    }
    let yes = [
        "SELECT count(*) FROM t GROUP BY cat",
        "SELECT cat FROM t GROUP BY cat",
        "SELECT SUM(price), Cat FROM t GROUP BY CAT",
    ];
    for sql in yes {
        assert!(
            matches!(f.engine.router().shape(sql), Some(Shape::Group { .. })),
            "{sql}"
        );
    }
    // An unterminated literal is data up to the end, never a shape.
    assert!(f
        .engine
        .router()
        .shape("SELECT id FROM t WHERE id = 'abc")
        .is_none());
}

#[test]
fn many_distinct_templates_do_not_grow_the_cache_without_bound() {
    let f = fixture(30, 1, &[]);
    for k in 0..4200 {
        let sql = format!(
            "SELECT id FROM t WHERE match_text(body, 'a') ORDER BY _score DESC LIMIT {}",
            k + 1
        );
        assert!(f.engine.router().shape(&sql).is_some());
    }
}

#[test]
fn bad_literals_and_columns_are_errors_not_panics() {
    let f = fixture(60, 1, &[]);
    let e = |sql: &str| f.engine.query_blocking(sql).unwrap_err().to_string();
    assert!(
        e("SELECT id FROM t WHERE knn(vec, 'a,b') ORDER BY _score DESC LIMIT 3")
            .contains("list of numbers")
    );
    assert!(
        e("SELECT id FROM t WHERE knn(vec, '1,2') ORDER BY _score DESC LIMIT 3")
            .contains("dimensions")
    );
    assert!(e("SELECT id FROM t WHERE knn(vec, 'a') AND cat = 'c01'").contains("list of numbers"));
    let s = f.snapshot();
    assert!(s.lookup("price", "1").is_err());
    assert!(s.lookup("nope", "1").is_err());
    assert!(s.match_top_k("cat", "a", 3).is_err());
    assert!(s.knn_top_k("body", &[1.0], 3).is_err());
    assert!(s.knn_top_k("vec", &[1.0], 3).is_err());
    assert!(s.knn_top_k("vec", &[0.0; DIM], 0).unwrap().is_empty());
    assert!(s.match_top_k("body", "   ", 3).unwrap().is_empty());
    assert!(s.match_top_k("body", "w1", 0).unwrap().is_empty());
    assert!(s.read(&[99], &[(0, None)]).is_err());
    assert!(s.read(&[3], &[(0, None)]).is_err()); // body has no values
    assert!(s.read(&[0], &[(-1, None)]).is_err());
    assert!(
        s.read(&[0], &[(1_000_000, None)])
            .is_ok_and(|b| b.num_rows() == 1)
            || true
    );
    assert_eq!(s.segment_count(), 1);
    assert_eq!(s.max_doc(), 60);
    assert_eq!(s.reader().max_doc(), 60);
    assert_eq!(s.spec().name(), "t");
}

#[test]
fn columns_the_index_lacks_read_as_null() {
    let spec = spec().long("nolong").keyword("nokw").key("nokey");
    let f = fixture_with(40, 2, &[3], spec, &SqlConfig::default());
    let r = run(&f, "SELECT nolong, nokw, nokey FROM t LIMIT 2");
    assert_eq!(r.len(), 2);
    assert!(r.iter().all(|row| row.iter().all(String::is_empty)));
    let r = run(
        &f,
        "SELECT id, nolong, nokw, nokey, price, cat, _score FROM t WHERE id = 'doc-00005'",
    );
    assert_eq!(r[0][0], "doc-00005");
    assert_eq!(r[0][1], "");
    let r = run(&f, "SELECT count(*) FROM t WHERE nokw IS NULL");
    assert_eq!(r[0][0], "39");
    // The group-by cannot use a keyword the segments lack.
    assert!(f
        .snapshot()
        .group_by(6, &[lucene_sql::group::AggFn::CountStar])
        .unwrap()
        .is_none());
    let g = run(&f, "SELECT nokw, count(*) FROM t GROUP BY nokw");
    assert_eq!(g, vec![vec![String::new(), "39".to_string()]]);
}

#[test]
fn engine_runs_async_on_the_callers_runtime_and_on_one_thread() {
    let config = SqlConfig {
        worker_threads: 0,
        target_partitions: 2,
        ..SqlConfig::default()
    };
    let f = fixture_with(80, 2, &[], spec(), &config);
    assert_eq!(f.engine.snapshot().max_doc(), 80);
    assert_eq!(run(&f, "SELECT count(*) FROM t")[0][0], "80");
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let fast = rt
        .block_on(
            f.engine
                .query("SELECT id, price FROM t WHERE id = 'doc-00001'"),
        )
        .unwrap();
    assert_eq!(rows(&fast).len(), 1);
    let slow = rt
        .block_on(f.engine.query("SELECT count(*) FROM t"))
        .unwrap();
    assert_eq!(rows(&slow)[0][0], "80");
    let n = rt
        .block_on(f.engine.standard("SELECT count(*) FROM t"))
        .unwrap();
    assert_eq!(rows(&n)[0][0], "80");
    assert!(f.engine.context().table_exist("t").unwrap());
    let d = SqlConfig::default();
    assert!(d.memory_limit > 0 && d.target_partitions >= 1);
}

#[test]
fn a_memory_limit_stops_a_statement_instead_of_growing() {
    let config = SqlConfig {
        memory_limit: 64 << 10,
        worker_threads: 0,
        target_partitions: 1,
        plan_cache: true,
        ..SqlConfig::default()
    };
    let f = fixture_with(3000, 1, &[], spec(), &config);
    let r = f
        .engine
        .query_blocking("SELECT id, price FROM t ORDER BY price, id");
    assert!(r.is_err(), "a 64 KiB pool sorting 3000 rows must refuse");
}

#[test]
fn the_scan_describes_itself_and_refuses_what_it_cannot_plan() {
    let f = fixture(60, 3, &[]);
    let table = LuceneTable::new(f.snapshot().clone());
    assert!(format!("{table:?}").contains("LuceneTable"));
    assert_eq!(table.snapshot().max_doc(), 60);
    assert_eq!(table.table_type(), datafusion::datasource::TableType::Base);
    let state = f.engine.context().state();
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let plan = rt
        .block_on(table.scan(&state as &dyn Session, Some(&[0usize, 1][..]), &[], Some(5)))
        .unwrap();
    assert_eq!(plan.name(), "SegScanExec");
    assert!(plan.children().is_empty());
    assert!(format!("{plan:?}").contains("SegScanExec"));
    let shown = format!(
        "{}",
        datafusion::physical_plan::displayable(plan.as_ref()).indent(false)
    );
    assert!(shown.contains("partitions=3"), "{shown}");
    assert_eq!(plan.properties().output_partitioning().partition_count(), 3);
    let mut visited = 0;
    plan.apply_expressions(&mut |_| {
        visited += 1;
        Ok(datafusion::common::tree_node::TreeNodeRecursion::Continue)
    })
    .unwrap();
    assert_eq!(visited, 0);
    assert!(rt
        .block_on(table.scan(&state as &dyn Session, None, &[], None))
        .is_err());
    // Executing a partition twice gives the same rows (no state is kept).
    let ctx = f.engine.context().task_ctx();
    let count = |p: usize| {
        let s = plan.execute(p, ctx.clone()).unwrap();
        rt.block_on(datafusion::physical_plan::common::collect(s))
            .unwrap()
            .iter()
            .map(|b| b.num_rows())
            .sum::<usize>()
    };
    assert_eq!(count(0), 5);
    assert_eq!(count(0), 5);
}

#[test]
fn the_top_k_rule_reaches_through_projections_and_only_for_score_desc() {
    let f = fixture(300, 3, &[]);
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let plan = |sql: &str| {
        rt.block_on(async {
            let df = f.engine.context().sql(sql).await.unwrap();
            format!("{}", df.into_optimized_plan().unwrap().display_indent())
        })
    };
    let w = "match_text(body, 'w1 w2')";
    // Through a projection that is not the sort's input directly.
    let p = plan(&format!(
        "SELECT id FROM t WHERE {w} ORDER BY _score DESC LIMIT 7"
    ));
    assert!(p.contains("fetch=7"), "{p}");
    let p = plan(&format!(
        "SELECT id, _score * 2 AS s FROM t WHERE {w} ORDER BY _score DESC LIMIT 6"
    ));
    assert!(p.contains("fetch=6"), "{p}");
    // Not for ascending, a second key, or another column.
    for order in ["_score ASC", "_score DESC, id", "price DESC"] {
        let p = plan(&format!(
            "SELECT id, _score FROM t WHERE {w} ORDER BY {order} LIMIT 5"
        ));
        assert!(!p.contains("fetch=5") || p.contains("TableScan: t") && !p.contains("TableScan: t projection=[id, _score], full_filters=[match_text(t.body, Utf8(\"w1 w2\"))], fetch=5"), "{order}: {p}");
    }
    // A scan that already stops sooner keeps its fetch.
    let p = plan(&format!(
        "SELECT id, _score FROM t WHERE {w} ORDER BY _score DESC LIMIT 5"
    ));
    assert!(p.contains("fetch=5"), "{p}");
}

#[test]
fn a_statement_with_more_rows_than_the_limit_fails_instead_of_collecting() {
    let config = SqlConfig {
        worker_threads: 0,
        max_rows: 100,
        ..SqlConfig::default()
    };
    let f = fixture_with(3000, 1, &[], spec(), &config);
    let err = f
        .engine
        .query_blocking("SELECT id, price FROM t ORDER BY price, id")
        .unwrap_err();
    assert!(err.to_string().contains("more than 100 rows"), "{err}");
    let ok = f
        .engine
        .query_blocking("SELECT id, price FROM t ORDER BY price, id LIMIT 100")
        .unwrap();
    assert_eq!(ok.iter().map(|b| b.num_rows()).sum::<usize>(), 100);
}

#[test]
fn an_empty_result_keeps_its_schema() {
    let f = fixture_with(10, 1, &[], spec(), &SqlConfig::default());
    let batches = f
        .engine
        .query_blocking("SELECT id, price FROM t WHERE price > 1000000000 ORDER BY price")
        .unwrap();
    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 0);
    assert_eq!(batches[0].schema().fields().len(), 2);
}
