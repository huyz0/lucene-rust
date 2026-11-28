//! The Arrow export returns what the table holds, for every filter shape,
//! with and without deletions, in one reader or in partitions.

mod common;

use arrow::array::RecordBatch;
use arrow::util::display::array_value_to_string;
use common::*;
use lucene_sql::export::{Filter, Literal, Op, Scan, ScanOptions};
use lucene_sql::SqlConfig;

fn collect(scan: &Scan) -> Vec<Vec<String>> {
    let batches: Vec<RecordBatch> = scan.open().map(|b| b.unwrap()).collect();
    sorted(rows(&batches))
}

fn expected(f: &Fixture, keep: impl Fn(&Doc) -> bool) -> Vec<Vec<String>> {
    sorted(
        f.live()
            .filter(|d| keep(d))
            .map(|d| {
                vec![
                    d.id.clone(),
                    d.price.to_string(),
                    if d.cat.is_empty() {
                        String::new()
                    } else {
                        d.cat.clone()
                    },
                ]
            })
            .collect(),
    )
}

fn options(filters: Vec<Filter>, dictionary: bool) -> ScanOptions {
    ScanOptions {
        columns: Some(vec!["id".into(), "price".into(), "cat".into()]),
        filters,
        dictionary,
        batch_rows: 64,
        ..ScanOptions::default()
    }
}

fn cmp(column: &str, op: Op, value: Literal) -> Filter {
    Filter::Cmp {
        column: column.into(),
        op,
        value,
    }
}

fn docs_with_gaps() -> Vec<Doc> {
    let mut docs = corpus(700);
    for (i, d) in docs.iter_mut().enumerate() {
        if i % 7 == 0 {
            d.cat = String::new();
        }
    }
    docs
}

fn fixture_gaps() -> Fixture {
    fixture_docs(
        docs_with_gaps(),
        3,
        &[3, 77, 400, 699],
        spec(),
        &SqlConfig::default(),
    )
}

#[test]
fn a_full_scan_is_the_live_table_as_dictionary_or_text() {
    let f = fixture_gaps();
    for dictionary in [true, false] {
        let scan = Scan::new(f.snapshot().clone(), &options(vec![], dictionary)).unwrap();
        assert_eq!(collect(&scan), expected(&f, |_| true), "dict={dictionary}");
    }
}

#[test]
fn filters_on_long_keyword_key_and_nulls_select_the_same_rows_as_the_corpus() {
    let f = fixture_gaps();
    let check = |filters: Vec<Filter>, keep: &dyn Fn(&Doc) -> bool| {
        for dictionary in [true, false] {
            let scan =
                Scan::new(f.snapshot().clone(), &options(filters.clone(), dictionary)).unwrap();
            assert_eq!(collect(&scan), expected(&f, keep), "{filters:?}");
        }
    };
    let long = |v: i64| Literal::Long(v);
    let text = |v: &str| Literal::Str(v.to_string());
    check(vec![cmp("price", Op::Ge, long(500))], &|d| d.price >= 500);
    check(vec![cmp("price", Op::Lt, long(100))], &|d| d.price < 100);
    check(vec![cmp("price", Op::Eq, long(7))], &|d| d.price == 7);
    check(vec![cmp("price", Op::Ne, long(7))], &|d| d.price != 7);
    check(
        vec![
            cmp("price", Op::Gt, long(100)),
            cmp("price", Op::Le, long(300)),
        ],
        &|d| d.price > 100 && d.price <= 300,
    );
    check(vec![cmp("price", Op::Ge, long(i64::MIN))], &|_| true);
    check(vec![cmp("price", Op::Le, long(i64::MAX))], &|_| true);
    check(vec![cmp("price", Op::Gt, long(i64::MAX))], &|_| false);
    check(vec![cmp("cat", Op::Eq, text("c03"))], &|d| d.cat == "c03");
    check(vec![cmp("cat", Op::Ne, text("c03"))], &|d| {
        !d.cat.is_empty() && d.cat != "c03"
    });
    check(vec![cmp("cat", Op::Lt, text("c05"))], &|d| {
        !d.cat.is_empty() && d.cat.as_str() < "c05"
    });
    check(vec![cmp("cat", Op::Ge, text("c05x"))], &|d| {
        !d.cat.is_empty() && d.cat.as_str() >= "c05x"
    });
    check(vec![cmp("cat", Op::Eq, text("zzz"))], &|_| false);
    check(
        vec![Filter::In {
            column: "cat".into(),
            values: vec![text("c01"), text("c07"), text("nope")],
        }],
        &|d| d.cat == "c01" || d.cat == "c07",
    );
    check(
        vec![Filter::In {
            column: "price".into(),
            values: vec![long(1), long(2), long(3), long(10), long(11), long(900)],
        }],
        &|d| [1, 2, 3, 10, 11, 900].contains(&d.price),
    );
    check(
        vec![Filter::IsNull {
            column: "cat".into(),
        }],
        &|d| d.cat.is_empty(),
    );
    check(
        vec![Filter::IsNotNull {
            column: "cat".into(),
        }],
        &|d| !d.cat.is_empty(),
    );
    check(vec![cmp("id", Op::Eq, text("doc-00010"))], &|d| {
        d.id == "doc-00010"
    });
    check(
        vec![Filter::In {
            column: "id".into(),
            values: vec![text("doc-00010"), text("doc-00003"), text("doc-00500")],
        }],
        &|d| ["doc-00010", "doc-00003", "doc-00500"].contains(&d.id.as_str()),
    );
    check(
        vec![
            cmp("cat", Op::Eq, text("c03")),
            cmp("price", Op::Ge, long(300)),
            cmp("id", Op::Ne, text("x")).clone(),
        ][..2]
            .to_vec(),
        &|d| d.cat == "c03" && d.price >= 300,
    );
}

#[test]
fn a_limit_stops_the_scan_and_partitions_cover_the_table_once() {
    let f = fixture_gaps();
    let mut o = options(vec![], true);
    o.limit = Some(100);
    let scan = Scan::new(f.snapshot().clone(), &o).unwrap();
    let n: usize = scan.open().map(|b| b.unwrap().num_rows()).sum();
    assert_eq!(n, 100);

    let scan = Scan::new(f.snapshot().clone(), &options(vec![], true)).unwrap();
    let groups = scan.partitions(2);
    assert_eq!(groups.len(), 2);
    let mut all = Vec::new();
    for g in groups {
        let batches: Vec<RecordBatch> = scan.open_segments(g).map(|b| b.unwrap()).collect();
        all.extend(rows(&batches));
    }
    assert_eq!(sorted(all), expected(&f, |_| true));
}

#[test]
fn projection_order_and_errors() {
    let f = fixture_gaps();
    let mut o = options(vec![], true);
    o.columns = Some(vec!["price".into(), "id".into()]);
    let scan = Scan::new(f.snapshot().clone(), &o).unwrap();
    let b = scan.open().next().unwrap().unwrap();
    assert_eq!(b.schema().field(0).name(), "price");
    assert_eq!(
        array_value_to_string(b.column(1), 0).unwrap().len(),
        "doc-00000".len()
    );
    // Default columns: every readable one.
    let all = Scan::new(f.snapshot().clone(), &ScanOptions::default()).unwrap();
    assert_eq!(all.schema().fields().len(), 3);
    for bad in [
        vec!["body".to_string()],
        vec!["_score".to_string()],
        vec!["nope".to_string()],
    ] {
        let o = ScanOptions {
            columns: Some(bad),
            ..ScanOptions::default()
        };
        assert!(Scan::new(f.snapshot().clone(), &o).is_err());
    }
    for bad in [
        cmp("price", Op::Eq, Literal::Str("x".into())),
        cmp("cat", Op::Eq, Literal::Long(1)),
        cmp("id", Op::Gt, Literal::Str("x".into())),
        cmp("body", Op::Eq, Literal::Str("x".into())),
        cmp("nope", Op::Eq, Literal::Long(1)),
        Filter::IsNull {
            column: "id".into(),
        },
    ] {
        assert!(Scan::new(f.snapshot().clone(), &options(vec![bad], true)).is_err());
    }
}

#[test]
fn dropping_a_reader_early_stops_its_thread() {
    let f = fixture_gaps();
    let scan = Scan::new(f.snapshot().clone(), &options(vec![], true)).unwrap();
    let mut r = scan.open();
    assert!(r.next().is_some());
    drop(r);
}

#[test]
fn batches_come_out_in_document_order_whatever_the_thread_count() {
    let f = fixture_gaps();
    let ids = |threads: usize| -> Vec<String> {
        let mut o = options(vec![], true);
        o.threads = threads;
        let scan = Scan::new(f.snapshot().clone(), &o).unwrap();
        let batches: Vec<RecordBatch> = scan.open().map(|b| b.unwrap()).collect();
        rows(&batches).into_iter().map(|r| r[0].clone()).collect()
    };
    let one = ids(1);
    assert_eq!(one, f.live().map(|d| d.id.clone()).collect::<Vec<_>>());
    for t in [2, 3, 8] {
        assert_eq!(ids(t), one, "threads={t}");
    }
}

#[test]
fn no_columns_returns_row_counts_only() {
    let f = fixture_gaps();
    let mut o = options(vec![cmp("price", Op::Ge, Literal::Long(500))], true);
    o.columns = Some(vec![]);
    let scan = Scan::new(f.snapshot().clone(), &o).unwrap();
    assert_eq!(scan.schema().fields().len(), 0);
    let n: usize = scan.open().map(|b| b.unwrap().num_rows()).sum();
    assert_eq!(n, f.live().filter(|d| d.price >= 500).count());
}

fn streamed(engine: &std::sync::Arc<lucene_sql::SqlEngine>, sql: &str) -> Vec<Vec<String>> {
    let batches: Vec<RecordBatch> = engine
        .stream_blocking(sql)
        .unwrap()
        .map(|b| b.unwrap())
        .collect();
    sorted(rows(&batches))
}

#[test]
fn a_sql_stream_returns_what_the_statement_returns() {
    let f = fixture_gaps();
    let engine = std::sync::Arc::new(
        lucene_sql::SqlEngine::new(f.snapshot().clone(), &SqlConfig::default()).unwrap(),
    );
    for sql in [
        // The router's point read, DataFusion's scan, a group-by on the kernels
        // and one the kernels refuse (documents without a `cat`).
        "SELECT id, price FROM t WHERE id = 'doc-00010'",
        "SELECT id, price FROM t WHERE price > 900",
        "SELECT cat, count(*), sum(price) FROM t GROUP BY cat",
        "SELECT price % 7 AS m, count(*) FROM t GROUP BY price % 7",
    ] {
        assert_eq!(streamed(&engine, sql), sorted(run(&f, sql)), "{sql}");
    }
    assert!(engine.stream_blocking("SELEC nope").is_err());
    assert!(engine.stream_blocking("DROP TABLE t").is_err());
}
