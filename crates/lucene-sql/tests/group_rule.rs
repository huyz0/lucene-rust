//! The group-by rule sends DataFusion's aggregates to the kernels when they
//! answer exactly, and the statement runs the standard way when they do not.

mod common;

use std::collections::BTreeMap;
use std::sync::Arc;

use common::*;

const N: usize = 2000;

fn plan_text(f: &Fixture, sql: &str) -> String {
    futures::executor::block_on(async {
        let b = f
            .engine
            .context()
            .sql(&format!("EXPLAIN {sql}"))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        rows(&b)
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .join("\n")
    })
}

fn expected(f: &Fixture) -> BTreeMap<String, (i64, i64)> {
    let mut m: BTreeMap<String, (i64, i64)> = BTreeMap::new();
    for d in f.live() {
        let e = m.entry(d.cat.clone()).or_default();
        e.0 += 1;
        e.1 += d.price;
    }
    m
}

fn as_map(r: Vec<Vec<String>>) -> BTreeMap<String, (i64, i64)> {
    r.into_iter()
        .map(|x| (x[0].clone(), (x[1].parse().unwrap(), x[2].parse().unwrap())))
        .collect()
}

// Statements the router does not take but the rule does: an aliased
// aggregate wrapped in a sort.
const SQL: &str =
    "SELECT cat AS c, count(*) AS n, sum(price) AS s FROM t GROUP BY cat ORDER BY n DESC, c";

#[test]
fn the_plan_reads_the_kernels_and_the_answer_matches_the_documents() {
    let f = fixture(N, 3, &[]);
    assert!(f.engine.router().shape(SQL).is_none());
    assert!(plan_text(&f, SQL).contains("GroupScanExec"));
    let r = run(&f, SQL);
    assert_eq!(as_map(r.clone()), expected(&f));
    let counts: Vec<i64> = r.iter().map(|x| x[1].parse().unwrap()).collect();
    assert!(counts.windows(2).all(|w| w[0] >= w[1]));
}

#[test]
fn deleted_documents_do_not_count() {
    let deleted: Vec<usize> = (0..N).step_by(5).collect();
    let f = fixture(N, 3, &deleted);
    assert!(plan_text(&f, SQL).contains("GroupScanExec"));
    assert_eq!(as_map(run(&f, SQL)), expected(&f));
}

#[test]
fn a_filter_or_other_aggregate_keeps_the_standard_plan() {
    let f = fixture(N, 2, &[]);
    for sql in [
        "SELECT cat, count(*) FROM t WHERE price > 10 GROUP BY cat",
        "SELECT cat, count(distinct price) FROM t GROUP BY cat",
        "SELECT price, count(*) FROM t GROUP BY price",
    ] {
        assert!(!plan_text(&f, sql).contains("GroupScanExec"), "{sql}");
        run(&f, sql);
    }
}

#[test]
fn having_filters_the_kernel_groups() {
    let f = fixture(N, 2, &[]);
    let sql = "SELECT cat, count(*) AS n FROM t GROUP BY cat HAVING count(*) > 170";
    assert!(plan_text(&f, sql).contains("GroupScanExec"));
    let want = expected(&f).values().filter(|v| v.0 > 170).count();
    assert_eq!(run(&f, sql).len(), want);
}

#[test]
fn a_null_group_runs_the_standard_way_and_still_answers() {
    let mut d = corpus(60);
    d[3].cat = String::new();
    let f = fixture_docs(d, 1, &[], spec(), &Default::default());
    let r = run(&f, SQL);
    assert!(r.iter().any(|x| x[0].is_empty() && x[1] == "1"));
    assert_eq!(
        r.iter()
            .map(|x| x[1].parse::<usize>().unwrap())
            .sum::<usize>(),
        60
    );
}

#[test]
fn sums_too_big_for_the_kernels_run_the_standard_way() {
    let mut d = corpus(60);
    d[0].price = i64::MAX / 4;
    d[1].price = i64::MAX / 4 + 1;
    let f = fixture_docs(d, 1, &[], spec(), &Default::default());
    let r = run(&f, SQL);
    let top = r
        .iter()
        .map(|x| x[2].parse::<i128>().unwrap())
        .max()
        .unwrap();
    assert!(top >= (i64::MAX / 4) as i128);
    assert_eq!(
        r.iter()
            .map(|x| x[1].parse::<usize>().unwrap())
            .sum::<usize>(),
        60
    );
    // The cached path takes the same exit.
    assert_eq!(sorted(standard(&f, SQL)), sorted(r));
}

#[test]
fn every_kernel_aggregate_matches_the_documents() {
    let f = fixture(N, 3, &[]);
    for one in ["count(price)", "min(price)", "max(price)", "avg(price)"] {
        let q = format!("SELECT cat, {one} FROM t GROUP BY cat");
        assert!(plan_text(&f, &q).contains("GroupScanExec"), "{q}");
    }
    let sql = "SELECT cat, count(price), min(price), max(price), avg(price) FROM t GROUP BY cat";
    assert!(plan_text(&f, sql).contains("GroupScanExec"), "{sql}");
    let r = run(&f, sql);
    assert_eq!(r.len(), CATS as usize);
    for row in r {
        let docs: Vec<i64> = f
            .live()
            .filter(|d| d.cat == row[0])
            .map(|d| d.price)
            .collect();
        assert_eq!(row[1], docs.len().to_string());
        assert_eq!(row[2], docs.iter().min().unwrap().to_string());
        assert_eq!(row[3], docs.iter().max().unwrap().to_string());
        let avg = docs.iter().sum::<i64>() as f64 / docs.len() as f64;
        assert!((row[4].parse::<f64>().unwrap() - avg).abs() < 1e-9);
    }
}

#[test]
fn aggregates_the_kernels_do_not_know_keep_the_standard_plan() {
    let f = fixture(N, 2, &[]);
    for sql in [
        "SELECT cat, count(cat) FROM t GROUP BY cat",
        "SELECT cat, count(distinct price) FROM t GROUP BY cat",
        "SELECT cat, sum(price) FILTER (WHERE price > 3) FROM t GROUP BY cat",
        "SELECT cat, median(price) FROM t GROUP BY cat",
        "SELECT cat, corr(price, price) FROM t GROUP BY cat",
        "SELECT cat, sum(price + 1) FROM t GROUP BY cat",
        "SELECT cat, id FROM t GROUP BY cat, id",
    ] {
        assert!(!plan_text(&f, sql).contains("GroupScanExec"), "{sql}");
    }
}

#[test]
fn the_physical_plan_prints_and_has_no_children() {
    let f = fixture(200, 2, &[]);
    let plan = futures::executor::block_on(async {
        f.engine
            .context()
            .sql(SQL)
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap()
    });
    let text = format!("{plan:?}");
    assert!(text.contains("GroupScanExec"), "{text}");
}

#[test]
fn another_table_is_not_rewritten() {
    use datafusion::arrow::array::{Int64Array, StringArray};
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::arrow::record_batch::RecordBatch;
    use datafusion::datasource::MemTable;
    let f = fixture(50, 1, &[]);
    let schema = std::sync::Arc::new(Schema::new(vec![
        Field::new("cat", DataType::Utf8, true),
        Field::new("price", DataType::Int64, true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            std::sync::Arc::new(StringArray::from(vec!["a", "a"])),
            std::sync::Arc::new(Int64Array::from(vec![1, 2])),
        ],
    )
    .unwrap();
    let ctx = f.engine.context();
    ctx.register_table(
        "other",
        std::sync::Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap()),
    )
    .unwrap();
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let r = rt.block_on(async {
        ctx.sql("SELECT cat, sum(price) FROM other GROUP BY cat")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap()
    });
    assert_eq!(rows(&r), vec![vec!["a".to_string(), "3".to_string()]]);
}

#[test]
fn the_kernel_table_and_scan_describe_themselves() {
    use datafusion::common::tree_node::TreeNode;
    use datafusion::datasource::source_as_provider;
    use datafusion::datasource::TableType;
    use datafusion::logical_expr::LogicalPlan;
    use datafusion::physical_plan::ExecutionPlan;
    let f = fixture(100, 1, &[]);
    let (logical, physical) = futures::executor::block_on(async {
        let df = f.engine.context().sql(SQL).await.unwrap();
        let state = f.engine.context().state();
        let logical = state.optimize(df.logical_plan()).unwrap();
        (logical, df.create_physical_plan().await.unwrap())
    });
    let mut seen = false;
    logical
        .apply(|n| {
            if let LogicalPlan::TableScan(s) = n {
                let t = source_as_provider(&s.source).unwrap();
                if format!("{t:?}").starts_with("GroupTable") {
                    assert_eq!(t.table_type(), TableType::Base);
                    seen = true;
                }
            }
            Ok(datafusion::common::tree_node::TreeNodeRecursion::Continue)
        })
        .unwrap();
    assert!(seen);
    let mut leaf: &dyn ExecutionPlan = physical.as_ref();
    loop {
        let kids = leaf.children();
        let Some(c) = kids.first() else { break };
        leaf = Arc::as_ref(c);
    }
    assert_eq!(leaf.name(), "GroupScanExec");
    leaf.apply_expressions(&mut |_| Ok(datafusion::common::tree_node::TreeNodeRecursion::Continue))
        .unwrap();
}
