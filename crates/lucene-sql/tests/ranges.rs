//! Comparisons of a long column with an integer reach the scan as one range
//! search; the count must be what a brute-force pass over the documents says,
//! and the plan must hold the range instead of a filter above the scan.

mod common;

use common::{fixture, fixture_points, run, standard, Doc};

type Keep = Box<dyn Fn(&Doc) -> bool>;

fn expect(docs: &[&Doc], f: impl Fn(&Doc) -> bool) -> Vec<Vec<String>> {
    vec![vec![docs.iter().filter(|d| f(d)).count().to_string()]]
}

#[test]
fn range_comparisons_count_what_a_pass_over_the_documents_counts() {
    let f = fixture(1500, 3, &[4, 77, 900]);
    let live: Vec<&Doc> = f.live().collect();
    let cases: Vec<(&str, Keep)> = vec![
        ("price > 900", Box::new(|d| d.price > 900)),
        ("price >= 900", Box::new(|d| d.price >= 900)),
        ("price < 5", Box::new(|d| d.price < 5)),
        ("price <= 5", Box::new(|d| d.price <= 5)),
        ("price = 500", Box::new(|d| d.price == 500)),
        ("800 < price", Box::new(|d| 800 < d.price)),
        (
            "price BETWEEN 100 AND 200",
            Box::new(|d| (100..=200).contains(&d.price)),
        ),
        (
            "price > 100 AND price < 200 AND cat = 'c03'",
            Box::new(|d| d.price > 100 && d.price < 200 && d.cat == "c03"),
        ),
        ("price > 9223372036854775807", Box::new(|_| false)),
        ("price < -9223372036854775808", Box::new(|_| false)),
        ("price >= -9223372036854775808", Box::new(|_| true)),
        ("price <= 9223372036854775807", Box::new(|_| true)),
    ];
    for (predicate, keep) in cases {
        let sql = format!("SELECT count(*) FROM t WHERE {predicate}");
        assert_eq!(run(&f, &sql), expect(&live, keep), "{predicate}");
    }
}

#[test]
fn a_range_with_a_match_scores_only_documents_in_the_range() {
    let f = fixture(1500, 3, &[]);
    let rows = run(
        &f,
        "SELECT price, _score FROM t WHERE match_text(body, 'w1 w2 w3') AND price > 500 \
         ORDER BY _score DESC LIMIT 20",
    );
    assert!(!rows.is_empty());
    assert!(rows.iter().all(|r| r[0].parse::<i64>().unwrap() > 500));
}

#[test]
fn the_range_is_in_the_scan_not_above_it() {
    let f = fixture(200, 2, &[]);
    let plan = f
        .engine
        .context()
        .sql("EXPLAIN SELECT count(*) FROM t WHERE price > 10")
        .await_plan();
    assert!(plan.contains("Range"), "{plan}");
    assert!(!plan.contains("FilterExec"), "{plan}");
}

trait AwaitPlan {
    fn await_plan(self) -> String;
}

impl<
        F: std::future::Future<Output = datafusion::error::Result<datafusion::prelude::DataFrame>>,
    > AwaitPlan for F
{
    fn await_plan(self) -> String {
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let batches = rt.block_on(async { self.await.unwrap().collect().await.unwrap() });
        common::rows(&batches)
            .into_iter()
            .map(|r| r.join(" "))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[test]
fn ranges_over_points_count_the_same_as_over_doc_values() {
    let with_points = fixture_points(1500, 3, &[4, 77, 900]);
    let values_only = fixture(1500, 3, &[4, 77, 900]);
    for predicate in [
        "price > 900",
        "price <= 5",
        "price = 500",
        "price BETWEEN 100 AND 200",
        "price > 100 AND price < 200 AND cat = 'c03'",
        "price > 9223372036854775807",
        "price >= -9223372036854775808",
    ] {
        let sql = format!("SELECT count(*) FROM t WHERE {predicate}");
        assert_eq!(
            run(&with_points, &sql),
            run(&values_only, &sql),
            "{predicate}"
        );
    }
}

#[test]
fn a_filtered_group_by_runs_on_the_kernels_and_agrees_with_the_standard_path() {
    for f in [
        fixture(1500, 3, &[4, 77, 900]),
        fixture_points(1500, 3, &[4, 77, 900]),
    ] {
        for filter in [
            "price > 900",
            "price BETWEEN 100 AND 200",
            "price = 500",
            "cat = 'c03' AND price < 300",
            "price > 5000",
        ] {
            let sql = format!(
                "SELECT cat, count(*), sum(price), min(price), max(price) FROM t \
                 WHERE {filter} GROUP BY cat ORDER BY cat"
            );
            assert_eq!(run(&f, &sql), standard(&f, &sql), "{filter}");
        }
        let plan = f
            .engine
            .context()
            .sql("EXPLAIN SELECT cat, count(*) FROM t WHERE price > 900 GROUP BY cat")
            .await_plan();
        assert!(plan.contains("GroupScanExec"), "{plan}");
        assert!(plan.contains("Range"), "{plan}");
    }
}
