//! The router answers counts, filtered group-bys and filtered searches; each
//! must return what DataFusion returns for the same statement, over indexes
//! with and without point trees and with deletions.

mod common;

use common::*;
use lucene_sql::router::Shape;

const N: usize = 2400;

fn shape(f: &Fixture, sql: &str) -> Option<Shape> {
    f.engine.router().shape(sql)
}

fn fixtures() -> Vec<Fixture> {
    let deleted: Vec<usize> = (0..N).step_by(11).collect();
    vec![
        fixture(N, 3, &deleted),
        fixture_points(N, 3, &deleted),
        fixture_points(N, 1, &[]),
    ]
}

const FILTERS: &[&str] = &[
    "price > 900",
    "price >= 900",
    "price < 5",
    "price <= 5",
    "price = 500",
    "price BETWEEN 100 AND 200",
    "price between 100 and 200 and cat = 'c03'",
    "price > 100 AND price < 200",
    "price > 200 AND price < 100",
    "price > -5",
    "price >= -9223372036854775808",
    "price > 9223372036854775807",
    "price < -9223372036854775808",
    "cat = 'c03'",
    "cat = 'nope'",
    "cat IN ('c01', 'c02', 'c07')",
    "cat IN ('c01', 'nope') AND price < 500",
    "(price > 100) AND (cat = 'c05' AND price <= 300)",
    "price = '77'",
];

#[test]
fn counts_agree_with_the_standard_path() {
    for f in fixtures() {
        let all = "SELECT count(*) FROM t";
        assert!(matches!(shape(&f, all), Some(Shape::Count { .. })));
        assert_eq!(run(&f, all), standard(&f, all));
        for filter in FILTERS {
            let sql = format!("SELECT count(*) FROM t WHERE {filter}");
            assert!(
                matches!(shape(&f, &sql), Some(Shape::Count { .. })),
                "{sql}"
            );
            assert_eq!(run(&f, &sql), standard(&f, &sql), "{sql}");
        }
    }
}

#[test]
fn counts_match_a_pass_over_the_documents() {
    let f = fixture_points(N, 3, &[4, 77, 900]);
    let live: Vec<&Doc> = f.live().collect();
    let n = |keep: &dyn Fn(&Doc) -> bool| live.iter().filter(|d| keep(d)).count().to_string();
    let cases: Vec<(&str, String)> = vec![
        (
            "price BETWEEN 100 AND 300",
            n(&|d| (100..=300).contains(&d.price)),
        ),
        (
            "price BETWEEN 100 AND 300 AND cat IN ('c01', 'c02')",
            n(&|d| (100..=300).contains(&d.price) && (d.cat == "c01" || d.cat == "c02")),
        ),
        ("price > 990", n(&|d| d.price > 990)),
    ];
    for (filter, want) in cases {
        let sql = format!("SELECT count(*) FROM t WHERE {filter}");
        assert_eq!(run(&f, &sql), vec![vec![want]], "{sql}");
    }
}

#[test]
fn an_alias_names_the_column() {
    let f = fixture_points(N, 2, &[]);
    for sql in [
        "SELECT count(*) AS n FROM t WHERE price > 10",
        "SELECT COUNT(*) AS \"N\" FROM t",
        "SELECT cat AS c, count(*) AS n, sum(price) AS s FROM t GROUP BY cat",
    ] {
        assert!(shape(&f, sql).is_some(), "{sql}");
        let fast = f.engine.query_blocking(sql).unwrap();
        let std = f.engine.standard_blocking(sql).unwrap();
        assert_eq!(fast[0].schema().fields(), std[0].schema().fields(), "{sql}");
        assert_eq!(rows(&fast), rows(&std), "{sql}");
    }
}

#[test]
fn filtered_group_bys_agree_with_the_standard_path() {
    for f in fixtures() {
        for filter in FILTERS.iter().filter(|x| !x.contains("cat")) {
            let sql = format!(
                "SELECT cat, count(*), sum(price), min(price), max(price), avg(price) FROM t \
                 WHERE {filter} GROUP BY cat"
            );
            assert!(
                matches!(shape(&f, &sql), Some(Shape::Group { .. })),
                "{sql}"
            );
            assert_eq!(sorted(run(&f, &sql)), sorted(standard(&f, &sql)), "{sql}");
        }
        let sql = "SELECT cat, count(*) FROM t WHERE cat IN ('c01', 'c02') GROUP BY cat";
        assert_eq!(sorted(run(&f, sql)), sorted(standard(&f, sql)));
    }
}

#[test]
fn filtered_searches_agree_with_the_standard_path() {
    for f in fixtures() {
        for filter in [
            "price > 500",
            "cat = 'c03'",
            "price BETWEEN 100 AND 700 AND cat IN ('c01', 'c03')",
        ] {
            let sql = format!(
                "SELECT id, price, _score FROM t WHERE match_text(body, 'w1 w2 w9') AND {filter} \
                 ORDER BY _score DESC LIMIT 15"
            );
            assert!(
                matches!(shape(&f, &sql), Some(Shape::Match { .. })),
                "{sql}"
            );
            let fast = run(&f, &sql);
            assert!(!fast.is_empty(), "{sql}");
            assert_eq!(sorted(fast), sorted(standard(&f, &sql)), "{sql}");

            let q = vec_text(&f.docs[5].vec);
            let sql = format!(
                "SELECT id, _score FROM t WHERE {filter} AND knn(vec, '{q}') \
                 ORDER BY _score DESC LIMIT 10"
            );
            assert!(matches!(shape(&f, &sql), Some(Shape::Knn { .. })), "{sql}");
            assert_eq!(sorted(run(&f, &sql)), sorted(standard(&f, &sql)), "{sql}");
        }
    }
}

#[test]
fn a_keyword_lookup_returns_every_document_with_the_value() {
    let f = fixture(N, 3, &[]);
    let sql = "SELECT id FROM t WHERE cat = 'c03'";
    assert!(shape(&f, sql).is_none());
    let rows = sorted(run(&f, sql));
    assert_eq!(rows, sorted(standard(&f, sql)));
    assert!(rows.len() > 1);
}

#[test]
fn what_is_not_a_filter_is_left_to_the_standard_path() {
    let f = fixture_points(N, 2, &[]);
    for sql in [
        "SELECT count(*) FROM t WHERE price > 1.5",
        "SELECT count(*) FROM t WHERE price > 5 LIMIT 1",
        "SELECT count(*) FROM t WHERE cat > 'c01'",
        "SELECT count(*), 'x' FROM t",
        "SELECT count(*) FROM t WHERE price > 5 AND body = 'w1'",
    ] {
        assert!(shape(&f, sql).is_none(), "{sql}");
    }
    // The statements the router declines still answer.
    for sql in [
        "SELECT count(*) FROM t WHERE price > 1.5",
        "SELECT count(*) FROM t WHERE cat > 'c05'",
    ] {
        assert_eq!(run(&f, sql), standard(&f, sql), "{sql}");
    }
}

#[test]
fn an_integer_that_is_not_an_operand_is_left_alone() {
    let f = fixture_points(N, 2, &[]);
    let sql = "SELECT id, _score FROM t WHERE match_text(body, 'w1') \
               ORDER BY _score DESC LIMIT 10";
    assert!(matches!(shape(&f, sql), Some(Shape::Match { k: 10, .. })));
    // A quoted number is a string literal: a key lookup of it.
    let by_key = "SELECT id FROM t WHERE id = '00012'";
    assert_eq!(run(&f, by_key), standard(&f, by_key));
    // An integer inside a quoted literal is part of it.
    let text = "SELECT count(*) FROM t WHERE cat = 'c > 5 and 3'";
    assert_eq!(run(&f, text), vec![vec!["0".to_string()]]);
}
