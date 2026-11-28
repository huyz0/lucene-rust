//! Statements the router translates beyond the first hot shapes (filter
//! trees, ordered row selects, global aggregates, paged searches) must
//! return what DataFusion returns, over indexes with and without point
//! trees, with deletions, and with documents that have no `cat` or `price`.

mod common;

use common::*;
use lucene_sql::router::Shape;

const N: usize = 1800;

/// Every seventh document has no `cat`, every ninth no `price`.
fn docs() -> Vec<Doc> {
    corpus(N)
        .into_iter()
        .enumerate()
        .map(|(i, mut d)| {
            if i % 7 == 3 {
                d.cat = String::new();
            }
            if i % 9 == 4 {
                d.price = NO_PRICE;
            }
            d
        })
        .collect()
}

fn fixtures() -> Vec<Fixture> {
    let deleted: Vec<usize> = (0..N).step_by(13).collect();
    vec![
        fixture_docs_points(docs(), 3, &deleted, false),
        fixture_docs_points(docs(), 3, &deleted, true),
        fixture_docs_points(docs(), 1, &[], true),
    ]
}

fn shape(f: &Fixture, sql: &str) -> Option<Shape> {
    f.engine.router().shape(sql)
}

/// An unordered page is any `limit` of the matching rows: the right number of
/// them, each one a match.
fn assert_page_of_matches(f: &Fixture, sql: &str, all: &str, limit: usize) {
    let page = run(f, sql);
    let every = standard(f, all);
    assert_eq!(page.len(), limit.min(every.len()), "{sql}");
    for row in &page {
        assert!(every.contains(row), "{sql}: {row:?}");
    }
}

/// The scores of a search's hits: hits with equal scores may tie in any order,
/// so two answers agree when their scores do.
fn scores(r: Vec<Vec<String>>) -> Vec<String> {
    r.into_iter().map(|row| row[1].clone()).collect()
}

const FILTERS: &[&str] = &[
    "cat = 'c03' OR cat = 'c04'",
    "cat = 'c03' OR price < 20",
    "(cat = 'c03' AND price < 500) OR (cat = 'c04' AND price >= 500)",
    "cat = 'c03' OR cat = 'c04' OR price BETWEEN 100 AND 120",
    "price > 100 AND (cat = 'c01' OR cat = 'c02')",
    "NOT cat = 'c03'",
    "cat <> 'c03'",
    "cat <> 'c03' AND price <> 500",
    "price <> 500",
    "NOT price > 500",
    "NOT price BETWEEN 100 AND 900",
    "price NOT BETWEEN 100 AND 900",
    "cat NOT IN ('c01', 'c02')",
    "cat IN ('c01', 'c02') OR price = 7",
    "cat IS NULL",
    "cat IS NOT NULL",
    "price IS NULL",
    "price IS NOT NULL AND price < 30",
    "cat IS NULL OR price IS NULL",
    "NOT cat IS NULL",
    "cat LIKE 'c0%'",
    "cat LIKE 'c1%' AND price < 300",
    "cat LIKE 'c03'",
    "cat LIKE '%'",
    "cat LIKE 'zz%'",
    "cat NOT LIKE 'c0%'",
    "cat LIKE 'c0%' OR price > 990",
    "(cat = 'c03')",
    "price = 5 OR price = 6 OR price = 7",
];

#[test]
fn filter_trees_agree_with_the_standard_path() {
    for f in fixtures() {
        for filter in FILTERS {
            let count = format!("SELECT count(*) FROM t WHERE {filter}");
            assert!(
                matches!(shape(&f, &count), Some(Shape::Count { .. })),
                "{count}"
            );
            assert_eq!(run(&f, &count), standard(&f, &count), "{count}");

            let group =
                format!("SELECT cat, count(*), sum(price) FROM t WHERE {filter} GROUP BY cat");
            assert!(
                matches!(shape(&f, &group), Some(Shape::Group { .. })),
                "{group}"
            );
            // A group-by with a null group is the standard path's.
            assert_eq!(
                sorted(run(&f, &group)),
                sorted(standard(&f, &group)),
                "{group}"
            );

            let rows = format!("SELECT price FROM t WHERE {filter} ORDER BY price DESC LIMIT 25");
            assert!(
                matches!(shape(&f, &rows), Some(Shape::Rows { .. })),
                "{rows}"
            );
            assert_eq!(run(&f, &rows), standard(&f, &rows), "{rows}");

            let any = format!("SELECT id, price, cat FROM t WHERE {filter} LIMIT 20");
            assert!(matches!(shape(&f, &any), Some(Shape::Rows { .. })), "{any}");
            assert_page_of_matches(
                &f,
                &any,
                &format!("SELECT id, price, cat FROM t WHERE {filter}"),
                20,
            );

            let global = format!("SELECT count(*), sum(price), min(price), max(price), avg(price), count(price) FROM t WHERE {filter}");
            assert!(
                matches!(shape(&f, &global), Some(Shape::Global { .. })),
                "{global}"
            );
            assert_eq!(run(&f, &global), standard(&f, &global), "{global}");
        }
    }
}

#[test]
fn filter_trees_in_searches() {
    for f in fixtures() {
        for filter in [
            "cat = 'c03' OR cat = 'c04'",
            "cat IS NOT NULL AND price <> 5",
            "cat LIKE 'c0%'",
        ] {
            let sql = format!(
                "SELECT id, _score FROM t WHERE match_text(body, 'w1 w2 w9') AND ({filter}) \
                 ORDER BY _score DESC LIMIT 15"
            );
            assert!(
                matches!(shape(&f, &sql), Some(Shape::Match { .. })),
                "{sql}"
            );
            let fast = run(&f, &sql);
            assert!(!fast.is_empty(), "{sql}");
            assert_eq!(scores(fast), scores(standard(&f, &sql)), "{sql}");
        }
    }
}

#[test]
fn row_selects_are_ordered_like_the_standard_path() {
    for f in fixtures() {
        for (order, columns) in [
            ("price", "price"),
            ("price ASC", "price"),
            ("price DESC", "price"),
            ("price ASC NULLS FIRST", "price"),
            ("price DESC NULLS LAST", "price"),
            ("price DESC NULLS FIRST", "price"),
            ("cat", "cat"),
            ("cat DESC", "cat"),
            ("cat ASC NULLS FIRST", "cat"),
            ("cat DESC NULLS LAST", "cat"),
            ("cat, price DESC", "cat, price"),
            ("price DESC, cat", "price, cat"),
            ("cat DESC NULLS LAST, price NULLS FIRST", "cat, price"),
        ] {
            for page in [
                "LIMIT 10",
                "LIMIT 10 OFFSET 20",
                "LIMIT 5 OFFSET 0",
                "LIMIT 0",
                "LIMIT 3 OFFSET 5000",
            ] {
                for filter in ["", "WHERE price > 300", "WHERE cat = 'c03' OR cat IS NULL"] {
                    let sql = format!("SELECT {columns} FROM t {filter} ORDER BY {order} {page}");
                    assert!(matches!(shape(&f, &sql), Some(Shape::Rows { .. })), "{sql}");
                    assert_eq!(run(&f, &sql), standard(&f, &sql), "{sql}");
                }
            }
        }
    }
}

#[test]
fn an_unordered_page_is_some_matching_documents() {
    for f in fixtures() {
        for (sql, all, limit) in [
            (
                "SELECT id, price FROM t LIMIT 7",
                "SELECT id, price FROM t",
                7,
            ),
            (
                "SELECT id FROM t WHERE price > 900 LIMIT 12",
                "SELECT id FROM t WHERE price > 900",
                12,
            ),
            (
                "SELECT cat FROM t WHERE cat = 'c01' LIMIT 4",
                "SELECT cat FROM t WHERE cat = 'c01'",
                4,
            ),
        ] {
            assert!(matches!(shape(&f, sql), Some(Shape::Rows { .. })), "{sql}");
            assert_page_of_matches(&f, sql, all, limit);
        }
    }
}

#[test]
fn key_lookups_return_every_listed_document() {
    for f in fixtures() {
        let sql = format!(
            "SELECT id, price FROM t WHERE id IN ('{}', '{}', '{}', 'nope')",
            id_of(3),
            id_of(700),
            id_of(13)
        );
        assert!(matches!(shape(&f, &sql), Some(Shape::Rows { .. })), "{sql}");
        assert_eq!(sorted(run(&f, &sql)), sorted(standard(&f, &sql)), "{sql}");
    }
}

#[test]
fn global_aggregates_agree_with_the_standard_path() {
    for f in fixtures() {
        for sql in [
            "SELECT sum(price), avg(price), min(price), max(price) FROM t",
            "SELECT count(price), count(*) FROM t",
            "SELECT sum(price) AS s, max(price) AS m FROM t WHERE price > 100",
            "SELECT avg(price) FROM t WHERE cat = 'c03'",
            "SELECT min(price), max(price) FROM t WHERE price > 5000",
            "SELECT sum(price), count(*) FROM t WHERE cat = 'nope'",
            "SELECT count(*), max(price) FROM t WHERE price BETWEEN 10 AND 20",
        ] {
            assert!(
                matches!(shape(&f, sql), Some(Shape::Global { .. })),
                "{sql}"
            );
            let fast = f.engine.query_blocking(sql).unwrap();
            let std = f.engine.standard_blocking(sql).unwrap();
            assert_eq!(fast[0].schema().fields(), std[0].schema().fields(), "{sql}");
            assert_eq!(rows(&fast), rows(&std), "{sql}");
        }
    }
}

#[test]
fn a_search_page_skips_the_offset() {
    for f in fixtures() {
        let sql = "SELECT id, _score FROM t WHERE match_text(body, 'w1 w2') \
                   ORDER BY _score DESC LIMIT 10 OFFSET 10";
        assert!(matches!(
            shape(&f, sql),
            Some(Shape::Match { offset: 10, .. })
        ));
        let fast = run(&f, sql);
        assert_eq!(fast.len(), 10);
        assert_eq!(scores(fast), scores(standard(&f, sql)));
    }
}

#[test]
fn what_stays_with_the_standard_path_still_answers() {
    let f = fixture_docs_points(docs(), 2, &[], true);
    for sql in [
        // Negating more than one condition, a pattern that is not a prefix,
        // keys the router cannot tell from missing, expressions, deep pages.
        "SELECT count(*) FROM t WHERE NOT (cat = 'c01' OR price > 5)",
        "SELECT count(*) FROM t WHERE cat LIKE '%1'",
        "SELECT count(*) FROM t WHERE cat LIKE 'c_1'",
        "SELECT count(*) FROM t WHERE id <> 'doc-00000001'",
        "SELECT count(*) FROM t WHERE id IS NULL",
        "SELECT id FROM t ORDER BY price + 1 LIMIT 3",
        "SELECT id FROM t ORDER BY price LIMIT 20000",
        "SELECT id FROM t ORDER BY price OFFSET 3",
        "SELECT id FROM t WHERE price > 5",
        "SELECT sum(price + 1) FROM t",
    ] {
        assert!(shape(&f, sql).is_none(), "{sql}");
        assert_eq!(run(&f, sql), standard(&f, sql), "{sql}");
    }
}
