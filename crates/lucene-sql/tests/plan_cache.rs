//! A cached plan gives the rows a freshly planned statement gives.

mod common;

use std::sync::Arc;

use common::*;
use lucene_sql::SqlConfig;

fn cached(f: &Fixture, sql: &str) -> Vec<Vec<String>> {
    rows(&f.engine.standard_blocking(sql).unwrap())
}

fn fresh(f: &Fixture, sql: &str) -> Vec<Vec<String>> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    rows(&rt.block_on(f.engine.uncached(sql)).unwrap())
}

#[test]
fn a_repeated_shape_is_planned_once_and_binds_each_literal() {
    let f = fixture(2000, 3, &[]);
    for round in 0..3 {
        for cat in ["c00", "c05", "c11", "zzz"] {
            let sql =
                format!("SELECT id, price FROM t WHERE cat = '{cat}' ORDER BY price, id LIMIT 4");
            assert_eq!(cached(&f, &sql), fresh(&f, &sql), "{sql} round {round}");
        }
    }
    let s = f.engine.plan_cache_stats().unwrap();
    assert_eq!(s.planned, 1);
    assert_eq!(s.hits, 11);
    assert_eq!(s.uncacheable, 0);
}

#[test]
fn a_cached_top_k_does_not_keep_the_last_runs_threshold() {
    // A common term first (a high threshold), then a rare one: a plan reused
    // without resetting its operators returned nothing for the second.
    let f = fixture(3000, 3, &[]);
    let q = |w: &str| {
        format!(
            "SELECT id, cat, _score FROM t WHERE match_text(body, '{w}') ORDER BY _score DESC LIMIT 10"
        )
    };
    for words in ["w1 w2", "w390", "w1", "w399 w398", "w2 w3 w4", "w350"] {
        let sql = q(words);
        let a = cached(&f, &sql);
        let b = fresh(&f, &sql);
        assert!(!b.is_empty(), "{words}");
        let scores = |r: &[Vec<String>]| r.iter().map(|x| x[2].clone()).collect::<Vec<_>>();
        assert_eq!(scores(&a), scores(&b), "{words}");
    }
    assert!(f.engine.plan_cache_stats().unwrap().hits >= 5);
}

#[test]
fn statements_whose_plan_depends_on_a_literal_are_not_cached() {
    let f = fixture(500, 2, &[]);
    for sql in [
        "SELECT id FROM t WHERE cat IN ('c01', 'c02') ORDER BY id LIMIT 5",
        "SELECT id FROM t WHERE lower(cat) = 'c01' ORDER BY id LIMIT 5",
        "SELECT id, 'tag' AS tag FROM t ORDER BY id LIMIT 2",
        "SELECT id FROM t WHERE cat = 'c01' OR cat = 'c02' ORDER BY id LIMIT 5",
        "SELECT id FROM t WHERE id LIKE 'doc-0001%' ORDER BY id LIMIT 5",
        "SELECT id FROM t WHERE 'a' = 'a' AND cat = 'c03' ORDER BY id LIMIT 3",
        "SELECT id FROM t WHERE price = 'not a number'",
    ] {
        for _ in 0..2 {
            let a = f.engine.standard_blocking(sql);
            let b = {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .build()
                    .unwrap();
                rt.block_on(f.engine.uncached(sql))
            };
            match (a, b) {
                (Ok(a), Ok(b)) => assert_eq!(rows(&a), rows(&b), "{sql}"),
                (Err(a), Err(b)) => assert_eq!(a.to_string(), b.to_string(), "{sql}"),
                _ => panic!("cached and fresh disagree on success: {sql}"),
            }
        }
    }
    let s = f.engine.plan_cache_stats().unwrap();
    assert!(s.uncacheable >= 6, "{s:?}");
}

#[test]
fn a_cached_plan_serves_concurrent_clients_with_their_own_literals() {
    let f = Arc::new(fixture(1500, 3, &[]));
    let expected: Vec<(String, Vec<Vec<String>>)> = (0..12)
        .map(|c| {
            let sql =
                format!("SELECT id, price FROM t WHERE cat = 'c{c:02}' ORDER BY price, id LIMIT 3");
            let r = fresh(&f, &sql);
            (sql, r)
        })
        .collect();
    let expected = Arc::new(expected);
    let threads: Vec<_> = (0..4)
        .map(|t| {
            let f = f.clone();
            let expected = expected.clone();
            std::thread::spawn(move || {
                for i in 0..60 {
                    let (sql, want) = &expected[(i * 5 + t) % expected.len()];
                    assert_eq!(&cached(&f, sql), want, "{sql}");
                }
            })
        })
        .collect();
    for t in threads {
        t.join().unwrap();
    }
}

#[test]
fn the_cache_is_bounded_and_can_be_turned_off() {
    let f = fixture(60, 1, &[]);
    for k in 1..1200 {
        let sql = format!("SELECT id FROM t WHERE cat = 'c01' ORDER BY id LIMIT {k}");
        let r = cached(&f, &sql);
        assert_eq!(r.len(), k.min(r.len()));
    }
    let s = f.engine.plan_cache_stats().unwrap();
    assert_eq!(s.planned, 1199);
    // An unterminated literal is the standard error, not a cached plan.
    assert!(f
        .engine
        .standard_blocking("SELECT id FROM t WHERE cat = 'c01")
        .is_err());

    let off = SqlConfig {
        plan_cache: false,
        ..SqlConfig::default()
    };
    let g = fixture_with(60, 1, &[], spec(), &off);
    assert!(g.engine.plan_cache_stats().is_none());
    assert_eq!(
        rows(
            &g.engine
                .standard_blocking("SELECT count(*) FROM t WHERE cat = 'c01'")
                .unwrap()
        )
        .len(),
        1
    );
}
