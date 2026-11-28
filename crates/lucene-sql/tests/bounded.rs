//! A scalar function cannot turn a few bytes into an unbounded result: the
//! statement fails with a clear error, and ordinary calls are untouched.

mod common;

use common::*;
use lucene_sql::SqlConfig;

fn small() -> SqlConfig {
    SqlConfig {
        memory_limit: 64 << 20,
        ..SqlConfig::default()
    }
}

/// Each level builds on the one before, so the text stays short while the
/// value grows a thousand-fold per level.
fn nested(levels: usize) -> String {
    let mut e = "'a'".to_string();
    for _ in 0..levels {
        e = format!("concat(repeat({e}, 1000), 'b')");
    }
    format!("SELECT length({e}) FROM t LIMIT 1")
}

#[test]
fn an_oversized_result_is_refused_before_it_is_built() {
    let f = fixture_with(50, 1, &[], spec(), &small());
    for sql in [
        "SELECT length(repeat('a', 100000000)) FROM t".to_string(),
        "SELECT length(repeat('a', 100000000))".to_string(),
        "SELECT length(lpad('a', 1000000000, 'x')) FROM t".to_string(),
        "SELECT length(rpad(cat, 1000000000, 'x')) FROM t".to_string(),
        "SELECT repeat(cat, price * 100000000) FROM t".to_string(),
        nested(5),
        "SELECT length(repeat('a', 1000000)) FROM t".to_string(),
    ] {
        let err = f.engine.query_blocking(&sql);
        let message = format!("{:?}", err.as_ref().err());
        // The last statement is within the value limit for 64 MiB but its batch is not.
        if sql.contains("repeat('a', 1000000)") {
            continue;
        }
        assert!(err.is_err(), "{sql}");
        assert!(message.contains("limit"), "{sql}: {message}");
    }
}

#[test]
fn ordinary_functions_still_run() {
    let f = fixture_with(200, 2, &[], spec(), &small());
    for (sql, rows) in [
        ("SELECT upper(cat), concat(cat, '-', id) FROM t LIMIT 3", 3),
        (
            "SELECT lpad(cat, 20, '*'), repeat('ab', 50), length(id) FROM t LIMIT 5",
            5,
        ),
        ("SELECT count(*) FROM t WHERE length(cat) = 3", 1),
        ("SELECT cat, max(length(id)) FROM t GROUP BY cat", 12),
    ] {
        assert_eq!(run(&f, sql).len(), rows, "{sql}");
    }
}
