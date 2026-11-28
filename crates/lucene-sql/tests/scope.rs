//! A statement stops when the request it runs for is cancelled or past its
//! deadline, on the router, the kernels and DataFusion's worker threads, and
//! a group-by is bounded by the engine's memory limit.

mod common;

use std::time::Duration;

use common::*;
use lucene_sql::SqlConfig;
use lucene_util::memory::{self, Scope};

const STATEMENTS: &[&str] = &[
    "SELECT count(*) FROM t WHERE price > 10",
    "SELECT cat, count(*) FROM t GROUP BY cat",
    "SELECT id, price FROM t WHERE price > 10 ORDER BY id LIMIT 5",
    "SELECT id, price FROM t ORDER BY price DESC, id LIMIT 5",
];

#[test]
fn a_cancelled_request_stops_every_path() {
    let f = fixture_points(2000, 3, &[]);
    for sql in STATEMENTS {
        let scope = Scope::new("test", 0, None);
        let _entered = memory::enter(scope.clone());
        assert!(f.engine.query_blocking(sql).is_ok(), "{sql}");
        scope.cancel();
        let err = f.engine.query_blocking(sql).unwrap_err();
        let stopped = err.stopped().unwrap_or_else(|| panic!("{sql}: {err}"));
        assert_eq!(stopped.reason, memory::Reason::Cancelled, "{sql}");
    }
}

#[test]
fn a_statement_past_its_deadline_stops() {
    let f = fixture_points(2000, 3, &[]);
    for sql in STATEMENTS {
        let scope = Scope::new("test", 0, None);
        scope.set_deadline(Duration::from_millis(1));
        let _entered = memory::enter(scope);
        std::thread::sleep(Duration::from_millis(5));
        let err = f.engine.query_blocking(sql).unwrap_err();
        assert!(err.stopped().is_some(), "{sql}: {err}");
    }
}

#[test]
fn statements_outside_a_scope_are_not_affected() {
    let f = fixture_points(500, 2, &[]);
    for sql in STATEMENTS {
        assert!(f.engine.query_blocking(sql).is_ok(), "{sql}");
    }
}

#[test]
fn a_group_by_is_bounded_by_the_memory_limit() {
    let config = SqlConfig {
        memory_limit: 256 * 4,
        ..SqlConfig::default()
    };
    let f = fixture_with(1000, 2, &[], spec(), &config);
    let sql = "SELECT cat, count(*) FROM t GROUP BY cat";
    assert!(f.engine.query_blocking(sql).is_err());
    let few = "SELECT cat, count(*) FROM t WHERE cat = 'c01' GROUP BY cat";
    assert!(f.engine.query_blocking(few).is_ok());
}
