//! The kernels answer a group-by only when the answer is what SQL gives.

mod common;

use common::*;
use lucene_sql::SqlConfig;

fn docs() -> Vec<Doc> {
    let mut d = corpus(200);
    for (i, doc) in d.iter_mut().enumerate() {
        if i % 10 == 0 {
            doc.cat = String::new();
        }
    }
    d
}

#[test]
fn documents_without_the_key_make_a_null_group_so_the_router_declines() {
    let f = fixture_docs(docs(), 2, &[], spec(), &SqlConfig::default());
    let sql = "SELECT cat, count(*) FROM t GROUP BY cat";
    assert!(f.engine.router().shape(sql).is_some());
    assert!(f
        .snapshot()
        .group_by(2, &[lucene_sql::group::AggFn::CountStar])
        .unwrap()
        .is_none());
    let r = run(&f, sql);
    assert_eq!(r.len(), CATS as usize + 1);
    assert!(r.iter().any(|row| row[0].is_empty() && row[1] == "20"));
}

#[test]
fn sums_the_kernels_cannot_hold_exactly_run_the_standard_way() {
    let mut d = corpus(60);
    d[0].price = i64::MAX / 4;
    d[1].price = i64::MAX / 4 + 1;
    let f = fixture_docs(d, 1, &[], spec(), &SqlConfig::default());
    let sql = "SELECT cat, sum(price), max(price) FROM t GROUP BY cat";
    assert!(f
        .snapshot()
        .group_by(2, &[lucene_sql::group::AggFn::Sum(1)])
        .unwrap()
        .is_none());
    let r = run(&f, sql);
    assert_eq!(sorted(r.clone()), sorted(standard(&f, sql)));
    let max = r
        .iter()
        .map(|x| x[2].parse::<i64>().unwrap())
        .max()
        .unwrap();
    assert_eq!(max, i64::MAX / 4 + 1);
}

#[test]
fn group_by_asks_for_the_right_columns() {
    let f = fixture(30, 1, &[]);
    use lucene_sql::group::AggFn;
    assert!(f.snapshot().group_by(1, &[AggFn::CountStar]).is_err());
    assert!(f.snapshot().group_by(2, &[AggFn::Sum(2)]).is_err());
    assert!(f.snapshot().group_by(99, &[]).is_err());
}

#[test]
fn averages_and_extremes_the_kernels_cannot_hold_run_the_standard_way() {
    use lucene_sql::group::AggFn;
    let mut d = corpus(60);
    d[0].price = i64::MAX / 4;
    let f = fixture_docs(d, 1, &[], spec(), &SqlConfig::default());
    for agg in [AggFn::Avg(1), AggFn::Min(1), AggFn::Max(1)] {
        // Exact only for groups whose values fit; the huge one makes it None.
        assert!(
            f.snapshot()
                .group_by(2, std::slice::from_ref(&agg))
                .unwrap()
                .is_none(),
            "{agg:?}"
        );
    }
    for sql in [
        "SELECT cat, avg(price) FROM t GROUP BY cat",
        "SELECT cat, min(price), max(price) FROM t GROUP BY cat",
    ] {
        assert_eq!(sorted(run(&f, sql)), sorted(standard(&f, sql)), "{sql}");
    }
}

#[test]
fn a_window_scan_has_a_null_score_and_search_columns_have_no_values() {
    use lucene_sql::columns::Columns;
    let f = fixture(10, 1, &[]);
    let r = run(
        &f,
        "SELECT id, price, cat, _score FROM t ORDER BY id LIMIT 2",
    );
    assert_eq!(r[0][3], "");
    let mut cols = Columns::new(f.snapshot());
    assert!(cols.for_window(&[3], 0, 0, 5, None).is_err());
    assert!(cols.for_window(&[99], 0, 0, 5, None).is_err());
    assert!(cols.for_docs(&[3], &[(0, None)]).is_err());
}
