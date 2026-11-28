//! The router, the DataFusion path and the native API must agree.

mod common;

use common::*;
use lucene_sql::router::Shape;

const N: usize = 3000;

fn shape(f: &Fixture, sql: &str) -> Option<Shape> {
    f.engine.router().shape(sql)
}

#[test]
fn point_read_agrees_across_paths() {
    let f = fixture(N, 3, &[]);
    for i in [0usize, 1, 999, 1000, 2999] {
        let sql = format!("SELECT id, price FROM t WHERE id = '{}'", id_of(i));
        assert!(matches!(shape(&f, &sql), Some(Shape::Point { .. })));
        let fast = run(&f, &sql);
        assert_eq!(fast, standard(&f, &sql));
        assert_eq!(fast, vec![vec![id_of(i), f.docs[i].price.to_string()]]);
    }
    let missing = "SELECT id, price FROM t WHERE id = 'nope'";
    assert!(run(&f, missing).is_empty());
    assert!(standard(&f, missing).is_empty());
    // Column order, a keyword column and the keyword as the key all route.
    let sql = format!("SELECT price, cat, id FROM t WHERE id = '{}'", id_of(7));
    assert!(shape(&f, &sql).is_some());
    assert_eq!(run(&f, &sql), standard(&f, &sql));
}

fn scores_and_ids(rows: &[Vec<String>]) -> (Vec<String>, Vec<String>) {
    (
        rows.iter().map(|r| r[1].clone()).collect(),
        rows.iter().map(|r| r[0].clone()).collect(),
    )
}

#[test]
fn text_top_k_agrees_across_paths() {
    let f = fixture(N, 3, &[]);
    for q in ["w1 w2", "w5", "w17 w3 w250", "W2 w2", "w399"] {
        for k in [1, 10, 37] {
            let sql = format!(
                "SELECT id, _score FROM t WHERE match_text(body, '{q}') ORDER BY _score DESC LIMIT {k}"
            );
            assert!(
                matches!(shape(&f, &sql), Some(Shape::Match { .. })),
                "{sql}"
            );
            let fast = run(&f, &sql);
            let std = standard(&f, &sql);
            assert_eq!(fast.len(), k.min(std.len()));
            let (fs, fi) = scores_and_ids(&fast);
            let (ss, si) = scores_and_ids(&std);
            assert_eq!(fs, ss, "{sql}");
            // The same documents (the scan runs the same search).
            let (mut fi, mut si) = (fi, si);
            fi.sort();
            si.sort();
            assert_eq!(fi, si, "{sql}");
        }
    }
    let none =
        "SELECT id, _score FROM t WHERE match_text(body, 'zzz') ORDER BY _score DESC LIMIT 5";
    assert!(run(&f, none).is_empty());
    assert!(standard(&f, none).is_empty());
}

#[test]
fn top_k_is_pushed_into_the_scan() {
    let f = fixture(N, 3, &[]);
    let sql = "EXPLAIN SELECT id, _score FROM t WHERE match_text(body, 'w1') ORDER BY _score DESC LIMIT 10";
    let plan = format!("{:?}", standard(&f, sql));
    assert!(plan.contains("fetch=Some(10)"), "{plan}");
    let logical = plan.contains("TableScan") || plan.contains("SegScanExec");
    assert!(logical, "{plan}");
}

#[test]
fn knn_agrees_by_recall_and_is_exact_against_the_brute_force_best() {
    let f = fixture(N, 3, &[]);
    for qi in [3usize, 500, 1777] {
        let q = &f.docs[qi].vec;
        let k = 10;
        let sql = format!(
            "SELECT id, _score FROM t WHERE knn(vec, '{}') ORDER BY _score DESC LIMIT {k}",
            vec_text(q)
        );
        assert!(matches!(shape(&f, &sql), Some(Shape::Knn { .. })));
        let fast = run(&f, &sql);
        let std = standard(&f, &sql);
        let ids = |r: &[Vec<String>]| r.iter().map(|x| x[0].clone()).collect::<Vec<_>>();
        let overlap = ids(&fast).iter().filter(|i| ids(&std).contains(i)).count();
        assert!(overlap * 10 >= k * 9, "recall {overlap}/{k}");
        // The query is a document: it is its own nearest neighbour.
        assert_eq!(fast[0][0], id_of(qi));
        // Recall against the exact answer.
        let mut exact: Vec<(f32, &Doc)> = f
            .docs
            .iter()
            .map(|d| {
                (
                    d.vec
                        .iter()
                        .zip(q)
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum::<f32>(),
                    d,
                )
            })
            .collect();
        exact.sort_by(|a, b| a.0.total_cmp(&b.0));
        let truth: Vec<String> = exact.iter().take(k).map(|e| e.1.id.clone()).collect();
        let hit = ids(&fast).iter().filter(|i| truth.contains(i)).count();
        assert!(hit * 10 >= k * 8, "recall vs exact {hit}/{k}");
    }
}

#[test]
fn group_by_agrees_across_paths() {
    let f = fixture(N, 3, &[]);
    let sql = "SELECT cat, count(*), sum(price), min(price), max(price), avg(price), count(price) FROM t GROUP BY cat";
    assert!(matches!(shape(&f, sql), Some(Shape::Group { .. })));
    let fast = run(&f, sql);
    assert_eq!(fast.len(), CATS as usize);
    assert_eq!(sorted(fast.clone()), sorted(standard(&f, sql)));
    // Against the documents themselves.
    let total: i64 = fast.iter().map(|r| r[1].parse::<i64>().unwrap()).sum();
    assert_eq!(total, N as i64);
    let reordered = "SELECT sum(price), cat FROM t GROUP BY cat";
    assert!(shape(&f, reordered).is_some());
    assert_eq!(sorted(run(&f, reordered)), sorted(standard(&f, reordered)));
}

#[test]
fn deleted_documents_vanish_from_every_path() {
    let deleted: Vec<usize> = (0..N).step_by(7).collect();
    let f = fixture(N, 3, &deleted);
    assert_eq!(f.snapshot().num_docs() as usize, N - deleted.len());
    // Point read of a deleted and a live document.
    let gone = format!("SELECT id, price FROM t WHERE id = '{}'", id_of(7 * 3));
    assert!(run(&f, &gone).is_empty());
    assert!(standard(&f, &gone).is_empty());
    let live = format!("SELECT id, price FROM t WHERE id = '{}'", id_of(8));
    assert_eq!(run(&f, &live).len(), 1);
    // Text search never returns a deleted document.
    let sql = "SELECT id, _score FROM t WHERE match_text(body, 'w1 w2 w3') ORDER BY _score DESC LIMIT 200";
    for rows in [run(&f, sql), standard(&f, sql)] {
        assert!(!rows.is_empty());
        for r in rows {
            let i: usize = r[0][4..].parse().unwrap();
            assert!(!i.is_multiple_of(7), "deleted {}", r[0]);
        }
    }
    // A full scan and the group-by count live documents only.
    let count = standard(&f, "SELECT count(*) FROM t");
    assert_eq!(count, vec![vec![(N - deleted.len()).to_string()]]);
    let g = "SELECT cat, count(*), sum(price) FROM t GROUP BY cat";
    let fast = run(&f, g);
    assert_eq!(sorted(fast.clone()), sorted(standard(&f, g)));
    let total: usize = fast.iter().map(|r| r[1].parse::<usize>().unwrap()).sum();
    assert_eq!(total, N - deleted.len());
    // k-NN: a deleted document is not its own neighbour.
    let q = &f.docs[0].vec;
    let sql = format!(
        "SELECT id FROM t WHERE knn(vec, '{}') ORDER BY _score DESC LIMIT 5",
        vec_text(q)
    );
    assert!(shape(&f, &sql).is_some());
    for r in run(&f, &sql) {
        assert_ne!(r[0], id_of(0));
    }
}

#[test]
fn filtered_knn_filters_before_it_takes_the_nearest() {
    let f = fixture(N, 3, &[]);
    let q = &f.docs[11].vec;
    let k = 8;
    let sql = format!(
        "SELECT id, cat, _score FROM t WHERE knn(vec, '{}') AND cat = 'c03' ORDER BY _score DESC LIMIT {k}",
        vec_text(q)
    );
    assert!(matches!(shape(&f, &sql), Some(Shape::Knn { .. })));
    let got = run(&f, &sql);
    assert_eq!(got.len(), k);
    assert!(got.iter().all(|r| r[1] == "c03"));
    let mut exact: Vec<(f32, &Doc)> = f
        .docs
        .iter()
        .filter(|d| d.cat == "c03")
        .map(|d| {
            (
                d.vec
                    .iter()
                    .zip(q)
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f32>(),
                d,
            )
        })
        .collect();
    exact.sort_by(|a, b| a.0.total_cmp(&b.0));
    let truth: Vec<String> = exact.iter().take(k).map(|e| e.1.id.clone()).collect();
    let hit = got.iter().filter(|r| truth.contains(&r[0])).count();
    assert!(hit * 10 >= k * 8, "filtered recall {hit}/{k}");
}

#[test]
fn near_misses_fall_back_with_the_same_rows() {
    let f = fixture(N, 3, &[]);
    let id = id_of(42);
    let near = [
        format!("SELECT id AS key, price FROM t WHERE id = '{id}'"),
        "SELECT id, _score FROM t WHERE match_text(body, 'w1') ORDER BY _score LIMIT 5".to_string(),
        "SELECT id, _score FROM t WHERE match_text(body, 'w1') ORDER BY _score DESC".to_string(),
        "SELECT id, _score FROM t WHERE match_text(body, 'w1') AND cat > 'c01' ORDER BY _score DESC LIMIT 5"
            .to_string(),
        "SELECT cat, count(*) FROM t GROUP BY cat HAVING count(*) > 1".to_string(),
        "SELECT cat, count(*) FROM t GROUP BY cat ORDER BY cat".to_string(),
        "SELECT cat, sum(price + 1) FROM t GROUP BY cat".to_string(),
        "SELECT DISTINCT cat FROM t".to_string(),
    ];
    for sql in near {
        assert!(shape(&f, &sql).is_none(), "should not route: {sql}");
        let ordered = sql.contains("ORDER BY");
        let a = run(&f, &sql);
        let b = standard(&f, &sql);
        if ordered && sql.contains("_score") {
            // Scores in order; ties may differ.
            let s = |r: &[Vec<String>]| r.iter().map(|x| x[1].clone()).collect::<Vec<_>>();
            assert_eq!(s(&a), s(&b), "{sql}");
        } else {
            assert_eq!(sorted(a), sorted(b), "{sql}");
        }
    }
}

#[test]
fn filters_the_router_now_translates_agree_with_the_standard_path() {
    let f = fixture(N, 3, &[]);
    let id = id_of(42);
    for sql in [
        format!("SELECT id, price FROM t WHERE id = '{id}' AND price > 0"),
        format!("SELECT id, price FROM t WHERE id = '{id}' LIMIT 1"),
        format!("SELECT id, price FROM t WHERE id IN ('{id}')"),
        format!("SELECT id, price FROM t WHERE price = 3 AND id = '{id}'"),
        "SELECT cat, count(*) FROM t WHERE price <> 500 GROUP BY cat".to_string(),
    ] {
        assert!(shape(&f, &sql).is_some(), "should route: {sql}");
        assert_eq!(sorted(run(&f, &sql)), sorted(standard(&f, &sql)), "{sql}");
    }
}

#[test]
fn routed_shapes_ignore_spacing_case_and_extra_parentheses_in_the_tree_not_the_text() {
    let f = fixture(300, 2, &[]);
    for sql in [
        "select  ID ,Price from T where ID='doc-00005'",
        "SELECT id, price\nFROM t\nWHERE id = 'doc-00005'",
        "SELECT \"id\", price FROM t WHERE (id = 'doc-00005')",
    ] {
        let routed = shape(&f, sql).is_some();
        assert_eq!(run(&f, sql), standard(&f, sql), "{sql}");
        // Parentheses are a node of the tree the filters see through.
        assert!(routed, "{sql}");
    }
    // A literal that merely looks like SQL is data.
    let weird = "SELECT id, price FROM t WHERE id = 'x'' OR ''1''=''1'";
    assert!(shape(&f, weird).is_none());
    assert!(run(&f, weird).is_empty());
}

#[test]
fn only_queries_run() {
    let f = fixture(50, 1, &[]);
    for sql in [
        "INSERT INTO t VALUES ('a', 1, 'c', 'b', 'v', 0.0)",
        "DELETE FROM t WHERE id = 'doc-00001'",
        "CREATE TABLE x AS SELECT * FROM t",
        "DROP TABLE t",
        "SET datafusion.execution.batch_size = 1",
        "COPY t TO '/tmp/x.csv'",
    ] {
        assert!(f.engine.query_blocking(sql).is_err(), "{sql}");
    }
    // The table is still there.
    assert_eq!(
        run(&f, "SELECT count(*) FROM t"),
        vec![vec!["50".to_string()]]
    );
}

#[test]
fn general_sql_runs_on_the_table() {
    let f = fixture(N, 3, &[]);
    let r = run(
        &f,
        "SELECT count(*), min(price), max(price) FROM t WHERE cat = 'c01'",
    );
    let c1: Vec<&Doc> = f.live().filter(|d| d.cat == "c01").collect();
    assert_eq!(r[0][0], c1.len().to_string());
    assert_eq!(
        r[0][1],
        c1.iter().map(|d| d.price).min().unwrap().to_string()
    );
    // A window function and a self join.
    let r = run(
        &f,
        "SELECT id, row_number() OVER (ORDER BY price, id) FROM t ORDER BY price, id LIMIT 3",
    );
    assert_eq!(r.len(), 3);
    assert_eq!(r[0][1], "1");
    let r = run(
        &f,
        "SELECT a.id, b.price FROM t a JOIN t b ON a.id = b.id WHERE a.id = 'doc-00010'",
    );
    assert_eq!(r, vec![vec![id_of(10), f.docs[10].price.to_string()]]);
    // Several segments are scanned: every document is counted once.
    assert_eq!(run(&f, "SELECT count(*) FROM t")[0][0], N.to_string());
    // ORDER BY a column reads doc values in order.
    let r = run(
        &f,
        "SELECT id, price FROM t ORDER BY price DESC, id LIMIT 1",
    );
    let top = f.docs.iter().map(|d| d.price).max().unwrap();
    assert_eq!(r[0][1], top.to_string());
}

#[test]
fn search_columns_and_bad_predicates_say_what_is_wrong() {
    let f = fixture(50, 1, &[]);
    let err = |sql: &str| f.engine.query_blocking(sql).unwrap_err().to_string();
    assert!(
        err("SELECT * FROM t").contains("search column"),
        "{}",
        err("SELECT * FROM t")
    );
    assert!(err("SELECT body FROM t").contains("search column"));
    assert!(
        err("SELECT id FROM t WHERE match_text(body, 'a') AND knn(vec, '1,2,3,4,5,6,7,8')")
            .contains("one match_text or knn")
    );
    assert!(
        err("SELECT id FROM t WHERE knn(vec, '1,2') ORDER BY _score DESC LIMIT 3")
            .contains("dimensions")
    );
    assert!(
        err("SELECT id FROM t WHERE match_text(body, 'a') OR cat = 'c01'").contains("match_text")
    );
}
