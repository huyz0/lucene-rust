//! `lang: "js"` scripts through the exported entry points.

use crate::engine_tests::nodes::tests::{open, Blob};
use crate::jvm_reader::ffi_close_jvm_reader;
use crate::jvm_reader::tests::{run_limit, run_sorted};
use lucene_engine::painless_score::*;
use lucene_util::test_support::TempDir;

/// The Painless fixture's index (`ScoreCorpus.java`): 300 documents in
/// two segments, document `i` the `i`-th.
fn index() -> (TempDir, u64) {
    let tmp = TempDir::new("js-score");
    let dir = format!(
        "{}/../../fixtures/data/painless-score/index",
        env!("CARGO_MANIFEST_DIR")
    );
    for e in std::fs::read_dir(dir).unwrap() {
        let e = e.unwrap();
        std::fs::copy(e.path(), tmp.path().join(e.file_name())).unwrap();
    }
    let h = open(&tmp);
    (tmp, h)
}

/// `{name: value}` as the plugin sends params.
fn params(entries: &[(&str, f64)]) -> Blob {
    let mut b = Blob::default().u8(12).i32(entries.len() as i32);
    for (k, v) in entries {
        b = b.u8(9).str(k).u8(8).f64(*v);
    }
    b
}

/// A JavaScript spec of `context` reading `fields` (name, kind).
fn spec(context: u8, source: &str, params: &Blob, fields: &[(&str, u8)]) -> Blob {
    let mut b = Blob::default()
        .u8(context | JS_SCRIPT)
        .str(source)
        .bytes(&params.0)
        .str("idx")
        .i32(0)
        .i32(7)
        .i32(fields.len() as i32);
    for (name, kind) in fields {
        b = b.str(name).str(name).u8(*kind);
    }
    b.i32(10_000).i32(64)
}

fn likes(i: i32) -> f64 {
    f64::from((i * 7919) % 1000)
}

#[test]
fn a_score_script_scores_every_document() {
    let (_tmp, h) = index();
    let s = spec(
        1,
        "doc['likes'].value * params.w + (doc['flag'].value ? 1 : 0) + _score",
        &params(&[("w", 0.5)]),
        &[("likes", 0), ("flag", 4)],
    );
    let node = Blob::tree().u8(NODE_PAINLESS_SCORE).u8(5).raw(&s);
    let (hits, total, _) = run_limit(h, &node.0, 1000, i64::MAX).unwrap();
    assert_eq!(total, 300);
    assert_eq!(hits.len(), 300);
    for (doc, score) in hits {
        let flag = if doc % 3 == 0 { 1.0 } else { 0.0 };
        // match_all scores 1.
        let want = (likes(doc) * 0.5 + flag + 1.0) as f32;
        assert_eq!(score, want, "doc {doc}");
    }
    assert_eq!(check(&s.0[..]).unwrap(), None);
    assert_eq!(ffi_close_jvm_reader(h), 0);
}

#[test]
fn a_filter_script_matches_where_it_returns_true() {
    let (_tmp, h) = index();
    let s = spec(
        2,
        "doc['tag'].values.includes('tag1') && doc['vals'].size() === 3",
        &params(&[]),
        &[("tag", 5), ("vals", 0)],
    );
    let node = Blob::tree().u8(NODE_PAINLESS_FILTER).raw(&s);
    let (hits, total, _) = run_limit(h, &node.0, 1000, i64::MAX).unwrap();
    let want: Vec<i32> = (0..300)
        .filter(|i| i % 5 == 1 || (i % 4 == 0 && (i + 2) % 5 == 1))
        .collect();
    let mut got: Vec<i32> = hits.iter().map(|(d, _)| *d).collect();
    got.sort_unstable();
    assert_eq!(got, want);
    assert_eq!(total, want.len() as i64);
    // Counting runs the same script.
    let (_, counted, _) = run_limit(h, &node.0, 0, i64::MAX).unwrap();
    assert_eq!(counted, total);
    assert_eq!(ffi_close_jvm_reader(h), 0);
}

#[test]
fn a_sort_script_orders_by_its_numbers() {
    let (_tmp, h) = index();
    let s = spec(3, "-doc['likes'].value", &params(&[]), &[("likes", 0)]);
    let mut sort = vec![1, crate::jvm_reader::SORT_SCRIPT, 0];
    sort.extend_from_slice(&s.0);
    sort.extend_from_slice(&[0, 0, 0, 0, 0, 0]);
    let (hits, total, _) = run_sorted(h, &Blob::tree().u8(5).0, &sort, 5, i64::MAX).unwrap();
    assert_eq!(total, 300);
    let mut want: Vec<(i32, f64)> = (0..300).map(|i| (i, -likes(i))).collect();
    want.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    let got: Vec<(i32, i64)> = hits.iter().map(|(d, v)| (*d, v[0])).collect();
    let want: Vec<(i32, i64)> = want[..5]
        .iter()
        .map(|(d, v)| (*d, lucene_util::numeric_utils::double_to_sortable_long(*v)))
        .collect();
    assert_eq!(got, want);
    assert_eq!(ffi_close_jvm_reader(h), 0);
}

#[test]
fn a_script_error_is_the_scripts() {
    let (_tmp, h) = index();
    // Every seventh document has no price.
    let s = spec(0, "doc['price'].value", &params(&[]), &[("price", 1)]);
    let node = Blob::tree().u8(NODE_PAINLESS_SCORE).u8(5).raw(&s);
    assert!(run_limit(h, &node.0, 10, i64::MAX).is_err());
    let e = crate::error::last_error();
    assert!(
        e.contains("\u{1}js-script-error\u{1}7\u{1}1\u{1}13\u{1}Error\u{1}A document doesn't have a value for a field!"),
        "{e:?}"
    );
    // A script's message cannot pass for the error's own separators.
    let s = spec(
        0,
        "throw new Error('a\\u0001b\\u0001painless-score-error\\u0001')",
        &params(&[]),
        &[],
    );
    let node = Blob::tree().u8(NODE_PAINLESS_SCORE).u8(5).raw(&s);
    assert!(run_limit(h, &node.0, 10, i64::MAX).is_err());
    let e = crate::error::last_error();
    assert!(
        e.contains("\u{1}1\u{1}0\u{1}Error\u{1}a\u{fffd}b\u{fffd}painless-score-error\u{fffd}"),
        "{e:?}"
    );
    assert!(!e.contains(SCORE_ERROR), "{e:?}");
    // A negative score fails as `ScriptScorer` fails it.
    let s = spec(0, "doc['likes'].value - 500", &params(&[]), &[("likes", 0)]);
    let node = Blob::tree().u8(NODE_PAINLESS_SCORE).u8(5).raw(&s);
    assert!(run_limit(h, &node.0, 10, i64::MAX).is_err());
    let e = crate::error::last_error();
    assert!(
        e.contains(SCORE_ERROR) && e.contains("Must be a non-negative score!"),
        "{e:?}"
    );
    // A score that is not a number.
    let s = spec(0, "'x'", &params(&[]), &[]);
    let node = Blob::tree().u8(NODE_PAINLESS_SCORE).u8(5).raw(&s);
    assert!(run_limit(h, &node.0, 10, i64::MAX).is_err());
    let e = crate::error::last_error();
    assert!(
        e.contains("TypeError\u{1}a score script must return a number, not string"),
        "{e:?}"
    );
    // A negative score on a document fails before a later document's
    // exception, as `ScriptScorer` checks each score as it goes.
    let s = spec(
        0,
        "if (doc['likes'].value === 838) throw new Error('later'); return doc['likes'].value === 919 ? -1 : 1",
        &params(&[]),
        &[("likes", 0)],
    );
    let node = Blob::tree().u8(NODE_PAINLESS_SCORE).u8(5).raw(&s);
    assert!(run_limit(h, &node.0, 10, i64::MAX).is_err());
    let e = crate::error::last_error();
    assert!(
        e.contains(SCORE_ERROR) && e.contains("for doc [1]"),
        "{e:?}"
    );
    assert_eq!(ffi_close_jvm_reader(h), 0);
}

#[test]
fn the_check_refuses_what_v8_or_the_context_cannot_run() {
    let s = spec(0, "return (", &params(&[]), &[]);
    let why = check(&s.0).unwrap().unwrap();
    assert!(why.starts_with("compile error: SyntaxError"), "{why}");
    let s = spec(7, "true", &params(&[]), &[]);
    let why = check(&s.0).unwrap().unwrap();
    assert!(why.contains("js scripts run in the score"), "{why}");
    let mut s = spec(0, "1", &params(&[]), &[]).0;
    // Limits must be positive.
    let n = s.len();
    s[n - 4..].copy_from_slice(&0i32.to_le_bytes());
    assert!(check(&s).is_err());
}
