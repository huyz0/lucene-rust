//! `CheckHits`: asserts a query's hits, their explanations and their
//! matches against what the searcher computes.
//!
//! Java's checks run inside a collector (`ExplanationAsserter`,
//! `MatchesAsserter`); here the hits are collected first and checked after,
//! since a collector of this port is handed `(doc, score)` and cannot call
//! back into the searcher it is collecting for. What is checked, and in
//! which order, is Java's. A failed check panics with Java's message.
//! `wrapUnderlyingReader` is [`query_utils::with_wrapped_underlying_reader`].

use std::collections::BTreeSet;

use lucene_search::collector::{ScoreMode, ScoringCollector, TopDocsCollector};
use lucene_search::explain::Explanation;
use lucene_search::index_searcher::IndexSearcher;
use lucene_search::query::{BooleanQuery, Clause};
use lucene_search::test_support::{self, NO_MORE_DOCS};
use lucene_search::top_docs::{ShardScoreDoc, TopDocs};
use lucene_search::Result;
use lucene_util::java_random::JavaRandom;

use super::query_utils;
use crate::util::lucene_test_case;

/// A collector keeping every `(global doc, score)` it is handed, under a
/// given score mode.
pub(crate) struct AllHits {
    pub(crate) hits: Vec<(i32, f32)>,
    mode: ScoreMode,
}

impl AllHits {
    pub(crate) fn new(mode: ScoreMode) -> Self {
        Self {
            hits: Vec::new(),
            mode,
        }
    }
}

impl ScoringCollector for AllHits {
    fn collect(&mut self, doc: i32, score: f32) {
        self.hits.push((doc, score));
    }
    fn score_mode(&self) -> ScoreMode {
        self.mode
    }
}

/// `searcher.search(query, collector)` with a collector of `mode`: every
/// hit, global, in collection order.
pub(crate) fn all_hits(
    searcher: &IndexSearcher<'_, '_>,
    query: &BooleanQuery,
    mode: ScoreMode,
) -> Result<Vec<(i32, f32)>> {
    let mut c = AllHits::new(mode);
    searcher.search_collector(query, &mut c)?;
    Ok(c.hits)
}

/// `checkNoMatchExplanations`: every document below `maxDoc` not in
/// `results` explains as a non-match.
///
/// # Errors
/// Whatever explaining reports.
pub fn check_no_match_explanations(
    query: &BooleanQuery,
    searcher: &IndexSearcher<'_, '_>,
    results: &[i32],
) -> Result<()> {
    let ignore: BTreeSet<i32> = results.iter().copied().collect();
    for doc in 0..searcher.max_doc() {
        if ignore.contains(&doc) {
            continue;
        }
        let exp = searcher.explain(query, doc)?;
        assert!(
            !exp.matched,
            "Explanation of [[{query:?}]] for #{doc} doesn't indicate non-match: {exp}"
        );
    }
    Ok(())
}

/// `checkHitCollector`: [`query_utils::check_with_searcher`], then the set
/// of documents a collector sees is exactly `results`.
///
/// # Errors
/// Whatever searching reports.
pub fn check_hit_collector(
    random: &mut JavaRandom,
    query: &BooleanQuery,
    searcher: &IndexSearcher<'_, '_>,
    results: &[i32],
) -> Result<()> {
    query_utils::check_with_searcher(random, query, searcher)?;
    let correct: BTreeSet<i32> = results.iter().copied().collect();
    let actual: BTreeSet<i32> = all_hits(searcher, query, ScoreMode::CompleteNoScores)?
        .into_iter()
        .map(|(d, _)| d)
        .collect();
    assert_eq!(correct, actual, "Simple: {query:?}");
    for edge in -1..2 {
        let actual: BTreeSet<i32> =
            query_utils::with_wrapped_underlying_reader(searcher, edge, |s| {
                all_hits(s, query, ScoreMode::CompleteNoScores)
            })?
            .into_iter()
            .map(|(d, _)| d)
            .collect();
        assert_eq!(correct, actual, "Wrap Reader {edge}: {query:?}");
    }
    Ok(())
}

/// `checkHits`: the top `max(10, 2 * results.len())` hits are exactly
/// `results` (as a set), then [`query_utils::check_with_searcher_wrap`]
/// with `rarely(random)` as `wrap`.
///
/// # Errors
/// Whatever searching reports.
pub fn check_hits(
    random: &mut JavaRandom,
    query: &BooleanQuery,
    searcher: &IndexSearcher<'_, '_>,
    results: &[i32],
) -> Result<()> {
    let n = 10.max(results.len() * 2);
    let hits = searcher.search(query, n)?.score_docs;
    let correct: BTreeSet<i32> = results.iter().copied().collect();
    let actual: BTreeSet<i32> = hits.iter().map(|h| h.doc).collect();
    assert_eq!(correct, actual, "{query:?}");
    let wrap = lucene_test_case::rarely(random, false, lucene_test_case::default_multiplier(false));
    query_utils::check_with_searcher_wrap(random, query, searcher, wrap)
}

/// `checkDocIds`: `hits` are `results`, in order.
pub fn check_doc_ids(mes: &str, results: &[i32], hits: &[ShardScoreDoc]) {
    assert_eq!(hits.len(), results.len(), "{mes} nr of hits");
    for (i, (&r, h)) in results.iter().zip(hits).enumerate() {
        assert_eq!(r, h.doc, "{mes} doc nrs for hit {i}");
    }
}

/// `checkHitsQuery`: both hit lists are `results`, in order, with equal
/// scores.
pub fn check_hits_query(
    query: &BooleanQuery,
    hits1: &[ShardScoreDoc],
    hits2: &[ShardScoreDoc],
    results: &[i32],
) {
    check_doc_ids("hits1", results, hits1);
    check_doc_ids("hits2", results, hits2);
    check_equal(query, hits1, hits2);
}

/// `checkEqual`: the same documents in the same order, scores within
/// `1e-6`.
pub fn check_equal(query: &BooleanQuery, hits1: &[ShardScoreDoc], hits2: &[ShardScoreDoc]) {
    const SCORE_TOLERANCE: f32 = 1.0e-6;
    assert_eq!(
        hits1.len(),
        hits2.len(),
        "Unequal lengths: hits1={},hits2={}",
        hits1.len(),
        hits2.len()
    );
    for (i, (h1, h2)) in hits1.iter().zip(hits2).enumerate() {
        assert!(
            h1.doc == h2.doc,
            "Hit {i} docnumbers don't match\n{}for query:{query:?}",
            hits2str(hits1, hits2, 0, 0)
        );
        assert!(
            (h1.score - h2.score).abs() <= SCORE_TOLERANCE,
            // SPACES-OK: Java's column layout.
            "Hit {i}, doc nrs {} and {}\nunequal       : {}\n           and: {}\nfor query:{query:?}",
            h1.doc,
            h2.doc,
            h1.score,
            h2.score
        );
    }
}

/// `hits2str`: the two hit lists side by side, rows `start..end` (`end`
/// `0`: every row).
pub fn hits2str(
    hits1: &[ShardScoreDoc],
    hits2: &[ShardScoreDoc],
    start: usize,
    end: usize,
) -> String {
    use std::fmt::Write;
    let (len1, len2) = (hits1.len(), hits2.len());
    let end = if end == 0 { len1.max(len2) } else { end };
    let mut sb = format!("Hits length1={len1}\tlength2={len2}\n");
    for i in start..end {
        let _ = write!(sb, "hit={i}:");
        match hits1.get(i) {
            Some(h) => {
                let _ = write!(sb, " doc{}={} shardIndex={}", h.doc, h.score, h.shard_index);
            }
            None => sb.push_str("               "),
        }
        sb.push_str(",\t");
        if let Some(h) = hits2.get(i) {
            let _ = write!(sb, " doc{}={} shardIndex={}", h.doc, h.score, h.shard_index);
        }
        sb.push('\n');
    }
    sb
}

/// `topdocsString`.
pub fn topdocs_string(docs: &TopDocs, start: usize, end: usize) -> String {
    use std::fmt::Write;
    let mut sb = format!(
        "TopDocs totalHits={}{} hits top={}\n",
        docs.total_hits.value,
        match docs.total_hits.relation {
            lucene_search::collector::TotalHitsRelation::EqualTo => "",
            lucene_search::collector::TotalHitsRelation::GreaterThanOrEqualTo => "+",
        },
        docs.score_docs.len()
    );
    let end = if end == 0 {
        docs.score_docs.len()
    } else {
        end.min(docs.score_docs.len())
    };
    for (i, d) in docs.score_docs.iter().enumerate().take(end).skip(start) {
        let _ = writeln!(sb, "\t{i}) doc={}\tscore={}", d.doc, d.score);
    }
    sb
}

/// `checkExplanations(query, defaultField, searcher, deep)`: every hit's
/// explanation is a match whose value is the hit's score
/// ([`verify_explanation`]).
///
/// # Errors
/// Whatever searching or explaining reports.
pub fn check_explanations(
    query: &BooleanQuery,
    searcher: &IndexSearcher<'_, '_>,
    deep: bool,
) -> Result<()> {
    let d = format!("{query:?}");
    for (doc, score) in all_hits(searcher, query, ScoreMode::Complete)? {
        let exp = searcher.explain(query, doc)?;
        verify_explanation(&d, doc, score, deep, &exp);
        assert!(
            exp.matched,
            "Explanation of [[{d}]] for #{doc} does not indicate match: {exp}"
        );
    }
    Ok(())
}

/// `checkMatches`: every hit has non-null `Matches`; within a segment, the
/// document before a hit that does not follow the previous one has none.
///
/// # Errors
/// Whatever searching or reading matches reports.
pub fn check_matches(query: &BooleanQuery, searcher: &IndexSearcher<'_, '_>) -> Result<()> {
    let clause = Clause::Boolean(Box::new(query.clone()));
    let mut collected_once = false;
    let mut last: Option<(usize, i32)> = None;
    for (doc, _) in all_hits(searcher, query, ScoreMode::CompleteNoScores)? {
        let leaf = searcher.segment_of(doc).expect("a hit is in a segment");
        let base = searcher.segments()[leaf].doc_base;
        let local = doc - base;
        // `doSetNextReader`: `lastCheckedDoc = -1` on each new leaf.
        let last_checked = match last {
            Some((l, d)) if l == leaf => d,
            _ => -1,
        };
        assert!(
            lucene_search::matches::matches(searcher, &clause, doc)?.is_some(),
            "Unexpected null Matches object in doc{local} for query {query:?}"
        );
        if collected_once && last_checked != local - 1 {
            assert!(
                lucene_search::matches::matches(searcher, &clause, doc - 1)?.is_none(),
                "Unexpected non-null Matches object in non-matching doc{local} for query {query:?}"
            );
        }
        collected_once = true;
        last = Some((leaf, local));
    }
    Ok(())
}

/// `.*, computed as .* from:`, matched against the whole of `descr`.
fn computed_from(descr: &str) -> bool {
    descr
        .find(", computed as ")
        .is_some_and(|i| descr[i + ", computed as ".len()..].ends_with(" from:"))
}

/// JUnit's `assertEquals(double expected, double actual, double delta)`.
fn doubles_equal(expected: f64, actual: f64, delta: f64) -> bool {
    expected.total_cmp(&actual).is_eq() || (expected - actual).abs() <= delta
}

/// `verifyExplanation`: the explanation's value is `score`; with `deep`,
/// a node with details combines them as its description says ("sum of:",
/// "product of:", "max of:", "max plus x times others of:", "computed as
/// ... from:").
///
/// # Panics
/// When the explanation does not account for the score.
pub fn verify_explanation(q: &str, doc: i32, score: f32, deep: bool, expl: &Explanation) {
    let value = expl.value;
    assert!(
        doubles_equal(f64::from(score), f64::from(value), 0.0),
        "{q}: score(doc={doc})={score} != explanationScore={value} Explanation: {expl}"
    );
    if !deep {
        return;
    }
    let detail = &expl.details;
    if expl.description.ends_with("computed from:") {
        return;
    }
    let descr = expl.description.to_lowercase();
    if descr.starts_with("score based on ") && descr.contains("child docs in range") {
        assert!(!detail.is_empty(), "Child doc explanations are missing");
    }
    if detail.is_empty() || !expl.matched {
        return;
    }
    if detail.len() == 1 && !computed_from(&descr) {
        if !expl.description.ends_with("with freq of:")
            && (score >= 0.0 || !expl.description.ends_with("times others of:"))
        {
            verify_explanation(q, doc, score, deep, &detail[0]);
        }
        return;
    }
    let mut x = 0.0f32;
    let product_of = descr.ends_with("product of:");
    let sum_of = descr.ends_with("sum of:");
    let max_of = descr.ends_with("max of:");
    let computed_of = descr.find("computed as").is_some_and(|i| i > 0) && computed_from(&descr);
    let mut max_times_others = false;
    if !(product_of || sum_of || max_of || computed_of) {
        if let Some(k) = descr.find("max plus ") {
            let k1 = k + "max plus ".len();
            if let Some(off) = descr[k1..].find(' ') {
                let k2 = k1 + off;
                if let Ok(v) = descr[k1..k2].trim().parse::<f32>() {
                    x = v;
                    if descr[k2..].trim() == "times others of:" {
                        max_times_others = true;
                    }
                }
            }
        }
    }
    assert!(
        product_of || sum_of || max_of || computed_of || max_times_others,
        "{q}: multi valued explanation description=\"{descr}\" must be 'max of plus x times others', \
         'computed as x from:' or end with 'product of' or 'sum of:' or 'max of:' - {expl}"
    );
    let mut sum = 0.0f64;
    let mut product = 1.0f32;
    let mut max = f32::NEG_INFINITY;
    let mut max_error = 0.0f64;
    for d in detail {
        let dval = d.value;
        verify_explanation(q, doc, dval, deep, d);
        product *= dval;
        sum += f64::from(dval);
        max = java_max(max, dval);
        if sum_of {
            max_error += f64::from(ulp(dval) * 2.0);
        }
    }
    let combined = if product_of {
        product
    } else if sum_of {
        sum as f32
    } else if max_of {
        max
    } else if max_times_others {
        (f64::from(max) + f64::from(x) * (sum - f64::from(max))) as f32
    } else {
        value
    };
    assert!(
        doubles_equal(f64::from(combined), f64::from(value), max_error),
        "{q}: actual subDetails combined=={combined} != value={value} Explanation: {expl}"
    );
}

/// `Math.max(float, float)`: NaN if either is, `0.0` over `-0.0`.
fn java_max(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else if a == b {
        if a.is_sign_negative() {
            b
        } else {
            a
        }
    } else {
        a.max(b)
    }
}

/// `Math.ulp(float)`.
fn ulp(f: f32) -> f32 {
    if f.is_nan() {
        return f32::NAN;
    }
    if f.is_infinite() {
        return f32::INFINITY;
    }
    let a = f.abs();
    if a == f32::MAX {
        return 2f32.powi(104);
    }
    f32::from_bits(a.to_bits() + 1) - a
}

/// `checkTopScores`: the top 1 and top 10 are the same whether pruning is
/// allowed or not, then the scorers' exposed maximum scores and block
/// boundaries bound the scores they produce (`doCheckMaxScores`).
///
/// # Errors
/// Whatever searching reports.
pub fn check_top_scores(
    random: &mut JavaRandom,
    query: &BooleanQuery,
    searcher: &IndexSearcher<'_, '_>,
) -> Result<()> {
    do_check_top_scores(query, searcher, 1)?;
    do_check_top_scores(query, searcher, 10)?;
    do_check_max_scores(random, query, searcher)
}

fn top_docs(
    searcher: &IndexSearcher<'_, '_>,
    query: &BooleanQuery,
    n: usize,
    threshold: u64,
) -> Result<Vec<ShardScoreDoc>> {
    let mut c = TopDocsCollector::with_total_hits_threshold(n, threshold);
    searcher.search_collector(query, &mut c)?;
    Ok(c.top_docs()
        .iter()
        .map(|h| ShardScoreDoc::new(h.doc_id, h.score))
        .collect())
}

fn do_check_top_scores(
    query: &BooleanQuery,
    searcher: &IndexSearcher<'_, '_>,
    num_hits: usize,
) -> Result<()> {
    let complete = top_docs(searcher, query, num_hits, u64::MAX)?;
    let top_scores = top_docs(searcher, query, num_hits, 1)?;
    check_equal(query, &complete, &top_scores);
    Ok(())
}

fn do_check_max_scores(
    random: &mut JavaRandom,
    query: &BooleanQuery,
    searcher: &IndexSearcher<'_, '_>,
) -> Result<()> {
    let leaves = searcher.segments().len();
    // Boundaries and maximum scores while iterating every match.
    for leaf in 0..leaves {
        test_support::with_leaf(searcher, query, leaf, |ctx| {
            let s1 = ctx.scorer(ScoreMode::Complete)?;
            let s2 = ctx.scorer(ScoreMode::TopScores)?;
            let (mut s1, mut s2) = match (s1, s2) {
                (None, s2) => {
                    assert!(s2.is_none_or(|mut s| s.exact_next().unwrap() == NO_MORE_DOCS));
                    return Ok(());
                }
                (Some(mut s1), None) => {
                    assert_eq!(s1.exact_next()?, NO_MORE_DOCS);
                    return Ok(());
                }
                (Some(s1), Some(s2)) => (s1, s2),
            };
            let mut up_to = -1;
            let mut max_score = 0.0f32;
            let mut min_score = 0.0f32;
            loop {
                let doc2 = s2.next_doc()?;
                let mut doc1 = s1.next_doc()?;
                while doc1 < doc2 {
                    if !s1.two_phase() || s1.matches()? {
                        assert!(s1.score()? < min_score);
                    }
                    doc1 = s1.next_doc()?;
                }
                assert_eq!(doc1, doc2);
                if doc2 == NO_MORE_DOCS {
                    break;
                }
                if doc2 > up_to {
                    up_to = s2.advance_shallow(doc2)?;
                    assert!(up_to >= doc2);
                    max_score = s2.max_score(up_to)?;
                }
                if !s2.two_phase() || s2.matches()? {
                    assert!(!s1.two_phase() || s1.matches()?);
                    let score = s2.score()?;
                    let s1_score = s1.score()?;
                    assert!(doubles_equal(f64::from(s1_score), f64::from(score), 0.0));
                    assert!(score <= max_score, "{score} > {max_score} up to {up_to}");
                    if score >= min_score && random.next_int_bounded(10) == 0 {
                        min_score = score;
                        s2.set_min_competitive_score(min_score)?;
                    }
                }
            }
            Ok(())
        })?;
    }
    // Advancing.
    for leaf in 0..leaves {
        test_support::with_leaf(searcher, query, leaf, |ctx| {
            let s1 = ctx.scorer(ScoreMode::Complete)?;
            let s2 = ctx.scorer(ScoreMode::TopScores)?;
            let (mut s1, mut s2) = match (s1, s2) {
                (None, s2) => {
                    assert!(s2.is_none_or(|mut s| s.exact_next().unwrap() == NO_MORE_DOCS));
                    return Ok(());
                }
                (Some(mut s1), None) => {
                    assert_eq!(s1.exact_next()?, NO_MORE_DOCS);
                    return Ok(());
                }
                (Some(s1), Some(s2)) => (s1, s2),
            };
            let mut up_to = -1;
            let mut min_score = 0.0f32;
            let mut max_score = 0.0f32;
            loop {
                let mut doc2 = s2.doc_id();
                let advance;
                let target;
                if random.next_boolean() {
                    advance = false;
                    target = doc2 + 1;
                } else {
                    advance = true;
                    // Java's int arithmetic: before the first document
                    // (`-1`) the bound wraps to `MIN_VALUE`, and the target
                    // wraps round to `NO_MORE_DOCS`.
                    let delta =
                        (1 + random.next_int_bounded(512)).min(NO_MORE_DOCS.wrapping_sub(doc2));
                    target = s2.doc_id().wrapping_add(delta);
                }
                if target > up_to && random.next_boolean() {
                    let delta = random.next_int_bounded(512).min(NO_MORE_DOCS - target);
                    up_to = target + delta;
                    let m = s2.advance_shallow(target)?;
                    assert!(m >= target);
                    max_score = s2.max_score(up_to)?;
                }
                doc2 = if advance {
                    s2.advance(target)?
                } else {
                    s2.next_doc()?
                };
                let mut doc1 = s1.advance(target)?;
                while doc1 < doc2 {
                    if !s1.two_phase() || s1.matches()? {
                        assert!(s1.score()? < min_score);
                    }
                    doc1 = s1.next_doc()?;
                }
                assert_eq!(doc1, doc2);
                if doc2 == NO_MORE_DOCS {
                    break;
                }
                if !s2.two_phase() || s2.matches()? {
                    assert!(!s1.two_phase() || s1.matches()?);
                    let score = s2.score()?;
                    let s1_score = s1.score()?;
                    assert!(doubles_equal(f64::from(s1_score), f64::from(score), 0.0));
                    if doc2 > up_to {
                        up_to = s2.advance_shallow(doc2)?;
                        assert!(up_to >= doc2);
                        max_score = s2.max_score(up_to)?;
                    }
                    assert!(score <= max_score);
                    if score >= min_score && random.next_int_bounded(10) == 0 {
                        min_score = score;
                        s2.set_min_competitive_score(min_score)?;
                    }
                }
            }
            Ok(())
        })?;
    }
    Ok(())
}
