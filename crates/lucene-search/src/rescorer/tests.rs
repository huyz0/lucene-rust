#![cfg(test)]
use super::*;

fn sd(doc: i32, score: f32) -> ShardScoreDoc {
    ShardScoreDoc::new(doc, score)
}

#[test]
fn ordering_is_scores_then_docs() {
    let got = select_sorted(vec![sd(3, 1.0), sd(1, 2.0), sd(2, 1.0), sd(4, f32::NAN)], 3);
    assert_eq!(got.iter().map(|h| h.doc).collect::<Vec<_>>(), vec![1, 2, 3]);
    assert_eq!(
        by_score_then_doc(&sd(1, f32::NAN), &sd(2, f32::NAN)),
        Ordering::Less
    );
    assert_eq!(
        by_score_then_doc(&sd(1, 0.0), &sd(2, f32::NAN)),
        Ordering::Less
    );
}

#[test]
fn query_rescore_modes_combine_as_opensearch() {
    use QueryRescoreMode::*;
    assert_eq!(Avg.combine(1.0, 2.0), 1.5);
    assert_eq!(Max.combine(1.0, 2.0), 2.0);
    assert_eq!(Min.combine(1.0, 2.0), 1.0);
    assert_eq!(Total.combine(1.0, 2.0), 3.0);
    assert_eq!(Multiply.combine(1.5, 2.0), 3.0);
    // Math.max/min: NaN wins, -0.0 is below 0.0.
    assert!(Max.combine(f32::NAN, 1.0).is_nan());
    assert!(Min.combine(0.0, -0.0).is_sign_negative());
}

#[test]
fn float_compare_is_javas() {
    assert_eq!(java_float_compare(-0.0, 0.0), Ordering::Less);
    assert_eq!(
        java_float_compare(f32::NAN, f32::INFINITY),
        Ordering::Greater
    );
    assert_eq!(java_float_compare(1.0, 1.0), Ordering::Equal);
}

#[test]
fn weight_combination_narrows_once() {
    let r = QueryRescorer::with_weight(BooleanQuery::new(), 2.0);
    let first = 0.1f32;
    let second = 0.3f32;
    assert_eq!(
        (r.combine)(first, true, second),
        (f64::from(first) + 2.0 * f64::from(second)) as f32
    );
    assert_eq!((r.combine)(first, false, 9.0), first);
}

#[test]
fn rescore_top_n_needs_n() {
    assert!(
        RescoreTopNQuery::new(BooleanQuery::new(), crate::values_source::constant(1.0), 0).is_err()
    );
    let mut m = MatchScores {
        matches: vec![(1, 2.0), (5, 3.0)],
        at: 0,
    };
    use crate::values_source::DoubleValues;
    assert!(!m.advance_exact(0).unwrap());
    assert!(m.advance_exact(5).unwrap());
    assert_eq!(m.double_value().unwrap(), 3.0);
    assert!(!m.advance_exact(9).unwrap());
    assert_eq!(m.double_value().unwrap(), 0.0);
}

/// A `RescoreTopNQuery` clause anywhere in a query -- under a boolean,
/// a dis-max, a boost, a constant-score wrapper -- is rewritten to its
/// `DocAndScoreQuery` by the searcher, and searches like the standalone
/// query; unrewritten, the tree refuses it. A values-source sort that
/// needs the searcher sorts once rewritten, and only over its reader.
#[test]
fn rescore_clauses_are_rewritten_wherever_they_sit() {
    use crate::directory_reader::DirectoryReader;
    use crate::query::{BoostQuery, Clause, ConstantScoreQuery, DisjunctionMaxQuery, TermQuery};
    let dir = lucene_store::FsDirectory::open(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/values_rescore_index"
    ));
    let reader = DirectoryReader::open(&dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let norms = vec![None; segments.len()];
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let ctx = ValuesContext::new(&searcher);
    let term = |w: &str| Clause::Term(TermQuery::new("body", w.as_bytes().to_vec()));
    let mut inner = BooleanQuery::new();
    inner.must.push(term("w0"));
    let rtn = RescoreTopNQuery::new(inner, crate::values_source::from_float_field("f"), 6).unwrap();
    assert_eq!(rtn, rtn.clone());
    assert!(format!("{rtn:?}").starts_with("RescoreTopNQuery:"));
    let alone = rtn.search(&ctx, 20).unwrap();
    assert!(!alone.score_docs.is_empty());

    let wrap = |c: Clause| {
        let mut q = BooleanQuery::new();
        q.must.push(c);
        q
    };
    let wrapped = [
        wrap(Clause::from(rtn.clone())),
        wrap(Clause::DisjunctionMax(Box::new(DisjunctionMaxQuery {
            disjuncts: vec![Clause::from(rtn.clone())],
            tie_breaker: 0.0,
        }))),
        wrap(Clause::Boost(Box::new(BoostQuery {
            inner: Box::new(Clause::from(rtn.clone())),
            boost: 1.0,
        }))),
        wrap(Clause::Boolean(Box::new(wrap(Clause::from(rtn.clone()))))),
    ];
    for q in &wrapped {
        assert!(has_rescore_clauses(q));
        let got = searcher.search(q, 20).unwrap();
        let docs = |t: &TopDocs| {
            t.score_docs
                .iter()
                .map(|h| (h.doc, h.score.to_bits()))
                .collect::<Vec<_>>()
        };
        assert_eq!(docs(&got), docs(&alone));
        assert_eq!(searcher.count(q).unwrap(), alone.total_hits.value);
    }
    let constant = wrap(Clause::ConstantScore(Box::new(ConstantScoreQuery {
        inner: Box::new(Clause::from(rtn.clone())),
        score: 2.0,
    })));
    assert_eq!(
        searcher.search(&constant, 20).unwrap().score_docs.len(),
        alone.score_docs.len()
    );
    let plain = wrap(term("w1"));
    assert!(!has_rescore_clauses(&plain));
    assert!(rewrite_rescore_clauses(&plain, &ctx).unwrap().is_none());
    // The tree itself refuses an unrewritten one.
    let err = crate::search_boolean_query_scored_segment(
        &segments[0],
        &wrapped[0],
        None,
        None,
        &mut crate::collector::TopDocsCollector::new(5),
    );
    assert!(err.is_err());

    // A query-backed sort: refused until rewritten; the rewritten key
    // is bound to this reader.
    let sort = vec![crate::values_source::double_sort_field(
        crate::values_source::from_query(wrap(term("w2"))),
        true,
        0.0,
    )];
    let readers = reader.segment_readers();
    let all = wrap(Clause::MatchAllDocs(crate::query::MatchAllDocsQuery::new(
        i32::MAX,
    )));
    let run = |s: &[crate::top_field::SortField]| {
        crate::top_field::search_sorted(&segments, readers, &all, &norms, s, 5, u64::MAX, None)
    };
    assert!(run(&sort).is_err());
    let rewritten = crate::top_field::rewrite_sort(&sort, &ctx).unwrap();
    assert_ne!(rewritten, sort);
    assert_eq!(
        crate::top_field::rewrite_sort(&rewritten, &ctx).unwrap(),
        rewritten
    );
    assert!(format!("{:?}", rewritten[0].rewritten).contains("RewrittenSource"));
    let top = run(&rewritten).unwrap();
    assert_eq!(top.hits.len(), 5);
    // Over one segment only: the other leaves are unknown to the key.
    let one = IndexSearcher::new(&segments[1..], &norms[1..]).unwrap();
    let other = crate::top_field::rewrite_sort(&sort, &ValuesContext::new(&one)).unwrap();
    assert!(run(&other).is_err());
}
