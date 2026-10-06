use super::*;
use crate::extended_query::{MultiTermSource, RewriteMethod};
use crate::query::{PrefixQuery, SpanQuery};

fn t(w: &str) -> SpanNode {
    SpanNode::term("f", w)
}

/// Every span query's `toString`, as Lucene prints it.
#[test]
fn the_queries_print_as_java_does() {
    let near = SpanNode::near(vec![t("a"), t("b")], 2, true).unwrap();
    assert_eq!(near.to_string(), "spanNear([f:a, f:b], 2, true)");
    let or = SpanNode::or(vec![t("a"), near.clone()]).unwrap();
    assert_eq!(
        or.to_string(),
        "spanOr([f:a, spanNear([f:a, f:b], 2, true)])"
    );
    assert_eq!(SpanNode::first(t("a"), 3).to_string(), "spanFirst(f:a, 3)");
    assert_eq!(
        SpanNode::position_range(t("a"), 1, 4).to_string(),
        "spanPosRange(f:a, 1, 4)"
    );
    assert_eq!(
        SpanNode::not(t("a"), t("b"), -1, 2).unwrap().to_string(),
        "spanNot(f:a, f:b, -1, 2)"
    );
    assert_eq!(
        SpanNode::not_within(t("a"), t("b"), 3).unwrap().to_string(),
        "spanNot(f:a, f:b, 3, 3)"
    );
    assert_eq!(
        SpanNode::containing(near.clone(), t("c"))
            .unwrap()
            .to_string(),
        "SpanContaining(spanNear([f:a, f:b], 2, true), f:c)"
    );
    assert_eq!(
        SpanNode::within(near, t("c")).unwrap().to_string(),
        "SpanWithin(spanNear([f:a, f:b], 2, true), f:c)"
    );
    assert_eq!(
        SpanNode::field_masking(SpanNode::term("g", "x"), "f").to_string(),
        "mask(g:x) as f"
    );
    let prefix = SpanNode::multi_term(MultiTermQuery::new(
        MultiTermSource::Prefix(PrefixQuery::new("f", "ap")),
        RewriteMethod::default(),
    ));
    assert_eq!(prefix.to_string(), "SpanMultiTermQueryWrapper(f:ap*)");
    assert_eq!(term_to_string(b"abc"), "abc");
    assert_eq!(term_to_string(&[0x61, 0xff, 0x01]), "[61 ff 1]");
}

/// The constructors' field checks; a field-less clause (none here) and
/// negative distances are allowed, as Java's are.
#[test]
fn clauses_of_different_fields_are_refused() {
    let other = SpanNode::term("g", "a");
    assert!(SpanNode::near(vec![t("a"), other.clone()], 0, true).is_err());
    assert!(SpanNode::or(vec![t("a"), other.clone()]).is_err());
    assert!(SpanNode::not(t("a"), other.clone(), 0, 0).is_err());
    assert!(SpanNode::containing(t("a"), other.clone()).is_err());
    assert!(SpanNode::within(t("a"), other.clone()).is_err());
    assert!(SpanNode::near(Vec::new(), 0, true).is_ok());
    assert!(SpanNode::not(t("a"), t("b"), -5, -5).is_ok());
    // A masked clause reports the masking field.
    let masked = SpanNode::field_masking(other, "f");
    assert_eq!(masked.field(), Some("f"));
    assert!(SpanNode::near(vec![t("a"), masked], 0, true).is_ok());
    assert_eq!(
        SpanNode::Near {
            clauses: Vec::new(),
            slop: 0,
            in_order: true
        }
        .field(),
        None
    );
}

/// A multi-term wrapper anywhere in the tree asks for the searcher's
/// rewrite; a query of the core span queries converts as it is.
#[test]
fn rewrite_needs_and_conversions() {
    let prefix = SpanNode::multi_term(MultiTermQuery::new(
        MultiTermSource::Prefix(PrefixQuery::new("f", "ap")),
        RewriteMethod::default(),
    ));
    assert!(prefix.needs_rewrite());
    assert!(SpanNode::first(prefix.clone(), 2).needs_rewrite());
    assert!(SpanNode::not(t("a"), prefix.clone(), 0, 0)
        .unwrap()
        .needs_rewrite());
    let check = SpanNode::PayloadCheck(Box::new(payloads::SpanPayloadCheckQuery::new(
        prefix.clone(),
        Vec::new(),
    )));
    assert!(check.needs_rewrite());
    assert!(!SpanNode::near(vec![t("a"), t("b")], 1, false)
        .unwrap()
        .needs_rewrite());
    assert_eq!(check.field(), Some("f"));

    let core = SpanQuery::span_near(
        [
            SpanQuery::span_term("f", "a"),
            SpanQuery::span_or([SpanQuery::span_term("f", "b")]),
        ],
        3,
        false,
    );
    assert_eq!(
        SpanNode::from(&core),
        SpanNode::Near {
            clauses: vec![
                t("a"),
                SpanNode::Or {
                    clauses: vec![t("b")]
                }
            ],
            slop: 3,
            in_order: false,
        }
    );
    let clause = crate::query::Clause::from(t("a"));
    assert!(matches!(clause, crate::query::Clause::Extended(_)));
}

/// The statistics a weight takes: a not query's include side only; a
/// containment's both sides; a masked query's own; the statistics pass reads
/// the exclude side too.
#[test]
fn weight_terms_follow_extract_term_states() {
    let q = SpanNode::not(SpanNode::containing(t("a"), t("b")).unwrap(), t("c"), 0, 0).unwrap();
    let mut w = Vec::new();
    weight_terms(&q, &mut w);
    let names: Vec<&[u8]> = w.iter().map(|(_, t)| t.as_slice()).collect();
    assert_eq!(names, [b"a".as_slice(), b"b"]);
    let mut all = Vec::new();
    all_terms(&q, &mut all);
    assert_eq!(all.len(), 3);
    let masked = SpanNode::field_masking(SpanNode::term("g", "x"), "f");
    assert_eq!(weight_query(&masked), &SpanNode::term("g", "x"));
    let mut m = Vec::new();
    weight_terms(&masked, &mut m);
    assert_eq!(m, [("g".to_string(), b"x".to_vec())]);
}

fn fixture() -> crate::directory_reader::DirectoryReader {
    let dir = std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/spans/index"
    ));
    crate::directory_reader::DirectoryReader::open(&lucene_store::FsDirectory::open(dir)).unwrap()
}

fn leaf<'a>(seg: &crate::multi_segment::OpenSegment<'a>) -> LeafContext<'a> {
    LeafContext {
        fields: seg.fields,
        doc_in: seg.doc_in,
        pos_in: seg.pos_in,
        pay_in: seg.pay_in,
        live_docs: seg.live_docs,
        points: None,
        norms: None,
        global: None,
        max_doc: seg.max_doc,
        cache: None,
        reader: None,
        similarity: None,
    }
}

/// Every span of every document: `(doc, start, end, width)`.
fn walk(spans: &mut dyn Spans) -> Vec<(i32, i32, i32, i32)> {
    let mut out = Vec::new();
    while spans.next_doc().unwrap() != NO_MORE_DOCS {
        if !spans.matches().unwrap() {
            continue;
        }
        assert_eq!((spans.start_position(), spans.end_position()), (-1, -1));
        while spans.next_start_position().unwrap() != NO_MORE_POSITIONS {
            out.push((
                spans.doc_id(),
                spans.start_position(),
                spans.end_position(),
                spans.width(),
            ));
        }
        assert_eq!(spans.start_position(), NO_MORE_POSITIONS);
        assert_eq!(spans.end_position(), NO_MORE_POSITIONS);
    }
    out
}

/// Counts the leaves a spans collects.
#[derive(Default)]
struct Leaves(Vec<(String, i32, Option<Vec<u8>>)>);

impl SpanCollector for Leaves {
    fn collect_leaf(&mut self, leaf: &mut TermSpans<'_>, position: i32) -> Result<()> {
        let payload = leaf.payload()?.map(<[u8]>::to_vec);
        self.0.push((
            String::from_utf8_lossy(&leaf.term.1).into_owned(),
            position,
            payload,
        ));
        Ok(())
    }
    fn reset(&mut self) {
        self.0.clear();
    }
}

/// The spans' own contract, walked directly: positions `-1` before a
/// document's first span and `NO_MORE_POSITIONS` after its last, for the
/// spans a scorer only ever reads through its frequency (containment and
/// payload spans nested, payload spans at the top); collection; the
/// refusals.
#[test]
fn spans_walk_and_collect_by_their_contract() {
    let reader = fixture();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let body = |w: &str| SpanNode::term("body", w);
    // A containment's spans are its source's.
    let near = SpanNode::near(vec![body("apple"), body("cat")], 4, true).unwrap();
    let containing = SpanNode::containing(near.clone(), body("bank")).unwrap();
    // The segment the containment matches in.
    let ctx = segments
        .iter()
        .map(leaf)
        .find(|ctx| {
            spans_with(ctx, &containing, false)
                .unwrap()
                .is_some_and(|mut c| !walk(&mut c).is_empty())
        })
        .expect("a segment with a containing span");
    let mut c = spans_with(&ctx, &containing, false).unwrap().unwrap();
    let spans = walk(&mut c);
    let mut b = spans_with(&ctx, &near, false).unwrap().unwrap();
    let bigs = walk(&mut b);
    assert!(spans.iter().all(|s| bigs.contains(s)), "{spans:?} {bigs:?}");
    let mut w = spans_with(
        &ctx,
        &SpanNode::within(near.clone(), body("bank")).unwrap(),
        false,
    )
    .unwrap()
    .unwrap();
    let littles = walk(&mut w);
    assert!(littles.iter().all(|&(_, s, e, wd)| e == s + 1 && wd == 0));
    // Collected: the big spans' terms, then the little one's.
    let mut c = spans_with(&ctx, &containing, false).unwrap().unwrap();
    let mut leaves = Leaves::default();
    while c.next_doc().unwrap() != NO_MORE_DOCS {
        if c.matches().unwrap() {
            c.next_start_position().unwrap();
            c.collect(&mut leaves).unwrap();
            break;
        }
    }
    let names: Vec<&str> = leaves.0.iter().map(|(t, _, _)| t.as_str()).collect();
    assert_eq!(names, ["apple", "cat", "bank"]);

    // A payload score query's spans at the top: the inner's, collected
    // with payloads.
    let pay = SpanNode::term("pay", "apple");
    let inner = spans_with(&ctx, &pay, true).unwrap().unwrap();
    let mut p = payloads::PayloadSpans::new(
        inner,
        payloads::PayloadFunction::Sum,
        payloads::PayloadDecoder::Float,
    );
    assert!(p.two_phase());
    assert!(p.match_cost() > 0.0 && p.cost() > 0);
    let walked = walk(&mut p);
    let mut plain = spans_with(&ctx, &pay, false).unwrap().unwrap();
    assert_eq!(walked, walk(&mut plain));
    let inner = spans_with(&ctx, &pay, true).unwrap().unwrap();
    let mut p = payloads::PayloadSpans::new(
        inner,
        payloads::PayloadFunction::Max,
        payloads::PayloadDecoder::Float,
    );
    p.next_doc().unwrap();
    assert!(p.matches().unwrap());
    p.next_start_position().unwrap();
    let mut leaves = Leaves::default();
    p.collect(&mut leaves).unwrap();
    assert_eq!(leaves.0.len(), 1);
    assert_eq!(leaves.0[0].0, "apple");

    // The pulsed singleton's payloads come from its decoded posting.
    let zeta = SpanNode::term("pay", "zeta");
    let mut z = None;
    for seg in &segments {
        let ctx = leaf(seg);
        if let Some(s) = spans_with(&ctx, &zeta, true).unwrap() {
            z = Some((ctx, s));
        }
    }
    let (_, mut z) = z.expect("zeta is in one segment");
    z.next_doc().unwrap();
    z.next_start_position().unwrap();
    let mut leaves = Leaves::default();
    z.collect(&mut leaves).unwrap();
    assert_eq!(leaves.0.len(), 1);

    // Refusals: a field without positions, one near clause, an unrewritten
    // wrapper.
    assert!(segments.iter().map(leaf).any(|ctx| matches!(
        spans_with(&ctx, &SpanNode::term("id", "d1"), false),
        Err(Error::IllegalState(_))
    )));
    let one = SpanNode::Near {
        clauses: vec![body("apple")],
        slop: 0,
        in_order: true,
    };
    assert!(matches!(
        spans_with(&ctx, &one, false),
        Err(Error::IllegalArgument(_))
    ));
    let prefix = SpanNode::multi_term(MultiTermQuery::new(
        MultiTermSource::Prefix(PrefixQuery::new("body", "ap")),
        RewriteMethod::default(),
    ));
    assert!(matches!(
        spans_with(&ctx, &prefix, false),
        Err(Error::IllegalArgument(_))
    ));
    let mut none = Vec::new();
    weight_terms(&prefix, &mut none);
    assert!(none.is_empty());
}

/// Searched through the searcher: a multi-term wrapper rewritten anywhere
/// in the tree, every wrapper kind; a top-terms rewrite keeps its size; a
/// filter clause scores nothing; another similarity names itself.
#[test]
fn searched_rewritten_and_explained() {
    use crate::extended_query::TermRangeQuery;
    use crate::query::{BooleanQuery, Clause, RegexpQuery, WildcardQuery};
    let reader = fixture();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let norms = vec![None; segments.len()];
    let mut searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let prefix = |p: &str| {
        SpanNode::multi_term(MultiTermQuery::new(
            MultiTermSource::Prefix(PrefixQuery::new("body", p)),
            RewriteMethod::default(),
        ))
    };
    let wrapped = [
        SpanNode::first(prefix("ap"), 4),
        SpanNode::position_range(prefix("ap"), 1, 5),
        SpanNode::not(prefix("ba"), SpanNode::term("body", "x"), 0, 0).unwrap(),
        SpanNode::containing(prefix("ap"), prefix("ap")).unwrap(),
        SpanNode::within(prefix("ap"), prefix("ap")).unwrap(),
        SpanNode::field_masking(prefix("ap"), "body"),
        SpanNode::or(vec![prefix("ap"), prefix("ca")]).unwrap(),
        SpanNode::PayloadCheck(Box::new(payloads::SpanPayloadCheckQuery::new(
            prefix("ap"),
            vec![None],
        ))),
        SpanNode::PayloadScore(Box::new(payloads::PayloadScoreQuery::new(
            prefix("ap"),
            payloads::PayloadFunction::Sum,
            payloads::PayloadDecoder::Float,
            false,
        ))),
    ];
    for q in wrapped {
        assert!(q.needs_rewrite());
        let r = q.rewrite(&searcher).unwrap();
        assert!(!r.needs_rewrite(), "{r}");
        let bq = BooleanQuery {
            must: vec![Clause::from(q.clone())],
            ..Default::default()
        };
        searcher.search(&bq, 5).unwrap();
    }
    let top = SpanNode::multi_term(MultiTermQuery::new(
        MultiTermSource::Prefix(PrefixQuery::new("body", "ap")),
        RewriteMethod::TopTermsBoostOnlyBoolean(1),
    ));
    let SpanNode::Or { clauses } = top.rewrite(&searcher).unwrap() else {
        panic!("a disjunction")
    };
    assert_eq!(clauses, [SpanNode::term("body", "ape")]);
    for source in [
        MultiTermSource::Wildcard(WildcardQuery::new("body", "ca?")),
        MultiTermSource::Regexp(RegexpQuery::new("body", "ca.")),
        MultiTermSource::TermRange(TermRangeQuery::new(
            "body",
            Some(b"*".to_vec()),
            None,
            true,
            false,
        )),
    ] {
        let q = SpanNode::multi_term(MultiTermQuery::new(source, RewriteMethod::default()));
        assert!(
            q.to_string().starts_with("SpanMultiTermQueryWrapper(body:"),
            "{q}"
        );
    }
    assert_eq!(
        multi_term_string(&MultiTermSource::TermRange(TermRangeQuery::new(
            "body",
            Some(b"*".to_vec()),
            None,
            true,
            false
        ))),
        "body:[\\* TO *}"
    );

    // An `INT` comparison of one-byte payloads is refused; more collected
    // terms than payloads to match is no match.
    let check = |inner: SpanNode, kind, pays: Vec<Option<Vec<u8>>>| BooleanQuery {
        must: vec![Clause::from(SpanNode::PayloadCheck(Box::new(
            payloads::SpanPayloadCheckQuery::with(inner, pays, kind, payloads::MatchOperation::Gt),
        )))],
        ..Default::default()
    };
    let short = check(
        SpanNode::term("pay", "apple"),
        payloads::PayloadType::Int,
        vec![Some(0i32.to_be_bytes().to_vec())],
    );
    assert!(matches!(
        searcher.search(&short, 5),
        Err(Error::IllegalArgument(_))
    ));
    let pair = SpanNode::near(
        vec![
            SpanNode::term("pay", "apple"),
            SpanNode::term("pay", "bank"),
        ],
        3,
        false,
    )
    .unwrap();
    let fewer = check(pair, payloads::PayloadType::String, vec![Some(vec![0])]);
    assert_eq!(searcher.search(&fewer, 5).unwrap().total_hits.value, 0);

    // A filter scores nothing: only the optional term's score is left.
    let near = SpanNode::near(
        vec![
            SpanNode::term("body", "apple"),
            SpanNode::term("body", "bank"),
        ],
        2,
        false,
    )
    .unwrap();
    let filtered = BooleanQuery {
        filter: vec![Clause::from(near.clone())],
        ..Default::default()
    };
    let td = searcher.search(&filtered, 50).unwrap();
    assert!(!td.score_docs.is_empty());
    assert!(td.score_docs.iter().all(|d| d.score == 0.0));

    // Another similarity: its scores, its name, no breakdown.
    let classic = crate::similarities::ClassicSimilarity::default();
    searcher.set_similarity(&classic);
    let scored = BooleanQuery {
        must: vec![Clause::from(near)],
        ..Default::default()
    };
    let td = searcher.search(&scored, 1).unwrap();
    let doc = td.score_docs[0].doc;
    let e = searcher.explain(&scored, doc).unwrap();
    assert_eq!(e.value, td.score_docs[0].score);
    assert!(
        e.description.ends_with("[Similarity], result of:"),
        "{}",
        e.description
    );
    assert!(e.details.is_empty());
}
