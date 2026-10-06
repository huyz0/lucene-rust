//! `SpanWeight` and `SpanScorer` (`lucene-queries`,
//! `org.apache.lucene.queries.spans`) for [`crate::spans::SpanNode`], and
//! `PayloadScoreQuery`'s `PayloadSpanWeight`/`PayloadSpanScorer`: the
//! query's [`crate::spans::Spans`] over the segment as a two-phase scorer,
//! scored by the similarity over the sloppy frequency of each document's
//! spans (`sum(1 / (1 + width))`), times the payload function's score
//! where it is a payload score query; and `SpanWeight.matches`.

use std::sync::Arc;

use super::{BoxScorer, LeafContext, Mode, Scorer, NO_MORE_DOCS};
use crate::explain::{java_float, Explanation};
use crate::field_norms::FieldNormsCursor;
use crate::similarities::{CollectionStatistics, SimScorer, TermStatistics};
use crate::spans::payloads::{self, PayloadFunction, PayloadSpans};
use crate::spans::{self, BoxSpans, SpanNode, Spans};
use crate::{similarity, Result};

/// `SpanWeight.buildSimWeight`'s inputs: the field, its collection
/// statistics and the statistics of the query's terms that have a document,
/// in `Term` order.
struct SimInputs {
    field: String,
    collection: CollectionStatistics,
    terms: Vec<TermStatistics>,
}

/// `buildSimWeight(query, searcher, termStates, boost)`'s statistics:
/// `None` where Java's `simScorer` is `null` (no term has a document, or
/// no document has the field).
fn sim_inputs(ctx: &LeafContext<'_>, q: &SpanNode) -> Result<Option<SimInputs>> {
    let Some(field) = q.field() else {
        return Ok(None);
    };
    let mut keys = Vec::new();
    spans::weight_terms(q, &mut keys);
    keys.sort_unstable();
    keys.dedup();
    let mut terms = Vec::with_capacity(keys.len());
    for (f, t) in &keys {
        if let Some(entry) = super::extended::term_entry(ctx, f, t)? {
            if entry.doc_freq > 0 {
                terms.push(entry.term_statistics());
            }
        }
    }
    if terms.is_empty() {
        return Ok(None);
    }
    let first = keys
        .iter()
        .find(|(f, _)| f == field)
        .map(|(_, t)| t.as_slice())
        .unwrap_or_default();
    let Some(collection) = super::extended::term_entry(ctx, field, first)?
        .as_ref()
        .and_then(super::extended::collection_of)
    else {
        return Ok(None);
    };
    Ok(Some(SimInputs {
        field: field.to_string(),
        collection,
        terms,
    }))
}

/// The spans a scorer reads: a payload score query's collect payloads.
enum ScorerSpans<'a> {
    Plain(BoxSpans<'a>),
    Payload {
        spans: PayloadSpans<'a>,
        function: PayloadFunction,
        include_span_score: bool,
    },
}

/// Forwards to the scorer's spans without a virtual call of its own.
macro_rules! on_spans {
    ($spans:expr, $s:ident => $e:expr) => {
        match $spans {
            ScorerSpans::Plain($s) => $e,
            ScorerSpans::Payload { spans: $s, .. } => $e,
        }
    };
}

/// `SpanScorer` (and `PayloadSpanScorer`).
pub(crate) struct SpanScorer<'a> {
    spans: ScorerSpans<'a>,
    sim: Option<Arc<dyn SimScorer>>,
    norms: Option<FieldNormsCursor<'a, 'a>>,
    freq: f32,
    last_scored_doc: i32,
}

impl<'a> SpanScorer<'a> {
    /// `ensureFreq()` / `setFreqCurrentDoc()`.
    fn ensure_freq(&mut self) -> Result<()> {
        let doc = on_spans!(&self.spans, s => s.doc_id());
        if self.last_scored_doc != doc {
            self.freq = if self.sim.is_some() {
                on_spans!(&mut self.spans, s => s.sloppy_freq())?
            } else {
                1.0
            };
            self.last_scored_doc = doc;
        }
        Ok(())
    }

    fn norm(&mut self, doc: i32) -> Result<i64> {
        Ok(match self.norms.as_mut() {
            Some(n) => n.norm_long(doc)?.unwrap_or(1),
            None => 1,
        })
    }

    /// `SpanScorer.scoreCurrentDoc()`.
    fn span_score(&mut self) -> Result<f32> {
        let doc = on_spans!(&self.spans, s => s.doc_id());
        let norm = self.norm(doc)?;
        Ok(self.sim.as_ref().map_or(0.0, |s| s.score(self.freq, norm)))
    }
}

impl Scorer for SpanScorer<'_> {
    fn doc_id(&self) -> i32 {
        on_spans!(&self.spans, s => s.doc_id())
    }
    fn next_doc(&mut self) -> Result<i32> {
        on_spans!(&mut self.spans, s => s.next_doc())
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        on_spans!(&mut self.spans, s => s.advance(target))
    }
    fn cost(&self) -> i64 {
        on_spans!(&self.spans, s => s.cost())
    }
    fn two_phase(&self) -> bool {
        on_spans!(&self.spans, s => s.two_phase())
    }
    fn matches(&mut self) -> Result<bool> {
        on_spans!(&mut self.spans, s => s.matches())
    }
    fn match_cost(&self) -> f32 {
        on_spans!(&self.spans, s => s.match_cost())
    }
    fn score(&mut self) -> Result<f32> {
        self.ensure_freq()?;
        match &self.spans {
            ScorerSpans::Plain(_) => self.span_score(),
            ScorerSpans::Payload {
                spans,
                function,
                include_span_score,
            } => {
                let p =
                    payloads::payload_score(*function, spans.payloads_seen, spans.payload_score);
                if *include_span_score {
                    Ok(self.span_score()? * p)
                } else {
                    Ok(p)
                }
            }
        }
    }
    /// `getMaxScore`: unbounded, as Java's.
    fn max_score(&mut self, _up_to: i32) -> Result<f32> {
        Ok(f32::INFINITY)
    }
}

/// The weight `q` scores with and the spans it reads: a top-level
/// `FieldMaskingSpanQuery` creates its masked query's weight; a payload
/// score query's spans are its inner query's, its similarity the inner
/// weight's.
fn build_scorer<'a>(
    ctx: &LeafContext<'a>,
    q: &SpanNode,
    boost: f32,
    mode: Mode,
) -> Result<Option<SpanScorer<'a>>> {
    let (weight_q, payload) = match q {
        SpanNode::PayloadScore(p) if mode.needs_scores() => {
            (spans::weight_query(&p.inner), Some(p))
        }
        SpanNode::PayloadScore(p) => (spans::weight_query(&p.inner), None),
        other => (spans::weight_query(other), None),
    };
    // `PayloadSpanWeight.getSpans(context, Postings.PAYLOADS)`.
    let Some(s) = spans::spans_with(ctx, weight_q, payload.is_some())? else {
        return Ok(None);
    };
    let (sim, norms) = if mode.needs_scores() {
        match sim_inputs(ctx, weight_q)? {
            Some(inputs) => (
                Some(super::extended::sim_scorer(
                    ctx,
                    &inputs.field,
                    boost,
                    &inputs.collection,
                    &inputs.terms,
                )),
                super::extended::norms_cursor(ctx, &inputs.field),
            ),
            None => (None, None),
        }
    } else {
        (None, None)
    };
    let spans = match payload {
        Some(p) => ScorerSpans::Payload {
            spans: PayloadSpans::new(s, p.function, p.decoder.clone()),
            function: p.function,
            include_span_score: p.include_span_score,
        },
        None => ScorerSpans::Plain(s),
    };
    Ok(Some(SpanScorer {
        spans,
        sim,
        norms,
        freq: 0.0,
        last_scored_doc: -1,
    }))
}

/// `SpanWeight.scorerSupplier(context)`: `None` when the query has no spans
/// in this segment.
pub(crate) fn span_node<'a>(
    ctx: &LeafContext<'a>,
    q: &SpanNode,
    boost: f32,
    mode: Mode,
) -> Result<Option<BoxScorer<'a>>> {
    Ok(build_scorer(ctx, q, boost, mode)?.map(|s| -> BoxScorer<'a> { Box::new(s) }))
}

/// `BM25Similarity`'s `BM25Scorer.explain(freq, norm)` for the default
/// similarity: `score(freq=..)` from the boost, the idf (one term's, or the
/// sum of several) and the tf.
fn bm25_explanation(
    value: f32,
    freq: f32,
    norm: Option<i64>,
    boost: f32,
    inputs: &SimInputs,
) -> Explanation {
    let doc_count = inputs.collection.doc_count;
    let mut idf_details = Vec::with_capacity(inputs.terms.len());
    let mut idf_acc = 0.0f64;
    for t in &inputs.terms {
        let idf = similarity::idf(t.doc_freq, doc_count);
        idf_acc += f64::from(idf);
        idf_details.push(crate::explain::idf_explanation(idf, t.doc_freq, doc_count));
    }
    let idf_node = if idf_details.len() == 1 {
        idf_details.remove(0)
    } else {
        Explanation::match_(idf_acc as f32, "idf, sum of:").with_details(idf_details)
    };
    let (field_length, avgdl, dl_desc) = match norm {
        Some(n) => {
            let avgdl = (inputs.collection.sum_total_term_freq as f64 / doc_count as f64) as f32;
            let dl = similarity::decode_norm(n);
            let desc = if (n as u8) > 39 {
                "dl, length of field (approximate)"
            } else {
                "dl, length of field"
            };
            (dl, avgdl, desc)
        }
        None => (
            similarity::UNNORMED_FIELD_LENGTH,
            similarity::UNNORMED_FIELD_LENGTH,
            "dl, length of field",
        ),
    };
    let norm_inverse = similarity::norm_inverse(
        field_length,
        avgdl,
        similarity::DEFAULT_K1,
        similarity::DEFAULT_B,
    );
    let tf = 1.0 - 1.0 / (1.0 + freq * norm_inverse);
    let tf_node = Explanation::match_(
        tf,
        "tf, computed as freq / (freq + k1 * (1 - b + b * dl / avgdl)) from:",
    )
    .with_details(vec![
        Explanation::match_(freq, format!("phraseFreq={}", java_float(freq))),
        Explanation::match_(similarity::DEFAULT_K1, "k1, term saturation parameter"),
        Explanation::match_(similarity::DEFAULT_B, "b, length normalization parameter"),
        Explanation::match_(field_length, dl_desc),
        Explanation::match_(avgdl, "avgdl, average length of field"),
    ]);
    let mut subs = Vec::with_capacity(3);
    if boost != 1.0 {
        subs.push(Explanation::match_(boost, "boost"));
    }
    subs.push(idf_node);
    subs.push(tf_node);
    Explanation::match_(
        value,
        format!(
            "score(freq={}), computed as boost * idf * tf from:",
            java_float(freq)
        ),
    )
    .with_details(subs)
}

/// `SpanWeight.explain(context, doc)` (and `PayloadSpanWeight.explain`):
/// the weight's explanation of the document's sloppy frequency, or `"no
/// matching term"`. Deletions are not consulted, as Java's are not.
pub(crate) fn explain_span_node(
    ctx: &LeafContext<'_>,
    q: &SpanNode,
    boost: f32,
    doc: i32,
) -> Result<Explanation> {
    if let SpanNode::PayloadScore(p) = q {
        let Some(mut scorer) = build_scorer(ctx, q, boost, Mode::Complete)? else {
            return Ok(Explanation::no_match("No match"));
        };
        if !advance_to(&mut scorer, doc)? {
            return Ok(Explanation::no_match("No match"));
        }
        let score = scorer.score()?;
        let ScorerSpans::Payload {
            spans, function, ..
        } = &scorer.spans
        else {
            return Err(crate::Error::IllegalState(
                "a payload score query's scorer holds no payload spans".into(),
            ));
        };
        let payload_expl =
            payloads::payload_explanation(*function, spans.payloads_seen, spans.payload_score);
        if p.include_span_score {
            let inner = explain_span_node(ctx, &p.inner, boost, doc)?;
            return Ok(Explanation::match_(score, "PayloadSpanQuery, product of:")
                .with_details(vec![inner, payload_expl]));
        }
        return Ok(payload_expl);
    }
    let weight_q = spans::weight_query(q);
    let Some(mut scorer) = build_scorer(ctx, q, boost, Mode::Complete)? else {
        return Ok(Explanation::no_match("no matching term"));
    };
    if !advance_to(&mut scorer, doc)? {
        return Ok(Explanation::no_match("no matching term"));
    }
    let Some(inputs) = sim_inputs(ctx, weight_q)? else {
        return Ok(Explanation::match_(
            0.0,
            format!("match {weight_q} in {doc} without score"),
        ));
    };
    scorer.ensure_freq()?;
    let freq = scorer.freq;
    let value = scorer.span_score()?;
    let norm = match scorer.norms.as_mut() {
        Some(n) => n.norm_long(doc)?,
        None => None,
    };
    // Another similarity's own breakdown is not ported: its top node only.
    let (name, detail) = match ctx.similarity {
        Some(s) if !s.is_default_bm25() => ("Similarity", None),
        _ => (
            "BM25Similarity",
            Some(bm25_explanation(value, freq, norm, boost, &inputs)),
        ),
    };
    let top = Explanation::match_(
        value,
        format!("weight({weight_q} in {doc}) [{name}], result of:"),
    );
    Ok(match detail {
        Some(d) => top.with_details(vec![d]),
        None => top,
    })
}

/// `SpanWeight.explain` of a weight created without scores (a filter
/// clause's, `COMPLETE_NO_SCORES`: no term statistics, so no similarity): a
/// match is `"match q in doc without score"`. A payload score query's weight
/// without scores is its inner query's.
pub(crate) fn explain_span_unscored(
    ctx: &LeafContext<'_>,
    q: &SpanNode,
    doc: i32,
) -> Result<Explanation> {
    let mut q = q;
    while let SpanNode::PayloadScore(p) = q {
        q = &p.inner;
    }
    let Some(mut scorer) = build_scorer(ctx, q, 1.0, Mode::NoScores)? else {
        return Ok(Explanation::no_match("no matching term"));
    };
    if !advance_to(&mut scorer, doc)? {
        return Ok(Explanation::no_match("no matching term"));
    }
    Ok(Explanation::match_(
        0.0,
        format!("match {} in {doc} without score", spans::weight_query(q)),
    ))
}

/// `scorer.iterator().advance(doc) == doc` for a two-phase spans: the
/// first confirmed document at or after `doc` is `doc`.
fn advance_to(scorer: &mut SpanScorer<'_>, doc: i32) -> Result<bool> {
    let mut d = scorer.advance(doc)?;
    while d != NO_MORE_DOCS && !scorer.matches()? {
        d = scorer.next_doc()?;
    }
    Ok(d == doc)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The matches iterators before their first and after their last match
    /// report `-1`s, and a sub-match's query is its term's (the span
    /// query's before the first).
    #[test]
    fn span_matches_iterators_outside_their_matches() {
        use crate::matches::MatchesIterator;
        let span_q = Arc::new(crate::query::Clause::from(SpanNode::term("f", "a")));
        let term = Arc::new(crate::query::Clause::Term(crate::TermQuery::new("f", "a")));
        let mut it = SpanMatchesIterator {
            query: Arc::clone(&span_q),
            spans: Arc::new(vec![SpanMatch {
                start: 2,
                end: 4,
                terms: vec![
                    TermMatch {
                        query: Arc::clone(&term),
                        position: 2,
                        start_offset: 5,
                        end_offset: 6,
                    },
                    TermMatch {
                        query: Arc::clone(&term),
                        position: 4,
                        start_offset: 9,
                        end_offset: 12,
                    },
                ],
            }]),
            at: 0,
        };
        let outside = |it: &SpanMatchesIterator| {
            [
                it.start_position(),
                it.end_position(),
                it.start_offset(),
                it.end_offset(),
            ]
        };
        assert_eq!(outside(&it), [-1; 4]);
        assert!(it.next().unwrap());
        assert_eq!(outside(&it), [2, 4, 5, 12]);
        assert_eq!(it.query(), span_q.as_ref());
        let mut sub = it.sub_matches().unwrap().unwrap();
        assert_eq!(sub.query(), span_q.as_ref(), "before the first term");
        assert_eq!(sub.start_position(), -1);
        assert!(sub.next().unwrap());
        assert_eq!((sub.start_position(), sub.end_offset()), (2, 6));
        assert_eq!(sub.query(), term.as_ref());
        assert!(sub.next().unwrap());
        assert_eq!((sub.end_position(), sub.start_offset()), (4, 9));
        assert!(!sub.next().unwrap());
        assert!(!it.next().unwrap());
        assert_eq!(outside(&it), [2, 4, 5, 12], "stays on the last");
    }

    /// No matches where the segment has no spans for the query, or the
    /// document has none.
    #[test]
    fn span_matches_of_a_missing_term_or_document_are_none() {
        let reader = reader();
        let opened = reader.open_segments().unwrap();
        let segs = opened.as_open_segments();
        let seg = &segs[0];
        let ctx = LeafContext {
            fields: seg.fields,
            doc_in: seg.doc_in,
            pos_in: seg.pos_in,
            pay_in: seg.pay_in,
            live_docs: None,
            points: None,
            norms: None,
            global: None,
            max_doc: seg.max_doc,
            cache: None,
            reader: seg.reader,
            similarity: None,
        };
        let q = Arc::new(crate::query::Clause::MatchNoDocs(Default::default()));
        let missing = SpanNode::term("body", "nosuch");
        assert!(span_matches(&ctx, &missing, Arc::clone(&q), 0)
            .unwrap()
            .is_none());
        let empty_or = SpanNode::Or {
            clauses: Vec::new(),
        };
        assert!(span_matches(&ctx, &empty_or, Arc::clone(&q), 0)
            .unwrap()
            .is_none());
        let apple = SpanNode::term("body", "apple");
        let found = (0..seg.max_doc.unwrap())
            .filter(|&d| {
                span_matches(&ctx, &apple, Arc::clone(&q), d)
                    .unwrap()
                    .is_some()
            })
            .count();
        assert!(found > 0 && found < usize::try_from(seg.max_doc.unwrap()).unwrap());
    }

    fn reader() -> crate::directory_reader::DirectoryReader {
        let dir = std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/spans/index"
        ));
        crate::directory_reader::DirectoryReader::open(&lucene_store::FsDirectory::open(dir))
            .unwrap()
    }

    /// The weight's statistics where Java's `simScorer` is `null`; a scorer
    /// without norms explains BM25 over unnormed lengths; the scorer's
    /// bounds and costs.
    #[test]
    fn statistics_bounds_and_unnormed_explanations() {
        let reader = reader();
        let opened = reader.open_segments().unwrap();
        let segments = opened.as_open_segments();
        let seg = &segments[0];
        let ctx = LeafContext {
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
        };
        let empty = SpanNode::Near {
            clauses: Vec::new(),
            slop: 0,
            in_order: true,
        };
        assert!(sim_inputs(&ctx, &empty).unwrap().is_none());
        assert!(sim_inputs(&ctx, &SpanNode::term("body", "nosuch"))
            .unwrap()
            .is_none());
        assert!(sim_inputs(&ctx, &SpanNode::term("nofield", "a"))
            .unwrap()
            .is_none());
        assert!(span_node(&ctx, &empty, 1.0, Mode::Complete)
            .unwrap()
            .is_none());

        let q = SpanNode::term("body", "apple");
        let mut s = build_scorer(&ctx, &q, 2.0, Mode::Complete)
            .unwrap()
            .unwrap();
        assert_eq!(s.max_score(NO_MORE_DOCS).unwrap(), f32::INFINITY);
        assert!(!s.two_phase());
        assert!(s.cost() > 0 && s.match_cost() > 0.0);
        let doc = s.next_doc().unwrap();
        let score = s.score().unwrap();
        let e = explain_span_node(&ctx, &q, 2.0, doc).unwrap();
        assert_eq!(e.value, score);
        let text = e.to_string();
        assert!(text.contains("2.0 = boost"), "{text}");
        assert!(text.contains("1.0 = dl, length of field"), "{text}");
        assert_eq!(
            explain_span_node(&ctx, &q, 1.0, NO_MORE_DOCS - 1)
                .unwrap()
                .description,
            "no matching term"
        );
        // Without scores: frequency 1, no similarity.
        let mut n = build_scorer(&ctx, &q, 1.0, Mode::NoScores)
            .unwrap()
            .unwrap();
        n.next_doc().unwrap();
        assert_eq!(n.score().unwrap(), 0.0);
        let e = explain_span_unscored(&ctx, &q, doc).unwrap();
        assert_eq!(
            e.description,
            format!("match body:apple in {doc} without score")
        );
        assert!(
            !explain_span_unscored(&ctx, &q, NO_MORE_DOCS - 1)
                .unwrap()
                .matched
        );
        assert!(
            !explain_span_unscored(&ctx, &SpanNode::term("body", "nosuch"), 0)
                .unwrap()
                .matched
        );
    }
}

// ---------------------------------------------------------------------------
// SpanWeight.matches
// ---------------------------------------------------------------------------

/// `SpanWeight.matches`' `TermMatch`: one leaf occurrence a span collected.
struct TermMatch {
    query: Arc<crate::query::Clause>,
    position: i32,
    start_offset: i32,
    end_offset: i32,
}

/// The anonymous `termCollector`: each leaf's term, position and offsets.
#[derive(Default)]
struct InnerTerms(Vec<TermMatch>);

impl spans::SpanCollector for InnerTerms {
    fn collect_leaf(&mut self, leaf: &mut spans::TermSpans<'_>, position: i32) -> Result<()> {
        let (start_offset, end_offset) = leaf
            .occurrence()?
            .map_or((-1, -1), |o| (o.start_offset, o.end_offset));
        let (field, term) = &leaf.term;
        self.0.push(TermMatch {
            query: Arc::new(crate::query::Clause::Term(crate::TermQuery::new(
                field.clone(),
                term.clone(),
            ))),
            position,
            start_offset,
            end_offset,
        });
        Ok(())
    }
    fn reset(&mut self) {
        self.0.clear();
    }
}

/// One span of the document: `[startPosition, endPosition - 1]` and its
/// inner terms in position order (`collectInnerTerms`' stable sort).
struct SpanMatch {
    start: i32,
    end: i32,
    terms: Vec<TermMatch>,
}

/// `SpanWeight.matches`' iterator over the document's spans, read when the
/// matches were asked for: the same sequence Java's lazy one walks.
struct SpanMatchesIterator {
    query: Arc<crate::query::Clause>,
    spans: Arc<Vec<SpanMatch>>,
    /// The next span: `next` makes the current one `at - 1`.
    at: usize,
}

impl SpanMatchesIterator {
    fn current(&self) -> Option<&SpanMatch> {
        self.at.checked_sub(1).and_then(|i| self.spans.get(i))
    }
}

impl crate::matches::MatchesIterator for SpanMatchesIterator {
    fn next(&mut self) -> Result<bool> {
        if self.at < self.spans.len() {
            self.at += 1;
            Ok(true)
        } else {
            Ok(false)
        }
    }
    fn start_position(&self) -> i32 {
        self.current().map_or(-1, |m| m.start)
    }
    fn end_position(&self) -> i32 {
        self.current().map_or(-1, |m| m.end)
    }
    /// `innerTerms[0].startOffset`.
    fn start_offset(&self) -> i32 {
        self.current()
            .and_then(|m| m.terms.first())
            .map_or(-1, |t| t.start_offset)
    }
    /// `innerTerms[innerTermCount - 1].endOffset`.
    fn end_offset(&self) -> i32 {
        self.current()
            .and_then(|m| m.terms.last())
            .map_or(-1, |t| t.end_offset)
    }
    /// One match per inner term, each its own `TermQuery`'s.
    fn sub_matches(&mut self) -> Result<Option<crate::matches::BoxMatchesIterator>> {
        let terms = self.current().map_or_else(Vec::new, |m| {
            m.terms
                .iter()
                .map(|t| {
                    (
                        Arc::clone(&t.query),
                        [t.position, t.position, t.start_offset, t.end_offset],
                    )
                })
                .collect()
        });
        Ok(Some(Box::new(TermMatchesIterator {
            terms,
            span_query: Arc::clone(&self.query),
            at: 0,
        })))
    }
    fn query(&self) -> &crate::query::Clause {
        &self.query
    }
}

/// `getSubMatches()`' iterator: the span's terms, each `[position,
/// position]` with its offsets, `getQuery()` a `TermQuery` of its term.
struct TermMatchesIterator {
    terms: Vec<(Arc<crate::query::Clause>, [i32; 4])>,
    span_query: Arc<crate::query::Clause>,
    at: usize,
}

impl TermMatchesIterator {
    fn span(&self) -> [i32; 4] {
        self.at
            .checked_sub(1)
            .and_then(|i| self.terms.get(i))
            .map_or([-1; 4], |t| t.1)
    }
}

impl crate::matches::MatchesIterator for TermMatchesIterator {
    fn next(&mut self) -> Result<bool> {
        if self.at < self.terms.len() {
            self.at += 1;
            Ok(true)
        } else {
            Ok(false)
        }
    }
    fn start_position(&self) -> i32 {
        self.span()[0]
    }
    fn end_position(&self) -> i32 {
        self.span()[1]
    }
    fn start_offset(&self) -> i32 {
        self.span()[2]
    }
    fn end_offset(&self) -> i32 {
        self.span()[3]
    }
    /// The current term's `TermQuery`; before the first, the span query's.
    fn query(&self) -> &crate::query::Clause {
        self.at
            .checked_sub(1)
            .and_then(|i| self.terms.get(i))
            .map_or(self.span_query.as_ref(), |t| t.0.as_ref())
    }
}

/// `SpanWeight.matches(context, doc)` (inherited by `PayloadScoreQuery`'s
/// and `SpanPayloadCheckQuery`'s weights): `MatchesUtils.forField` over the
/// weight's field, each span of `doc` (leaf-local) with its inner terms'
/// offsets and, as sub-matches, the terms themselves. `query` is what the
/// iterator's `getQuery()` reports: the weight's query (a top-level
/// `FieldMaskingSpanQuery`'s weight is its masked query's).
///
/// Java's `spans.advance(doc)` confirms the document (a `Spans` moves to
/// matching documents only); here the two-phase view's `matches()` does.
/// The spans are walked when the matches are asked for rather than as the
/// iterator moves: the same spans, in the same order.
pub(crate) fn span_matches(
    ctx: &LeafContext<'_>,
    q: &SpanNode,
    query: Arc<crate::query::Clause>,
    doc: i32,
) -> Result<Option<crate::matches::BoxMatches>> {
    let Some(field) = q.field() else {
        return Ok(None);
    };
    // `getSpans(context, Postings.OFFSETS)`: offsets are read per occurrence.
    let Some(mut s) = spans::spans_with(ctx, q, false)? else {
        return Ok(None);
    };
    if s.advance(doc)? != doc || !s.matches()? {
        return Ok(None);
    }
    let mut found = Vec::new();
    let mut collector = InnerTerms::default();
    while s.next_start_position()? != spans::NO_MORE_POSITIONS {
        collector.0.clear();
        s.collect(&mut collector)?;
        let mut terms = std::mem::take(&mut collector.0);
        terms.sort_by_key(|t| t.position);
        found.push(SpanMatch {
            start: s.start_position(),
            end: s.end_position().wrapping_sub(1),
            terms,
        });
    }
    let found = Arc::new(found);
    crate::matches::for_field(
        field,
        Box::new(move || {
            Ok(Some(Box::new(SpanMatchesIterator {
                query: Arc::clone(&query),
                spans: Arc::clone(&found),
                at: 0,
            }) as crate::matches::BoxMatchesIterator))
        }),
    )
}
