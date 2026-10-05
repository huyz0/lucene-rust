//! `SpanWeight` and `SpanScorer` (`lucene-queries`,
//! `org.apache.lucene.queries.spans`) for [`crate::spans::SpanNode`], and
//! `PayloadScoreQuery`'s `PayloadSpanWeight`/`PayloadSpanScorer`: the
//! query's [`crate::spans::Spans`] over the segment as a two-phase scorer,
//! scored by the similarity over the sloppy frequency of each document's
//! spans (`sum(1 / (1 + width))`), times the payload function's score
//! where it is a payload score query.

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

impl<'a> ScorerSpans<'a> {
    fn spans(&self) -> &dyn Spans {
        match self {
            ScorerSpans::Plain(s) => s.as_ref(),
            ScorerSpans::Payload { spans, .. } => spans,
        }
    }
    fn spans_mut(&mut self) -> &mut (dyn Spans + 'a) {
        match self {
            ScorerSpans::Plain(s) => s.as_mut(),
            ScorerSpans::Payload { spans, .. } => spans,
        }
    }
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
        let doc = self.spans.spans().doc_id();
        if self.last_scored_doc != doc {
            self.freq = if self.sim.is_some() {
                spans::sloppy_freq(self.spans.spans_mut())?
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
        let doc = self.spans.spans().doc_id();
        let norm = self.norm(doc)?;
        Ok(self.sim.as_ref().map_or(0.0, |s| s.score(self.freq, norm)))
    }
}

impl Scorer for SpanScorer<'_> {
    fn doc_id(&self) -> i32 {
        self.spans.spans().doc_id()
    }
    fn next_doc(&mut self) -> Result<i32> {
        self.spans.spans_mut().next_doc()
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        self.spans.spans_mut().advance(target)
    }
    fn cost(&self) -> i64 {
        self.spans.spans().cost()
    }
    fn two_phase(&self) -> bool {
        self.spans.spans().two_phase()
    }
    fn matches(&mut self) -> Result<bool> {
        self.spans.spans_mut().matches()
    }
    fn match_cost(&self) -> f32 {
        self.spans.spans().match_cost()
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
            unreachable!("a payload score query scoring builds payload spans")
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

/// `scorer.iterator().advance(doc) == doc` for a two-phase spans: the
/// first confirmed document at or after `doc` is `doc`.
fn advance_to(scorer: &mut SpanScorer<'_>, doc: i32) -> Result<bool> {
    let mut d = scorer.advance(doc)?;
    while d != NO_MORE_DOCS && !scorer.matches()? {
        d = scorer.next_doc()?;
    }
    Ok(d == doc)
}
