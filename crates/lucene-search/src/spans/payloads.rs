//! `lucene-queries`' payloads package (`org.apache.lucene.queries.payloads`,
//! Lucene 10.5.0): [`PayloadScoreQuery`], which scores a span query's
//! matches by the payloads of the terms in them through a
//! [`PayloadFunction`] and a [`PayloadDecoder`]; and
//! [`SpanPayloadCheckQuery`], which keeps only the spans whose terms carry
//! the given payloads (equal, or compared as ints, floats or strings --
//! `PayloadMatcherFactory`'s matchers).

use std::sync::Arc;

use super::{AcceptStatus, BoxSpans, SpanCollector, SpanFilter, SpanNode, Spans, TermSpans};
use crate::explain::Explanation;
use crate::{Error, Result};

/// `PayloadFunction` and its four implementations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PayloadFunction {
    /// `MinPayloadFunction`.
    Min,
    /// `MaxPayloadFunction`.
    Max,
    /// `AveragePayloadFunction`.
    Average,
    /// `SumPayloadFunction`.
    Sum,
}

/// `Math.max(float, float)`: NaN if either is, `0.0` over `-0.0`.
fn java_max(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else if a == 0.0 && b == 0.0 {
        if a.is_sign_negative() {
            b
        } else {
            a
        }
    } else if a >= b {
        a
    } else {
        b
    }
}

/// `Math.min(float, float)`: NaN if either is, `-0.0` over `0.0`.
fn java_min(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else if a == 0.0 && b == 0.0 {
        if a.is_sign_negative() {
            a
        } else {
            b
        }
    } else if a <= b {
        a
    } else {
        b
    }
}

impl PayloadFunction {
    /// The class's `getSimpleName()`.
    pub fn name(self) -> &'static str {
        match self {
            PayloadFunction::Min => "MinPayloadFunction",
            PayloadFunction::Max => "MaxPayloadFunction",
            PayloadFunction::Average => "AveragePayloadFunction",
            PayloadFunction::Sum => "SumPayloadFunction",
        }
    }

    /// `currentScore(docId, field, start, end, numPayloadsSeen,
    /// currentScore, currentPayloadScore)`.
    pub fn current_score(
        self,
        num_payloads_seen: i32,
        current_score: f32,
        current_payload_score: f32,
    ) -> f32 {
        match self {
            PayloadFunction::Min if num_payloads_seen != 0 => {
                java_min(current_payload_score, current_score)
            }
            PayloadFunction::Max if num_payloads_seen != 0 => {
                java_max(current_payload_score, current_score)
            }
            PayloadFunction::Min | PayloadFunction::Max => current_payload_score,
            PayloadFunction::Average | PayloadFunction::Sum => {
                current_payload_score + current_score
            }
        }
    }

    /// `docScore(docId, field, numPayloadsSeen, payloadScore)`.
    pub fn doc_score(self, num_payloads_seen: i32, payload_score: f32) -> f32 {
        if num_payloads_seen <= 0 {
            return 1.0;
        }
        match self {
            PayloadFunction::Average => payload_score / num_payloads_seen as f32,
            _ => payload_score,
        }
    }

    /// `explain(docId, field, numPayloadsSeen, payloadScore)`.
    pub fn explain(self, num_payloads_seen: i32, payload_score: f32) -> Explanation {
        Explanation::match_(
            self.doc_score(num_payloads_seen, payload_score),
            format!("{}.docScore()", self.name()),
        )
    }
}

/// A payload decoder's function: the payload (`None` when the position
/// has none) to a factor.
pub type DecodeFn = dyn Fn(Option<&[u8]>) -> f32 + Send + Sync;

/// `PayloadDecoder`.
#[derive(Clone)]
pub enum PayloadDecoder {
    /// `PayloadDecoder.FLOAT_DECODER`: `1` without a payload, else its
    /// first byte as a signed `byte` widened to `float` -- Java's lambda
    /// reads `bytes.bytes[bytes.offset]`, not a float encoding.
    Float,
    /// A caller's decoder; equal only to itself, as a lambda is.
    Custom(Arc<DecodeFn>),
}

impl PayloadDecoder {
    /// `computePayloadFactor(payload)`.
    pub fn compute_payload_factor(&self, payload: Option<&[u8]>) -> f32 {
        match self {
            PayloadDecoder::Float => payload
                .and_then(|p| p.first())
                .map_or(1.0, |&b| f32::from(b as i8)),
            PayloadDecoder::Custom(f) => f(payload),
        }
    }
}

impl std::fmt::Debug for PayloadDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PayloadDecoder::Float => f.write_str("FLOAT_DECODER"),
            PayloadDecoder::Custom(_) => f.write_str("PayloadDecoder"),
        }
    }
}

impl PartialEq for PayloadDecoder {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (PayloadDecoder::Float, PayloadDecoder::Float) => true,
            (PayloadDecoder::Custom(a), PayloadDecoder::Custom(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}

/// `PayloadScoreQuery`.
#[derive(Debug, Clone, PartialEq)]
pub struct PayloadScoreQuery {
    pub inner: SpanNode,
    pub function: PayloadFunction,
    pub decoder: PayloadDecoder,
    pub include_span_score: bool,
}

impl PayloadScoreQuery {
    /// `new PayloadScoreQuery(wrappedQuery, function, decoder,
    /// includeSpanScore)`.
    pub fn new(
        inner: SpanNode,
        function: PayloadFunction,
        decoder: PayloadDecoder,
        include_span_score: bool,
    ) -> Self {
        PayloadScoreQuery {
            inner,
            function,
            decoder,
            include_span_score,
        }
    }
}

impl std::fmt::Display for PayloadScoreQuery {
    /// `toString(field)`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PayloadScoreQuery({}, function: {}, includeSpanScore: {})",
            self.inner,
            self.function.name(),
            self.include_span_score
        )
    }
}

/// `SpanPayloadCheckQuery.PayloadType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PayloadType {
    Int,
    Float,
    String,
}

/// `SpanPayloadCheckQuery.MatchOperation`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatchOperation {
    Eq,
    Gt,
    Gte,
    Lt,
    Lte,
}

impl std::fmt::Display for PayloadType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            PayloadType::Int => "INT",
            PayloadType::Float => "FLOAT",
            PayloadType::String => "STRING",
        })
    }
}

impl std::fmt::Display for MatchOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            MatchOperation::Eq => "EQ",
            MatchOperation::Gt => "GT",
            MatchOperation::Gte => "GTE",
            MatchOperation::Lt => "LT",
            MatchOperation::Lte => "LTE",
        })
    }
}

/// `SpanPayloadCheckQuery`.
#[derive(Debug, Clone, PartialEq)]
pub struct SpanPayloadCheckQuery {
    pub inner: SpanNode,
    /// `payloadToMatch`: one entry per term the span collects, in collection
    /// order; `None` asks for no payload there.
    pub payload_to_match: Vec<Option<Vec<u8>>>,
    pub payload_type: PayloadType,
    pub operation: MatchOperation,
}

impl SpanPayloadCheckQuery {
    /// `new SpanPayloadCheckQuery(match, payloadToMatch)`: string type,
    /// equality.
    pub fn new(inner: SpanNode, payload_to_match: Vec<Option<Vec<u8>>>) -> Self {
        Self::with(
            inner,
            payload_to_match,
            PayloadType::String,
            MatchOperation::Eq,
        )
    }

    /// `new SpanPayloadCheckQuery(match, payloadToMatch, payloadType,
    /// operation)`.
    pub fn with(
        inner: SpanNode,
        payload_to_match: Vec<Option<Vec<u8>>>,
        payload_type: PayloadType,
        operation: MatchOperation,
    ) -> Self {
        SpanPayloadCheckQuery {
            inner,
            payload_to_match,
            payload_type,
            operation,
        }
    }
}

impl std::fmt::Display for SpanPayloadCheckQuery {
    /// `toString(field)`: each payload as `Term.toString(bytes)` (its UTF-8
    /// text, else the bytes' hex), a `null` one as `null`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SpanPayloadCheckQuery({}, payloadRef: ", self.inner)?;
        for p in &self.payload_to_match {
            match p {
                Some(bytes) => f.write_str(&super::term_to_string(bytes))?,
                None => f.write_str("null")?,
            }
            f.write_str(";")?;
        }
        write!(
            f,
            ", payloadType:{};, operation:{};)",
            self.payload_type, self.operation
        )
    }
}

/// `BitUtil.VH_BE_INT.get(bytes, 0)`.
fn be_int(b: &[u8]) -> Result<i32> {
    let a: [u8; 4] = b.get(..4).and_then(|s| s.try_into().ok()).ok_or_else(|| {
        Error::IllegalArgument(format!("an INT payload needs 4 bytes, got {}", b.len()))
    })?;
    Ok(i32::from_be_bytes(a))
}

/// `BitUtil.VH_BE_FLOAT.get(bytes, 0)`.
fn be_float(b: &[u8]) -> Result<f32> {
    let a: [u8; 4] = b.get(..4).and_then(|s| s.try_into().ok()).ok_or_else(|| {
        Error::IllegalArgument(format!("a FLOAT payload needs 4 bytes, got {}", b.len()))
    })?;
    Ok(f32::from_be_bytes(a))
}

/// `String.compareTo` of the two byte strings decoded as UTF-8 (malformed
/// sequences replaced, as `new String(bytes, UTF_8)` does): by UTF-16 unit.
fn java_string_compare(a: &[u8], b: &[u8]) -> std::cmp::Ordering {
    let (a, b) = (String::from_utf8_lossy(a), String::from_utf8_lossy(b));
    a.encode_utf16().cmp(b.encode_utf16())
}

/// `PayloadMatcherFactory.createMatcherForOpAndType(type, op)
/// .comparePayload(source, payload)`.
///
/// # Errors
/// An `INT` or `FLOAT` comparison of a payload (or threshold) shorter than
/// four bytes, where Java's `VarHandle` read is out of bounds.
pub fn compare_payload(
    payload_type: PayloadType,
    op: MatchOperation,
    source: &[u8],
    payload: &[u8],
) -> Result<bool> {
    use std::cmp::Ordering;
    if op == MatchOperation::Eq {
        return Ok(source == payload);
    }
    Ok(match payload_type {
        PayloadType::Int => {
            let (val, thresh) = (be_int(payload)?, be_int(source)?);
            match op {
                MatchOperation::Lt => val < thresh,
                MatchOperation::Lte => val <= thresh,
                MatchOperation::Gt => val > thresh,
                _ => val >= thresh,
            }
        }
        PayloadType::Float => {
            let (val, thresh) = (be_float(payload)?, be_float(source)?);
            match op {
                MatchOperation::Lt => val < thresh,
                MatchOperation::Lte => val <= thresh,
                MatchOperation::Gt => val > thresh,
                _ => val >= thresh,
            }
        }
        PayloadType::String => {
            let res = java_string_compare(payload, source);
            match op {
                MatchOperation::Lt => res == Ordering::Less,
                MatchOperation::Lte => res != Ordering::Greater,
                MatchOperation::Gt => res == Ordering::Greater,
                _ => res != Ordering::Less,
            }
        }
    })
}

/// `SpanPayloadCheckQuery.PayloadChecker`.
struct PayloadChecker {
    payload_to_match: Arc<[Option<Vec<u8>>]>,
    payload_type: PayloadType,
    operation: MatchOperation,
    upto: usize,
    matches: bool,
    /// A matcher's error, raised after the collection.
    error: Option<Error>,
}

impl SpanCollector for PayloadChecker {
    fn collect_leaf(&mut self, leaf: &mut TermSpans<'_>, _position: i32) -> Result<()> {
        if !self.matches {
            return Ok(());
        }
        if self.upto >= self.payload_to_match.len() {
            self.matches = false;
            return Ok(());
        }
        let payload = leaf.payload()?;
        let want = &self.payload_to_match[self.upto];
        self.upto += 1;
        self.matches = match (want, payload) {
            (None, p) => p.is_none(),
            (Some(_), None) => false,
            (Some(w), Some(p)) => match compare_payload(self.payload_type, self.operation, w, p) {
                Ok(m) => m,
                Err(e) => {
                    self.error = Some(e);
                    false
                }
            },
        };
        Ok(())
    }

    fn reset(&mut self) {
        self.upto = 0;
        self.matches = true;
    }
}

impl SpanFilter for PayloadChecker {
    /// The anonymous `FilterSpans.accept`: collect, then `match()`.
    fn accept<S: super::Spans>(&mut self, candidate: &mut S) -> Result<AcceptStatus> {
        self.reset();
        candidate.collect(self)?;
        if let Some(e) = self.error.take() {
            return Err(e);
        }
        Ok(
            if self.matches && self.upto == self.payload_to_match.len() {
                AcceptStatus::Yes
            } else {
                AcceptStatus::No
            },
        )
    }
}

/// `SpanPayloadCheckWeight.getSpans`.
pub(crate) fn check_spans<'a>(q: &SpanPayloadCheckQuery, inner: BoxSpans<'a>) -> BoxSpans<'a> {
    BoxSpans::boxed(super::FilterSpans::new(
        inner,
        PayloadChecker {
            payload_to_match: q.payload_to_match.clone().into(),
            payload_type: q.payload_type,
            operation: q.operation,
            upto: 0,
            matches: true,
            error: None,
        },
    ))
}

/// `PayloadScoreQuery.PayloadSpans`: the spans unchanged (`accept` is
/// always `YES`), collecting each span's payloads into the function's
/// running score.
pub(crate) struct PayloadSpans<'a> {
    inner: BoxSpans<'a>,
    function: PayloadFunction,
    decoder: PayloadDecoder,
    at_first_in_current_doc: bool,
    start_pos: i32,
    pub(crate) payloads_seen: i32,
    pub(crate) payload_score: f32,
}

/// The collector half of `PayloadSpans`.
struct PayloadCollector<'d> {
    function: PayloadFunction,
    decoder: &'d PayloadDecoder,
    payloads_seen: i32,
    payload_score: f32,
}

impl SpanCollector for PayloadCollector<'_> {
    fn collect_leaf(&mut self, leaf: &mut TermSpans<'_>, _position: i32) -> Result<()> {
        let payload = leaf.payload()?;
        let factor = self.decoder.compute_payload_factor(payload);
        self.payload_score =
            self.function
                .current_score(self.payloads_seen, self.payload_score, factor);
        self.payloads_seen = self.payloads_seen.wrapping_add(1);
        Ok(())
    }

    fn reset(&mut self) {}
}

impl<'a> PayloadSpans<'a> {
    pub(crate) fn new(
        inner: BoxSpans<'a>,
        function: PayloadFunction,
        decoder: PayloadDecoder,
    ) -> Self {
        PayloadSpans {
            inner,
            function,
            decoder,
            at_first_in_current_doc: false,
            start_pos: -1,
            payloads_seen: 0,
            payload_score: 0.0,
        }
    }
}

impl Spans for PayloadSpans<'_> {
    fn doc_id(&self) -> i32 {
        self.inner.doc_id()
    }
    fn next_doc(&mut self) -> Result<i32> {
        self.inner.next_doc()
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        self.inner.advance(target)
    }
    fn cost(&self) -> i64 {
        self.inner.cost()
    }
    /// `FilterSpans`' `twoPhaseCurrentDocMatches` with an `accept` that is
    /// always `YES`: the inner's first span.
    fn matches(&mut self) -> Result<bool> {
        if !self.inner.matches()? {
            return Ok(false);
        }
        self.at_first_in_current_doc = false;
        self.start_pos = self.inner.next_start_position()?;
        if self.start_pos == super::NO_MORE_POSITIONS {
            self.start_pos = -1;
            return Ok(false);
        }
        self.at_first_in_current_doc = true;
        Ok(true)
    }
    fn match_cost(&self) -> f32 {
        self.inner.match_cost()
    }
    fn two_phase(&self) -> bool {
        true
    }
    fn next_start_position(&mut self) -> Result<i32> {
        if self.at_first_in_current_doc {
            self.at_first_in_current_doc = false;
            return Ok(self.start_pos);
        }
        self.start_pos = self.inner.next_start_position()?;
        Ok(self.start_pos)
    }
    // SENTINEL: `-1` = "before the document's first span", `Spans`' own
    // contract (`startPosition`/`endPosition` before `nextStartPosition`);
    // callers compare it as a position, below every real one, as Java's do.
    fn start_position(&self) -> i32 {
        if self.at_first_in_current_doc {
            -1
        } else {
            self.start_pos
        }
    }
    // SENTINEL: `-1` = "before the document's first span", `Spans`' own
    // contract (`startPosition`/`endPosition` before `nextStartPosition`);
    // callers compare it as a position, below every real one, as Java's do.
    fn end_position(&self) -> i32 {
        if self.at_first_in_current_doc {
            -1
        } else if self.start_pos != super::NO_MORE_POSITIONS {
            self.inner.end_position()
        } else {
            super::NO_MORE_POSITIONS
        }
    }
    fn width(&self) -> i32 {
        self.inner.width()
    }
    fn collect(&mut self, collector: &mut dyn SpanCollector) -> Result<()> {
        self.inner.collect(collector)
    }
    /// `doStartCurrentDoc()`.
    fn do_start_current_doc(&mut self) {
        self.payload_score = 0.0;
        self.payloads_seen = 0;
    }
    /// `doCurrentSpans()`: `in.collect(this)`.
    fn do_current_spans(&mut self) -> Result<()> {
        let mut c = PayloadCollector {
            function: self.function,
            decoder: &self.decoder,
            payloads_seen: self.payloads_seen,
            payload_score: self.payload_score,
        };
        self.inner.collect(&mut c)?;
        self.payloads_seen = c.payloads_seen;
        self.payload_score = c.payload_score;
        Ok(())
    }
}

/// `PayloadSpanScorer.getPayloadScore()`: the function's document score,
/// `0` where it is negative or NaN.
pub(crate) fn payload_score(
    function: PayloadFunction,
    payloads_seen: i32,
    payload_score: f32,
) -> f32 {
    let score = function.doc_score(payloads_seen, payload_score);
    if score >= 0.0 {
        score
    } else {
        0.0
    }
}

/// `PayloadSpanScorer.getPayloadExplanation()`.
pub(crate) fn payload_explanation(
    function: PayloadFunction,
    payloads_seen: i32,
    payload_score: f32,
) -> Explanation {
    let expl = function.explain(payloads_seen, payload_score);
    if expl.value < 0.0 {
        Explanation::match_(0.0, "truncated score, max of:")
            .with_details(vec![Explanation::match_(0.0, "minimum score"), expl])
    } else if expl.value.is_nan() {
        Explanation::match_(
            0.0,
            "payload score, computed as (score == NaN ? 0 : score) since NaN is an illegal score from:",
        )
        .with_details(vec![expl])
    } else {
        expl
    }
}

#[cfg(test)]
mod tests;
