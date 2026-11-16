//! `Explanation`-equivalent (`org.apache.lucene.search.Explanation`) and
//! `explain_clause`, this port's `IndexSearcher.explain(query, doc)`
//! counterpart: a per-doc, human-readable breakdown of exactly how a
//! [`crate::Clause`]'s score for one document was computed.
//!
//! **This task does not change any scoring behavior.** Every arm below
//! recomputes a score using the *exact same* functions and arguments the
//! already-verified `search_*_scored` functions in `lib.rs` already call
//! ([`crate::similarity::idf`], [`crate::similarity::tf_norm`],
//! [`crate::term_doc_freqs`], [`crate::dismax_scores`]'s max+tie-breaker
//! formula, etc) — `explain_clause`'s reported top-level `value` is therefore
//! bit-for-bit identical to what `search_term_query_scored`/
//! `search_boolean_query_scored`/`search_disjunction_max_query_scored` already
//! produce for the same doc, not a second, independently-computed
//! approximation. This crate's own unit tests below assert that equality
//! directly (`assert_eq!`, not an epsilon comparison) against those functions'
//! actual output.
//!
//! ## [`Explanation`]'s shape
//!
//! Every `description` string this module emits is real Lucene 10.5.0's
//! verbatim (`"weight(field:term in doc) [BM25Similarity], result of:"`,
//! `"score(freq=..), computed as boost * idf * tf from:"`, `"idf, computed as
//! log(1 + (N - n + 0.5) / (n + 0.5)) from:"`, `"sum of:"`, `"max of:"`, ...)
//! -- downstream tooling parses these, so they are a compatibility contract,
//! not prose. [`Explanation`]'s [`std::fmt::Display`] is likewise a port of
//! `Explanation.toString()`'s two-space-per-level indented rendering.
//!
//! Mirrors real Lucene's `Explanation` class exactly: `value` (the computed
//! score contribution), `description` (what this node represents),
//! `details` (child `Explanation`s the value was derived from), and `matched`
//! (real Lucene's own `isMatch()`/internal `match` boolean — `true` for every
//! node built via [`Explanation::match_`], `false` with `value == 0.0` for
//! every node built via [`Explanation::no_match`], the same
//! `Explanation.match(...)`/`Explanation.noMatch(...)` factory-method split
//! real Lucene's own class provides).
//!
//! ## Which [`crate::Clause`] variants get a "real" explanation vs a flat one
//!
//! - **Real, detailed explanations** (mirroring real Lucene's own
//!   `TermWeight.explain`/`BooleanWeight.explain`/`PhraseWeight.explain`/
//!   `DisjunctionMaxQuery`'s explain/`ConstantScoreWeight.explain`/
//!   `BoostQuery.BoostWeight.explain` as closely as this port's own scoring
//!   math allows): `Clause::Term`, `Clause::Boolean`, `Clause::Phrase`,
//!   `Clause::DisjunctionMax`, `Clause::ConstantScore`, `Clause::Boost`.
//! - **Flat, one-level explanations** ("matches, constant score 1.0" or "no
//!   match" — these clauses have no single term's frequency/idf to break
//!   down further, see each query type's own doc comment in `query.rs` for
//!   why they're unscored): `Clause::Wildcard`, `Clause::Prefix`,
//!   `Clause::Fuzzy`, `Clause::Regexp`.
//! - **The scorer's value, one level** (`SpanWeight.explain`'s value without
//!   its per-term breakdown): `Clause::Span`, `Clause::MultiPhrase`.

use std::collections::HashMap;

use lucene_codecs::blocktree::BlockTreeFields;
use lucene_codecs::postings::{DocInput, PayInput, PosInput};
use lucene_util::fixed_bit_set::FixedBitSet;

use crate::points_query::PointsInput;
use crate::query::{BoostQuery, ConstantScoreQuery, DisjunctionMaxQuery, PhraseQuery, TermQuery};
use crate::{similarity, BooleanQuery, Clause, FieldNorms, GlobalStats, Result};

/// The Rust analogue of real Lucene's `Explanation` — see this module's doc
/// comment for the exact shape and factory-method correspondence.
#[derive(Debug, Clone, PartialEq)]
pub struct Explanation {
    /// Real Lucene's `isMatch()` — `true` for a matching node (built via
    /// [`Explanation::match_`]), `false` for a non-matching node (built via
    /// [`Explanation::no_match`], always paired with `value == 0.0`).
    pub matched: bool,
    /// The computed score contribution this node represents. Always `0.0`
    /// for a non-matching node, same as real `Explanation.noMatch`.
    pub value: f32,
    /// Human-readable description of what this node's `value` represents.
    pub description: String,
    /// Child explanations `value` was derived from — empty for a leaf node.
    pub details: Vec<Explanation>,
    /// The value as the `long` Java's `Explanation` holds for a count (a
    /// term's `n`, a field's `N`), which prints with no decimal point.
    long_value: Option<i64>,
    /// The value as the `double` a function query's explanation holds,
    /// printed as `Double.toString` prints it.
    double_value: Option<f64>,
}

thread_local! {
    /// The segment being explained, `(maxDoc, docBase)`: set by
    /// [`crate::index_searcher::IndexSearcher::explain`] for the length of one
    /// explanation, so a clause that needs the segment's size (a block join's
    /// parent filter) or its base (its description) can read them where this
    /// module's signatures do not carry them.
    static LEAF: std::cell::Cell<Option<(i32, i32)>> = const { std::cell::Cell::new(None) };
}

/// The reader of the segment being explained and the searcher's similarity
/// (when not the default BM25): what a clause reading doc values or the
/// similarity itself (a function query) explains through.
type LeafState = (
    Option<crate::directory_reader::SegmentReader>,
    Option<std::sync::Arc<dyn crate::similarities::Similarity>>,
);

thread_local! {
    static LEAF_READER: std::cell::RefCell<LeafState> =
        const { std::cell::RefCell::new((None, None)) };
}

thread_local! {
    /// The boost the weights being explained were created with
    /// (`createWeight(searcher, scoreMode, boost)`): the product of the
    /// `BoostQuery`s around them, which a term weight's BM25 explanation
    /// reports as its `boost` node and scores with. Set by [`explain_boost`]
    /// for the clauses whose explanation takes it (see [`takes_boost`]).
    static BOOST: std::cell::Cell<f32> = const { std::cell::Cell::new(1.0) };
}

thread_local! {
    /// Whether the weights being explained were created without scores
    /// (`ScoreMode.COMPLETE_NO_SCORES`): `BooleanWeight` creates its
    /// `FILTER` and `MUST_NOT` clauses' weights so. Set by [`with_no_scores`].
    static NO_SCORES: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The boost the clause being explained was weighted with.
fn weight_boost() -> f32 {
    BOOST.with(std::cell::Cell::get)
}

/// Whether the clause being explained was weighted without scores.
fn no_scores() -> bool {
    NO_SCORES.with(std::cell::Cell::get)
}

/// Runs `f` explaining weights created without scores, restoring the
/// previous mode on every exit, an unwind included.
fn with_no_scores<R>(f: impl FnOnce() -> R) -> R {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            NO_SCORES.with(|n| n.set(self.0));
        }
    }
    let _restore = Restore(NO_SCORES.with(|n| n.replace(true)));
    f()
}

/// `similarity.getClass().getSimpleName()` of the searcher being explained:
/// a weight's explanation prints it whatever its scorer.
fn similarity_name() -> String {
    with_leaf_reader(|_, sim| {
        sim.map_or("BM25Similarity", |s| s.simple_name())
            .to_string()
    })
}

/// A weight created without scores, explained through the dummy
/// `SimScorer` its query assigns (`TermWeight`'s scores `0`,
/// `PhraseWeight`'s `1`) and `SimScorer.explain`'s default layout:
/// `weight(q in doc) [Sim], result of:` over `score(freq=f), with freq of:`
/// over `freq`, whatever the searcher's similarity -- only its name shows.
fn explain_dummy_scorer(clause: &Clause, doc: i32, value: f32, freq: Explanation) -> Explanation {
    Explanation::match_(
        value,
        format!(
            "weight({} in {doc}) [{}], result of:",
            describe_clause(clause),
            similarity_name()
        ),
    )
    .with_details(vec![Explanation::match_(
        value,
        format!("score(freq={}), with freq of:", java_float(freq.value)),
    )
    .with_details(vec![freq])])
}

/// Runs `f` with `(max_doc, doc_base)` as the segment being explained,
/// `reader` its reader and `similarity` the searcher's.
pub(crate) fn with_leaf<R>(
    max_doc: Option<i32>,
    doc_base: i32,
    reader: Option<&crate::directory_reader::SegmentReader>,
    similarity: Option<&dyn crate::similarities::Similarity>,
    f: impl FnOnce() -> R,
) -> R {
    /// Restores the previous segment on every exit, an unwind included: a
    /// panic caught at the FFI boundary must not leave a pooled JVM thread
    /// explaining against a stale segment.
    struct Restore {
        leaf: Option<(i32, i32)>,
        reader: Option<LeafState>,
    }
    impl Drop for Restore {
        fn drop(&mut self) {
            LEAF.with(|l| l.set(self.leaf));
            if let Some(state) = self.reader.take() {
                LEAF_READER.with(|r| r.replace(state));
            }
        }
    }
    let state = (reader.cloned(), similarity.and_then(|s| s.shared()));
    // The guard first, holding what it must put back: a swap that panics
    // (the reader state borrowed by an enclosing `with_leaf_reader`) then
    // still restores the half already swapped.
    let mut restore = Restore {
        leaf: LEAF.with(std::cell::Cell::get),
        reader: None,
    };
    LEAF.with(|l| l.set(max_doc.map(|m| (m, doc_base))));
    restore.reader = Some(LEAF_READER.with(|r| r.replace(state)));
    f()
}

/// Runs `f` with `boost` as the boost the clauses it explains were weighted
/// with, restoring the previous one on every exit, an unwind included.
fn with_boost<R>(boost: f32, f: impl FnOnce() -> R) -> R {
    struct Restore(f32);
    impl Drop for Restore {
        fn drop(&mut self) {
            BOOST.with(|b| b.set(self.0));
        }
    }
    let _restore = Restore(BOOST.with(|b| b.replace(boost)));
    f()
}

/// Runs `f` with the reader of the segment being explained and the
/// searcher's similarity, where a caller set them.
pub(crate) fn with_leaf_reader<R>(
    f: impl FnOnce(
        Option<&crate::directory_reader::SegmentReader>,
        Option<&dyn crate::similarities::Similarity>,
    ) -> R,
) -> R {
    LEAF_READER.with(|r| {
        let state = r.borrow();
        f(state.0.as_ref(), state.1.as_deref())
    })
}

/// The segment being explained, if a caller set it.
pub(crate) fn leaf() -> Option<(i32, i32)> {
    LEAF.with(std::cell::Cell::get)
}

impl Explanation {
    /// Real `Explanation.match(value, description, details...)`-equivalent:
    /// builds a matching, leaf explanation (`details` empty; use
    /// [`Self::with_details`] to attach children).
    pub fn match_(value: f32, description: impl Into<String>) -> Self {
        Self {
            matched: true,
            value,
            description: description.into(),
            details: Vec::new(),
            long_value: None,
            double_value: None,
        }
    }

    /// `Explanation.match(long, description)`: a count, printed as Java
    /// prints a `Long` (`7991`, not `7991.0`); [`Self::value`] holds it as a
    /// float.
    pub fn match_long(value: i64, description: impl Into<String>) -> Self {
        Self {
            long_value: Some(value),
            ..Self::match_(value as f32, description)
        }
    }

    /// `Explanation.match(double, description)`: a `double`, printed as
    /// Java prints a `Double`; [`Self::value`] holds it as a float.
    pub fn match_double(value: f64, description: impl Into<String>) -> Self {
        Self {
            double_value: Some(value),
            ..Self::match_(value as f32, description)
        }
    }

    /// `Explanation.match(other.getValue(), description, details)`: a match
    /// holding `other`'s value as `other` holds it (a `Float`, `Long` or
    /// `Double`).
    pub fn match_value_of(other: &Explanation, description: impl Into<String>) -> Self {
        Self {
            matched: true,
            value: other.value,
            description: description.into(),
            details: Vec::new(),
            long_value: other.long_value,
            double_value: other.double_value,
        }
    }

    /// `getValue().toString()`: the value as Java prints the `Number` this
    /// node holds (`7991` for a `Long`, `0.5` for a `Float` or a `Double`).
    pub fn value_string(&self) -> String {
        match (self.long_value, self.double_value) {
            (Some(v), _) => v.to_string(),
            (None, Some(d)) => crate::function::java_double(d),
            (None, None) => java_float(self.value),
        }
    }

    /// `getValue().doubleValue()`.
    pub fn double_value(&self) -> f64 {
        match (self.double_value, self.long_value) {
            (Some(d), _) => d,
            (None, Some(l)) => l as f64,
            _ => f64::from(self.value),
        }
    }

    /// Real `Explanation.noMatch(description, details...)`-equivalent: a
    /// non-matching explanation, `value` fixed at `0.0` — real Lucene's own
    /// convention (a non-match has no score to report).
    pub fn no_match(description: impl Into<String>) -> Self {
        Self {
            matched: false,
            value: 0.0,
            description: description.into(),
            details: Vec::new(),
            long_value: None,
            double_value: None,
        }
    }

    /// Builder-style: attaches `details` (child explanations) to this node.
    pub fn with_details(mut self, details: Vec<Explanation>) -> Self {
        self.details = details;
        self
    }
}

impl std::fmt::Display for Explanation {
    /// Port of real `Explanation.toString()`: one line per node,
    /// `"{value} = {description}"`, each nesting level indented by two
    /// spaces, every line (including the last) terminated by `\n` -- the
    /// exact rendering `IndexSearcher.explain(...).toString()` produces and
    /// that downstream tooling parses.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.fmt_at_depth(f, 0)
    }
}

impl Explanation {
    fn fmt_at_depth(&self, f: &mut std::fmt::Formatter<'_>, depth: usize) -> std::fmt::Result {
        for _ in 0..depth {
            f.write_str("  ")?;
        }
        writeln!(f, "{} = {}", self.value_string(), self.description)?;
        for detail in &self.details {
            detail.fmt_at_depth(f, depth + 1)?;
        }
        Ok(())
    }
}

/// Renders `v` the way Java's `Float.toString`/`String.valueOf(float)` does:
/// the shortest decimal that round-trips, always with a decimal point
/// (`2.0`), in computerized scientific notation outside `[1e-3, 1e7)`
/// (`1.0E-4`), with Java's choices where Rust's formatter differs (a tie
/// goes to the even digit; the smallest subnormals take two digits:
/// `Float.MIN_VALUE` is `1.4E-45`).
pub(crate) fn java_float(v: f32) -> String {
    lucene_util::geo::java_float_string(v)
}

/// Renders `clause` the way real Lucene's `Query.toString()` does for the
/// corresponding Java query class -- the string every `ConstantScoreWeight`/
/// `BooleanWeight` explanation embeds (`"no match on required clause (" +
/// c.query() + ")"`, `getQuery().toString() + " doesn't match id " + doc`,
/// ...). Real Lucene's `toString(String field)` omits the `field:` prefix
/// when the field equals the enclosing "default" field; explanations always
/// call the no-argument `toString()`, i.e. default field `""`, so every
/// field is printed.
pub fn describe_clause(clause: &Clause) -> String {
    fn term(bytes: &[u8]) -> String {
        String::from_utf8_lossy(bytes).into_owned()
    }
    match clause {
        // `FieldExistsQuery.toString`.
        Clause::Exists(q) => format!("FieldExistsQuery [field={}]", q.field),
        // The function queries' own `toString`.
        Clause::Extended(q) => match q.as_ref() {
            #[cfg(any(test, feature = "asserting"))]
            crate::extended_query::ExtendedQuery::Test(q) if q.delegates_query_semantics() => {
                describe_clause(q.query())
            }
            crate::extended_query::ExtendedQuery::Function(f) => format!("{f:?}"),
            crate::extended_query::ExtendedQuery::FunctionRange(f) => format!("{f:?}"),
            crate::extended_query::ExtendedQuery::FunctionMatch(f) => format!("{f:?}"),
            crate::extended_query::ExtendedQuery::FunctionScore(f) => format!("{f:?}"),
            crate::extended_query::ExtendedQuery::Interval(f) => f.to_string(),
            crate::extended_query::ExtendedQuery::Span(f) => f.to_string(),
            _ => q.name().to_string(),
        },
        Clause::Term(q) => format!("{}:{}", q.field, term(&q.term)),
        Clause::Phrase(q) => {
            let body = q
                .terms
                .iter()
                .map(|t| term(t))
                .collect::<Vec<_>>()
                .join(" ");
            let slop = if q.slop == 0 {
                String::new()
            } else {
                format!("~{}", q.slop)
            };
            format!("{}:\"{body}\"{slop}", q.field)
        }
        Clause::MultiPhrase(q) => {
            let body = q
                .term_arrays
                .iter()
                .map(|alts| {
                    if alts.len() == 1 {
                        term(&alts[0])
                    } else {
                        format!(
                            "({})",
                            alts.iter().map(|t| term(t)).collect::<Vec<_>>().join(" ")
                        )
                    }
                })
                .collect::<Vec<_>>()
                .join(" ");
            let slop = if q.slop == 0 {
                String::new()
            } else {
                format!("~{}", q.slop)
            };
            format!("{}:\"{body}\"{slop}", q.field)
        }
        // `BooleanQuery.toString`: a sub-boolean in parentheses, the whole in
        // parentheses only under a minimum-should-match.
        Clause::Boolean(q) => {
            let mut parts = Vec::new();
            for c in &q.must {
                parts.push(format!("+{}", describe_sub_clause(c)));
            }
            // `Occur.FILTER.toString()` is `"#"`.
            for c in &q.filter {
                parts.push(format!("#{}", describe_sub_clause(c)));
            }
            for c in &q.should {
                parts.push(describe_sub_clause(c));
            }
            for c in &q.must_not {
                parts.push(format!("-{}", describe_sub_clause(c)));
            }
            if q.minimum_should_match == 0 {
                parts.join(" ")
            } else {
                format!("({})~{}", parts.join(" "), q.minimum_should_match)
            }
        }
        Clause::DisjunctionMax(q) => {
            let body = q
                .disjuncts
                .iter()
                .map(describe_sub_clause)
                .collect::<Vec<_>>()
                .join(" | ");
            let tie = if q.tie_breaker == 0.0 {
                String::new()
            } else {
                format!("~{}", java_float(q.tie_breaker))
            };
            format!("({body}){tie}")
        }
        Clause::ConstantScore(q) => format!("ConstantScore({})", describe_clause(&q.inner)),
        Clause::Boost(q) => format!("({})^{}", describe_clause(&q.inner), java_float(q.boost)),
        Clause::Wildcard(q) => format!("{}:{}", q.field, term(&q.pattern)),
        Clause::Prefix(q) => format!("{}:{}*", q.field, term(&q.prefix)),
        Clause::Fuzzy(q) => format!("{}:{}~{}", q.field, term(&q.term), q.max_edits),
        Clause::Regexp(q) => format!("{}:/{}/", q.field, q.pattern),
        Clause::Span(q) => describe_span(q),
        Clause::PointsRange(q) => format!("{}:[{} TO {}]", q.field, q.min, q.max),
        Clause::MatchAllDocs(_) => "*:*".to_string(),
        Clause::MatchNoDocs(q) => format!("MatchNoDocsQuery(\"{}\")", q.reason),
        Clause::TermInSet(q) => format!(
            "{}:({})",
            q.field,
            q.terms
                .iter()
                .map(|t| term(t))
                .collect::<Vec<_>>()
                .join(" ")
        ),
    }
}

/// A clause nested in a `BooleanQuery` or a `DisjunctionMaxQuery`: both
/// Java `toString`s wrap a sub-boolean in parentheses ("wrap sub-bools in
/// parens").
fn describe_sub_clause(c: &Clause) -> String {
    match c {
        Clause::Boolean(_) => format!("({})", describe_clause(c)),
        _ => describe_clause(c),
    }
}

/// `describe_clause`'s [`crate::query::SpanQuery`] arm, mirroring real
/// `SpanTermQuery`/`SpanNearQuery`/`SpanOrQuery`'s own `toString`.
fn describe_span(span: &crate::query::SpanQuery) -> String {
    use crate::query::SpanQuery;
    match span {
        SpanQuery::SpanTerm { field, term } => {
            format!("spanTerm({field}:{})", String::from_utf8_lossy(term))
        }
        SpanQuery::SpanNear {
            clauses,
            slop,
            in_order,
        } => format!(
            "spanNear([{}], {slop}, {in_order})",
            clauses
                .iter()
                .map(describe_span)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        SpanQuery::SpanOr { clauses } => format!(
            "spanOr([{}])",
            clauses
                .iter()
                .map(describe_span)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// Real `IndexSearcher.explain(query, doc)`-equivalent for one already-opened
/// segment and one already-resolved [`Clause`] (`query`), matching whatever
/// `must`/`should`/`must_not`/nesting the clause tree describes — see this
/// module's doc comment for exactly which variants get a detailed vs flat
/// explanation, and `lib.rs`'s `search_*_scored` functions this mirrors
/// (`search_term_query_scored`, `search_boolean_query_scored`,
/// `search_phrase_query_scored`, `search_disjunction_max_query_scored`).
///
/// - `doc`: the single doc ID to explain (real Lucene's `explain(query,
///   int doc)` signature takes exactly one doc for exactly this reason —
///   explain is a diagnostic tool for one result, not a bulk scoring path).
/// - `norms`: same contract as [`crate::search_boolean_query_scored`]'s —
///   per-field real norms, falling back to
///   [`crate::similarity::UNNORMED_FIELD_LENGTH`] for an unlisted field or
///   when `norms` itself is `None`.
#[allow(clippy::too_many_arguments)]
pub fn explain_clause(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    clause: &Clause,
    doc: i32,
    norms: Option<&HashMap<String, FieldNorms<'_>>>,
) -> Result<Explanation> {
    explain_clause_with_stats(
        fields, doc_in, pos_in, pay_in, live_docs, None, clause, doc, norms, None,
    )
}

/// [`explain_clause`] with reader-wide statistics (`IndexSearcher.explain`
/// over several segments): a term's `n` and `N`, a phrase's per-term `idf`s
/// and a multi-term or extended query's scorer all read `global` where it
/// holds the term, as the scored search does, so the explanation's value is
/// the score the multi-segment search gives the document. `None` is the
/// segment's own statistics. `points` is the segment's BKD points, which a
/// [`Clause::PointsRange`] needs (without them it reports
/// [`crate::Error::MissingPointsInput`], as [`explain_clause`] always does).
/// A matching boolean's value is its scorer's score for the document, as
/// `BooleanWeight.explain` pulls one: the clauses' sum rounds differently.
#[allow(clippy::too_many_arguments)]
pub fn explain_clause_with_stats(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    points: Option<&PointsInput<'_>>,
    clause: &Clause,
    doc: i32,
    norms: Option<&HashMap<String, FieldNorms<'_>>>,
    global: Option<&GlobalStats>,
) -> Result<Explanation> {
    match clause {
        // `ConstantScoreWeight.explain`: the scorer reaches the document or
        // not (it needs the segment's reader, which `with_leaf` sets).
        Clause::Exists(q) => {
            let name = format!("FieldExistsQuery [field={}]", q.field);
            Ok(
                match scorer_score(
                    fields, doc_in, pos_in, pay_in, live_docs, points, norms, global, clause, doc,
                )? {
                    Some(score) => Explanation::match_(score, name),
                    None => Explanation::no_match(format!("{name} doesn't match id {doc}")),
                },
            )
        }
        // The score the scorer tree gives the document, as one match.
        Clause::Extended(q) => {
            // A test query's weight is a `FilterWeight`: the wrapped
            // query's explanation.
            #[cfg(any(test, feature = "asserting"))]
            if let crate::extended_query::ExtendedQuery::Test(t) = q.as_ref() {
                return explain_clause_with_stats(
                    fields,
                    doc_in,
                    pos_in,
                    pay_in,
                    live_docs,
                    points,
                    t.query(),
                    doc,
                    norms,
                    global,
                );
            }
            if let crate::extended_query::ExtendedQuery::Interval(iq) = q.as_ref() {
                let ctx = interval_context(
                    fields, doc_in, pos_in, pay_in, live_docs, points, norms, global,
                );
                return crate::exec::intervals::explain_interval(&ctx, iq, 1.0, doc);
            }
            if let crate::extended_query::ExtendedQuery::Span(sq) = q.as_ref() {
                return explain_span(
                    fields, doc_in, pos_in, pay_in, live_docs, points, norms, global, sq, 1.0, doc,
                );
            }
            // `SynonymQuery.createWeight` without scores: the rewritten
            // boolean of its terms' `TermQuery`s, also without scores.
            if let crate::extended_query::ExtendedQuery::Synonym(sq) = q.as_ref() {
                if no_scores() {
                    let disjunction = Clause::from(BooleanQuery {
                        should: sq
                            .terms
                            .iter()
                            .map(|(t, _)| Clause::Term(TermQuery::new(sq.field.clone(), t.clone())))
                            .collect(),
                        ..BooleanQuery::default()
                    })
                    .rewrite();
                    return explain_clause_with_stats(
                        fields,
                        doc_in,
                        pos_in,
                        pay_in,
                        live_docs,
                        points,
                        &disjunction,
                        doc,
                        norms,
                        global,
                    );
                }
            }
            if let Some(e) = crate::exec::join::explain_extended(
                fields, doc_in, pos_in, pay_in, live_docs, points, norms, global, q, doc,
            )? {
                return Ok(e);
            }
            let hits = crate::exec::extended::resolve(
                fields, doc_in, pos_in, pay_in, live_docs, points, norms, global, q, true,
            )?;
            Ok(match hits.iter().find(|(d, _)| *d == doc) {
                Some(&(_, score)) => {
                    Explanation::match_(score, format!("{}, result of:", q.name()))
                }
                None => Explanation::no_match(format!("no match on {}", q.name())),
            })
        }
        Clause::Term(query) => {
            let clause_norms = norms.and_then(|m| m.get(&query.field));
            explain_term(fields, doc_in, live_docs, query, doc, clause_norms, global)
        }
        Clause::Phrase(query) => {
            crate::require_phrase_positions(fields, query)?;
            let clause_norms = norms.and_then(|m| m.get(&query.field));
            explain_phrase(
                fields,
                doc_in,
                pos_in,
                pay_in,
                live_docs,
                query,
                doc,
                clause_norms,
                global,
            )
        }
        Clause::Boolean(nested) => explain_boolean(
            fields, doc_in, pos_in, pay_in, live_docs, points, nested, doc, norms, global,
        ),
        Clause::DisjunctionMax(nested) => explain_dismax(
            fields, doc_in, pos_in, pay_in, live_docs, points, nested, doc, norms, global,
        ),
        Clause::ConstantScore(nested) => explain_constant_score(
            fields, doc_in, pos_in, pay_in, live_docs, points, nested, doc,
        ),
        Clause::Boost(nested) => explain_boost(
            fields, doc_in, pos_in, pay_in, live_docs, points, nested, doc, norms, global,
        ),
        Clause::Wildcard(query) => {
            let matched = crate::wildcard_doc_ids(fields, doc_in, live_docs, query)?.contains(&doc);
            Ok(explain_flat_match(matched, clause, doc))
        }
        Clause::Prefix(query) => {
            let matched = crate::prefix_doc_ids(fields, doc_in, live_docs, query)?.contains(&doc);
            Ok(explain_flat_match(matched, clause, doc))
        }
        Clause::Fuzzy(query) => {
            let matched = crate::fuzzy_doc_ids(fields, doc_in, live_docs, query)?.contains(&doc);
            Ok(explain_flat_match(matched, clause, doc))
        }
        Clause::Regexp(query) => {
            let matched = crate::regexp_doc_ids(fields, doc_in, live_docs, query)?.contains(&doc);
            Ok(explain_flat_match(matched, clause, doc))
        }
        // `SpanWeight.explain`: the score the scorer gives the document (no
        // per-term breakdown yet), or `"no matching term"`.
        Clause::Span(query) => {
            let hits = crate::exec::span::resolve_span(
                fields, doc_in, pos_in, pay_in, live_docs, norms, global, query,
            )?;
            Ok(match hits.iter().find(|(d, _)| *d == doc) {
                // `SpanWeight` without scores has no similarity.
                Some(_) if no_scores() => Explanation::match_(
                    0.0,
                    format!("match {} in {doc} without score", describe_clause(clause)),
                ),
                Some(&(_, score)) => Explanation::match_(
                    score,
                    format!(
                        "weight({} in {doc}) [BM25Similarity], result of:",
                        describe_clause(clause)
                    ),
                ),
                None => Explanation::no_match("no matching term"),
            })
        }
        Clause::PointsRange(query) => {
            let matched = crate::points_range_doc_ids(points, live_docs, query)?.contains(&doc);
            Ok(explain_flat_match(matched, clause, doc))
        }
        Clause::MatchAllDocs(query) => {
            let matched = crate::match_all_doc_ids(live_docs, query.max_doc).contains(&doc);
            Ok(explain_flat_match(matched, clause, doc))
        }
        // `MatchNoDocsQuery`'s weight: `Explanation.noMatch(reason)`.
        Clause::MatchNoDocs(q) => Ok(Explanation::no_match(q.reason.clone())),
        Clause::MultiPhrase(query) if no_scores() && !query.term_arrays.is_empty() => {
            // `MultiPhraseQuery.rewrite`: one position is a boolean of its
            // terms, explained as one created without scores.
            if let [terms] = query.term_arrays.as_slice() {
                let disjunction = Clause::from(BooleanQuery {
                    should: terms
                        .iter()
                        .map(|t| Clause::Term(TermQuery::new(query.field.clone(), t.clone())))
                        .collect(),
                    ..BooleanQuery::default()
                })
                .rewrite();
                return explain_clause_with_stats(
                    fields,
                    doc_in,
                    pos_in,
                    pay_in,
                    live_docs,
                    points,
                    &disjunction,
                    doc,
                    norms,
                    global,
                );
            }
            Ok(
                match crate::multi_phrase_freq(
                    fields, doc_in, pos_in, pay_in, live_docs, query, doc,
                )? {
                    Some(freq) => explain_phrase_unscored(clause, doc, freq),
                    None => Explanation::no_match("no matching terms"),
                },
            )
        }
        Clause::MultiPhrase(query) => {
            crate::require_multi_phrase_positions(fields, query)?;
            // No per-position breakdown yet (real `MultiPhraseWeight.explain`
            // has one): this reports the real score the scorer produces for
            // this doc under the searcher's similarity, obtained from the
            // scorer itself so the two can never disagree, rather than a
            // re-derivation that could drift. `PhraseWeight.explain`'s two
            // no-matches: the approximation not on the document ("no
            // matching terms"), or its positions not matching ("no matching
            // phrase").
            with_leaf_reader(|_, similarity| {
                let mut ctx = interval_context(
                    fields, doc_in, pos_in, pay_in, live_docs, points, norms, global,
                );
                ctx.similarity = similarity;
                ctx.max_doc = leaf().map(|(max_doc, _)| max_doc);
                let Some(mut s) = crate::exec::extended::multi_phrase(
                    &ctx,
                    query,
                    1.0,
                    crate::exec::Mode::Complete,
                )?
                else {
                    return Ok(Explanation::no_match("no matching terms"));
                };
                // The scorer is exact (it visits matches only): whether the
                // approximation is on `doc` is asked separately.
                if s.advance(doc)? != doc || !s.matches()? {
                    let on =
                        crate::exec::extended::multi_phrase_approximation_on(&ctx, query, doc)?;
                    return Ok(Explanation::no_match(if on {
                        "no matching phrase"
                    } else {
                        "no matching terms"
                    }));
                }
                Ok(Explanation::match_(
                    s.score()?,
                    format!(
                        "weight({} in {doc}) [{}], result of:",
                        describe_clause(clause),
                        similarity_name()
                    ),
                ))
            })
        }
        Clause::TermInSet(query) => {
            let matched =
                crate::term_in_set_doc_ids(fields, doc_in, live_docs, query)?.contains(&doc);
            Ok(explain_flat_match(matched, clause, doc))
        }
    }
}

/// The segment as a scorer tree sees it, for the weights whose explanation
/// runs their own scorer.
#[allow(clippy::too_many_arguments)]
fn interval_context<'a>(
    fields: &'a BlockTreeFields,
    doc_in: Option<&'a DocInput<'a>>,
    pos_in: Option<&'a PosInput<'a>>,
    pay_in: Option<&'a PayInput<'a>>,
    live_docs: Option<&'a FixedBitSet>,
    points: Option<&'a PointsInput<'a>>,
    norms: Option<&'a HashMap<String, FieldNorms<'a>>>,
    global: Option<&'a GlobalStats>,
) -> crate::exec::LeafContext<'a> {
    crate::exec::LeafContext {
        fields,
        doc_in,
        pos_in,
        pay_in,
        live_docs,
        points,
        norms,
        global,
        max_doc: None,
        cache: None,
        reader: None,
        similarity: None,
    }
}

/// A leaf clause with no per-term breakdown (`Wildcard`/`Prefix`/`Fuzzy`/
/// `Regexp` — see this module's doc comment): matches score exactly
/// `1.0` (same flat constant every `clause_scores` arm for these variants
/// already reports), non-matches are a clean `no_match` at `0.0`.
///
/// The descriptions are real `ConstantScoreWeight.explain`'s verbatim --
/// `getQuery().toString()` on a match, `getQuery() + " doesn't match id " +
/// doc` on a non-match -- since every one of these clauses is a
/// `MultiTermQuery`/`SpanQuery` that real Lucene rewrites to a
/// constant-scoring weight before explaining it.
fn explain_flat_match(matched: bool, clause: &Clause, doc: i32) -> Explanation {
    let described = describe_clause(clause);
    if matched {
        // `ConstantScoreWeight.explain`: `Explanation.match(score,
        // getQuery().toString() + (score == 1f ? "" : "^" + score))` -- these
        // clauses always score a flat `1.0` here, so the `^score` suffix is
        // never appended.
        Explanation::match_(1.0, described)
    } else {
        Explanation::no_match(format!("{described} doesn't match id {doc}"))
    }
}

/// Whether `clause` matches `doc` at all — used by [`explain_boolean`]/
/// [`explain_dismax`]/[`explain_constant_score`] to decide "does this
/// sub-clause participate" without needing its full score breakdown.
#[allow(clippy::too_many_arguments)]
fn clause_matches(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    points: Option<&PointsInput<'_>>,
    clause: &Clause,
    doc: i32,
) -> Result<bool> {
    Ok(
        crate::resolve_clause_docs(fields, doc_in, pos_in, pay_in, live_docs, points, clause)?
            .contains(&doc),
    )
}

/// [`Clause::Term`]'s explanation: mirrors real `TermWeight.explain`/
/// `BM25Scorer.explain` — a `weight(field:term in doc) [BM25Similarity]` node
/// wrapping a `score(freq=...)` node, itself wrapping `idf` (with `n`/`N`
/// leaf details) and `tf` (with `freq`/`k1`/`b`/`dl`/`avgdl` leaf details),
/// every description string verbatim from Java. `value` is computed via the exact same
/// [`similarity::idf`]/[`similarity::tf_norm`] calls, in the same order, as
/// [`crate::term_doc_scores`] — bit-for-bit identical to
/// `search_term_query_scored`'s own output for this doc (verified by this
/// module's own unit tests).
fn explain_term(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &TermQuery,
    doc: i32,
    norms: Option<&FieldNorms<'_>>,
    global: Option<&GlobalStats>,
) -> Result<Explanation> {
    let Some(field_terms) = fields.field(&query.field) else {
        return Ok(Explanation::no_match("no matching term"));
    };
    let Some(stats) = field_terms.try_seek_exact(&query.term)? else {
        return Ok(Explanation::no_match("no matching term"));
    };
    let doc_freqs = crate::term_doc_freqs(fields, doc_in, live_docs, query)?;
    let Some(&(_, freq)) = doc_freqs.iter().find(|&&(d, _)| d == doc) else {
        return Ok(Explanation::no_match("no matching term"));
    };
    // A `TermWeight` without scores reads postings without frequencies
    // (`freq()` is 1) and explains through its dummy `SimScorer` (score 0),
    // under any similarity.
    if no_scores() {
        return Ok(explain_dummy_scorer(
            &Clause::Term(query.clone()),
            doc,
            0.0,
            Explanation::match_(1.0, "freq, occurrences of term within document"),
        ));
    }

    // `TermWeight` takes `searcher.collectionStatistics`/`termStatistics`:
    // reader-wide when the caller gathered them.
    let (doc_freq, doc_count) = match global.and_then(|g| g.term(&query.field, &query.term)) {
        Some(g) => (g.doc_freq, g.doc_count),
        None => (stats.doc_freq as i64, field_terms.doc_count as i64),
    };
    // Under the searcher's similarity, when it is not the default BM25:
    // `docScorer.explain(freq, norm)` of `similarity.scorer(boost,
    // collectionStats, termStats)`.
    if let Some(e) = explain_under_similarity(
        fields,
        global,
        &query.field,
        std::slice::from_ref(&query.term),
        &Clause::Term(query.clone()),
        doc,
        Explanation::match_(freq as f32, "freq, occurrences of term within document"),
        norms,
    )? {
        return Ok(e);
    }
    let (field_length, avg_field_length) = match norms {
        Some(fn_) => (fn_.field_length(doc)?, fn_.avg_field_length),
        None => (
            similarity::UNNORMED_FIELD_LENGTH,
            similarity::UNNORMED_FIELD_LENGTH,
        ),
    };
    let idf = similarity::idf(doc_freq, doc_count);
    // `BM25Scorer.explain`/`explainTF` verbatim: the reported score is
    // `weight - weight / (1 + freq * normInverse)` -- the *same* expression the
    // scorer evaluates, which is why this stays bit-identical to
    // `search_term_query_scored`'s output -- while the `tf` sub-explanation is
    // Lucene's own `1f - 1f / (1 + freq * normInverse)`. Real Lucene notes that
    // it deliberately does not present this as a "product of", because the
    // rewrite introduces a rounding difference against `idf * tf`.
    let norm_inverse = similarity::norm_inverse(
        field_length,
        avg_field_length,
        similarity::DEFAULT_K1,
        similarity::DEFAULT_B,
    );
    let tf_norm = 1.0 - 1.0 / (1.0 + freq as f32 * norm_inverse);
    // `BM25Scorer`'s `weight = boost * idf`.
    let boost = weight_boost();
    let value = similarity::do_score(boost * idf, freq as f32, norm_inverse);

    let idf_explanation = idf_explanation(idf, doc_freq, doc_count);

    let tf_explanation = Explanation::match_(
        tf_norm,
        "tf, computed as freq / (freq + k1 * (1 - b + b * dl / avgdl)) from:",
    )
    .with_details(vec![
        Explanation::match_(freq as f32, "freq, occurrences of term within document"),
        Explanation::match_(similarity::DEFAULT_K1, "k1, term saturation parameter"),
        Explanation::match_(similarity::DEFAULT_B, "b, length normalization parameter"),
        Explanation::match_(field_length, dl_description(field_length)),
        Explanation::match_(avg_field_length, "avgdl, average length of field"),
    ]);

    let mut subs = Vec::with_capacity(3);
    if boost != 1.0 {
        subs.push(Explanation::match_(boost, "boost"));
    }
    subs.push(idf_explanation);
    subs.push(tf_explanation);
    let score_explanation = Explanation::match_(
        value,
        format!(
            "score(freq={}), computed as boost * idf * tf from:",
            java_float(freq as f32)
        ),
    )
    .with_details(subs);

    Ok(Explanation::match_(
        value,
        format!(
            "weight({} in {doc}) [BM25Similarity], result of:",
            describe_clause(&Clause::Term(query.clone()))
        ),
    )
    .with_details(vec![score_explanation]))
}

/// One term's `idf` node, `TFIDFSimilarity.idfExplain(CollectionStatistics,
/// TermStatistics)`.
pub(crate) fn tfidf_idf_explanation(
    tfidf: &dyn crate::similarities::TfIdfSimilarity,
    doc_freq: i64,
    doc_count: i64,
) -> Explanation {
    Explanation::match_(
        tfidf.idf(doc_freq, doc_count),
        "idf, computed as log((docCount+1)/(docFreq+1)) + 1 from:",
    )
    .with_details(vec![
        Explanation::match_long(doc_freq, "docFreq, number of documents containing term"),
        Explanation::match_long(doc_count, "docCount, total number of documents with field"),
    ])
}

/// `TFIDFScorer.explain` (`explainScore`): `idf` is the weight's idf node,
/// `freq` the frequency the weight hands it (a term's `freq`, a phrase's
/// `phraseFreq`), `boost` the weight's, folded into `queryWeight = boost *
/// idf` and reported as its own node when not 1, in the scorer's order of
/// operations (`queryWeight * tf * norm`) over the similarity's own
/// `normTable`.
pub(crate) fn explain_tfidf(
    tfidf: &dyn crate::similarities::TfIdfSimilarity,
    boost: f32,
    idf: Explanation,
    freq: Explanation,
    encoded_norm: i64,
) -> Explanation {
    let freq_value = freq.value;
    let tf = tfidf.tf(freq_value);
    let tf_explanation = Explanation::match_(
        tf,
        format!("tf(freq={}), with freq of:", java_float(freq_value)),
    )
    .with_details(vec![freq]);
    let table = tfidf.norm_table();
    // `normTable[(int) (encodedNorm & 0xFF)]`.
    let norm = table[(encoded_norm & 0xFF) as usize];
    let query_weight = boost * idf.value;
    let value = query_weight * tf * norm;
    let mut subs = Vec::with_capacity(4);
    if boost != 1.0 {
        subs.push(Explanation::match_(boost, "boost"));
    }
    subs.extend([idf, tf_explanation, Explanation::match_(norm, "fieldNorm")]);
    Explanation::match_(
        value,
        format!("score(freq={}), product of:", java_float(freq_value)),
    )
    .with_details(subs)
}

/// `BM25Scorer.explain` of a `BM25Similarity` with its own `k1` and `b`:
/// `value` is the scorer's score, `idf` the weight's idf node, `freq` the
/// frequency the weight hands it; `dl` is `LENGTH_TABLE[norm]` and `avgdl`
/// the collection's (`BM25Similarity.avgFieldLength`), which the scorer's
/// cache was built from.
pub(crate) fn explain_bm25(
    bm25: &crate::similarities::Bm25Similarity,
    boost: f32,
    value: f32,
    idf: Explanation,
    freq: Explanation,
    encoded_norm: i64,
    collection: &crate::similarities::CollectionStatistics,
) -> Explanation {
    let (k1, b) = (bm25.k1(), bm25.b());
    let norm_byte = (encoded_norm & 0xFF) as u8;
    #[allow(clippy::cast_possible_truncation)]
    let avgdl = (collection.sum_total_term_freq as f64 / collection.doc_count as f64) as f32;
    let doclen = lucene_util::small_float::byte4_to_int(norm_byte) as f32;
    let norm_inverse = 1.0 / (k1 * ((1.0 - b) + b * doclen / avgdl));
    let freq_value = freq.value;
    let dl = if norm_byte > 39 {
        "dl, length of field (approximate)"
    } else {
        "dl, length of field"
    };
    let tf = Explanation::match_(
        1.0 - 1.0 / (1.0 + freq_value * norm_inverse),
        "tf, computed as freq / (freq + k1 * (1 - b + b * dl / avgdl)) from:",
    )
    .with_details(vec![
        freq,
        Explanation::match_(k1, "k1, term saturation parameter"),
        Explanation::match_(b, "b, length normalization parameter"),
        Explanation::match_(doclen, dl),
        Explanation::match_(avgdl, "avgdl, average length of field"),
    ]);
    let mut subs = Vec::with_capacity(3);
    if boost != 1.0 {
        subs.push(Explanation::match_(boost, "boost"));
    }
    subs.extend([idf, tf]);
    Explanation::match_(
        value,
        format!(
            "score(freq={}), computed as boost * idf * tf from:",
            java_float(freq_value)
        ),
    )
    .with_details(subs)
}

/// `idfExplain(collectionStats, termStats[])`'s node over one idf node per
/// term: the one term's, or `description` over their sum in a double,
/// narrowed once.
pub(crate) fn summed_idf_explanation(mut idfs: Vec<Explanation>, description: &str) -> Explanation {
    if idfs.len() == 1 {
        return idfs.remove(0);
    }
    let sum: f64 = idfs.iter().map(|e| f64::from(e.value)).sum();
    #[allow(clippy::cast_possible_truncation)]
    let sum = sum as f32;
    Explanation::match_(sum, description).with_details(idfs)
}

/// A term's or a phrase's explanation under the searcher's similarity when
/// it is not the default BM25 (`None` under the default): `TermWeight`'s
/// and `PhraseWeight`'s `weight(q in doc) [Similarity], result of:` over
/// `similarity.scorer(boost, collectionStats, termStats...).explain(freq,
/// norm)`. A `TFIDFSimilarity` and a `BM25Similarity` (here, or unwrapped
/// from a `PerFieldSimilarityWrapper`) explain their breakdowns from the
/// weight's statistics; any other similarity explains through its own
/// [`crate::similarities::SimScorer::explain`]. The value is the scorer's:
/// the hit's score.
#[allow(clippy::too_many_arguments)]
fn explain_under_similarity(
    fields: &BlockTreeFields,
    global: Option<&GlobalStats>,
    field: &str,
    terms: &[Vec<u8>],
    clause: &Clause,
    doc: i32,
    freq: Explanation,
    norms: Option<&FieldNorms<'_>>,
) -> Result<Option<Explanation>> {
    with_leaf_reader(|_, sim| {
        let Some(sim) = sim else {
            return Ok(None);
        };
        let mut ctx = interval_context(fields, None, None, None, None, None, None, global);
        ctx.max_doc = leaf().map(|(max_doc, _)| max_doc);
        let mut entries = Vec::with_capacity(terms.len());
        for term in terms {
            // Present: the caller found every term in the segment.
            if let Some(entry) = crate::exec::extended::term_entry(&ctx, field, term)? {
                entries.push((term, entry));
            }
        }
        let Some((_, first)) = entries.first() else {
            return Ok(None);
        };
        let collection = first.collection_statistics();
        let term_stats: Vec<_> = entries.iter().map(|(t, e)| e.term_statistics(t)).collect();
        // `LeafSimScorer.getNormValue`: `1` for a field without norms.
        let encoded = match norms {
            Some(n) => n.cursor().norm_long(doc)?.unwrap_or(1),
            None => 1,
        };
        let boost = weight_boost();
        let score = if let Some(tfidf) = sim.as_tfidf(field) {
            let idfs = term_stats
                .iter()
                .map(|t| tfidf_idf_explanation(tfidf, t.doc_freq, collection.doc_count))
                .collect();
            let idf = summed_idf_explanation(idfs, "idf(), sum of:");
            explain_tfidf(tfidf, boost, idf, freq, encoded)
        } else {
            let scorer = sim.scorer(field, boost, &collection, &term_stats);
            let value = scorer.score(freq.value, encoded);
            if let Some(bm25) = sim.as_bm25(field) {
                let idfs = term_stats
                    .iter()
                    .map(|t| {
                        let idf = crate::similarities::Bm25Similarity::idf(
                            t.doc_freq,
                            collection.doc_count,
                        );
                        idf_explanation(idf, t.doc_freq, collection.doc_count)
                    })
                    .collect();
                let idf = summed_idf_explanation(idfs, "idf, sum of:");
                explain_bm25(bm25, boost, value, idf, freq, encoded, &collection)
            } else {
                scorer.explain(freq, encoded)
            }
        };
        Ok(Some(
            Explanation::match_(
                score.value,
                format!(
                    "weight({} in {doc}) [{}], result of:",
                    describe_clause(clause),
                    sim.simple_name()
                ),
            )
            .with_details(vec![score]),
        ))
    })
}

/// One term's `idf` node, verbatim from `BM25Similarity.idfExplain(FieldStats,
/// TermStats)` -- the description and both leaf-detail descriptions are real
/// Lucene's exact strings (`N`/`n`, not `docCount`/`docFreq`).
pub(crate) fn idf_explanation(idf: f32, doc_freq: i64, doc_count: i64) -> Explanation {
    Explanation::match_(
        idf,
        "idf, computed as log(1 + (N - n + 0.5) / (n + 0.5)) from:",
    )
    .with_details(vec![
        Explanation::match_long(doc_freq, "n, number of documents containing term"),
        Explanation::match_long(doc_count, "N, total number of documents with field"),
    ])
}

/// `BM25Similarity.BM25Scorer.explainTF`'s `dl` leaf description: real Lucene
/// prints `"dl, length of field (approximate)"` when the *encoded* norm byte
/// is `> 39`, and `"dl, length of field"` otherwise. This port only carries
/// the already-decoded length, but `BM25Similarity.LENGTH_TABLE[i] == i` for
/// every `i <= 39` and the table is monotonically non-decreasing, so
/// `field_length > 39.0` is exactly the same predicate (asserted against the
/// real table in this module's tests).
fn dl_description(field_length: f32) -> &'static str {
    if field_length > 39.0 {
        "dl, length of field (approximate)"
    } else {
        "dl, length of field"
    }
}

/// [`Clause::Phrase`]'s explanation: same shape as [`explain_term`], but the
/// idf is the sum of every constituent term's own idf and `tfNorm` uses the
/// doc's phrase frequency in place of a single term's `freq` — mirroring
/// [`crate::search_phrase_query_scored`]'s exact formula (see that
/// function's doc comment). A single-term phrase delegates straight to
/// [`explain_term`], same degenerate-case convention
/// `search_phrase_query_scored` itself uses.
#[allow(clippy::too_many_arguments)]
fn explain_phrase(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &PhraseQuery,
    doc: i32,
    norms: Option<&FieldNorms<'_>>,
    global: Option<&GlobalStats>,
) -> Result<Explanation> {
    if query.terms.is_empty() {
        return Ok(Explanation::no_match(
            "PhraseQuery with no terms matches nothing",
        ));
    }
    if query.terms.len() == 1 {
        let term_query = TermQuery::new(query.field.clone(), query.terms[0].clone());
        return explain_term(fields, doc_in, live_docs, &term_query, doc, norms, global);
    }
    let Some(field_terms) = fields.field(&query.field) else {
        return Ok(Explanation::no_match("no matching terms"));
    };
    let Some(pos_in) = pos_in else {
        return Err(crate::Error::MissingPosInput);
    };

    // `BM25Similarity.idfExplain(TermStatistics[])` sums in a double and
    // casts once: an `f32` sum of three or more idfs can be an ulp off.
    let mut idf_acc = 0.0f64;
    let mut idf_details = Vec::with_capacity(query.terms.len());
    for term in &query.terms {
        let Some(stats) = field_terms.try_seek_exact(term)? else {
            return Ok(Explanation::no_match("no matching terms"));
        };
        let (doc_freq, doc_count) = match global.and_then(|g| g.term(&query.field, term)) {
            Some(g) => (g.doc_freq, g.doc_count),
            None => (stats.doc_freq as i64, field_terms.doc_count as i64),
        };
        let term_idf = similarity::idf(doc_freq, doc_count);
        idf_acc += f64::from(term_idf);
        idf_details.push(idf_explanation(term_idf, doc_freq, doc_count));
    }
    let idf_sum = idf_acc as f32;

    let mut per_term_docs: Vec<Vec<i32>> = Vec::with_capacity(query.terms.len());
    let mut per_term_maps: Vec<HashMap<i32, Vec<i32>>> = Vec::with_capacity(query.terms.len());
    for term in &query.terms {
        let Some((docs, positions, spans)) = crate::term_doc_positions(
            fields,
            doc_in,
            pos_in,
            pay_in,
            live_docs,
            &query.field,
            term,
        )?
        else {
            return Ok(Explanation::no_match("no matching terms"));
        };
        // Not a hot path (one document, on demand): rebuild the doc -> positions
        // map the explain code expects from the aligned vectors the phrase
        // reader now returns.
        per_term_maps.push(
            docs.iter()
                .copied()
                .zip(spans.iter())
                .map(|(d, &(start, end))| (d, positions[start as usize..end as usize].to_vec()))
                .collect::<std::collections::HashMap<i32, Vec<i32>>>(),
        );
        per_term_docs.push(docs);
    }

    if !per_term_docs.iter().all(|docs| docs.contains(&doc)) {
        return Ok(Explanation::no_match("no matching terms"));
    }

    let term_positions: Vec<Vec<i32>> = per_term_maps
        .iter()
        .map(|m| {
            m.get(&doc)
                .cloned()
                .expect("doc came from the conjunction of every term's own doc list")
        })
        .collect();
    let term_positions: Vec<&[i32]> = term_positions.iter().map(|v| v.as_slice()).collect();
    // `PhraseScorer.score()`'s frequency: `ExactPhraseMatcher`'s match count
    // for `slop == 0`, `SloppyPhraseMatcher`'s summed `1/(1+matchLength)`
    // otherwise -- see `crate::sloppy_phrase::sloppy_phrase_freq`.
    let phrase_freq = if query.slop == 0 {
        crate::phrase_freq_exact(&term_positions) as f32
    } else {
        crate::sloppy_phrase::sloppy_phrase_freq(
            &term_positions,
            &crate::sloppy_phrase::PhraseRepeats::for_phrase(&query.terms),
            query.slop,
        )
    };
    if phrase_freq == 0.0 {
        return Ok(Explanation::no_match("no matching phrase"));
    }
    // `PhraseWeight` without scores: no term statistics, so `getStats` is
    // `null` and the weight's dummy `SimScorer` scores every match `1`.
    if no_scores() {
        return Ok(explain_phrase_unscored(
            &Clause::Phrase(query.clone()),
            doc,
            phrase_freq,
        ));
    }
    // Under the searcher's similarity, when it is not the default BM25:
    // `PhraseWeight.explain`'s `docScorer.explain(phraseFreq, norm)`.
    if let Some(e) = explain_under_similarity(
        fields,
        global,
        &query.field,
        &query.terms,
        &Clause::Phrase(query.clone()),
        doc,
        Explanation::match_(
            phrase_freq,
            format!("phraseFreq={}", java_float(phrase_freq)),
        ),
        norms,
    )? {
        return Ok(e);
    }

    let (field_length, avg_field_length) = match norms {
        Some(fn_) => (fn_.field_length(doc)?, fn_.avg_field_length),
        None => (
            similarity::UNNORMED_FIELD_LENGTH,
            similarity::UNNORMED_FIELD_LENGTH,
        ),
    };
    // Same `doScore`/`explainTF` split as `explain_term` above.
    let norm_inverse = similarity::norm_inverse(
        field_length,
        avg_field_length,
        similarity::DEFAULT_K1,
        similarity::DEFAULT_B,
    );
    let tf_norm = 1.0 - 1.0 / (1.0 + phrase_freq * norm_inverse);
    let value = similarity::do_score(idf_sum, phrase_freq, norm_inverse);

    // `BM25Similarity.idfExplain(FieldStats, TermStats[])` -- the phrase's idf
    // is the sum of every constituent term's own idf, reported under Java's
    // exact `"idf, sum of:"` description with one child per term.
    let idf_node = Explanation::match_(idf_sum, "idf, sum of:").with_details(idf_details);

    // `PhraseWeight.explain` builds `Explanation.match(freq, "phraseFreq=" +
    // freq)` and hands it to the very same `BM25Scorer.explain`/`explainTF`
    // pair `explain_term` uses, so the `tf`/`score` nodes below are the
    // term case's strings verbatim, only with the phrase freq inside.
    let tf_explanation = Explanation::match_(
        tf_norm,
        "tf, computed as freq / (freq + k1 * (1 - b + b * dl / avgdl)) from:",
    )
    .with_details(vec![
        Explanation::match_(
            phrase_freq,
            format!("phraseFreq={}", java_float(phrase_freq)),
        ),
        Explanation::match_(similarity::DEFAULT_K1, "k1, term saturation parameter"),
        Explanation::match_(similarity::DEFAULT_B, "b, length normalization parameter"),
        Explanation::match_(field_length, dl_description(field_length)),
        Explanation::match_(avg_field_length, "avgdl, average length of field"),
    ]);

    let score_explanation = Explanation::match_(
        value,
        format!(
            "score(freq={}), computed as boost * idf * tf from:",
            java_float(phrase_freq)
        ),
    )
    .with_details(vec![idf_node, tf_explanation]);

    Ok(Explanation::match_(
        value,
        format!(
            "weight({} in {doc}) [BM25Similarity], result of:",
            describe_clause(&Clause::Phrase(query.clone()))
        ),
    )
    .with_details(vec![score_explanation]))
}

/// `PhraseWeight.explain` of a weight created without scores (a
/// `PhraseQuery`'s or a `MultiPhraseQuery`'s): its dummy `SimScorer`, which
/// scores `1`, over the matcher's frequency.
fn explain_phrase_unscored(clause: &Clause, doc: i32, freq: f32) -> Explanation {
    explain_dummy_scorer(
        clause,
        doc,
        1.0,
        Explanation::match_(freq, format!("phraseFreq={}", java_float(freq))),
    )
}

/// [`Clause::Boolean`]'s explanation: mirrors real `BooleanWeight.explain` --
/// `no_match` when the doc doesn't satisfy `must`'s conjunction / the
/// `should`'s disjunction (when `must` is empty) / `minimum_should_match`, or
/// falls in `must_not`'s exclusion (see [`crate::matched_boolean_docs`], the
/// same matched-doc-set computation `search_boolean_query`/
/// `search_boolean_query_scored` already use); otherwise a `sum of:` node
/// whose value is the sum of every `must` clause's own explanation (always
/// included, since a matching doc satisfies every `must` clause by
/// definition) plus every `should` clause that itself matches this doc --
/// exactly [`crate::search_boolean_query_scored`]'s own summation (`must`
/// chained with `should`, via [`crate::clause_scores`]), so `value` is
/// bit-for-bit identical to that function's output for this doc.
#[allow(clippy::too_many_arguments)]
fn explain_boolean(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    points: Option<&PointsInput<'_>>,
    query: &BooleanQuery,
    doc: i32,
    norms: Option<&HashMap<String, FieldNorms<'_>>>,
    global: Option<&GlobalStats>,
) -> Result<Explanation> {
    // `BooleanWeight.explain`'s exact control flow: build every clause's own
    // explanation first, tracking `fail` (a required clause that didn't match,
    // or a prohibited clause that did), `matchCount` and `shouldMatchCount`,
    // then pick one of Java's four outcomes from those counters.
    let mut details: Vec<Explanation> = Vec::new();
    let mut failing_optionals: Vec<Explanation> = Vec::new();
    let mut fail = false;
    let mut match_count = 0usize;
    let mut should_match_count = 0usize;
    let mut total = 0.0f32;

    for clause in &query.must {
        let e = explain_clause_with_stats(
            fields, doc_in, pos_in, pay_in, live_docs, points, clause, doc, norms, global,
        )?;
        if e.matched {
            match_count += 1;
            total += e.value;
            details.push(e);
        } else {
            fail = true;
            details.push(
                Explanation::no_match(format!(
                    "no match on required clause ({})",
                    describe_clause(clause)
                ))
                .with_details(vec![e]),
            );
        }
    }
    // `BooleanWeight.explain`'s `FILTER` arm, verbatim:
    //
    //     subs.add(Explanation.match(0f, "match on required clause, product of:",
    //         Explanation.match(0f, Occur.FILTER + " clause"), e));
    //
    // `Occur.FILTER + " clause"` is `"# clause"`. The wrapper's value is `0f`,
    // which is what keeps a filter clause out of the parent's sum -- the
    // clause's own explanation `e` survives as a child for diagnosis, but its
    // value is not what the parent adds up. `matchCount` counts it (Java
    // increments for every non-prohibited match); `shouldMatchCount` does not.
    for clause in &query.filter {
        // A function-score query that does not score is its wrapped query
        // (`FunctionScoreQuery.createWeight` returns the inner weight).
        let unscored = match clause {
            Clause::Extended(e) => match e.as_ref() {
                crate::extended_query::ExtendedQuery::FunctionScore(f) => f.in_query.as_ref(),
                _ => clause,
            },
            _ => clause,
        };
        let e = with_no_scores(|| {
            explain_clause_with_stats(
                fields, doc_in, pos_in, pay_in, live_docs, points, unscored, doc, norms, global,
            )
        })?;
        if e.matched {
            match_count += 1;
            details.push(
                Explanation::match_(0.0, "match on required clause, product of:")
                    .with_details(vec![Explanation::match_(0.0, "# clause"), e]),
            );
        } else {
            fail = true;
            details.push(
                Explanation::no_match(format!(
                    "no match on required clause ({})",
                    describe_clause(clause)
                ))
                .with_details(vec![e]),
            );
        }
    }
    for clause in &query.should {
        let e = explain_clause_with_stats(
            fields, doc_in, pos_in, pay_in, live_docs, points, clause, doc, norms, global,
        )?;
        if e.matched {
            match_count += 1;
            should_match_count += 1;
            total += e.value;
            details.push(e);
        } else {
            failing_optionals.push(
                Explanation::no_match(format!(
                    "no match on optional clause ({})",
                    describe_clause(clause)
                ))
                .with_details(vec![e]),
            );
        }
    }
    for clause in &query.must_not {
        let e = with_no_scores(|| {
            explain_clause_with_stats(
                fields, doc_in, pos_in, pay_in, live_docs, points, clause, doc, norms, global,
            )
        })?;
        if e.matched {
            fail = true;
            details.push(
                Explanation::no_match(format!(
                    "match on prohibited clause ({})",
                    describe_clause(clause)
                ))
                .with_details(vec![e]),
            );
        }
    }

    if fail {
        return Ok(Explanation::no_match(
            "Failure to meet condition(s) of required/prohibited clause(s)",
        )
        .with_details(details));
    }
    if match_count == 0 {
        details.extend(failing_optionals);
        return Ok(Explanation::no_match("No matching clauses").with_details(details));
    }
    if should_match_count < query.minimum_should_match {
        details.extend(failing_optionals);
        return Ok(Explanation::no_match(format!(
            "Failure to match minimum number of optional clauses: {}, matched: {should_match_count}",
            query.minimum_should_match
        ))
        .with_details(details));
    }
    // `BooleanWeight.explain`: "in order to make sure that explanations have
    // the same value as the score, we pull a scorer and use it to compute the
    // score" -- the sum above rounds differently from the scorer's.
    let clause = Clause::Boolean(Box::new(query.clone()));
    let value = scorer_score(
        fields, doc_in, pos_in, pay_in, live_docs, points, norms, global, &clause, doc,
    )?
    .unwrap_or(total);
    Ok(Explanation::match_(value, "sum of:").with_details(details))
}

/// `weight.scorer(context)` advanced to `doc`: its score, `None` when the
/// scorer does not match it. The weight is created without scores where the
/// clause being explained was (a filter's or an exclusion's subtree), so a
/// boolean there sums its clauses' dummy scores.
#[allow(clippy::too_many_arguments)]
fn scorer_score(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    points: Option<&PointsInput<'_>>,
    norms: Option<&HashMap<String, FieldNorms<'_>>>,
    global: Option<&GlobalStats>,
    clause: &Clause,
    doc: i32,
) -> Result<Option<f32>> {
    with_leaf_reader(|reader, similarity| {
        let ctx = crate::exec::LeafContext {
            fields,
            doc_in,
            pos_in,
            pay_in,
            live_docs,
            points,
            norms,
            global,
            max_doc: leaf().map(|(max_doc, _)| max_doc),
            cache: None,
            reader,
            similarity,
        };
        // `BooleanWeight.explain`/`ConstantScoreWeight.explain` pull the
        // weight's own scorer, which no `AssertingWeight` wraps (a
        // no-scores weight's own scorer is scored, its sub-scorers are not).
        let Some(mut s) = crate::exec::build::build_unchecked(
            &ctx,
            clause,
            weight_boost(),
            if no_scores() {
                crate::exec::Mode::NoScores
            } else {
                crate::exec::Mode::Complete
            },
            false,
        )?
        else {
            return Ok(None);
        };
        if crate::exec::exact_advance(&mut *s, doc)? != doc {
            return Ok(None);
        }
        Ok(Some(s.score()?))
    })
}

/// [`Clause::DisjunctionMax`]'s explanation: mirrors real
/// `DisjunctionMaxQuery`'s explain -- the matching disjunct with the highest
/// score, plus `tie_breaker * sum(every other matching disjunct's score)`;
/// see [`crate::dismax_scores`]'s doc comment for the exact formula this
/// mirrors (`max + tieBreaker * sum(rest)`), computed here from the same
/// per-disjunct explanations (in `query.disjuncts`' own order, same
/// summation order `dismax_scores` uses, so floating-point results match
/// bit-for-bit).
#[allow(clippy::too_many_arguments)]
fn explain_dismax(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    points: Option<&PointsInput<'_>>,
    query: &DisjunctionMaxQuery,
    doc: i32,
    norms: Option<&HashMap<String, FieldNorms<'_>>>,
    global: Option<&GlobalStats>,
) -> Result<Explanation> {
    // `DisjunctionMaxWeight.explain`: sub-explanations of every *matching*
    // disjunct on a match, of every non-matching one on a no-match.
    let mut subs_on_match = Vec::new();
    let mut subs_on_no_match = Vec::new();
    for clause in &query.disjuncts {
        let e = explain_clause_with_stats(
            fields, doc_in, pos_in, pay_in, live_docs, points, clause, doc, norms, global,
        )?;
        if e.matched {
            subs_on_match.push(e);
        } else if subs_on_match.is_empty() {
            subs_on_no_match.push(e);
        }
    }
    if subs_on_match.is_empty() {
        return Ok(Explanation::no_match("No matching clause").with_details(subs_on_no_match));
    }

    let value = crate::dismax_value(subs_on_match.iter().map(|e| e.value), query.tie_breaker);

    // `DisjunctionMaxWeight.explain`'s exact description switch.
    let desc = if query.tie_breaker == 0.0 {
        "max of:".to_string()
    } else {
        format!(
            "max plus {} times others of:",
            java_float(query.tie_breaker)
        )
    };
    Ok(Explanation::match_(value, desc).with_details(subs_on_match))
}

/// [`Clause::ConstantScore`]'s explanation: mirrors real
/// `ConstantScoreWeight.explain` -- matches iff the wrapped clause matches
/// (its own score discarded entirely), always scoring exactly
/// `nested.score`, same as [`crate::clause_scores`]'s `Clause::ConstantScore`
/// arm.
#[allow(clippy::too_many_arguments)]
fn explain_constant_score(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    points: Option<&PointsInput<'_>>,
    nested: &ConstantScoreQuery,
    doc: i32,
) -> Result<Explanation> {
    if !clause_matches(
        fields,
        doc_in,
        pos_in,
        pay_in,
        live_docs,
        points,
        &nested.inner,
        doc,
    )? {
        return Ok(Explanation::no_match(format!(
            "ConstantScore({}) doesn't match id {doc}",
            describe_clause(&nested.inner)
        )));
    }
    // `ConstantScoreWeight.explain`: `getQuery().toString() + (score == 1f ?
    // "" : "^" + score)`.
    let described = format!("ConstantScore({})", describe_clause(&nested.inner));
    let description = if nested.score == 1.0 {
        described
    } else {
        format!("{described}^{}", java_float(nested.score))
    };
    Ok(Explanation::match_(nested.score, description))
}

/// [`Clause::Boost`]'s explanation: mirrors real `BoostQuery.BoostWeight.
/// explain` -- matches iff the wrapped clause matches, scoring the wrapped
/// clause's own score multiplied by `boost`, same as [`crate::clause_scores`]'s
/// `Clause::Boost` arm (`inner.value * nested.boost`, same multiplication
/// order, so bit-for-bit identical).
#[allow(clippy::too_many_arguments)]
fn explain_boost(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    points: Option<&PointsInput<'_>>,
    nested: &BoostQuery,
    doc: i32,
    norms: Option<&HashMap<String, FieldNorms<'_>>>,
    global: Option<&GlobalStats>,
) -> Result<Explanation> {
    // An interval query's weight takes the boosts around it itself.
    let mut boost = nested.boost;
    let mut inner: &Clause = &nested.inner;
    while let Clause::Boost(b) = inner {
        boost *= b.boost;
        inner = &b.inner;
    }
    if let Clause::Extended(e) = inner {
        if let crate::extended_query::ExtendedQuery::Interval(iq) = e.as_ref() {
            let ctx = interval_context(
                fields, doc_in, pos_in, pay_in, live_docs, points, norms, global,
            );
            return crate::exec::intervals::explain_interval(&ctx, iq, boost, doc);
        }
        // A span query's weight takes the boost into its similarity.
        if let crate::extended_query::ExtendedQuery::Span(sq) = e.as_ref() {
            return explain_span(
                fields, doc_in, pos_in, pay_in, live_docs, points, norms, global, sq, boost, doc,
            );
        }
    }
    // Term weights take the boost into their similarity's scorer
    // (`similarity.scorer(boost, ...)`), and a boolean or dis-max of them
    // hands it down (`BooleanWeight` creates its clauses' weights with it):
    // explained with the boost, not as a product with it.
    if takes_boost(inner) {
        return with_boost(weight_boost() * boost, || {
            explain_clause_with_stats(
                fields, doc_in, pos_in, pay_in, live_docs, points, inner, doc, norms, global,
            )
        });
    }
    // A function query's weight takes the boost itself.
    if let Some(e) = crate::exec::function::explain_boosted(
        fields,
        doc_in,
        pos_in,
        pay_in,
        live_docs,
        points,
        norms,
        global,
        &nested.inner,
        nested.boost,
        doc,
    )? {
        return Ok(e);
    }
    let inner = explain_clause_with_stats(
        fields,
        doc_in,
        pos_in,
        pay_in,
        live_docs,
        points,
        &nested.inner,
        doc,
        norms,
        global,
    )?;
    if !inner.matched {
        return Ok(Explanation::no_match(format!(
            "{} doesn't match id {doc}",
            describe_clause(&Clause::Boost(Box::new(nested.clone())))
        )));
    }
    let value = inner.value * nested.boost;
    Ok(Explanation::match_(value, "product of:")
        .with_details(vec![inner, Explanation::match_(nested.boost, "boost")]))
}

/// `SpanWeight.explain` over the segment as a scorer tree sees it, with the
/// searcher's similarity.
#[allow(clippy::too_many_arguments)]
fn explain_span(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    points: Option<&PointsInput<'_>>,
    norms: Option<&HashMap<String, FieldNorms<'_>>>,
    global: Option<&GlobalStats>,
    q: &crate::spans::SpanNode,
    boost: f32,
    doc: i32,
) -> Result<Explanation> {
    with_leaf_reader(|_, similarity| {
        let mut ctx = interval_context(
            fields, doc_in, pos_in, pay_in, live_docs, points, norms, global,
        );
        ctx.similarity = similarity;
        ctx.max_doc = leaf().map(|(max_doc, _)| max_doc);
        // A span query's weight without scores has no similarity
        // (`buildSimWeight` with no term states).
        if no_scores() {
            return crate::exec::spans::explain_span_unscored(&ctx, q, doc);
        }
        crate::exec::spans::explain_span_node(&ctx, q, boost, doc)
    })
}

/// Whether `clause`'s explanation takes its weight's boost (the term
/// weights, under any similarity, and the booleans, dis-maxes and boosts of
/// them).
fn takes_boost(clause: &Clause) -> bool {
    match clause {
        Clause::Term(_) | Clause::MatchNoDocs(_) => true,
        Clause::Boost(b) => takes_boost(&b.inner),
        // A boolean's `FILTER` and `MUST_NOT` weights are created without
        // scores: whatever they are, the boost reaches no explanation of
        // theirs.
        Clause::Boolean(b) => b.must.iter().chain(&b.should).all(takes_boost),
        Clause::DisjunctionMax(d) => d.disjuncts.iter().all(takes_boost),
        _ => false,
    }
}

#[cfg(test)]
mod tests;
