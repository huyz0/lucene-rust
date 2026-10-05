//! `lucene-queries`' intervals (`org.apache.lucene.queries.intervals`,
//! Lucene 10.5.0): [`IntervalsSource`] and its factory [`Intervals`],
//! [`IntervalQuery`] and its scoring functions, the minimum-interval
//! iterators ([`iterators`]), their `Matches` ([`mod@matches`]) and
//! `IntervalBuilder`'s analyzed-text sources ([`builder`]).
//!
//! # The shape
//!
//! Java's `IntervalsSource` subclasses are the variants of one enum,
//! [`IntervalsSource`], each with the Java class's `toString`, `minExtent`,
//! `pullUpDisjunctions` and `equals`. The factory functions are
//! [`Intervals`]' associated functions, named as Java names them; the ones
//! that pull disjunctions up (`phrase`, `maxgaps`, `containing`,
//! `containedBy`, `notContaining`, `notContainedBy`, `within`, `before`,
//! `after`) can fail as Java's do when the expansion passes
//! `IndexSearcher.getMaxClauseCount()`, so they return a [`Result`].
//!
//! The iterators are Java's, method for method: an
//! [`iterators::IntervalIterator`] per source, composed per segment, moved
//! document by document and interval by interval. [`IntervalQuery`] scores
//! a document as `IntervalScorer` does: the sloppy frequency of its
//! intervals, `sum(1 / max(width - minExtent + 1, 1))` rounded to `float`
//! at each step, through the saturation (`pivot`) or sigmoid (`pivot`,
//! `exp`) function.
//!
//! # What differs from Java
//!
//! - `DisjunctionIntervalsSource` keeps its sources in a `HashSet`, so
//!   Java's order of a disjunction's iterators is its sources' hash order
//!   (`BytesRef.hashCode` is seeded per JVM). Here they keep their first
//!   occurrence's order. Which source a disjunction reports when two of
//!   them hold the same interval -- its `gaps()`, its matches' sub-matches --
//!   follows that order in both; the intervals themselves, and so every
//!   hit and score, do not depend on it.
//! - `Intervals.term(term, Predicate<BytesRef>)`'s predicate is a
//!   [`PayloadFilter`], and like Java's `equals` it is not compared.
//! - A term's positions are read a document at a time (a pulsed singleton
//!   term's one document only when it is live); a payload-filtered term's
//!   occurrences, payloads included, likewise.

use std::fmt;
use std::sync::Arc;

use lucene_util::automaton::Automaton;

use crate::extended_query::{MultiTermSource, TermRangeQuery, MAX_CLAUSE_COUNT};
use crate::query::{PrefixQuery, RegexpQuery, WildcardQuery};
use crate::{Error, Result};

pub mod builder;
pub mod iterators;
pub mod matches;

/// `Intervals.DEFAULT_MAX_EXPANSIONS`.
pub const DEFAULT_MAX_EXPANSIONS: usize = 128;

/// `IntervalIterator.NO_MORE_INTERVALS`.
pub const NO_MORE_INTERVALS: i32 = i32::MAX;

/// The predicate of `Intervals.term(term, Predicate<BytesRef>)`: called with
/// each position's payload, `None` where the position has none (Java's
/// `getPayload() == null`).
#[derive(Clone)]
pub struct PayloadFilter(pub Arc<PayloadPredicate>);

/// A payload filter's predicate.
pub type PayloadPredicate = dyn Fn(Option<&[u8]>) -> bool + Send + Sync;

impl PayloadFilter {
    pub fn new(f: impl Fn(Option<&[u8]>) -> bool + Send + Sync + 'static) -> Self {
        PayloadFilter(Arc::new(f))
    }

    pub(crate) fn test(&self, payload: Option<&[u8]>) -> bool {
        (self.0)(payload)
    }
}

impl fmt::Debug for PayloadFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("PayloadFilter")
    }
}

/// The terms a `MultiTermIntervalsSource` expands to, before its field is
/// known: each is the automaton `Intervals`' factory compiles.
#[derive(Debug, Clone)]
pub enum MultiTermPattern {
    /// `Intervals.prefix`: `PrefixQuery.toAutomaton(prefix)`.
    Prefix(Vec<u8>),
    /// `Intervals.wildcard`: `WildcardQuery.toAutomaton`.
    Wildcard(Vec<u8>),
    /// `Intervals.regexp`: `new RegExp(text).toAutomaton()`.
    Regexp(String),
    /// `Intervals.range`: `TermRangeQuery.toAutomaton`.
    Range {
        lower: Option<Vec<u8>>,
        upper: Option<Vec<u8>>,
        include_lower: bool,
        include_upper: bool,
    },
    /// `Intervals.fuzzyTerm` / `Intervals.multiterm(CompiledAutomaton)`:
    /// an automaton over code points (`binary == false`) or bytes.
    Automaton {
        automaton: Arc<Automaton>,
        binary: bool,
    },
}

impl PartialEq for MultiTermPattern {
    fn eq(&self, other: &Self) -> bool {
        use MultiTermPattern as P;
        match (self, other) {
            (P::Prefix(a), P::Prefix(b)) | (P::Wildcard(a), P::Wildcard(b)) => a == b,
            (P::Regexp(a), P::Regexp(b)) => a == b,
            (
                P::Range {
                    lower: l1,
                    upper: u1,
                    include_lower: il1,
                    include_upper: iu1,
                },
                P::Range {
                    lower: l2,
                    upper: u2,
                    include_lower: il2,
                    include_upper: iu2,
                },
            ) => l1 == l2 && u1 == u2 && il1 == il2 && iu1 == iu2,
            (
                P::Automaton {
                    automaton: a1,
                    binary: b1,
                },
                P::Automaton {
                    automaton: a2,
                    binary: b2,
                },
            ) => b1 == b2 && **a1 == **a2,
            _ => false,
        }
    }
}

impl MultiTermPattern {
    /// The term enumeration over `field` (`CompiledAutomaton.getTermsEnum`).
    pub(crate) fn source(&self, field: &str) -> MultiTermSource {
        match self {
            MultiTermPattern::Prefix(p) => {
                MultiTermSource::Prefix(PrefixQuery::new(field, p.clone()))
            }
            MultiTermPattern::Wildcard(p) => {
                MultiTermSource::Wildcard(WildcardQuery::new(field, p.clone()))
            }
            MultiTermPattern::Regexp(p) => {
                MultiTermSource::Regexp(RegexpQuery::new(field, p.clone()))
            }
            MultiTermPattern::Range {
                lower,
                upper,
                include_lower,
                include_upper,
            } => MultiTermSource::TermRange(TermRangeQuery::new(
                field,
                lower.clone(),
                upper.clone(),
                *include_lower,
                *include_upper,
            )),
            MultiTermPattern::Automaton { automaton, binary } => {
                MultiTermSource::Automaton(crate::extended_query::AutomatonQuery {
                    field: field.to_string(),
                    automaton: Arc::clone(automaton),
                    binary: *binary,
                })
            }
        }
    }
}

/// `FilteredIntervalsSource`'s two filters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntervalFilterKind {
    /// `MaxGaps`: `it.gaps() <= n`.
    MaxGaps(i32),
    /// `MaxWidth`: `(it.end() - it.start()) + 1 <= n`.
    MaxWidth(i32),
}

/// `IntervalsSource`: one variant per Java implementation.
#[derive(Debug, Clone)]
pub enum IntervalsSource {
    /// `TermIntervalsSource`.
    Term(Vec<u8>),
    /// `PayloadFilteredTermIntervalsSource`.
    PayloadFilteredTerm {
        term: Vec<u8>,
        filter: PayloadFilter,
    },
    /// `BlockIntervalsSource` (a phrase): already flattened.
    Block(Vec<IntervalsSource>),
    /// `DisjunctionIntervalsSource`: already simplified, and whether it
    /// pulls its sources up into an enclosing gap-sensitive source.
    Disjunction {
        sources: Vec<IntervalsSource>,
        pull_up: bool,
    },
    /// `OrderedIntervalsSource`.
    Ordered(Vec<IntervalsSource>),
    /// `UnorderedIntervalsSource`.
    Unordered(Vec<IntervalsSource>),
    /// `RepeatingIntervalsSource`: `count` copies of `source`, named
    /// `ORDERED`/`UNORDERED` when it is the whole of one (the name is not
    /// part of equality, as in Java).
    Repeating {
        source: Box<IntervalsSource>,
        count: i32,
        name: Option<&'static str>,
    },
    /// `FilteredIntervalsSource`: `MaxGaps` or `MaxWidth`.
    Filtered {
        source: Box<IntervalsSource>,
        filter: IntervalFilterKind,
    },
    /// `ExtendedIntervalsSource`.
    Extended {
        source: Box<IntervalsSource>,
        before: i32,
        after: i32,
    },
    /// `OffsetIntervalsSource`: one position before (`before`) or after
    /// each interval of `source`.
    Offset {
        source: Box<IntervalsSource>,
        before: bool,
    },
    /// `FixedFieldIntervalsSource`.
    FixedField {
        field: String,
        source: Box<IntervalsSource>,
    },
    /// `NoMatchIntervalsSource`.
    NoMatch(String),
    /// `ContainingIntervalsSource`.
    Containing {
        big: Box<IntervalsSource>,
        small: Box<IntervalsSource>,
    },
    /// `ContainedByIntervalsSource`.
    ContainedBy {
        small: Box<IntervalsSource>,
        big: Box<IntervalsSource>,
    },
    /// `NotContainingIntervalsSource`.
    NotContaining {
        minuend: Box<IntervalsSource>,
        subtrahend: Box<IntervalsSource>,
    },
    /// `NotContainedByIntervalsSource`.
    NotContainedBy {
        minuend: Box<IntervalsSource>,
        subtrahend: Box<IntervalsSource>,
    },
    /// `OverlappingIntervalsSource`.
    Overlapping {
        source: Box<IntervalsSource>,
        reference: Box<IntervalsSource>,
    },
    /// `NonOverlappingIntervalsSource`.
    NonOverlapping {
        minuend: Box<IntervalsSource>,
        subtrahend: Box<IntervalsSource>,
    },
    /// `MinimumShouldMatchIntervalsSource`.
    MinimumShouldMatch {
        sources: Vec<IntervalsSource>,
        min_should_match: i32,
    },
    /// `MultiTermIntervalsSource`.
    MultiTerm {
        pattern: MultiTermPattern,
        max_expansions: usize,
        /// The string `toString` and the expansion error show.
        name: String,
    },
}

fn same_set(a: &[IntervalsSource], b: &[IntervalsSource]) -> bool {
    a.len() == b.len() && a.iter().all(|x| b.contains(x)) && b.iter().all(|x| a.contains(x))
}

impl PartialEq for IntervalsSource {
    /// Each Java class's `equals`.
    fn eq(&self, other: &Self) -> bool {
        use IntervalsSource as S;
        match (self, other) {
            (S::Term(a), S::Term(b)) => a == b,
            // `PayloadFilteredTermIntervalsSource.equals` compares the term only.
            (S::PayloadFilteredTerm { term: a, .. }, S::PayloadFilteredTerm { term: b, .. }) => {
                a == b
            }
            (S::Block(a), S::Block(b))
            | (S::Ordered(a), S::Ordered(b))
            | (S::Unordered(a), S::Unordered(b)) => a == b,
            // A `HashSet`'s equality: the same sources, in any order.
            (S::Disjunction { sources: a, .. }, S::Disjunction { sources: b, .. }) => {
                same_set(a, b)
            }
            (
                S::Repeating {
                    source: a,
                    count: ca,
                    ..
                },
                S::Repeating {
                    source: b,
                    count: cb,
                    ..
                },
            ) => ca == cb && a == b,
            (
                S::Filtered {
                    source: a,
                    filter: fa,
                },
                S::Filtered {
                    source: b,
                    filter: fb,
                },
            ) => fa == fb && a == b,
            (
                S::Extended {
                    source: a,
                    before: ba,
                    after: aa,
                },
                S::Extended {
                    source: b,
                    before: bb,
                    after: ab,
                },
            ) => ba == bb && aa == ab && a == b,
            (
                S::Offset {
                    source: a,
                    before: ba,
                },
                S::Offset {
                    source: b,
                    before: bb,
                },
            ) => ba == bb && a == b,
            (
                S::FixedField {
                    field: fa,
                    source: a,
                },
                S::FixedField {
                    field: fb,
                    source: b,
                },
            ) => fa == fb && a == b,
            (S::NoMatch(a), S::NoMatch(b)) => a == b,
            (S::Containing { big: a1, small: a2 }, S::Containing { big: b1, small: b2 })
            | (S::ContainedBy { small: a1, big: a2 }, S::ContainedBy { small: b1, big: b2 })
            | (
                S::NotContaining {
                    minuend: a1,
                    subtrahend: a2,
                },
                S::NotContaining {
                    minuend: b1,
                    subtrahend: b2,
                },
            )
            | (
                S::NotContainedBy {
                    minuend: a1,
                    subtrahend: a2,
                },
                S::NotContainedBy {
                    minuend: b1,
                    subtrahend: b2,
                },
            )
            | (
                S::Overlapping {
                    source: a1,
                    reference: a2,
                },
                S::Overlapping {
                    source: b1,
                    reference: b2,
                },
            )
            | (
                S::NonOverlapping {
                    minuend: a1,
                    subtrahend: a2,
                },
                S::NonOverlapping {
                    minuend: b1,
                    subtrahend: b2,
                },
            ) => a1 == b1 && a2 == b2,
            (
                S::MinimumShouldMatch {
                    sources: a,
                    min_should_match: ma,
                },
                S::MinimumShouldMatch {
                    sources: b,
                    min_should_match: mb,
                },
            ) => ma == mb && a == b,
            (
                S::MultiTerm {
                    pattern: pa,
                    max_expansions: ma,
                    name: na,
                },
                S::MultiTerm {
                    pattern: pb,
                    max_expansions: mb,
                    name: nb,
                },
            ) => ma == mb && na == nb && pa == pb,
            _ => false,
        }
    }
}

/// Java's `String.compareTo`: UTF-16 code unit order.
fn java_string_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    a.encode_utf16().cmp(b.encode_utf16())
}

fn join(sources: &[IntervalsSource]) -> String {
    sources
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

impl fmt::Display for IntervalsSource {
    /// Each Java class's `toString`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use IntervalsSource as S;
        match self {
            S::Term(t) => f.write_str(&String::from_utf8_lossy(t)),
            S::PayloadFilteredTerm { term, .. } => {
                write!(f, "PAYLOAD_FILTERED({})", String::from_utf8_lossy(term))
            }
            S::Block(s) => write!(f, "BLOCK({})", join(s)),
            S::Ordered(s) => write!(f, "ORDERED({})", join(s)),
            S::Unordered(s) => write!(f, "UNORDERED({})", join(s)),
            S::Disjunction { sources, .. } => {
                let mut parts: Vec<String> = sources.iter().map(ToString::to_string).collect();
                parts.sort_by(|a, b| java_string_cmp(a, b));
                write!(f, "or({})", parts.join(","))
            }
            S::Repeating {
                source,
                count,
                name,
            } => {
                let s = source.to_string();
                let mut out = s.clone();
                for _ in 1..*count {
                    out.push(',');
                    out.push_str(&s);
                }
                match name {
                    Some(n) => write!(f, "{n}({out})"),
                    None => f.write_str(&out),
                }
            }
            S::Filtered { source, filter } => match filter {
                IntervalFilterKind::MaxGaps(n) => write!(f, "MAXGAPS/{n}({source})"),
                IntervalFilterKind::MaxWidth(n) => write!(f, "MAXWIDTH/{n}({source})"),
            },
            S::Extended {
                source,
                before,
                after,
            } => write!(f, "EXTEND({source},{before},{after})"),
            S::Offset { source, before } => {
                if *before {
                    write!(f, "PRECEDING({source})")
                } else {
                    write!(f, "FOLLOWING({source})")
                }
            }
            S::FixedField { field, source } => write!(f, "FIELD({field},{source})"),
            S::NoMatch(reason) => write!(f, "NOMATCH({reason})"),
            S::Containing { big, small } => write!(f, "CONTAINING({big},{small})"),
            S::ContainedBy { small, big } => write!(f, "CONTAINED_BY({small},{big})"),
            S::NotContaining {
                minuend,
                subtrahend,
            } => write!(f, "NOT_CONTAINING({minuend},{subtrahend})"),
            S::NotContainedBy {
                minuend,
                subtrahend,
            } => write!(f, "NOT_CONTAINED_BY({minuend},{subtrahend})"),
            S::Overlapping { source, reference } => write!(f, "OVERLAPPING({source},{reference})"),
            S::NonOverlapping {
                minuend,
                subtrahend,
            } => write!(f, "NON_OVERLAPPING({minuend},{subtrahend})"),
            S::MinimumShouldMatch {
                sources,
                min_should_match,
            } => write!(f, "AtLeast({}~{min_should_match})", join(sources)),
            S::MultiTerm { name, .. } => write!(f, "MultiTerm({name})"),
        }
    }
}

/// Pushes `s` unless an equal source is already there (a Java `Set`'s add).
fn push_distinct(out: &mut Vec<IntervalsSource>, s: IntervalsSource) {
    if !out.contains(&s) {
        out.push(s);
    }
}

impl IntervalsSource {
    /// `IntervalsSource.minExtent()`: the minimum width of an interval this
    /// source returns. Java's `int` sums wrap; so do these.
    pub fn min_extent(&self) -> i32 {
        use IntervalsSource as S;
        match self {
            S::Term(_) | S::PayloadFilteredTerm { .. } | S::Offset { .. } | S::MultiTerm { .. } => {
                1
            }
            S::NoMatch(_) => 0,
            S::Block(s) | S::Ordered(s) | S::Unordered(s) => s
                .iter()
                .fold(0i32, |acc, x| acc.wrapping_add(x.min_extent())),
            S::Disjunction { sources, .. } => sources
                .iter()
                .fold(i32::MAX, |acc, x| acc.min(x.min_extent())),
            S::Repeating { source, .. } | S::Filtered { source, .. } => source.min_extent(),
            S::Extended {
                source,
                before,
                after,
            } => {
                let m = before
                    .wrapping_add(source.min_extent())
                    .wrapping_add(*after);
                if m < 0 {
                    i32::MAX
                } else {
                    m
                }
            }
            S::FixedField { source, .. } => source.min_extent(),
            S::Containing { big, .. } => big.min_extent(),
            S::ContainedBy { small, .. } => small.min_extent(),
            S::Overlapping { source, .. } => source.min_extent(),
            S::NotContaining { minuend, .. }
            | S::NotContainedBy { minuend, .. }
            | S::NonOverlapping { minuend, .. } => minuend.min_extent(),
            S::MinimumShouldMatch {
                sources,
                min_should_match,
            } => {
                let mut extents: Vec<i32> = sources.iter().map(Self::min_extent).collect();
                extents.sort_unstable();
                let take = usize::try_from(*min_should_match).unwrap_or(0);
                extents
                    .iter()
                    .take(take)
                    .fold(0i32, |acc, x| acc.wrapping_add(*x))
            }
        }
    }

    /// `IntervalsSource.pullUpDisjunctions()`.
    pub fn pull_up_disjunctions(&self) -> Result<Vec<IntervalsSource>> {
        use IntervalsSource as S;
        Ok(match self {
            S::Disjunction { sources, pull_up } => {
                if *pull_up {
                    sources.clone()
                } else {
                    vec![self.clone()]
                }
            }
            // `Disjunctions.pullUp(subSources, OrderedIntervalsSource::new)`:
            // the constructor, not `build`.
            S::Ordered(subs) => pull_up_list(subs, |l| Ok(S::Ordered(l)))?,
            S::Unordered(subs) => pull_up_list(subs, |l| Ok(S::Unordered(l)))?,
            S::Filtered {
                source,
                filter: IntervalFilterKind::MaxWidth(w),
            } => {
                let w = *w;
                pull_up_one(source, |s| S::Filtered {
                    source: Box::new(s),
                    filter: IntervalFilterKind::MaxWidth(w),
                })?
            }
            S::Extended {
                source,
                before,
                after,
            } => {
                let inner = source.pull_up_disjunctions()?;
                if inner.is_empty() {
                    vec![self.clone()]
                } else {
                    let mut out = Vec::new();
                    for s in inner {
                        push_distinct(
                            &mut out,
                            S::Extended {
                                source: Box::new(s),
                                before: *before,
                                after: *after,
                            },
                        );
                    }
                    out
                }
            }
            S::FixedField { field, source } => {
                let inner = source.pull_up_disjunctions()?;
                if inner.len() == 1 {
                    vec![self.clone()]
                } else {
                    let mut out = Vec::new();
                    for s in inner {
                        push_distinct(
                            &mut out,
                            S::FixedField {
                                field: field.clone(),
                                source: Box::new(s),
                            },
                        );
                    }
                    out
                }
            }
            S::Containing { big, small } => pull_up_one(big, |s| S::Containing {
                big: Box::new(s),
                small: small.clone(),
            })?,
            S::ContainedBy { small, big } => pull_up_one(big, |s| S::ContainedBy {
                small: small.clone(),
                big: Box::new(s),
            })?,
            S::Overlapping { source, reference } => pull_up_list(
                &[source.as_ref().clone(), reference.as_ref().clone()],
                |mut l| {
                    let reference = l.pop().unwrap_or_else(|| S::NoMatch(String::new()));
                    let source = l.pop().unwrap_or_else(|| S::NoMatch(String::new()));
                    Ok(S::Overlapping {
                        source: Box::new(source),
                        reference: Box::new(reference),
                    })
                },
            )?,
            _ => vec![self.clone()],
        })
    }
}

/// `Disjunctions.splitDisjunctions`: the source's pulled-up disjuncts, those
/// of `minExtent() == 1` merged into one disjunction first.
fn split_disjunctions(source: &IntervalsSource) -> Result<Vec<IntervalsSource>> {
    let mut singletons = Vec::new();
    let mut non_singletons = Vec::new();
    for disj in source.pull_up_disjunctions()? {
        if disj.min_extent() == 1 {
            singletons.push(disj);
        } else {
            non_singletons.push(disj);
        }
    }
    let mut split = Vec::new();
    if !singletons.is_empty() {
        split.push(Intervals::or(singletons)?);
    }
    split.extend(non_singletons);
    Ok(split)
}

/// `Disjunctions.pullUp(List, Function)`: every combination of the sources'
/// disjuncts, each made into one source by `function`.
fn pull_up_list(
    sources: &[IntervalsSource],
    function: impl Fn(Vec<IntervalsSource>) -> Result<IntervalsSource>,
) -> Result<Vec<IntervalsSource>> {
    let mut rewritten: Vec<Vec<IntervalsSource>> = vec![Vec::new()];
    for source in sources {
        let disjuncts = split_disjunctions(source)?;
        if disjuncts.len() == 1 {
            for l in &mut rewritten {
                l.push(disjuncts[0].clone());
            }
        } else {
            if rewritten.len().saturating_mul(disjuncts.len()) > MAX_CLAUSE_COUNT {
                return Err(Error::IllegalArgument(
                    "Too many disjunctions to expand".into(),
                ));
            }
            let mut to_add = Vec::with_capacity(rewritten.len().saturating_mul(disjuncts.len()));
            for disj in &disjuncts {
                for sub in &rewritten {
                    let mut l = sub.clone();
                    l.push(disj.clone());
                    to_add.push(l);
                }
            }
            rewritten = to_add;
        }
    }
    rewritten.into_iter().map(function).collect()
}

/// `Disjunctions.pullUp(IntervalsSource, Function)`.
fn pull_up_one(
    source: &IntervalsSource,
    function: impl Fn(IntervalsSource) -> IntervalsSource,
) -> Result<Vec<IntervalsSource>> {
    Ok(split_disjunctions(source)?
        .into_iter()
        .map(function)
        .collect())
}

/// `Intervals`: the factory functions, named as Java names them.
pub struct Intervals;

impl Intervals {
    /// `Intervals.term(BytesRef)`.
    pub fn term(term: impl Into<Vec<u8>>) -> IntervalsSource {
        IntervalsSource::Term(term.into())
    }

    /// `Intervals.term(BytesRef, Predicate<BytesRef>)`: the term's
    /// positions whose payload `filter` accepts.
    pub fn term_with_payload_filter(
        term: impl Into<Vec<u8>>,
        filter: PayloadFilter,
    ) -> IntervalsSource {
        IntervalsSource::PayloadFilteredTerm {
            term: term.into(),
            filter,
        }
    }

    /// `Intervals.phrase(String...)`.
    pub fn phrase_terms(terms: &[&str]) -> Result<IntervalsSource> {
        if terms.len() == 1 {
            return Ok(Self::term(terms[0]));
        }
        Self::phrase(terms.iter().map(|t| Self::term(*t)).collect())
    }

    /// `Intervals.phrase(IntervalsSource...)`: `BlockIntervalsSource.build`.
    pub fn phrase(sources: Vec<IntervalsSource>) -> Result<IntervalsSource> {
        if sources.len() == 1 {
            return Ok(sources
                .into_iter()
                .next()
                .unwrap_or_else(|| Self::no_intervals("")));
        }
        let blocks = pull_up_list(&sources, |l| Ok(block(l)))?;
        Self::or(blocks)
    }

    /// `Intervals.or(IntervalsSource...)`: a disjunction that pulls itself
    /// up into an enclosing gap-sensitive source.
    pub fn or(sources: Vec<IntervalsSource>) -> Result<IntervalsSource> {
        Self::or_with_rewrite(true, sources)
    }

    /// `Intervals.or(boolean rewrite, IntervalsSource...)`
    /// (`DisjunctionIntervalsSource.create`).
    pub fn or_with_rewrite(
        rewrite: bool,
        sources: Vec<IntervalsSource>,
    ) -> Result<IntervalsSource> {
        let simplified = simplify(sources)?;
        if simplified.len() == 1 {
            return Ok(simplified
                .into_iter()
                .next()
                .unwrap_or_else(|| Self::no_intervals("")));
        }
        Ok(IntervalsSource::Disjunction {
            sources: simplified,
            pull_up: rewrite,
        })
    }

    fn multi_term(
        pattern: MultiTermPattern,
        max_expansions: usize,
        name: String,
    ) -> Result<IntervalsSource> {
        if max_expansions > MAX_CLAUSE_COUNT {
            return Err(Error::IllegalArgument(format!(
                "maxExpansions [{max_expansions}] cannot be greater than \
                 BooleanQuery.getMaxClauseCount [{MAX_CLAUSE_COUNT}]"
            )));
        }
        Ok(IntervalsSource::MultiTerm {
            pattern,
            max_expansions,
            name,
        })
    }

    /// `Intervals.prefix(BytesRef, int)`.
    pub fn prefix(prefix: impl Into<Vec<u8>>, max_expansions: usize) -> Result<IntervalsSource> {
        let prefix = prefix.into();
        let name = format!("{}*", String::from_utf8_lossy(&prefix));
        Self::multi_term(MultiTermPattern::Prefix(prefix), max_expansions, name)
    }

    /// `Intervals.wildcard(BytesRef, int)`.
    pub fn wildcard(pattern: impl Into<Vec<u8>>, max_expansions: usize) -> Result<IntervalsSource> {
        let pattern = pattern.into();
        let name = String::from_utf8_lossy(&pattern).into_owned();
        Self::multi_term(MultiTermPattern::Wildcard(pattern), max_expansions, name)
    }

    /// `Intervals.regexp(BytesRef, int)`.
    pub fn regexp(pattern: &str, max_expansions: usize) -> Result<IntervalsSource> {
        // The pattern must parse, as `new RegExp(..).toAutomaton()` does here.
        lucene_codecs::regexp::RegexpPattern::new(pattern.as_bytes())?;
        Self::multi_term(
            MultiTermPattern::Regexp(pattern.to_string()),
            max_expansions,
            pattern.to_string(),
        )
    }

    /// `Intervals.range(BytesRef, BytesRef, boolean, boolean, int)`.
    pub fn range(
        lower: Option<Vec<u8>>,
        upper: Option<Vec<u8>>,
        include_lower: bool,
        include_upper: bool,
        max_expansions: usize,
    ) -> Result<IntervalsSource> {
        let name = format!(
            "{{{},{}}}",
            lower.as_deref().map_or_else(
                || "* ".to_string(),
                |l| String::from_utf8_lossy(l).into_owned()
            ),
            upper.as_deref().map_or_else(
                || "*".to_string(),
                |u| String::from_utf8_lossy(u).into_owned()
            ),
        );
        Self::multi_term(
            MultiTermPattern::Range {
                lower,
                upper,
                include_lower,
                include_upper,
            },
            max_expansions,
            name,
        )
    }

    /// `Intervals.fuzzyTerm(String, int, int, boolean, int)`: the terms
    /// within `max_edits` of `term` (`FuzzyQuery.getFuzzyAutomaton`).
    pub fn fuzzy_term(
        term: &str,
        max_edits: i32,
        prefix_length: i32,
        transpositions: bool,
        max_expansions: usize,
    ) -> Result<IntervalsSource> {
        if !(0..=2).contains(&max_edits) {
            return Err(Error::IllegalArgument(format!(
                "max edits must be 0..2, inclusive; got: {max_edits}"
            )));
        }
        if prefix_length < 0 {
            return Err(Error::IllegalArgument(
                "prefixLength cannot be less than 0".into(),
            ));
        }
        let code_points: Vec<char> = term.chars().collect();
        let p = usize::try_from(prefix_length)
            .unwrap_or(0)
            .min(code_points.len());
        let prefix: String = code_points[..p].iter().collect();
        let suffix: Vec<i32> = code_points[p..].iter().map(|&c| c as i32).collect();
        let lev = lucene_util::automaton::LevenshteinAutomata::with_alphabet(
            suffix,
            0x10FFFF,
            transpositions,
        )
        .map_err(|e| Error::IllegalArgument(format!("{e:?}")))?;
        let automaton = lev
            .to_automaton_with_prefix(max_edits, &prefix)
            .ok_or_else(|| Error::IllegalArgument(format!("max edits {max_edits}")))?;
        Self::multi_term(
            MultiTermPattern::Automaton {
                automaton: Arc::new(automaton),
                binary: false,
            },
            max_expansions,
            format!("{term}~{max_edits}"),
        )
    }

    /// `Intervals.multiterm(CompiledAutomaton, int, String)`.
    pub fn multiterm(
        automaton: Automaton,
        binary: bool,
        max_expansions: usize,
        pattern: &str,
    ) -> Result<IntervalsSource> {
        Self::multi_term(
            MultiTermPattern::Automaton {
                automaton: Arc::new(automaton),
                binary,
            },
            max_expansions,
            pattern.to_string(),
        )
    }

    /// `Intervals.maxwidth(int, IntervalsSource)`.
    pub fn maxwidth(width: i32, source: IntervalsSource) -> IntervalsSource {
        IntervalsSource::Filtered {
            source: Box::new(source),
            filter: IntervalFilterKind::MaxWidth(width),
        }
    }

    /// `Intervals.maxgaps(int, IntervalsSource)`: each pulled-up disjunct
    /// filtered.
    pub fn maxgaps(gaps: i32, source: IntervalsSource) -> Result<IntervalsSource> {
        let filtered = source
            .pull_up_disjunctions()?
            .into_iter()
            .map(|s| IntervalsSource::Filtered {
                source: Box::new(s),
                filter: IntervalFilterKind::MaxGaps(gaps),
            })
            .collect();
        Self::or(filtered)
    }

    /// `Intervals.extend(IntervalsSource, int, int)`.
    pub fn extend(source: IntervalsSource, before: i32, after: i32) -> IntervalsSource {
        IntervalsSource::Extended {
            source: Box::new(source),
            before,
            after,
        }
    }

    /// `Intervals.ordered(IntervalsSource...)` (`OrderedIntervalsSource.build`).
    pub fn ordered(sources: Vec<IntervalsSource>) -> IntervalsSource {
        if sources.len() == 1 {
            return sources
                .into_iter()
                .next()
                .unwrap_or_else(|| Self::no_intervals(""));
        }
        // `deduplicate`: runs of equal sources become one repeating source.
        let mut deduplicated: Vec<IntervalsSource> = Vec::new();
        let mut current: Vec<IntervalsSource> = Vec::new();
        for source in sources {
            if current.is_empty() || current[0] == source {
                current.push(source);
            } else {
                deduplicated.push(repeating(current[0].clone(), current.len()));
                current.clear();
                current.push(source);
            }
        }
        if let Some(first) = current.first() {
            deduplicated.push(repeating(first.clone(), current.len()));
        }
        name_single(&mut deduplicated, "ORDERED");
        if deduplicated.len() == 1 {
            return deduplicated
                .into_iter()
                .next()
                .unwrap_or_else(|| Self::no_intervals(""));
        }
        IntervalsSource::Ordered(deduplicated)
    }

    /// `Intervals.unordered(IntervalsSource...)` (`UnorderedIntervalsSource.build`).
    pub fn unordered(sources: Vec<IntervalsSource>) -> IntervalsSource {
        if sources.len() == 1 {
            return sources
                .into_iter()
                .next()
                .unwrap_or_else(|| Self::no_intervals(""));
        }
        // `deduplicate`: a `LinkedHashMap` of counts, first occurrence order.
        let mut counts: Vec<(IntervalsSource, usize)> = Vec::new();
        for source in sources {
            match counts.iter_mut().find(|(s, _)| *s == source) {
                Some((_, n)) => *n = n.saturating_add(1),
                None => counts.push((source, 1)),
            }
        }
        let mut deduplicated: Vec<IntervalsSource> =
            counts.into_iter().map(|(s, n)| repeating(s, n)).collect();
        name_single(&mut deduplicated, "UNORDERED");
        if deduplicated.len() == 1 {
            return deduplicated
                .into_iter()
                .next()
                .unwrap_or_else(|| Self::no_intervals(""));
        }
        IntervalsSource::Unordered(deduplicated)
    }

    /// `Intervals.unorderedNoOverlaps(a, b)`.
    pub fn unordered_no_overlaps(
        a: IntervalsSource,
        b: IntervalsSource,
    ) -> Result<IntervalsSource> {
        Self::or(vec![
            Self::ordered(vec![a.clone(), b.clone()]),
            Self::ordered(vec![b, a]),
        ])
    }

    /// `Intervals.fixField(String, IntervalsSource)`.
    pub fn fix_field(field: impl Into<String>, source: IntervalsSource) -> IntervalsSource {
        IntervalsSource::FixedField {
            field: field.into(),
            source: Box::new(source),
        }
    }

    /// `Intervals.nonOverlapping(minuend, subtrahend)`.
    pub fn non_overlapping(
        minuend: IntervalsSource,
        subtrahend: IntervalsSource,
    ) -> IntervalsSource {
        IntervalsSource::NonOverlapping {
            minuend: Box::new(minuend),
            subtrahend: Box::new(subtrahend),
        }
    }

    /// `Intervals.overlapping(source, reference)`.
    pub fn overlapping(source: IntervalsSource, reference: IntervalsSource) -> IntervalsSource {
        IntervalsSource::Overlapping {
            source: Box::new(source),
            reference: Box::new(reference),
        }
    }

    /// `Intervals.notWithin(minuend, positions, subtrahend)`.
    pub fn not_within(
        minuend: IntervalsSource,
        positions: i32,
        subtrahend: IntervalsSource,
    ) -> IntervalsSource {
        Self::non_overlapping(minuend, Self::extend(subtrahend, positions, positions))
    }

    /// `Intervals.within(source, positions, reference)`.
    pub fn within(
        source: IntervalsSource,
        positions: i32,
        reference: IntervalsSource,
    ) -> Result<IntervalsSource> {
        Self::contained_by(source, Self::extend(reference, positions, positions))
    }

    /// `Intervals.notContaining(minuend, subtrahend)`.
    pub fn not_containing(
        minuend: IntervalsSource,
        subtrahend: IntervalsSource,
    ) -> Result<IntervalsSource> {
        let pulled = pull_up_one(&minuend, |s| IntervalsSource::NotContaining {
            minuend: Box::new(s),
            subtrahend: Box::new(subtrahend.clone()),
        })?;
        Self::or(pulled)
    }

    /// `Intervals.containing(big, small)`.
    pub fn containing(big: IntervalsSource, small: IntervalsSource) -> Result<IntervalsSource> {
        let pulled = pull_up_one(&big, |s| IntervalsSource::Containing {
            big: Box::new(s),
            small: Box::new(small.clone()),
        })?;
        Self::or(pulled)
    }

    /// `Intervals.notContainedBy(small, big)`.
    pub fn not_contained_by(
        small: IntervalsSource,
        big: IntervalsSource,
    ) -> Result<IntervalsSource> {
        let pulled = pull_up_one(&big, |s| IntervalsSource::NotContainedBy {
            minuend: Box::new(small.clone()),
            subtrahend: Box::new(s),
        })?;
        Self::or(pulled)
    }

    /// `Intervals.containedBy(small, big)`.
    pub fn contained_by(small: IntervalsSource, big: IntervalsSource) -> Result<IntervalsSource> {
        let pulled = pull_up_one(&big, |s| IntervalsSource::ContainedBy {
            small: Box::new(small.clone()),
            big: Box::new(s),
        })?;
        Self::or(pulled)
    }

    /// `Intervals.atLeast(int, IntervalsSource...)`.
    pub fn at_least(min_should_match: i32, sources: Vec<IntervalsSource>) -> IntervalsSource {
        let n = i32::try_from(sources.len()).unwrap_or(i32::MAX);
        if min_should_match == n {
            return Self::unordered(sources);
        }
        if min_should_match > n {
            let listed = sources
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            return IntervalsSource::NoMatch(format!(
                "Too few sources to match minimum of [{min_should_match}]: [{listed}]"
            ));
        }
        IntervalsSource::MinimumShouldMatch {
            sources,
            min_should_match,
        }
    }

    /// `Intervals.before(source, reference)`.
    pub fn before(source: IntervalsSource, reference: IntervalsSource) -> Result<IntervalsSource> {
        let offset = IntervalsSource::Offset {
            source: Box::new(reference),
            before: true,
        };
        Self::contained_by(source, Self::extend(offset, i32::MAX, 0))
    }

    /// `Intervals.after(source, reference)`.
    pub fn after(source: IntervalsSource, reference: IntervalsSource) -> Result<IntervalsSource> {
        let offset = IntervalsSource::Offset {
            source: Box::new(reference),
            before: false,
        };
        Self::contained_by(source, Self::extend(offset, 0, i32::MAX))
    }

    /// `Intervals.noIntervals(String)`.
    pub fn no_intervals(reason: &str) -> IntervalsSource {
        IntervalsSource::NoMatch(reason.to_string())
    }
}

/// `new BlockIntervalsSource(sources)`: nested blocks flattened (`flatten`).
fn block(sources: Vec<IntervalsSource>) -> IntervalsSource {
    let mut flattened = Vec::with_capacity(sources.len());
    for s in sources {
        match s {
            IntervalsSource::Block(inner) => flattened.extend(inner),
            other => flattened.push(other),
        }
    }
    IntervalsSource::Block(flattened)
}

/// `RepeatingIntervalsSource.build`.
fn repeating(source: IntervalsSource, count: usize) -> IntervalsSource {
    if count == 1 {
        return source;
    }
    IntervalsSource::Repeating {
        source: Box::new(source),
        count: i32::try_from(count).unwrap_or(i32::MAX),
        name: None,
    }
}

/// `deduplicate`'s `setName`, when the only source left is a repeat.
fn name_single(deduplicated: &mut [IntervalsSource], label: &'static str) {
    if let [IntervalsSource::Repeating { name, .. }] = deduplicated {
        *name = Some(label);
    }
}

/// `DisjunctionIntervalsSource.simplify`: nested disjunctions pulled up,
/// duplicates dropped.
fn simplify(sources: Vec<IntervalsSource>) -> Result<Vec<IntervalsSource>> {
    let mut simplified = Vec::with_capacity(sources.len());
    for source in sources {
        if matches!(source, IntervalsSource::Disjunction { .. }) {
            for s in source.pull_up_disjunctions()? {
                push_distinct(&mut simplified, s);
            }
        } else {
            push_distinct(&mut simplified, source);
        }
    }
    Ok(simplified)
}

/// `IntervalScoreFunction`: how a sloppy frequency becomes a score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IntervalScoreFunction {
    /// `SaturationFunction`: `w * (1 - k / (k + S))`.
    Saturation { pivot: f32 },
    /// `SigmoidFunction`: `w * (1 - k^a / (S^a + k^a))`, in `double`.
    Sigmoid { pivot: f32, exp: f32, pivot_pa: f64 },
}

impl IntervalScoreFunction {
    /// `IntervalScoreFunction.saturationFunction(pivot)`.
    pub fn saturation(pivot: f32) -> Result<Self> {
        if pivot <= 0.0 || !pivot.is_finite() {
            return Err(Error::IllegalArgument(format!(
                "pivot must be > 0, got: {}",
                crate::explain::java_float(pivot)
            )));
        }
        Ok(IntervalScoreFunction::Saturation { pivot })
    }

    /// `IntervalScoreFunction.sigmoidFunction(pivot, exp)`.
    pub fn sigmoid(pivot: f32, exp: f32) -> Result<Self> {
        if pivot <= 0.0 || !pivot.is_finite() {
            return Err(Error::IllegalArgument(format!(
                "pivot must be > 0, got: {}",
                crate::explain::java_float(pivot)
            )));
        }
        if exp <= 0.0 || !exp.is_finite() {
            return Err(Error::IllegalArgument(format!(
                "exp must be > 0, got: {}",
                crate::explain::java_float(exp)
            )));
        }
        Ok(IntervalScoreFunction::Sigmoid {
            pivot,
            exp,
            pivot_pa: f64::from(pivot).powf(f64::from(exp)),
        })
    }

    /// `scorer(weight).score(freq, norm)`.
    pub fn score(&self, weight: f32, freq: f32) -> f32 {
        match *self {
            IntervalScoreFunction::Saturation { pivot } => {
                weight * (1.0f32 - pivot / (pivot + freq))
            }
            IntervalScoreFunction::Sigmoid { exp, pivot_pa, .. } => {
                let pow = f64::from(freq).powf(f64::from(exp));
                (f64::from(weight) * (1.0f64 - pivot_pa / (pow + pivot_pa))) as f32
            }
        }
    }

    /// `explain(interval, weight, sloppyFreq)`.
    pub fn explain(
        &self,
        interval: &str,
        weight: f32,
        sloppy_freq: f32,
    ) -> crate::explain::Explanation {
        use crate::explain::Explanation;
        let score = self.score(weight, sloppy_freq);
        match *self {
            IntervalScoreFunction::Saturation { pivot } => Explanation::match_(
                score,
                "Saturation function on interval frequency, computed as w * S / (S + k) from:",
            )
            .with_details(vec![
                Explanation::match_(weight, "w, weight of this function"),
                Explanation::match_(
                    pivot,
                    "k, pivot feature value that would give a score contribution equal to w/2",
                ),
                Explanation::match_(
                    sloppy_freq,
                    format!("S, the sloppy frequency of the interval query {interval}"),
                ),
            ]),
            IntervalScoreFunction::Sigmoid { pivot, exp, .. } => Explanation::match_(
                score,
                "Sigmoid function on interval frequency, computed as w * S^a / (S^a + k^a) from:",
            )
            .with_details(vec![
                Explanation::match_(weight, "w, weight of this function"),
                Explanation::match_(
                    pivot,
                    "k, pivot feature value that would give a score contribution equal to w/2",
                ),
                Explanation::match_(
                    exp,
                    "a, exponent, higher values make the function grow slower before k and faster after k",
                ),
                Explanation::match_(
                    sloppy_freq,
                    format!("S, the sloppy frequency of the interval query {interval}"),
                ),
            ]),
        }
    }
}

/// `IntervalQuery`: the documents holding an interval of `source` in
/// `field`, scored by the sloppy frequency of their intervals.
#[derive(Debug, Clone)]
pub struct IntervalQuery {
    pub field: String,
    pub source: IntervalsSource,
    pub score_function: IntervalScoreFunction,
}

impl PartialEq for IntervalQuery {
    /// `IntervalQuery.equals`: the field and the source (not the scoring
    /// function).
    fn eq(&self, other: &Self) -> bool {
        self.field == other.field && self.source == other.source
    }
}

impl IntervalQuery {
    /// `new IntervalQuery(field, source)`: saturation with pivot 1.
    pub fn new(field: impl Into<String>, source: IntervalsSource) -> Self {
        IntervalQuery {
            field: field.into(),
            source,
            score_function: IntervalScoreFunction::Saturation { pivot: 1.0 },
        }
    }

    /// `new IntervalQuery(field, source, pivot)`.
    pub fn with_pivot(
        field: impl Into<String>,
        source: IntervalsSource,
        pivot: f32,
    ) -> Result<Self> {
        Ok(IntervalQuery {
            field: field.into(),
            source,
            score_function: IntervalScoreFunction::saturation(pivot)?,
        })
    }

    /// `new IntervalQuery(field, source, pivot, exp)`.
    pub fn with_pivot_and_exp(
        field: impl Into<String>,
        source: IntervalsSource,
        pivot: f32,
        exp: f32,
    ) -> Result<Self> {
        Ok(IntervalQuery {
            field: field.into(),
            source,
            score_function: IntervalScoreFunction::sigmoid(pivot, exp)?,
        })
    }

    /// `IntervalQuery.toString(field)`.
    pub fn to_string_with_field(&self, field: &str) -> String {
        if self.field == field {
            self.source.to_string()
        } else {
            format!("{}:{}", self.field, self.source)
        }
    }
}

impl fmt::Display for IntervalQuery {
    /// `Query.toString()`: `toString("")`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_string_with_field(""))
    }
}

impl From<IntervalQuery> for crate::query::Clause {
    fn from(q: IntervalQuery) -> Self {
        crate::query::Clause::Extended(Box::new(crate::extended_query::ExtendedQuery::Interval(q)))
    }
}

#[cfg(test)]
mod tests;
