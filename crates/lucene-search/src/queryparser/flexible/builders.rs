//! `StandardQueryTreeBuilder`: the processed node tree as a query, bottom
//! up. Java finds a node's builder by its class, then its superclasses
//! (`QuotedFieldQueryNode` builds as a `FieldQueryNode`, `BooleanModifierNode`
//! as a `ModifierQueryNode`); the match below keeps that lookup.

use super::nodes::{IntervalFn, Kind, Modifier, Node};
use super::processors::{Result, QUERY_NODE_EXCEPTION};
use super::StandardQueryParser;
use crate::extended_query::{ExtendedQuery, TermRangeQuery};
use crate::intervals::{IntervalQuery, Intervals, IntervalsSource, DEFAULT_MAX_EXPANSIONS};
use crate::query::{
    Clause, FuzzyQuery, MultiPhraseQuery, PhraseQuery, PrefixQuery, RegexpQuery, TermQuery,
    WildcardQuery,
};
use crate::query_visitor::Occur;
use crate::queryparser::classic::{
    check_regexp, float_to_edits, QueryParserError, MAX_CLAUSE_COUNT,
};
use crate::queryparser::{ParsedBoolean, ParsedQuery};

fn illegal(message: impl Into<String>) -> QueryParserError {
    QueryParserError::runtime_class("java.lang.IllegalArgumentException", message)
}

fn search_error(e: crate::Error) -> QueryParserError {
    match e {
        crate::Error::IllegalArgument(m) | crate::Error::InvalidQuery(m) => illegal(m),
        other => QueryParserError::runtime(other.to_string()),
    }
}

/// The JVM's message for `MinShouldMatchNodeBuilder`'s `(BooleanQuery) q`
/// over a query of another class, as HotSpot words it with Lucene on the
/// application class path (the module/loader suffix names the loader, so it
/// reads differently inside an OpenSearch plugin -- see the ledger).
fn class_cast_to_boolean(q: &ParsedQuery) -> String {
    let package = match q {
        ParsedQuery::Other(Clause::Extended(_)) => "org.apache.lucene.queries.intervals",
        _ => "org.apache.lucene.search",
    };
    let from = format!("{package}.{}", q.class_name());
    let to = "org.apache.lucene.search.BooleanQuery";
    format!(
        "class {from} cannot be cast to class {to} ({from} and {to} are in unnamed module of loader 'app')"
    )
}

/// HotSpot's helpful message for `booleanQuery.clauses()` when the group
/// built to `null` (Lucene is compiled with local variable names).
const MIN_SHOULD_MATCH_NPE: &str =
    "Cannot invoke \"org.apache.lucene.search.BooleanQuery.clauses()\" because \"booleanQuery\" is null";

/// `QueryTreeBuilder.build(node)`: the query, or `None` where Java's
/// builder returns `null`.
pub(crate) fn build(cfg: &StandardQueryParser, node: &Node) -> Result<Option<ParsedQuery>> {
    let field = |d: &super::nodes::FieldData| d.field.clone().unwrap_or_default();
    Ok(Some(match &node.kind {
        // `GroupQueryNodeBuilder`, `ModifierQueryNodeBuilder`: the child's
        Kind::Group | Kind::Modifier(_) | Kind::BooleanModifier(_) => {
            return build(cfg, &node.children[0]);
        }
        Kind::Field(d) | Kind::Quoted(d) => {
            ParsedQuery::Term(TermQuery::new(field(d), d.text.as_string().into_bytes()))
        }
        Kind::Fuzzy {
            data,
            similarity,
            prefix_length,
        } => {
            let text = data.text.as_string();
            let edits = float_to_edits(*similarity, text.chars().count());
            if *prefix_length < 0 {
                return Err(illegal("prefixLength cannot be negative."));
            }
            ParsedQuery::Fuzzy(
                FuzzyQuery::new(field(data), text.into_bytes())
                    .with_max_edits(edits)
                    .with_prefix_length(*prefix_length as usize),
            )
        }
        Kind::Wildcard(d) => ParsedQuery::with_rewrite(
            ParsedQuery::Wildcard(WildcardQuery::new(
                field(d),
                d.text.as_string().into_bytes(),
            )),
            cfg.multi_term_rewrite_method,
        ),
        Kind::PrefixWildcard(d) => {
            let mut units = d.text.units.clone();
            units.pop();
            ParsedQuery::with_rewrite(
                ParsedQuery::Prefix(PrefixQuery::new(
                    field(d),
                    String::from_utf16_lossy(&units).into_bytes(),
                )),
                cfg.multi_term_rewrite_method,
            )
        }
        Kind::Regexp { field, text } => {
            let text = text.as_string();
            check_regexp(&text)?;
            ParsedQuery::with_rewrite(
                ParsedQuery::Regexp(RegexpQuery::new(field.clone().unwrap_or_default(), text)),
                cfg.multi_term_rewrite_method,
            )
        }
        Kind::TermRange {
            lower_inclusive,
            upper_inclusive,
        } => {
            let bound = |n: &Node| {
                let t = n
                    .field_data()
                    .map(|d| d.text.as_string())
                    .unwrap_or_default();
                (!t.is_empty()).then(|| t.into_bytes())
            };
            ParsedQuery::with_rewrite(
                ParsedQuery::TermRange(TermRangeQuery::new(
                    node.get_field().unwrap_or_default(),
                    bound(&node.children[0]),
                    bound(&node.children[1]),
                    *lower_inclusive,
                    *upper_inclusive,
                )),
                cfg.multi_term_rewrite_method,
            )
        }
        Kind::MatchAll => ParsedQuery::MatchAll,
        Kind::MatchNoDocs => ParsedQuery::MatchNoDocs(String::new()),
        Kind::Boolean | Kind::And | Kind::Or => {
            let mut clauses = Vec::new();
            for child in &node.children {
                if let Some(q) = build(cfg, child)? {
                    if clauses.len() >= MAX_CLAUSE_COUNT {
                        return Err(QueryParserError::runtime_class(
                            QUERY_NODE_EXCEPTION,
                            "Too many boolean clauses, the maximum supported is 1,024".to_string(),
                        ));
                    }
                    let occur = match child.modifier() {
                        Some(Modifier::Req) => Occur::Must,
                        Some(Modifier::Not) => Occur::MustNot,
                        _ => Occur::Should,
                    };
                    clauses.push((occur, q));
                }
            }
            ParsedQuery::Boolean(ParsedBoolean {
                clauses,
                minimum_should_match: 0,
            })
        }
        Kind::Synonym => {
            let mut clauses = Vec::new();
            for child in &node.children {
                if let Some(q) = build(cfg, child)? {
                    clauses.push((Occur::Should, q));
                }
            }
            ParsedQuery::Boolean(ParsedBoolean {
                clauses,
                minimum_should_match: 0,
            })
        }
        Kind::Boost(value) => {
            let Some(child) = node.children.first() else {
                return Ok(None);
            };
            let Some(q) = build(cfg, child)? else {
                return Err(QueryParserError::runtime_class(
                    "java.lang.NullPointerException",
                    "null",
                ));
            };
            if !value.is_finite() || *value < 0.0 {
                return Err(illegal(format!(
                    "boost must be a positive float, got {}",
                    lucene_analysis::attributes::java_float_to_string(*value)
                )));
            }
            ParsedQuery::Boost(Box::new(q), *value)
        }
        Kind::TokenizedPhrase => phrase(node, 0)?,
        Kind::MultiPhrase => multi_phrase(node, 0)?,
        Kind::Slop(slop) => {
            let child = &node.children[0];
            // `PhraseQuery.Builder.setSlop` and `MultiPhraseQuery.Builder.setSlop`
            // word the refusal each their own way
            match child.kind {
                Kind::TokenizedPhrase if *slop < 0 => {
                    return Err(illegal(format!("Slop must be >= 0, got {slop}")))
                }
                Kind::TokenizedPhrase => phrase(child, *slop)?,
                _ if *slop < 0 => return Err(illegal("slop value cannot be negative")),
                _ => multi_phrase(child, *slop)?,
            }
        }
        Kind::MinShouldMatch(n) => {
            let group = &node.children[0];
            match build(cfg, group)? {
                Some(ParsedQuery::Boolean(b)) => ParsedQuery::Boolean(ParsedBoolean {
                    clauses: b.clauses,
                    minimum_should_match: usize::try_from(*n)
                        .map_err(|_| illegal("Minimum number should match must be >= 0"))?,
                }),
                Some(other) => {
                    return Err(QueryParserError::runtime_class(
                        "java.lang.ClassCastException",
                        class_cast_to_boolean(&other),
                    ))
                }
                None => {
                    return Err(QueryParserError::runtime_class(
                        "java.lang.NullPointerException",
                        MIN_SHOULD_MATCH_NPE,
                    ))
                }
            }
        }
        Kind::Interval { field, source } => {
            let Some(field) = field else {
                return Err(QueryParserError::runtime_class(
                    "java.lang.NullPointerException",
                    "Field must not be null for interval queries.",
                ));
            };
            let source = to_interval_source(cfg, field, source)?;
            ParsedQuery::Other(Clause::Extended(Box::new(ExtendedQuery::Interval(
                IntervalQuery::new(field.clone(), source),
            ))))
        }
        Kind::Deleted | Kind::NoTokenFound => {
            return Err(QueryParserError::runtime_class(
                QUERY_NODE_EXCEPTION,
                format!(
                    "Cannot convert query to lucene syntax: {} error: {}",
                    node.to_query_string(),
                    node.class_name()
                ),
            ))
        }
    }))
}

/// `PhraseQueryNodeBuilder` (and `SlopQueryNodeBuilder` over it).
fn phrase(node: &Node, slop: i32) -> Result<ParsedQuery> {
    let field = node.get_field().unwrap_or_default();
    let terms = node
        .children
        .iter()
        .filter_map(|c| c.field_data())
        .map(|d| (d.text.as_string().into_bytes(), d.position_increment));
    let q = PhraseQuery::with_positions(field, terms).map_err(search_error)?;
    Ok(ParsedQuery::Phrase(q.with_slop(slop as u32)))
}

/// `MultiPhraseQueryNodeBuilder`: the terms grouped by position, in order.
fn multi_phrase(node: &Node, slop: i32) -> Result<ParsedQuery> {
    let field = node.get_field().unwrap_or_default();
    let mut by_position: std::collections::BTreeMap<i32, Vec<Vec<u8>>> = Default::default();
    for d in node.children.iter().filter_map(|c| c.field_data()) {
        by_position
            .entry(d.position_increment)
            .or_default()
            .push(d.text.as_string().into_bytes());
    }
    let mut q =
        MultiPhraseQuery::with_positions(field, by_position.into_iter().map(|(p, t)| (t, p)))
            .map_err(search_error)?;
    q.slop = slop as u32;
    Ok(ParsedQuery::MultiPhrase(q))
}

/// `IntervalFunction.toIntervalSource(field, analyzer)`.
fn to_interval_source(
    cfg: &StandardQueryParser,
    field: &str,
    f: &IntervalFn,
) -> Result<IntervalsSource> {
    let src = |g: &IntervalFn| to_interval_source(cfg, field, g);
    let all = |gs: &[IntervalFn]| gs.iter().map(src).collect::<Result<Vec<_>>>();
    let e = search_error;
    Ok(match f {
        IntervalFn::AnalyzedText(term) => {
            let analyzer = cfg
                .analyzer
                .as_ref()
                .expect("checked by IntervalQueryNodeProcessor");
            crate::intervals::builder::analyzed_text(term, analyzer, field, 0, true).map_err(e)?
        }
        IntervalFn::AtLeast(n, gs) => Intervals::at_least(*n, all(gs)?),
        IntervalFn::MaxWidth(n, g) => Intervals::maxwidth(*n, src(g)?),
        IntervalFn::MaxGaps(n, g) => Intervals::maxgaps(*n, src(g)?).map_err(e)?,
        IntervalFn::Ordered(gs) => Intervals::ordered(all(gs)?),
        IntervalFn::Unordered(gs) => Intervals::unordered(all(gs)?),
        IntervalFn::UnorderedNoOverlaps(a, b) => {
            Intervals::unordered_no_overlaps(src(a)?, src(b)?).map_err(e)?
        }
        IntervalFn::Or(gs) => Intervals::or(all(gs)?).map_err(e)?,
        IntervalFn::Phrase(gs) => Intervals::phrase(all(gs)?).map_err(e)?,
        IntervalFn::After(a, b) => Intervals::after(src(a)?, src(b)?).map_err(e)?,
        IntervalFn::Before(a, b) => Intervals::before(src(a)?, src(b)?).map_err(e)?,
        IntervalFn::Containing(big, small) => {
            Intervals::containing(src(big)?, src(small)?).map_err(e)?
        }
        IntervalFn::NotContaining(m, s) => {
            Intervals::not_containing(src(m)?, src(s)?).map_err(e)?
        }
        IntervalFn::ContainedBy(small, big) => {
            Intervals::contained_by(src(small)?, src(big)?).map_err(e)?
        }
        IntervalFn::NotContainedBy(small, big) => {
            Intervals::not_contained_by(src(small)?, src(big)?).map_err(e)?
        }
        IntervalFn::Within(a, n, b) => Intervals::within(src(a)?, *n, src(b)?).map_err(e)?,
        IntervalFn::NotWithin(a, n, b) => Intervals::not_within(src(a)?, *n, src(b)?),
        IntervalFn::Overlapping(a, b) => Intervals::overlapping(src(a)?, src(b)?),
        IntervalFn::NonOverlapping(a, b) => Intervals::non_overlapping(src(a)?, src(b)?),
        IntervalFn::Extend(g, before, after) => Intervals::extend(src(g)?, *before, *after),
        IntervalFn::Wildcard(w, max) => {
            let max = if *max == 0 {
                DEFAULT_MAX_EXPANSIONS
            } else {
                usize::try_from(*max).unwrap_or(0)
            };
            Intervals::wildcard(w.as_bytes().to_vec(), max).map_err(e)?
        }
        IntervalFn::FuzzyTerm(term, edits, max) => {
            // `FuzzyQuery.defaultPrefixLength`, `defaultTranspositions`
            Intervals::fuzzy_term(term, *edits, 0, true, usize::try_from(*max).unwrap_or(0))
                .map_err(e)?
        }
    })
}

/// `IntervalQueryNode.toString()`: `field:function`, `null` for no field.
pub(crate) fn interval_node_string(field: Option<&str>, source: &IntervalFn) -> String {
    format!("{}:{}", field.unwrap_or("null"), source)
}

/// `AnalyzedText.requiresQuotes(term)`: `[\s]`, Java's ASCII white space.
fn requires_quotes(term: &str) -> bool {
    term.chars()
        .any(|c| matches!(c, ' ' | '\t' | '\n' | '\x0b' | '\x0c' | '\r'))
}

fn quoted(term: &str) -> String {
    if requires_quotes(term) {
        format!("\"{term}\"")
    } else {
        term.to_string()
    }
}

impl std::fmt::Display for IntervalFn {
    /// Each function's `toString()`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let join = |gs: &[IntervalFn]| {
            gs.iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(" ")
        };
        match self {
            IntervalFn::AnalyzedText(t) => write!(f, "{}", quoted(t)),
            IntervalFn::AtLeast(n, gs) => write!(f, "fn:atLeast({n} {})", join(gs)),
            IntervalFn::MaxWidth(n, g) => write!(f, "fn:maxwidth({n} {g})"),
            IntervalFn::MaxGaps(n, g) => write!(f, "fn:maxgaps({n} {g})"),
            IntervalFn::Ordered(gs) => write!(f, "fn:ordered({})", join(gs)),
            IntervalFn::Unordered(gs) => write!(f, "fn:unordered({})", join(gs)),
            IntervalFn::UnorderedNoOverlaps(a, b) => write!(f, "fn:unorderedNoOverlaps({a} {b})"),
            IntervalFn::Or(gs) => write!(f, "fn:or({})", join(gs)),
            IntervalFn::Phrase(gs) => write!(f, "fn:phrase({})", join(gs)),
            IntervalFn::After(a, b) => write!(f, "fn:after({a} {b})"),
            IntervalFn::Before(a, b) => write!(f, "fn:before({a} {b})"),
            IntervalFn::Containing(a, b) => write!(f, "fn:containing({a} {b})"),
            IntervalFn::NotContaining(a, b) => write!(f, "fn:notContaining({a} {b})"),
            IntervalFn::ContainedBy(a, b) => write!(f, "fn:containedBy({a} {b})"),
            IntervalFn::NotContainedBy(a, b) => write!(f, "fn:notContainedBy({a} {b})"),
            IntervalFn::Within(a, n, b) => write!(f, "fn:within({a} {n} {b})"),
            IntervalFn::NotWithin(a, n, b) => write!(f, "fn:notWithin({a} {n} {b})"),
            IntervalFn::Overlapping(a, b) => write!(f, "fn:overlapping({a} {b})"),
            IntervalFn::NonOverlapping(a, b) => write!(f, "fn:nonOverlapping({a} {b})"),
            IntervalFn::Extend(g, b, a) => write!(f, "fn:extend({g} {b} {a})"),
            IntervalFn::Wildcard(w, max) => {
                if *max == 0 {
                    write!(f, "fn:wildcard({w})")
                } else {
                    write!(f, "fn:wildcard({w} maxExpansions:{max})")
                }
            }
            // as Java prints it: the expansions straight after the edits
            IntervalFn::FuzzyTerm(t, edits, max) => {
                write!(f, "fn:fuzzyTerm({} {edits}{max})", quoted(t))
            }
        }
    }
}
