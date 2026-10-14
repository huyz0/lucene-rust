//! `lucene-queryparser`: query strings into queries.
//!
//! [`ParsedQuery`] is the query tree a parser yields, Lucene's query classes
//! as the parser builds them: a [`ParsedQuery::Boolean`] keeps its clauses in
//! the order they were added (which `BooleanQuery.toString` prints, and
//! [`crate::query::BooleanQuery`] does not keep), so the tree prints as
//! Lucene's does; [`ParsedQuery::to_clause`] is the query to run.
//!
//! - [`classic`]: `org.apache.lucene.queryparser.classic` -- `QueryParser`
//!   (the JavaCC grammar's lexer and parser, ported state for state) and
//!   `QueryParserBase`.

pub mod classic;
pub mod ext;
mod javacc;
pub mod simple;
pub mod surround;

use lucene_analysis::attributes::java_float_to_string;

use crate::extended_query::{
    ExtendedQuery, MultiTermQuery, MultiTermSource, RewriteMethod, SynonymQuery, TermRangeQuery,
};
use crate::query::{
    BooleanQuery, BoostQuery, Clause, FuzzyQuery, MatchAllDocsQuery, MatchNoDocsQuery,
    MultiPhraseQuery, PhraseQuery, PrefixQuery, RegexpQuery, SpanQuery, TermQuery, WildcardQuery,
};
use crate::query_visitor::Occur;

/// A query a parser built, as the Java query objects it stands for.
#[derive(Debug, Clone, PartialEq)]
pub enum ParsedQuery {
    /// `TermQuery`.
    Term(TermQuery),
    /// `PhraseQuery`.
    Phrase(PhraseQuery),
    /// `MultiPhraseQuery`.
    MultiPhrase(MultiPhraseQuery),
    /// `SynonymQuery`.
    Synonym(SynonymQuery),
    /// `WildcardQuery`.
    Wildcard(WildcardQuery),
    /// `PrefixQuery`.
    Prefix(PrefixQuery),
    /// `FuzzyQuery`.
    Fuzzy(FuzzyQuery),
    /// `RegexpQuery`.
    Regexp(RegexpQuery),
    /// `TermRangeQuery`.
    TermRange(TermRangeQuery),
    /// `MatchAllDocsQuery`.
    MatchAll,
    /// `MatchNoDocsQuery(reason)`.
    MatchNoDocs(String),
    /// `SpanTermQuery`, `SpanNearQuery` or `SpanOrQuery`.
    Span(SpanQuery),
    /// `BooleanQuery`: its clauses in insertion order.
    Boolean(ParsedBoolean),
    /// `BoostQuery`.
    Boost(Box<ParsedQuery>, f32),
    /// Any other query an analysis chain made (a graph query's spans).
    Other(Clause),
}

/// `BooleanQuery`, clause order kept.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ParsedBoolean {
    pub clauses: Vec<(Occur, ParsedQuery)>,
    pub minimum_should_match: usize,
}

impl ParsedQuery {
    /// The tree of a query `QueryBuilder` made. Its booleans hold clauses of
    /// one occur each (or `MUST` and `SHOULD` built in that order), so the
    /// buckets' order is their insertion order.
    pub fn from_clause(clause: Clause) -> Self {
        match clause {
            Clause::Term(t) => ParsedQuery::Term(t),
            Clause::Phrase(p) => ParsedQuery::Phrase(p),
            Clause::MultiPhrase(m) => ParsedQuery::MultiPhrase(m),
            Clause::Wildcard(w) => ParsedQuery::Wildcard(w),
            Clause::Prefix(p) => ParsedQuery::Prefix(p),
            Clause::Fuzzy(f) => ParsedQuery::Fuzzy(f),
            Clause::Regexp(r) => ParsedQuery::Regexp(r),
            Clause::MatchNoDocs(m) => ParsedQuery::MatchNoDocs(m.reason),
            Clause::Span(s) => ParsedQuery::Span(s),
            Clause::Boost(b) => {
                let BoostQuery { inner, boost } = *b;
                ParsedQuery::Boost(Box::new(ParsedQuery::from_clause(*inner)), boost)
            }
            Clause::Boolean(b) => {
                let BooleanQuery {
                    must,
                    filter,
                    should,
                    must_not,
                    minimum_should_match,
                } = *b;
                let clauses = must
                    .into_iter()
                    .map(|c| (Occur::Must, c))
                    .chain(filter.into_iter().map(|c| (Occur::Filter, c)))
                    .chain(should.into_iter().map(|c| (Occur::Should, c)))
                    .chain(must_not.into_iter().map(|c| (Occur::MustNot, c)))
                    .map(|(o, c)| (o, ParsedQuery::from_clause(c)))
                    .collect();
                ParsedQuery::Boolean(ParsedBoolean {
                    clauses,
                    minimum_should_match,
                })
            }
            Clause::Extended(e) => match *e {
                ExtendedQuery::Synonym(s) => ParsedQuery::Synonym(s),
                ExtendedQuery::MultiTerm(MultiTermQuery {
                    source: MultiTermSource::TermRange(r),
                    ..
                }) => ParsedQuery::TermRange(r),
                other => ParsedQuery::Other(Clause::Extended(Box::new(other))),
            },
            other => ParsedQuery::Other(other),
        }
    }

    /// The query to run. `max_doc` is the index's, which a
    /// `MatchAllDocsQuery` matches up to.
    pub fn to_clause(&self, max_doc: i32) -> Clause {
        match self {
            ParsedQuery::Term(t) => Clause::Term(t.clone()),
            ParsedQuery::Phrase(p) => Clause::Phrase(p.clone()),
            ParsedQuery::MultiPhrase(m) => Clause::MultiPhrase(m.clone()),
            ParsedQuery::Synonym(s) => {
                Clause::Extended(Box::new(ExtendedQuery::Synonym(s.clone())))
            }
            ParsedQuery::Wildcard(w) => Clause::Wildcard(w.clone()),
            ParsedQuery::Prefix(p) => Clause::Prefix(p.clone()),
            ParsedQuery::Fuzzy(f) => Clause::Fuzzy(f.clone()),
            ParsedQuery::Regexp(r) => Clause::Regexp(r.clone()),
            // `QueryParserBase.multiTermRewriteMethod`'s default.
            ParsedQuery::TermRange(r) => {
                Clause::Extended(Box::new(ExtendedQuery::MultiTerm(MultiTermQuery::new(
                    MultiTermSource::TermRange(r.clone()),
                    RewriteMethod::ConstantScoreBlended,
                ))))
            }
            ParsedQuery::MatchAll => Clause::MatchAllDocs(MatchAllDocsQuery::new(max_doc)),
            ParsedQuery::MatchNoDocs(reason) => {
                Clause::MatchNoDocs(MatchNoDocsQuery::new().with_reason(reason.clone()))
            }
            ParsedQuery::Span(s) => Clause::Span(s.clone()),
            ParsedQuery::Boolean(b) => {
                let mut q = BooleanQuery::new();
                for (occur, c) in &b.clauses {
                    let c = c.to_clause(max_doc);
                    match occur {
                        Occur::Must => q.must.push(c),
                        Occur::Filter => q.filter.push(c),
                        Occur::Should => q.should.push(c),
                        Occur::MustNot => q.must_not.push(c),
                    }
                }
                q.minimum_should_match = b.minimum_should_match;
                Clause::Boolean(Box::new(q))
            }
            ParsedQuery::Boost(q, boost) => {
                Clause::Boost(Box::new(BoostQuery::new(q.to_clause(max_doc), *boost)))
            }
            ParsedQuery::Other(c) => c.clone(),
        }
    }

    /// `Query.toString(field)`.
    pub fn to_string(&self, field: &str) -> String {
        match self {
            ParsedQuery::Boolean(b) => boolean_string(b, field),
            ParsedQuery::Boost(q, boost) => {
                format!("({})^{}", q.to_string(field), java_float_to_string(*boost))
            }
            ParsedQuery::Wildcard(w) => with_field(&w.field, field, term_text(&w.pattern)),
            ParsedQuery::Prefix(p) => with_field(&p.field, field, term_text(&p.prefix) + "*"),
            ParsedQuery::Fuzzy(f) => with_field(
                &f.field,
                field,
                format!("{}~{}", term_text(&f.term), f.max_edits),
            ),
            ParsedQuery::Regexp(r) => with_field(&r.field, field, format!("/{}/", r.pattern)),
            ParsedQuery::TermRange(r) => {
                let bound = |b: &Option<Vec<u8>>| match b {
                    None => "*".to_string(),
                    Some(b) => {
                        let t = term_text(b);
                        if t == "*" {
                            "\\*".to_string()
                        } else {
                            t
                        }
                    }
                };
                with_field(
                    &r.field,
                    field,
                    format!(
                        "{}{} TO {}{}",
                        if r.include_lower { '[' } else { '{' },
                        bound(&r.lower),
                        bound(&r.upper),
                        if r.include_upper { ']' } else { '}' }
                    ),
                )
            }
            ParsedQuery::MatchAll => "*:*".to_string(),
            ParsedQuery::MatchNoDocs(reason) => format!("MatchNoDocsQuery(\"{reason}\")"),
            ParsedQuery::Span(s) => span_string(s, field),
            ParsedQuery::Term(t) => {
                crate::query_builder::to_query_string(Some(&Clause::Term(t.clone())), field)
            }
            ParsedQuery::Phrase(_)
            | ParsedQuery::MultiPhrase(_)
            | ParsedQuery::Synonym(_)
            | ParsedQuery::Other(_) => {
                crate::query_builder::to_query_string(Some(&self.to_clause(0)), field)
            }
        }
    }

    /// The query's class tree: booleans and boosts opened up, every other
    /// query as `<SimpleClassName>(<toString(field)>)` (what the parser
    /// fixtures record).
    pub fn describe(&self, field: &str) -> String {
        match self {
            ParsedQuery::Boolean(b) => {
                let mut out = String::from("Boolean(");
                if b.minimum_should_match != 0 {
                    out.push_str(&format!("msm={} ", b.minimum_should_match));
                }
                for (i, (occur, c)) in b.clauses.iter().enumerate() {
                    if i != 0 {
                        out.push(' ');
                    }
                    out.push_str(occur_prefix(*occur));
                    out.push_str(&c.describe(field));
                }
                out.push(')');
                out
            }
            ParsedQuery::Boost(q, boost) => {
                format!(
                    "Boost({}^{})",
                    q.describe(field),
                    java_float_to_string(*boost)
                )
            }
            other => format!("{}({})", other.class_name(), other.to_string(field)),
        }
    }

    /// The Java query class's simple name.
    pub fn class_name(&self) -> &'static str {
        match self {
            ParsedQuery::Term(_) => "TermQuery",
            ParsedQuery::Phrase(_) => "PhraseQuery",
            ParsedQuery::MultiPhrase(_) => "MultiPhraseQuery",
            ParsedQuery::Synonym(_) => "SynonymQuery",
            ParsedQuery::Wildcard(_) => "WildcardQuery",
            ParsedQuery::Prefix(_) => "PrefixQuery",
            ParsedQuery::Fuzzy(_) => "FuzzyQuery",
            ParsedQuery::Regexp(_) => "RegexpQuery",
            ParsedQuery::TermRange(_) => "TermRangeQuery",
            ParsedQuery::MatchAll => "MatchAllDocsQuery",
            ParsedQuery::MatchNoDocs(_) => "MatchNoDocsQuery",
            ParsedQuery::Span(SpanQuery::SpanTerm { .. }) => "SpanTermQuery",
            ParsedQuery::Span(SpanQuery::SpanNear { .. }) => "SpanNearQuery",
            ParsedQuery::Span(SpanQuery::SpanOr { .. }) => "SpanOrQuery",
            ParsedQuery::Boolean(_) => "BooleanQuery",
            ParsedQuery::Boost(..) => "BoostQuery",
            ParsedQuery::Other(_) => "Query",
        }
    }
}

/// `SpanTermQuery`, `SpanNearQuery` and `SpanOrQuery`'s `toString(field)`.
fn span_string(s: &SpanQuery, field: &str) -> String {
    match s {
        SpanQuery::SpanTerm { field: f, term } => {
            if f == field {
                term_text(term)
            } else {
                format!("{f}:{}", term_text(term))
            }
        }
        SpanQuery::SpanNear {
            clauses,
            slop,
            in_order,
        } => {
            let parts: Vec<String> = clauses.iter().map(|c| span_string(c, field)).collect();
            format!("spanNear([{}], {slop}, {in_order})", parts.join(", "))
        }
        SpanQuery::SpanOr { clauses } => {
            let parts: Vec<String> = clauses.iter().map(|c| span_string(c, field)).collect();
            format!("spanOr([{}])", parts.join(", "))
        }
    }
}

fn occur_prefix(occur: Occur) -> &'static str {
    match occur {
        Occur::Must => "+",
        Occur::MustNot => "-",
        Occur::Filter => "#",
        Occur::Should => "",
    }
}

/// `BooleanQuery.toString(field)`.
fn boolean_string(b: &ParsedBoolean, field: &str) -> String {
    let msm = b.minimum_should_match;
    let mut out = String::new();
    let needs_parens = msm > 0;
    if needs_parens {
        out.push('(');
    }
    for (i, (occur, c)) in b.clauses.iter().enumerate() {
        out.push_str(occur_prefix(*occur));
        if let ParsedQuery::Boolean(_) = c {
            out.push('(');
            out.push_str(&c.to_string(field));
            out.push(')');
        } else {
            out.push_str(&c.to_string(field));
        }
        if i + 1 != b.clauses.len() {
            out.push(' ');
        }
    }
    if needs_parens {
        out.push(')');
    }
    if msm > 0 {
        out.push('~');
        out.push_str(&msm.to_string());
    }
    out
}

fn with_field(query_field: &str, field: &str, body: String) -> String {
    if query_field == field {
        body
    } else {
        format!("{query_field}:{body}")
    }
}

/// `Term.toString(BytesRef)`: the bytes as UTF-8, or, when they are not,
/// `BytesRef.toString()`'s `[hex hex ...]`.
pub(crate) fn term_text(bytes: &[u8]) -> String {
    match std::str::from_utf8(bytes) {
        Ok(s) => s.to_string(),
        Err(_) => {
            let hex: Vec<String> = bytes.iter().map(|b| format!("{b:x}")).collect();
            format!("[{}]", hex.join(" "))
        }
    }
}

#[cfg(test)]
mod tests;
