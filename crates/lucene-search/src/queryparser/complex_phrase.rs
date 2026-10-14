//! `org.apache.lucene.queryparser.complexPhrase`: phrases whose words may be
//! wildcards, fuzzy terms, ranges or nested booleans (`"(john jon*) smith~1"`).
//!
//! `ComplexPhraseQueryParser` is the classic parser with a hook: a quoted
//! phrase becomes a [`ComplexPhraseQuery`] holding its text, which a second
//! pass parses on the phrase's field (the classic parser's
//! [`crate::queryparser::classic::QueryParser::complex_phrase`] mode). Against
//! an index, [`ParsedQuery::rewrite_complex_phrases`] turns each into the
//! span query Lucene's `ComplexPhraseQuery.rewrite` makes, expanding its
//! multi-term parts as `IndexSearcher.rewrite` does: prefixes, wildcards,
//! regular expressions and ranges with `SCORING_BOOLEAN_REWRITE`, fuzzy terms
//! with `FuzzyQuery`'s top-terms blended rewrite (each term boosted by its
//! similarity).

use lucene_codecs::blocktree::SeekStatus;
use lucene_codecs::fuzzy::FuzzyMatch;
use lucene_codecs::wildcard::WildcardPattern;
use lucene_util::automaton::{CharacterRunAutomaton, RegExp};

use crate::multi_terms::MultiTerms;
use crate::query::{Clause, FuzzyQuery, TermQuery};
use crate::query_visitor::Occur;
use crate::queryparser::classic::QueryParserError;
use crate::queryparser::{ParsedBoolean, ParsedQuery};
use crate::reader::IndexReader;
use crate::spans::SpanNode;

/// `ComplexPhraseQuery`: a phrase's field, text, slop and order, and its
/// parsed contents once the second pass has run.
#[derive(Debug, Clone)]
pub struct ComplexPhraseQuery {
    pub field: String,
    pub text: String,
    pub slop: i32,
    pub in_order: bool,
    /// `contents[0]`: the phrase's text parsed on `field`.
    pub contents: Option<ParsedQuery>,
}

/// The term a clause that matched nothing becomes.
const DUMMY: &str = "Dummy clause because no terms found - must match nothing";

fn search(e: crate::Error) -> QueryParserError {
    QueryParserError::from_search(e)
}

/// `getClass().getName()` of a query a phrase's contents hold: the second
/// pass makes neither spans nor phrases (a nested phrase fails it), so
/// every such query is of `org.apache.lucene.search`.
fn java_class(q: &ParsedQuery) -> String {
    format!("org.apache.lucene.search.{}", q.class_name())
}

/// `equals`: field, text, slop and order, not the contents (which the
/// text determines).
impl PartialEq for ComplexPhraseQuery {
    fn eq(&self, other: &Self) -> bool {
        self.field == other.field
            && self.text == other.text
            && self.slop == other.slop
            && self.in_order == other.in_order
    }
}

impl ComplexPhraseQuery {
    /// `toString(field)`.
    pub fn to_string(&self, field: &str) -> String {
        let mut sb = String::new();
        if self.field != field {
            sb.push_str(&self.field);
            sb.push(':');
        }
        sb.push('"');
        sb.push_str(&self.text);
        sb.push('"');
        if self.slop != 0 {
            sb.push('~');
            sb.push_str(&self.slop.to_string());
        }
        sb
    }

    fn unknown(&self, q: &ParsedQuery) -> QueryParserError {
        QueryParserError::illegal_argument(format!(
            "Unknown query type \"{}\" found in phrase query string \"{}\"",
            java_class(q),
            self.text
        ))
    }

    /// `rewrite(indexSearcher)`, the multi-term parts expanded against
    /// `reader`.
    ///
    /// # Errors
    /// `IllegalArgumentException` for contents a phrase cannot hold, and the
    /// reader's errors.
    pub fn rewrite(&self, reader: &dyn IndexReader) -> Result<ParsedQuery, QueryParserError> {
        let Some(contents) = &self.contents else {
            return Err(QueryParserError::runtime(
                "ComplexPhraseQuery: the phrase was never parsed",
            ));
        };
        let bq = match contents {
            ParsedQuery::Term(_)
            | ParsedQuery::Synonym(_)
            | ParsedQuery::Prefix(_)
            | ParsedQuery::Wildcard(_)
            | ParsedQuery::Regexp(_)
            | ParsedQuery::TermRange(_)
            | ParsedQuery::Fuzzy(_) => return searcher_rewrite(contents, reader),
            ParsedQuery::Boolean(bq) => bq,
            other => return Err(self.unknown(other)),
        };
        let mut all_span_clauses = Vec::with_capacity(bq.clauses.len());
        let mut num_negatives = 0;
        for (occur, clause) in &bq.clauses {
            let mut qc = searcher_rewrite(clause, reader)?;
            if *occur == Occur::MustNot {
                num_negatives += 1;
            }
            while let ParsedQuery::Boost(inner, _) = qc {
                qc = *inner;
            }
            let span = match &qc {
                ParsedQuery::Synonym(s) => {
                    self.boolean_span(&convert(&s.field, s.terms.iter().map(|(t, _)| t)))?
                }
                ParsedQuery::Boolean(b) => self.boolean_span(b)?,
                ParsedQuery::MatchNoDocs(_) => {
                    SpanNode::term(self.field.clone(), DUMMY.as_bytes().to_vec())
                }
                ParsedQuery::Term(t) => SpanNode::term(t.field.clone(), t.term.clone()),
                other => return Err(self.unknown(other)),
            };
            all_span_clauses.push(span);
        }
        if num_negatives == 0 {
            return Ok(ParsedQuery::SpanNode(span_near(
                all_span_clauses,
                self.slop,
                self.in_order,
            )?));
        }
        let positive: Vec<SpanNode> = bq
            .clauses
            .iter()
            .zip(&all_span_clauses)
            .filter(|((o, _), _)| *o != Occur::MustNot)
            .map(|(_, s)| s.clone())
            .collect();
        let include = match <[SpanNode; 1]>::try_from(positive) {
            Ok([only]) => only,
            Err(positive) => span_near(positive, self.slop + num_negatives, self.in_order)?,
        };
        let exclude = span_near(all_span_clauses, self.slop, self.in_order)?;
        Ok(ParsedQuery::SpanNode(span_not(include, exclude)?))
    }

    /// The span of a boolean (or converted synonym) clause of the phrase:
    /// its first span clause, or the dummy term when it has none.
    fn boolean_span(&self, b: &ParsedBoolean) -> Result<SpanNode, QueryParserError> {
        let mut sc = Vec::new();
        self.add_complex_phrase_clause(&mut sc, b)?;
        Ok(if sc.is_empty() {
            SpanNode::term(self.field.clone(), DUMMY.as_bytes().to_vec())
        } else {
            sc.swap_remove(0)
        })
    }

    /// `addComplexPhraseClause(spanClauses, qc)`.
    fn add_complex_phrase_clause(
        &self,
        span_clauses: &mut Vec<SpanNode>,
        b: &ParsedBoolean,
    ) -> Result<(), QueryParserError> {
        let mut ors = Vec::new();
        let mut nots = Vec::new();
        for (occur, clause) in &b.clauses {
            let mut child = clause;
            while let ParsedQuery::Boost(inner, _) = child {
                child = inner;
            }
            let chosen = if *occur == Occur::MustNot {
                &mut nots
            } else {
                &mut ors
            };
            match child {
                ParsedQuery::Term(t) => {
                    chosen.push(SpanNode::term(t.field.clone(), t.term.clone()))
                }
                ParsedQuery::Boolean(b) => self.add_complex_phrase_clause(chosen, b)?,
                ParsedQuery::MatchNoDocs(_) => chosen.push(SpanNode::term(
                    self.field.clone(),
                    DUMMY.as_bytes().to_vec(),
                )),
                other => {
                    return Err(QueryParserError::illegal_argument(format!(
                        "Unknown query type:{}",
                        java_class(other)
                    )))
                }
            }
        }
        if ors.is_empty() {
            return Ok(());
        }
        let soq = span_or(ors)?;
        if nots.is_empty() {
            span_clauses.push(soq);
        } else {
            span_clauses.push(span_not(soq, span_or(nots)?)?);
        }
        Ok(())
    }
}

/// `SpanQuery.getField()`.
fn span_field(s: &SpanNode) -> Option<&str> {
    match s {
        SpanNode::Term { field, .. } => Some(field),
        SpanNode::Near { clauses, .. } | SpanNode::Or { clauses } => {
            clauses.first().and_then(span_field)
        }
        SpanNode::Not { include, .. } => span_field(include),
        _ => None,
    }
}

fn same_field() -> QueryParserError {
    QueryParserError::illegal_argument("Clauses must have same field.")
}

/// The clauses of a `SpanNearQuery` or `SpanOrQuery`, all of one field.
fn check_clauses(clauses: &[SpanNode]) -> Result<(), QueryParserError> {
    let mut field = None;
    for c in clauses {
        match (field, span_field(c)) {
            (None, f) => field = f,
            (Some(f), Some(g)) if f != g => return Err(same_field()),
            _ => {}
        }
    }
    Ok(())
}

/// `new SpanNearQuery(clauses, slop, inOrder)`.
fn span_near(
    clauses: Vec<SpanNode>,
    slop: i32,
    in_order: bool,
) -> Result<SpanNode, QueryParserError> {
    check_clauses(&clauses)?;
    Ok(SpanNode::Near {
        clauses,
        slop,
        in_order,
    })
}

/// `new SpanOrQuery(clauses)`.
fn span_or(clauses: Vec<SpanNode>) -> Result<SpanNode, QueryParserError> {
    check_clauses(&clauses)?;
    Ok(SpanNode::Or { clauses })
}

/// `new SpanNotQuery(include, exclude)`.
fn span_not(include: SpanNode, exclude: SpanNode) -> Result<SpanNode, QueryParserError> {
    if let (Some(a), Some(b)) = (span_field(&include), span_field(&exclude)) {
        if a != b {
            return Err(same_field());
        }
    }
    Ok(SpanNode::Not {
        include: Box::new(include),
        exclude: Box::new(exclude),
        pre: 0,
        post: 0,
    })
}

/// `convert(SynonymQuery)`: its terms as `SHOULD` term queries.
fn convert<'a>(field: &str, terms: impl Iterator<Item = &'a Vec<u8>>) -> ParsedBoolean {
    ParsedBoolean {
        clauses: terms
            .map(|t| {
                (
                    Occur::Should,
                    ParsedQuery::Term(TermQuery::new(field, t.clone())),
                )
            })
            .collect(),
        minimum_should_match: 0,
    }
}

/// `IndexSearcher.rewrite(q)` for what a phrase's contents can hold: the
/// multi-term queries expanded (the parser's second pass makes them with
/// `SCORING_BOOLEAN_REWRITE`; a fuzzy query keeps its own top-terms
/// rewrite), then `BooleanQuery.rewrite`.
pub(crate) fn searcher_rewrite(
    q: &ParsedQuery,
    reader: &dyn IndexReader,
) -> Result<ParsedQuery, QueryParserError> {
    let expanded = expand(q, reader)?;
    Ok(ParsedQuery::from_clause(
        expanded.to_clause(reader.max_doc()).rewrite(),
    ))
}

fn expand(q: &ParsedQuery, reader: &dyn IndexReader) -> Result<ParsedQuery, QueryParserError> {
    if let Some((field, terms)) = scoring_terms(q, reader)? {
        return Ok(ParsedQuery::Boolean(ParsedBoolean {
            clauses: terms
                .into_iter()
                .map(|t| {
                    (
                        Occur::Should,
                        ParsedQuery::Term(TermQuery::new(field.clone(), t)),
                    )
                })
                .collect(),
            minimum_should_match: 0,
        }));
    }
    Ok(match q {
        ParsedQuery::Fuzzy(f) => fuzzy_rewrite(f, reader)?,
        ParsedQuery::Boolean(b) => ParsedQuery::Boolean(ParsedBoolean {
            clauses: b
                .clauses
                .iter()
                .map(|(o, c)| Ok((*o, expand(c, reader)?)))
                .collect::<Result<_, QueryParserError>>()?,
            minimum_should_match: b.minimum_should_match,
        }),
        ParsedQuery::Boost(inner, boost) => {
            ParsedQuery::Boost(Box::new(expand(inner, reader)?), *boost)
        }
        other => other.clone(),
    })
}

/// The terms of `field` from the first `>= start` in term order, each passed
/// to `keep` until it returns `false` (the terms past what a query can
/// match): `TermRangeTermsEnum`'s seek to the lower bound and stop past the
/// upper, and a prefix's seek.
fn each_term(
    reader: &dyn IndexReader,
    field: &str,
    start: &[u8],
    keep: &mut dyn FnMut(&[u8]) -> Result<bool, QueryParserError>,
) -> Result<(), QueryParserError> {
    let Some(terms) = MultiTerms::get_terms(reader, field).map_err(search)? else {
        return Ok(());
    };
    let mut te = terms.iterator().map_err(search)?;
    if te.try_seek_ceil(start).map_err(search)? == SeekStatus::End {
        return Ok(());
    }
    while let Some(t) = te.term() {
        if !keep(t)? || te.next().map_err(search)?.is_none() {
            break;
        }
    }
    Ok(())
}

/// The terms of `field` starting with `literal`, each passed to `keep`: a
/// prefix's seek, then its walk while the terms still start with it.
fn each_prefixed(
    reader: &dyn IndexReader,
    field: &str,
    literal: &[u8],
    keep: &mut dyn FnMut(&[u8]) -> Result<(), QueryParserError>,
) -> Result<(), QueryParserError> {
    each_term(reader, field, literal, &mut |t: &[u8]| {
        let within = t.starts_with(literal);
        if within {
            keep(t)?;
        }
        Ok(within)
    })
}

/// A multi-term query's field and its matched terms.
type FieldTerms = (String, Vec<Vec<u8>>);

/// One more matched term, or `TooManyClauses` at the first past the limit
/// (`SCORING_BOOLEAN_REWRITE`'s `checkMaxClauseCount`), not after them all.
fn collect(out: &mut Vec<Vec<u8>>, t: &[u8]) -> Result<(), QueryParserError> {
    if out.len() >= crate::queryparser::classic::MAX_CLAUSE_COUNT {
        return Err(QueryParserError::too_many_clauses());
    }
    out.push(t.to_vec());
    Ok(())
}

/// `ScoringRewrite`: the matched terms of a prefix, wildcard, regular
/// expression or range, sorted; `None` for another query.
fn scoring_terms(
    q: &ParsedQuery,
    reader: &dyn IndexReader,
) -> Result<Option<FieldTerms>, QueryParserError> {
    let mut out = Vec::new();
    let field = match q {
        ParsedQuery::Prefix(p) => {
            each_term(reader, &p.field, &p.prefix, &mut |t: &[u8]| {
                let within = t.starts_with(&p.prefix);
                if within {
                    collect(&mut out, t)?;
                }
                Ok(within)
            })?;
            &p.field
        }
        ParsedQuery::Wildcard(w) => {
            let pattern = WildcardPattern::new(&w.pattern);
            each_prefixed(
                reader,
                &w.field,
                &pattern.literal_prefix(),
                &mut |t: &[u8]| {
                    if pattern.matches(t) {
                        collect(&mut out, t)?;
                    }
                    Ok(())
                },
            )?;
            &w.field
        }
        ParsedQuery::Regexp(r) => {
            let re = RegExp::with_flags(&r.pattern, RegExp::ALL, 0)
                .map_err(|e| QueryParserError::illegal_argument(e.to_string()))?;
            let a = re
                .to_automaton()
                .map_err(|e| QueryParserError::illegal_argument(e.to_string()))?;
            let run = CharacterRunAutomaton::new(&a)
                .map_err(|e| QueryParserError::illegal_argument(e.to_string()))?;
            // every term: Lucene intersects the automaton with the terms
            each_term(reader, &r.field, b"", &mut |t: &[u8]| {
                if let Ok(s) = std::str::from_utf8(t) {
                    if run.run(s) {
                        collect(&mut out, t)?;
                    }
                }
                Ok(true)
            })?;
            &r.field
        }
        ParsedQuery::TermRange(r) => {
            let start = r.lower.clone().unwrap_or_default();
            each_term(reader, &r.field, &start, &mut |t: &[u8]| {
                if r.accepts(t) {
                    collect(&mut out, t)?;
                }
                Ok(r.upper.as_deref().is_none_or(|u| t <= u))
            })?;
            &r.field
        }
        _ => return Ok(None),
    };
    Ok(Some((field.clone(), out)))
}

/// `FuzzyQuery.rewrite` with its default `TopTermsBlendedFreqScoringRewrite`:
/// the terms its Levenshtein automaton accepts (`FuzzyTermsEnum`), each
/// with its similarity boost, the `maxExpansions` best (at most the clause
/// limit), in term order, each boosted by its similarity unless 1, as
/// `BlendedTermQuery.rewrite`'s `BOOLEAN_REWRITE` makes them.
fn fuzzy_rewrite(
    f: &FuzzyQuery,
    reader: &dyn IndexReader,
) -> Result<ParsedQuery, QueryParserError> {
    let pattern = FuzzyMatch::new(&f.term, f.max_edits, f.prefix_length, f.transpositions);
    let mut scored: Vec<(f32, Vec<u8>)> = Vec::new();
    each_prefixed(
        reader,
        &f.field,
        pattern.literal_prefix(),
        &mut |t: &[u8]| {
            if let Some(boost) = pattern.boost(t) {
                scored.push((boost, t.to_vec()));
            }
            Ok(())
        },
    )?;
    // `TopTermsRewrite`: the best `maxExpansions` by boost, a tie to the
    // smaller term; then term order.
    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    scored.truncate(
        f.max_expansions
            .min(crate::queryparser::classic::MAX_CLAUSE_COUNT),
    );
    scored.sort_by(|a, b| a.1.cmp(&b.1));
    let mut clauses = Vec::with_capacity(scored.len());
    for (boost, term) in scored {
        // `TopTermsRewrite` truncates a negative boost to 0
        let boost = boost.max(0.0);
        let tq = TermQuery::new(f.field.clone(), term);
        clauses.push((
            Occur::Should,
            if boost == 1.0 {
                ParsedQuery::Term(tq)
            } else if boost == 0.0 {
                // `BoostQuery.rewrite` of a zero boost: a constant score, so
                // that no score is needed
                ParsedQuery::Boost(
                    Box::new(ParsedQuery::Other(Clause::ConstantScore(Box::new(
                        crate::query::ConstantScoreQuery::new(Clause::Term(tq), 1.0),
                    )))),
                    0.0,
                )
            } else {
                ParsedQuery::Boost(Box::new(ParsedQuery::Term(tq)), boost)
            },
        ));
    }
    Ok(ParsedQuery::Boolean(ParsedBoolean {
        clauses,
        minimum_should_match: 0,
    }))
}

impl ParsedQuery {
    /// Every [`ComplexPhraseQuery`] in the tree replaced by its rewrite
    /// against `reader`.
    ///
    /// # Errors
    /// A phrase's rewrite errors.
    pub fn rewrite_complex_phrases(
        &self,
        reader: &dyn IndexReader,
    ) -> Result<ParsedQuery, QueryParserError> {
        Ok(match self {
            ParsedQuery::ComplexPhrase(c) => c.rewrite(reader)?,
            ParsedQuery::Boolean(b) => ParsedQuery::Boolean(ParsedBoolean {
                clauses: b
                    .clauses
                    .iter()
                    .map(|(o, c)| Ok((*o, c.rewrite_complex_phrases(reader)?)))
                    .collect::<Result<_, QueryParserError>>()?,
                minimum_should_match: b.minimum_should_match,
            }),
            ParsedQuery::Boost(q, b) => {
                ParsedQuery::Boost(Box::new(q.rewrite_complex_phrases(reader)?), *b)
            }
            other => other.clone(),
        })
    }
}

/// A complex phrase as a [`Clause`] before its rewrite: it cannot run
/// without the index's terms.
pub(crate) fn unrewritten_clause() -> Clause {
    Clause::MatchNoDocs(
        crate::query::MatchNoDocsQuery::new()
            .with_reason("ComplexPhraseQuery: call rewrite_complex_phrases first"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_phrase_prints_its_field_text_and_slop() {
        let q = ComplexPhraseQuery {
            field: "f".into(),
            text: "a b*".into(),
            slop: 2,
            in_order: true,
            contents: None,
        };
        assert_eq!(q.to_string("f"), "\"a b*\"~2");
        assert_eq!(q.to_string("g"), "f:\"a b*\"~2");
        let q = ComplexPhraseQuery { slop: 0, ..q };
        assert_eq!(q.to_string("f"), "\"a b*\"");
    }

    #[test]
    fn span_builders_refuse_clauses_of_two_fields() {
        let a = SpanNode::term("f", b"a".to_vec());
        let g = SpanNode::term("g", b"a".to_vec());
        let msg = |r: Result<SpanNode, QueryParserError>| r.unwrap_err().message;
        assert_eq!(
            msg(span_not(a.clone(), g.clone())),
            "Clauses must have same field."
        );
        assert_eq!(
            msg(span_or(vec![a.clone(), g.clone()])),
            "Clauses must have same field."
        );
        assert_eq!(
            msg(span_near(vec![a.clone(), g.clone()], 0, true)),
            "Clauses must have same field."
        );
        // a span of no field (a span-first here) is checked against nothing
        let first = SpanNode::first(g, 1);
        assert_eq!(span_field(&first), None);
        assert!(span_not(a.clone(), first.clone()).is_ok());
        assert!(span_or(vec![first, a]).is_ok());
    }

    #[test]
    fn phrases_are_equal_by_field_text_slop_and_order_not_contents() {
        let a = ComplexPhraseQuery {
            field: "f".into(),
            text: "a b".into(),
            slop: 0,
            in_order: true,
            contents: None,
        };
        let parsed = ComplexPhraseQuery {
            contents: Some(ParsedQuery::MatchAll),
            ..a.clone()
        };
        assert_eq!(a, parsed);
        assert_ne!(
            a,
            ComplexPhraseQuery {
                slop: 1,
                ..a.clone()
            }
        );
        assert_ne!(
            a,
            ComplexPhraseQuery {
                in_order: false,
                ..a.clone()
            }
        );
        assert_ne!(
            a,
            ComplexPhraseQuery {
                text: "a".into(),
                ..a.clone()
            }
        );
        assert_ne!(
            a,
            ComplexPhraseQuery {
                field: "g".into(),
                ..a.clone()
            }
        );
    }

    /// Shapes `BooleanQuery.rewrite` never leaves in a rewritten clause (an
    /// all-negative boolean, a match-nothing clause, a match-all one), which
    /// Java's `addComplexPhraseClause` still handles.
    #[test]
    fn boolean_clauses_become_span_ors_with_the_dummy_for_nothing() {
        let phrase = ComplexPhraseQuery {
            field: "f".into(),
            text: "x".into(),
            slop: 0,
            in_order: true,
            contents: None,
        };
        let term = |t: &str| ParsedQuery::Term(TermQuery::new("f", t.as_bytes().to_vec()));
        let boolean = |clauses: Vec<(Occur, ParsedQuery)>| ParsedBoolean {
            clauses,
            minimum_should_match: 0,
        };
        let dummy = SpanNode::term("f", DUMMY.as_bytes().to_vec());
        assert_eq!(
            phrase
                .boolean_span(&boolean(vec![(Occur::MustNot, term("a"))]))
                .unwrap(),
            dummy
        );
        assert_eq!(
            phrase
                .boolean_span(&boolean(vec![
                    (Occur::Should, ParsedQuery::MatchNoDocs(String::new())),
                    (Occur::Should, term("a")),
                ]))
                .unwrap(),
            SpanNode::Or {
                clauses: vec![dummy, SpanNode::term("f", b"a".to_vec())]
            }
        );
        // a nested boolean's words join the span-or, a boost is looked
        // through, a negated word is the span-not's exclude
        let nested = ParsedQuery::Boolean(boolean(vec![(Occur::Should, term("b"))]));
        assert_eq!(
            phrase
                .boolean_span(&boolean(vec![
                    (Occur::Should, nested),
                    (Occur::Should, ParsedQuery::Boost(Box::new(term("c")), 2.0)),
                    (Occur::MustNot, term("d")),
                ]))
                .unwrap(),
            SpanNode::Not {
                include: Box::new(SpanNode::Or {
                    clauses: vec![
                        SpanNode::Or {
                            clauses: vec![SpanNode::term("f", b"b".to_vec())]
                        },
                        SpanNode::term("f", b"c".to_vec())
                    ]
                }),
                exclude: Box::new(SpanNode::Or {
                    clauses: vec![SpanNode::term("f", b"d".to_vec())]
                }),
                pre: 0,
                post: 0,
            }
        );
        assert_eq!(
            phrase
                .boolean_span(&boolean(vec![(Occur::Should, ParsedQuery::MatchAll)]))
                .unwrap_err()
                .message,
            "Unknown query type:org.apache.lucene.search.MatchAllDocsQuery"
        );
    }

    #[test]
    fn an_unrewritten_phrase_matches_nothing_and_says_why() {
        let q = ParsedQuery::ComplexPhrase(Box::new(ComplexPhraseQuery {
            field: "f".into(),
            text: "a".into(),
            slop: 0,
            in_order: true,
            contents: None,
        }));
        let Clause::MatchNoDocs(m) = q.to_clause(3) else {
            panic!("not a match-no-docs");
        };
        assert_eq!(
            m.reason,
            "ComplexPhraseQuery: call rewrite_complex_phrases first"
        );
    }
}
