//! `org.apache.lucene.queryparser.surround.query`: the surround query tree
//! ([`SrndQuery`]), its `toString`, and its rewrite into Lucene queries
//! against an index (`makeLuceneQueryField`, with Java's lazy
//! `RewriteQuery` nodes rewritten at once: [`SrndQuery::make_lucene_query_field`]).
//!
//! `BasicQueryFactory` caps the term and span-term queries a rewrite may
//! make; `SpanNearClauseFactory` gathers the span clauses of a distance
//! operand (a `HashMap` in Java, whose iteration order is hash-dependent
//! and so unspecified: the clauses are kept in the order they were added).

use lucene_analysis::attributes::java_float_to_string;
use lucene_codecs::wildcard::WildcardPattern;

use crate::query::{SpanQuery, TermQuery};
use crate::query_visitor::Occur;
use crate::queryparser::classic::QueryParserError;
use crate::queryparser::{ParsedBoolean, ParsedQuery};
use lucene_codecs::blocktree::SeekStatus;

use crate::reader::IndexReader;

/// `TooManyBasicQueries`.
fn too_many_basic_queries(max: i32) -> QueryParserError {
    QueryParserError::io_exception(
        "TooManyBasicQueries",
        format!("Exceeded maximum of {max} basic queries."),
    )
}

/// `BasicQueryFactory`: counts the basic queries a rewrite makes.
#[derive(Debug, Clone)]
pub struct BasicQueryFactory {
    max_basic_queries: i32,
    queries_made: i32,
}

impl Default for BasicQueryFactory {
    fn default() -> Self {
        Self::new(1024)
    }
}

impl BasicQueryFactory {
    /// `BasicQueryFactory(maxBasicQueries)`.
    pub fn new(max_basic_queries: i32) -> Self {
        BasicQueryFactory {
            max_basic_queries,
            queries_made: 0,
        }
    }

    /// `getNrQueriesMade()`.
    pub fn nr_queries_made(&self) -> i32 {
        self.queries_made
    }

    /// `getMaxBasicQueries()`.
    pub fn max_basic_queries(&self) -> i32 {
        self.max_basic_queries
    }

    fn check_max(&mut self) -> Result<(), QueryParserError> {
        if self.queries_made >= self.max_basic_queries {
            return Err(too_many_basic_queries(self.max_basic_queries));
        }
        self.queries_made += 1;
        Ok(())
    }

    /// `newTermQuery(term)`.
    pub fn new_term_query(
        &mut self,
        field: &str,
        term: &[u8],
    ) -> Result<ParsedQuery, QueryParserError> {
        self.check_max()?;
        Ok(ParsedQuery::Term(TermQuery::new(field, term.to_vec())))
    }

    /// `newSpanTermQuery(term)`.
    pub fn new_span_term_query(
        &mut self,
        field: &str,
        term: &[u8],
    ) -> Result<SpanQuery, QueryParserError> {
        self.check_max()?;
        Ok(SpanQuery::SpanTerm {
            field: field.to_string(),
            term: term.to_vec(),
        })
    }
}

impl std::fmt::Display for BasicQueryFactory {
    /// `toString()`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "org.apache.lucene.queryparser.surround.query.BasicQueryFactory(maxBasicQueries: {}, queriesMade: {})",
            self.max_basic_queries, self.queries_made
        )
    }
}

/// `SpanNearClauseFactory`: the span clauses of one distance operand.
struct SpanNearClauseFactory<'r> {
    reader: &'r dyn IndexReader,
    field: String,
    clauses: Vec<SpanQuery>,
}

impl<'r> SpanNearClauseFactory<'r> {
    fn new(reader: &'r dyn IndexReader, field: &str) -> Self {
        SpanNearClauseFactory {
            reader,
            field: field.to_string(),
            clauses: Vec::new(),
        }
    }

    /// `addSpanQueryWeighted(sq, weight)`: the weight is not used past
    /// this, only the key's presence.
    fn add_span_query(&mut self, sq: SpanQuery) {
        if !self.clauses.contains(&sq) {
            self.clauses.push(sq);
        }
    }

    /// `makeSpanClause()`.
    fn make_span_clause(&mut self) -> SpanQuery {
        if self.clauses.len() == 1 {
            self.clauses.remove(0)
        } else {
            SpanQuery::SpanOr {
                clauses: std::mem::take(&mut self.clauses),
            }
        }
    }
}

/// A surround query: its kind, and the weight `^w` gave it.
#[derive(Debug, Clone, PartialEq)]
pub struct SrndQuery {
    kind: Kind,
    weight: f32,
    weighted: bool,
}

/// `ComposedQuery`'s state.
#[derive(Debug, Clone, PartialEq)]
struct Composed {
    queries: Vec<SrndQuery>,
    infix: bool,
    op_name: String,
}

#[derive(Debug, Clone, PartialEq)]
enum Kind {
    /// `SrndTermQuery`.
    Term { text: String, quoted: bool },
    /// `SrndPrefixQuery`.
    Prefix {
        prefix: String,
        quoted: bool,
        truncator: char,
    },
    /// `SrndTruncQuery`.
    Trunc { truncated: String },
    /// `OrQuery`.
    Or(Composed),
    /// `AndQuery`.
    And(Composed),
    /// `NotQuery`.
    Not(Composed),
    /// `DistanceQuery`.
    Distance {
        composed: Composed,
        op_distance: i32,
        ordered: bool,
    },
    /// `FieldsQuery`.
    Fields {
        q: Box<SrndQuery>,
        field_names: Vec<String>,
        field_op: char,
    },
}

impl std::fmt::Display for SrndQuery {
    /// `toString()`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.render())
    }
}

impl SrndQuery {
    fn new(kind: Kind) -> Self {
        SrndQuery {
            kind,
            weight: 1.0,
            weighted: false,
        }
    }

    /// `new SrndTermQuery(termText, quoted)`.
    pub fn term(text: impl Into<String>, quoted: bool) -> Self {
        Self::new(Kind::Term {
            text: text.into(),
            quoted,
        })
    }

    /// `new SrndPrefixQuery(prefix, quoted, truncator)`.
    pub fn prefix(prefix: impl Into<String>, quoted: bool, truncator: char) -> Self {
        Self::new(Kind::Prefix {
            prefix: prefix.into(),
            quoted,
            truncator,
        })
    }

    /// `new SrndTruncQuery(truncated)`.
    ///
    /// # Errors
    /// `TooComplexToDeterminizeException` for a pattern whose automaton is
    /// too complex, as Java's constructor throws.
    pub fn trunc(truncated: impl Into<String>) -> Result<Self, QueryParserError> {
        let truncated = truncated.into();
        crate::query_visitor::wildcard_automaton_checked(truncated.as_bytes())
            .map_err(|e| QueryParserError::too_complex(e.to_string()))?;
        Ok(Self::new(Kind::Trunc { truncated }))
    }

    /// `new OrQuery(queries, infix, opName)`.
    pub fn or(queries: Vec<SrndQuery>, infix: bool, op_name: impl Into<String>) -> Self {
        Self::new(Kind::Or(Composed {
            queries,
            infix,
            op_name: op_name.into(),
        }))
    }

    /// `new AndQuery(queries, infix, opName)`.
    pub fn and(queries: Vec<SrndQuery>, infix: bool, op_name: impl Into<String>) -> Self {
        Self::new(Kind::And(Composed {
            queries,
            infix,
            op_name: op_name.into(),
        }))
    }

    /// `new NotQuery(queries, opName)`.
    pub fn not(queries: Vec<SrndQuery>, op_name: impl Into<String>) -> Self {
        Self::new(Kind::Not(Composed {
            queries,
            infix: true,
            op_name: op_name.into(),
        }))
    }

    /// `new DistanceQuery(queries, infix, opDistance, opName, ordered)`.
    pub fn distance(
        queries: Vec<SrndQuery>,
        infix: bool,
        op_distance: i32,
        op_name: impl Into<String>,
        ordered: bool,
    ) -> Self {
        Self::new(Kind::Distance {
            composed: Composed {
                queries,
                infix,
                op_name: op_name.into(),
            },
            op_distance,
            ordered,
        })
    }

    /// `new FieldsQuery(q, fieldNames, fieldOp)`.
    pub fn fields(q: SrndQuery, field_names: Vec<String>, field_op: char) -> Self {
        Self::new(Kind::Fields {
            q: Box::new(q),
            field_names,
            field_op,
        })
    }

    /// `setWeight(w)`.
    pub fn set_weight(&mut self, w: f32) {
        self.weight = w;
        self.weighted = true;
    }

    /// `isWeighted()`.
    pub fn is_weighted(&self) -> bool {
        self.weighted
    }

    /// `getWeight()`.
    pub fn weight(&self) -> f32 {
        self.weight
    }

    fn composed(&self) -> Option<&Composed> {
        match &self.kind {
            Kind::Or(c) | Kind::And(c) | Kind::Not(c) => Some(c),
            Kind::Distance { composed, .. } => Some(composed),
            _ => None,
        }
    }

    fn weight_to_string(&self, r: &mut String) {
        if self.weighted {
            r.push('^');
            r.push_str(&java_float_to_string(self.weight));
        }
    }

    fn render(&self) -> String {
        let mut r = String::new();
        match &self.kind {
            Kind::Term { text, quoted } => {
                if *quoted {
                    r.push('"');
                }
                r.push_str(text);
                if *quoted {
                    r.push('"');
                }
                self.weight_to_string(&mut r);
            }
            Kind::Prefix {
                prefix,
                quoted,
                truncator,
            } => {
                if *quoted {
                    r.push('"');
                }
                r.push_str(prefix);
                if *quoted {
                    r.push('"');
                }
                r.push(*truncator);
                self.weight_to_string(&mut r);
            }
            Kind::Trunc { truncated } => {
                r.push_str(truncated);
                self.weight_to_string(&mut r);
            }
            Kind::Fields {
                q,
                field_names,
                field_op,
            } => {
                r.push('(');
                for name in field_names {
                    r.push_str(name);
                    r.push(*field_op);
                }
                r.push_str(&q.render());
                r.push(')');
            }
            _ => {
                let Some(c) = self.composed() else {
                    return r;
                };
                if c.infix {
                    // Brackets are possibly redundant in the result.
                    r.push('(');
                    for (i, q) in c.queries.iter().enumerate() {
                        if i != 0 {
                            r.push(' ');
                            r.push_str(&c.op_name);
                            r.push(' ');
                        }
                        r.push_str(&q.render());
                    }
                    r.push(')');
                } else {
                    r.push_str(&c.op_name);
                    r.push('(');
                    for (i, q) in c.queries.iter().enumerate() {
                        if i != 0 {
                            r.push_str(", ");
                        }
                        r.push_str(&q.render());
                    }
                    r.push(')');
                }
                self.weight_to_string(&mut r);
            }
        }
        r
    }

    /// `isFieldsSubQueryAcceptable()`.
    pub fn is_fields_sub_query_acceptable(&self) -> bool {
        match &self.kind {
            Kind::Fields { .. } => false,
            _ => match self.composed() {
                // at least one subquery should be acceptable
                Some(c) => c.queries.iter().any(Self::is_fields_sub_query_acceptable),
                None => true,
            },
        }
    }

    /// Whether this is a `DistanceSubQuery`.
    fn is_distance_sub_query(&self) -> bool {
        matches!(
            self.kind,
            Kind::Term { .. }
                | Kind::Prefix { .. }
                | Kind::Trunc { .. }
                | Kind::Or(_)
                | Kind::Distance { .. }
        )
    }

    /// `distanceSubQueryNotAllowed()`: `None` when the query may be an
    /// operand of a distance operator.
    pub fn distance_sub_query_not_allowed(&self) -> Option<String> {
        match &self.kind {
            Kind::Term { .. } | Kind::Prefix { .. } | Kind::Trunc { .. } => None,
            Kind::Or(c) => {
                for q in &c.queries {
                    if q.is_distance_sub_query() {
                        if let Some(m) = q.distance_sub_query_not_allowed() {
                            return Some(m);
                        }
                    } else {
                        return Some(format!("subquery not allowed: {}", q.render()));
                    }
                }
                None
            }
            Kind::Distance { composed, .. } => {
                for q in &composed.queries {
                    if q.is_distance_sub_query() {
                        if let Some(m) = q.distance_sub_query_not_allowed() {
                            return Some(m);
                        }
                    } else {
                        return Some(format!(
                            "Operator {} does not allow subquery {}",
                            composed.op_name,
                            q.render()
                        ));
                    }
                }
                None
            }
            _ => None,
        }
    }

    // ------------------------------------------------------- the rewrite

    /// `makeLuceneQueryField(fieldName, qf)` followed by the rewrite of its
    /// `RewriteQuery` nodes against `reader`: the terms a term, prefix or
    /// truncation matches are expanded, and a distance operator becomes a
    /// `SpanNearQuery`.
    ///
    /// # Errors
    /// `TooManyBasicQueries` past `qf`'s limit, and the reader's errors.
    pub fn make_lucene_query_field(
        &self,
        field: Option<&str>,
        qf: &mut BasicQueryFactory,
        reader: &dyn IndexReader,
    ) -> Result<ParsedQuery, QueryParserError> {
        let q = self.make_lucene_query_field_no_boost(field, qf, reader)?;
        if !self.weighted {
            return Ok(q);
        }
        // `new BoostQuery(q, getWeight())`; weight may be at any level in a
        // SrndQuery
        if !self.weight.is_finite() || self.weight < 0.0 {
            return Err(QueryParserError::illegal_argument(format!(
                "boost must be a positive float, got {}",
                java_float_to_string(self.weight)
            )));
        }
        Ok(ParsedQuery::Boost(Box::new(q), self.weight))
    }

    fn make_lucene_query_field_no_boost(
        &self,
        field: Option<&str>,
        qf: &mut BasicQueryFactory,
        reader: &dyn IndexReader,
    ) -> Result<ParsedQuery, QueryParserError> {
        match &self.kind {
            Kind::Term { .. } | Kind::Prefix { .. } | Kind::Trunc { .. } => {
                // `SimpleTermRewriteQuery.rewrite`
                let field = field.unwrap_or_default();
                let mut sub = Vec::new();
                self.visit_matching_terms(reader, field, &mut |term| {
                    sub.push(qf.new_term_query(field, term)?);
                    Ok(())
                })?;
                Ok(match sub.len() {
                    0 => ParsedQuery::MatchNoDocs(String::new()),
                    1 => sub.remove(0),
                    _ => make_boolean_query(sub, Occur::Should),
                })
            }
            Kind::Or(c) => {
                make_composed(c, field, qf, reader).map(|s| make_boolean_query(s, Occur::Should))
            }
            Kind::And(c) => {
                make_composed(c, field, qf, reader).map(|s| make_boolean_query(s, Occur::Must))
            }
            Kind::Not(c) => {
                let mut subs = make_composed(c, field, qf, reader)?;
                let first = subs.remove(0);
                let mut clauses = vec![(Occur::Must, first)];
                clauses.extend(subs.into_iter().map(|q| (Occur::MustNot, q)));
                Ok(ParsedQuery::Boolean(ParsedBoolean {
                    clauses,
                    minimum_should_match: 0,
                }))
            }
            Kind::Distance { .. } => {
                // `DistanceRewriteQuery.rewrite`
                self.span_near_query(reader, field.unwrap_or_default(), qf)
            }
            Kind::Fields {
                q,
                field_names,
                field_op,
            } => {
                // `FieldsQuery.makeLuceneQueryNoBoost`
                if field_names.len() == 1 {
                    return q.make_lucene_query_field_no_boost(Some(&field_names[0]), qf, reader);
                }
                let queries: Vec<SrndQuery> = field_names
                    .iter()
                    .map(|name| SrndQuery::fields((**q).clone(), vec![name.clone()], *field_op))
                    .collect();
                SrndQuery::or(queries, true, "OR").make_lucene_query_field(None, qf, reader)
            }
        }
    }

    /// `SimpleTerm.visitMatchingTerms(reader, fieldName, mtv)`.
    fn visit_matching_terms(
        &self,
        reader: &dyn IndexReader,
        field: &str,
        mtv: &mut dyn FnMut(&[u8]) -> Result<(), QueryParserError>,
    ) -> Result<(), QueryParserError> {
        let search = |e: crate::Error| QueryParserError::from_search(e);
        let Some(terms) =
            crate::multi_terms::MultiTerms::get_terms(reader, field).map_err(search)?
        else {
            return Ok(());
        };
        let mut te = terms.iterator().map_err(search)?;
        match &self.kind {
            Kind::Term { text, .. } => {
                // check term presence in index here for symmetry with other
                // SimpleTerm's
                if te.try_seek_ceil(text.as_bytes()).map_err(search)? == SeekStatus::Found {
                    mtv(text.as_bytes())?;
                }
            }
            Kind::Prefix { prefix, .. } => {
                let prefix = prefix.as_bytes();
                if te.try_seek_ceil(prefix).map_err(search)? != SeekStatus::End {
                    while let Some(t) = te.term().map(<[u8]>::to_vec) {
                        if !t.starts_with(prefix) {
                            break;
                        }
                        mtv(&t)?;
                        if te.next().map_err(search)?.is_none() {
                            break;
                        }
                    }
                }
            }
            Kind::Trunc { truncated } => {
                let pattern = WildcardPattern::new(truncated.as_bytes());
                while let Some(t) = te.next().map_err(search)? {
                    if pattern.matches(t) {
                        let t = t.to_vec();
                        mtv(&t)?;
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// `SimpleTerm.addSpanQueries` and the other `DistanceSubQuery`s.
    fn add_span_queries(
        &self,
        sncf: &mut SpanNearClauseFactory<'_>,
        qf: &mut BasicQueryFactory,
    ) -> Result<(), QueryParserError> {
        match &self.kind {
            Kind::Term { .. } | Kind::Prefix { .. } | Kind::Trunc { .. } => {
                let field = sncf.field.clone();
                let mut found = Vec::new();
                self.visit_matching_terms(sncf.reader, &field, &mut |term| {
                    found.push(qf.new_span_term_query(&field, term)?);
                    Ok(())
                })?;
                for sq in found {
                    sncf.add_span_query(sq);
                }
                Ok(())
            }
            Kind::Or(c) => {
                for q in &c.queries {
                    q.add_span_queries(sncf, qf)?;
                }
                Ok(())
            }
            Kind::Distance { .. } => {
                let snq = self.span_near_query(sncf.reader, &sncf.field.clone(), qf)?;
                // `addSpanQuery`: a `MatchNoDocsQuery` adds nothing
                if let ParsedQuery::Span(sq) = snq {
                    sncf.add_span_query(sq);
                }
                Ok(())
            }
            _ => Err(QueryParserError::runtime("ClassCastException")),
        }
    }

    /// `DistanceQuery.getSpanNearQuery(reader, fieldName, qf)`.
    fn span_near_query(
        &self,
        reader: &dyn IndexReader,
        field: &str,
        qf: &mut BasicQueryFactory,
    ) -> Result<ParsedQuery, QueryParserError> {
        let Kind::Distance {
            composed,
            op_distance,
            ordered,
        } = &self.kind
        else {
            return Err(QueryParserError::runtime("ClassCastException"));
        };
        let mut span_clauses = Vec::with_capacity(composed.queries.len());
        let mut subs = composed.queries.iter();
        while let Some(q) = subs.next() {
            let mut sncf = SpanNearClauseFactory::new(reader, field);
            q.add_span_queries(&mut sncf, qf)?;
            if sncf.clauses.is_empty() {
                // distance operator requires all sub queries
                for rest in subs {
                    // produce evt. error messages but ignore results
                    rest.add_span_queries(&mut sncf, qf)?;
                    sncf.clauses.clear();
                }
                return Ok(ParsedQuery::MatchNoDocs(String::new()));
            }
            span_clauses.push(sncf.make_span_clause());
        }
        Ok(ParsedQuery::Span(SpanQuery::SpanNear {
            clauses: span_clauses,
            slop: u32::try_from(op_distance - 1).unwrap_or(0),
            in_order: *ordered,
        }))
    }
}

/// `ComposedQuery.makeLuceneSubQueriesField(fn, qf)`.
fn make_composed(
    c: &Composed,
    field: Option<&str>,
    qf: &mut BasicQueryFactory,
    reader: &dyn IndexReader,
) -> Result<Vec<ParsedQuery>, QueryParserError> {
    c.queries
        .iter()
        .map(|q| q.make_lucene_query_field(field, qf, reader))
        .collect()
}

/// `SrndBooleanQuery.makeBooleanQuery(queries, occur)`.
fn make_boolean_query(queries: Vec<ParsedQuery>, occur: Occur) -> ParsedQuery {
    ParsedQuery::Boolean(ParsedBoolean {
        clauses: queries.into_iter().map(|q| (occur, q)).collect(),
        minimum_should_match: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_basic_query_factory_counts_and_stops_at_its_maximum() {
        let mut qf = BasicQueryFactory::new(2);
        assert_eq!((qf.max_basic_queries(), qf.nr_queries_made()), (2, 0));
        assert!(qf.new_term_query("f", b"a").is_ok());
        assert!(qf.new_span_term_query("f", b"b").is_ok());
        assert_eq!(qf.nr_queries_made(), 2);
        let e = qf.new_term_query("f", b"c").unwrap_err();
        assert_eq!(e.class, "TooManyBasicQueries");
        assert_eq!(e.message, "Exceeded maximum of 2 basic queries.");
        assert_eq!(
            qf.to_string(),
            "org.apache.lucene.queryparser.surround.query.BasicQueryFactory(maxBasicQueries: 2, queriesMade: 2)"
        );
        assert_eq!(BasicQueryFactory::default().max_basic_queries(), 1024);
    }

    #[test]
    fn queries_print_as_the_parser_wrote_them() {
        let mut w = SrndQuery::term("a", true);
        w.set_weight(2.0);
        assert!(w.is_weighted());
        assert_eq!(w.weight(), 2.0);
        assert_eq!(w.to_string(), "\"a\"^2.0");
        let q = SrndQuery::prefix("ab", false, '*');
        assert_eq!(q.render(), "ab*");
        let t = SrndQuery::trunc("a?c*").unwrap();
        assert_eq!(t.to_string(), "a?c*");
        let or = SrndQuery::or(vec![q.clone(), t.clone()], false, "OR");
        assert_eq!(or.to_string(), "OR(ab*, a?c*)");
        let f = SrndQuery::fields(or.clone(), vec!["x".into(), "y".into()], ':');
        assert_eq!(f.to_string(), "(x:y:OR(ab*, a?c*))");
        assert!(!f.is_fields_sub_query_acceptable());
        assert!(or.is_fields_sub_query_acceptable());
        let not = SrndQuery::not(vec![q.clone(), t.clone()], "NOT");
        assert_eq!(not.to_string(), "(ab* NOT a?c*)");
        let and = SrndQuery::and(vec![q, t], true, "AND");
        assert_eq!(and.to_string(), "(ab* AND a?c*)");
        assert!(
            !SrndQuery::fields(and.clone(), vec!["x".into()], ':').is_fields_sub_query_acceptable()
        );
        assert!(!SrndQuery::or(vec![f.clone(), f], true, "OR").is_fields_sub_query_acceptable());
    }

    #[test]
    fn distance_operands_must_be_distance_sub_queries() {
        let term = || SrndQuery::term("a", false);
        let and = SrndQuery::and(vec![term(), term()], true, "AND");
        let d = SrndQuery::distance(vec![term(), and.clone()], true, 1, "W", true);
        assert_eq!(
            d.distance_sub_query_not_allowed().as_deref(),
            Some("Operator W does not allow subquery (a AND a)")
        );
        let or = SrndQuery::or(vec![term(), and], true, "OR");
        assert_eq!(
            or.distance_sub_query_not_allowed().as_deref(),
            Some("subquery not allowed: (a AND a)")
        );
        let nested = SrndQuery::distance(vec![term(), or], true, 2, "2N", false);
        assert!(nested.distance_sub_query_not_allowed().is_some());
        let fine = SrndQuery::distance(
            vec![term(), SrndQuery::or(vec![term(), term()], true, "OR")],
            true,
            1,
            "W",
            true,
        );
        assert_eq!(fine.distance_sub_query_not_allowed(), None);
        assert_eq!(term().distance_sub_query_not_allowed(), None);
    }

    #[test]
    fn a_truncation_too_complex_to_determinize_is_refused() {
        let pattern = format!("{}x", "?*".repeat(40));
        let e = SrndQuery::trunc(pattern).err();
        // either it determinizes within the default effort, or it is refused
        // with Lucene's exception
        if let Some(e) = e {
            assert_eq!(e.class, "TooComplexToDeterminizeException");
        }
    }
}
