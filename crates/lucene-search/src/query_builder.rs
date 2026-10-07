//! `org.apache.lucene.util.QueryBuilder`: queries built from analyzed text
//! -- a term, a boolean of terms, a synonym set, a phrase, a multi-phrase,
//! or, over a token graph (a multi-token synonym), a boolean of the graph's
//! paths -- as the analysis chain decides.
//!
//! A port of Lucene 10.5.0's `QueryBuilder` over this workspace's streaming
//! analysis model (`lucene_analysis::TokenStream`) and its graph helper
//! (`lucene_analysis::GraphTokenStreamFiniteStrings`). The token stream is
//! read once into its tokens' attributes and replayed, as Java replays it
//! through a `CachingTokenFilter`.
//!
//! Differences: `BoostAttribute` is not part of this attribute model, so
//! every token's boost is `1.0` (`DEFAULT_BOOST`) and no term or phrase is
//! wrapped in a `BoostQuery` for it; `null` (no tokens) is `None`.
//! [`to_query_string`] is `Query.toString(field)` for the queries this
//! builder makes.
//!
//! Verified against Lucene by `tests/query_builder_fixtures.rs`
//! (`fixtures/src/GenQueryBuilder.java`): the query's `toString` and its
//! hits and scores, over `StandardAnalyzer` text and canned token graphs.

use lucene_analysis::{
    AnalysisError, Analyzer, AttributeSource, GraphTokenStreamFiniteStrings, TokenStream,
};

use crate::extended_query::SynonymQuery;
use crate::query::{BooleanQuery, BoostQuery, Clause, MultiPhraseQuery, PhraseQuery, TermQuery};
use crate::query_visitor::Occur;
use crate::{Error, Result};

/// `BoostAttribute.DEFAULT_BOOST`.
const DEFAULT_BOOST: f32 = 1.0;

fn analysis(e: AnalysisError) -> Error {
    Error::IllegalArgument(format!("Error analyzing query text: {e}"))
}

/// `QueryBuilder`.
pub struct QueryBuilder<'a> {
    analyzer: &'a Analyzer,
    enable_position_increments: bool,
    enable_graph_queries: bool,
    auto_generate_multi_term_synonyms_phrase_query: bool,
}

impl<'a> QueryBuilder<'a> {
    /// `new QueryBuilder(analyzer)`.
    pub fn new(analyzer: &'a Analyzer) -> Self {
        Self {
            analyzer,
            enable_position_increments: true,
            enable_graph_queries: true,
            auto_generate_multi_term_synonyms_phrase_query: false,
        }
    }

    /// `getAnalyzer()`.
    pub fn analyzer(&self) -> &'a Analyzer {
        self.analyzer
    }

    /// `setAnalyzer(analyzer)`.
    pub fn set_analyzer(&mut self, analyzer: &'a Analyzer) {
        self.analyzer = analyzer;
    }

    /// `getEnablePositionIncrements()`.
    pub fn enable_position_increments(&self) -> bool {
        self.enable_position_increments
    }

    /// `setEnablePositionIncrements(enable)`: phrase positions follow the
    /// tokens' increments (holes kept), or are consecutive.
    pub fn set_enable_position_increments(&mut self, enable: bool) {
        self.enable_position_increments = enable;
    }

    /// `getEnableGraphQueries()`.
    pub fn enable_graph_queries(&self) -> bool {
        self.enable_graph_queries
    }

    /// `setEnableGraphQueries(v)`: whether a token spanning several positions
    /// makes a graph query.
    pub fn set_enable_graph_queries(&mut self, v: bool) {
        self.enable_graph_queries = v;
    }

    /// `getAutoGenerateMultiTermSynonymsPhraseQuery()`.
    pub fn auto_generate_multi_term_synonyms_phrase_query(&self) -> bool {
        self.auto_generate_multi_term_synonyms_phrase_query
    }

    /// `setAutoGenerateMultiTermSynonymsPhraseQuery(enable)`: a multi-term
    /// synonym's path is a phrase rather than a conjunction.
    pub fn set_auto_generate_multi_term_synonyms_phrase_query(&mut self, enable: bool) {
        self.auto_generate_multi_term_synonyms_phrase_query = enable;
    }

    /// `createBooleanQuery(field, queryText)`: `SHOULD` between positions.
    ///
    /// # Errors
    /// The analyzer's errors.
    pub fn create_boolean_query(&self, field: &str, query_text: &str) -> Result<Option<Clause>> {
        self.create_boolean_query_with(field, query_text, Occur::Should)
    }

    /// `createBooleanQuery(field, queryText, operator)`.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for an operator other than `SHOULD` or
    /// `MUST`, and the analyzer's errors.
    pub fn create_boolean_query_with(
        &self,
        field: &str,
        query_text: &str,
        operator: Occur,
    ) -> Result<Option<Clause>> {
        if operator != Occur::Should && operator != Occur::Must {
            return Err(Error::IllegalArgument(
                "invalid operator: only SHOULD or MUST are allowed".to_string(),
            ));
        }
        self.create_field_query_from_text(operator, field, query_text, false, 0)
    }

    /// `createPhraseQuery(field, queryText)`.
    ///
    /// # Errors
    /// The analyzer's errors.
    pub fn create_phrase_query(&self, field: &str, query_text: &str) -> Result<Option<Clause>> {
        self.create_phrase_query_with_slop(field, query_text, 0)
    }

    /// `createPhraseQuery(field, queryText, phraseSlop)`.
    ///
    /// # Errors
    /// The analyzer's errors.
    pub fn create_phrase_query_with_slop(
        &self,
        field: &str,
        query_text: &str,
        phrase_slop: u32,
    ) -> Result<Option<Clause>> {
        self.create_field_query_from_text(Occur::Must, field, query_text, true, phrase_slop)
    }

    /// `createMinShouldMatchQuery(field, queryText, fraction)`: a boolean of
    /// the positions, `(int) (fraction * clauses)` of which must match.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for a fraction outside `[0, 1]`, and the
    /// analyzer's errors.
    pub fn create_min_should_match_query(
        &self,
        field: &str,
        query_text: &str,
        fraction: f32,
    ) -> Result<Option<Clause>> {
        if fraction.is_nan() || !(0.0..=1.0).contains(&fraction) {
            return Err(Error::IllegalArgument(
                "fraction should be >= 0 and <= 1".to_string(),
            ));
        }
        if fraction == 1.0 {
            return self.create_boolean_query_with(field, query_text, Occur::Must);
        }
        let query =
            self.create_field_query_from_text(Occur::Should, field, query_text, false, 0)?;
        Ok(query.map(|q| match q {
            // `addMinShouldMatchToBoolean`.
            Clause::Boolean(mut b) => {
                let clauses = b.must.len() + b.filter.len() + b.should.len() + b.must_not.len();
                b.minimum_should_match = (fraction * clauses as f32) as usize;
                Clause::Boolean(b)
            }
            other => other,
        }))
    }

    /// `createFieldQuery(analyzer, operator, field, queryText, quoted,
    /// phraseSlop)`: the analyzer's tokens for `query_text`.
    fn create_field_query_from_text(
        &self,
        operator: Occur,
        field: &str,
        query_text: &str,
        quoted: bool,
        phrase_slop: u32,
    ) -> Result<Option<Clause>> {
        let mut source = self
            .analyzer
            .token_stream(field, query_text)
            .map_err(analysis)?;
        let query = self.create_field_query(&mut source, operator, field, quoted, phrase_slop);
        source.close().map_err(analysis)?;
        query
    }

    /// `createFieldQuery(source, operator, field, quoted, phraseSlop)`: the
    /// query the tokens of `source` make -- a term, a synonym set, a
    /// boolean of positions, a (multi-)phrase, or a graph query.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for an operator other than `SHOULD` or
    /// `MUST`, and `source`'s errors.
    pub fn create_field_query(
        &self,
        source: &mut dyn TokenStream,
        operator: Occur,
        field: &str,
        quoted: bool,
        phrase_slop: u32,
    ) -> Result<Option<Clause>> {
        if operator != Occur::Should && operator != Occur::Must {
            return Err(Error::IllegalArgument(
                "invalid operator: only SHOULD or MUST are allowed".to_string(),
            ));
        }
        // `CachingTokenFilter`: read once, replayed below.
        let tokens = read_tokens(source).map_err(analysis)?;

        // Phase 1: the number of tokens and positions, synonyms, a graph.
        let num_tokens = tokens.len();
        let mut position_count = 0;
        let mut has_synonyms = false;
        let mut is_graph = false;
        for t in &tokens {
            let inc = t.position_increment();
            if inc != 0 {
                position_count += inc;
            } else {
                has_synonyms = true;
            }
            if self.enable_graph_queries && t.position_length() > 1 {
                is_graph = true;
            }
        }

        // Phase 2: a term, a boolean, a phrase or a graph query.
        Ok(if num_tokens == 0 {
            None
        } else if num_tokens == 1 {
            Some(self.analyze_term(field, &tokens[0]))
        } else if is_graph {
            if quoted {
                Some(self.analyze_graph_phrase(&tokens, field, phrase_slop)?)
            } else {
                Some(self.analyze_graph_boolean(field, &tokens, operator)?)
            }
        } else if quoted && position_count > 1 {
            if has_synonyms {
                Some(self.analyze_multi_phrase(field, &tokens, phrase_slop)?)
            } else {
                Some(self.analyze_phrase(field, &tokens, phrase_slop)?)
            }
        } else if position_count == 1 {
            Some(self.analyze_boolean(field, &tokens)?)
        } else {
            Some(self.analyze_multi_boolean(field, &tokens, operator)?)
        })
    }

    /// `analyzeTerm`.
    fn analyze_term(&self, field: &str, token: &AttributeSource) -> Clause {
        new_term_query(field, token.term_bytes().to_vec(), DEFAULT_BOOST)
    }

    /// `analyzeBoolean`: one position, synonyms.
    fn analyze_boolean(&self, field: &str, tokens: &[AttributeSource]) -> Result<Clause> {
        new_synonym_query(
            field,
            tokens
                .iter()
                .map(|t| (t.term_bytes().to_vec(), DEFAULT_BOOST))
                .collect(),
        )
    }

    /// `add(field, q, current, operator)`: a position's terms.
    fn add(
        &self,
        field: &str,
        q: &mut BooleanQuery,
        current: &[(Vec<u8>, f32)],
        operator: Occur,
    ) -> Result<()> {
        let clause = match current {
            [] => return Ok(()),
            [(term, boost)] => new_term_query(field, term.clone(), *boost),
            _ => new_synonym_query(field, current.to_vec())?,
        };
        push(q, clause, operator);
        Ok(())
    }

    /// `analyzeMultiBoolean`: each position's terms, `operator` between
    /// them.
    fn analyze_multi_boolean(
        &self,
        field: &str,
        tokens: &[AttributeSource],
        operator: Occur,
    ) -> Result<Clause> {
        let mut q = BooleanQuery::new();
        let mut current: Vec<(Vec<u8>, f32)> = Vec::new();
        for t in tokens {
            if t.position_increment() != 0 {
                self.add(field, &mut q, &current, operator)?;
                current.clear();
            }
            current.push((t.term_bytes().to_vec(), DEFAULT_BOOST));
        }
        self.add(field, &mut q, &current, operator)?;
        Ok(Clause::Boolean(Box::new(q)))
    }

    /// `analyzePhrase`.
    fn analyze_phrase(&self, field: &str, tokens: &[AttributeSource], slop: u32) -> Result<Clause> {
        let mut position = -1;
        let mut terms = Vec::with_capacity(tokens.len());
        for t in tokens {
            if self.enable_position_increments {
                position += t.position_increment();
            } else {
                position += 1;
            }
            terms.push((t.term_bytes().to_vec(), position));
        }
        let q = PhraseQuery::with_positions(field, terms)?.with_slop(slop);
        Ok(Clause::Phrase(q))
    }

    /// `analyzeMultiPhrase`: stacked terms as one position's alternatives.
    fn analyze_multi_phrase(
        &self,
        field: &str,
        tokens: &[AttributeSource],
        slop: u32,
    ) -> Result<Clause> {
        let mut position = -1;
        let mut entries: Vec<(Vec<Vec<u8>>, i32)> = Vec::new();
        let mut multi_terms: Vec<Vec<u8>> = Vec::new();
        let add = |entries: &mut Vec<(Vec<Vec<u8>>, i32)>, terms: Vec<Vec<u8>>, position| {
            // `add(terms)` without a position: the one after the last.
            let at = if self.enable_position_increments {
                position
            } else {
                entries.last().map_or(0, |e| e.1 + 1)
            };
            entries.push((terms, at));
        };
        for t in tokens {
            let inc = t.position_increment();
            if inc > 0 && !multi_terms.is_empty() {
                add(&mut entries, std::mem::take(&mut multi_terms), position);
            }
            position += inc;
            multi_terms.push(t.term_bytes().to_vec());
        }
        add(&mut entries, multi_terms, position);
        let mut q = MultiPhraseQuery::with_positions(field, entries)?;
        q.slop = slop;
        Ok(Clause::MultiPhrase(q))
    }

    /// `analyzeGraphBoolean`: the graph cut at its articulation points; a
    /// segment with side paths is a disjunction of its paths' queries, one
    /// without a synonym set of its terms.
    fn analyze_graph_boolean(
        &self,
        field: &str,
        tokens: &[AttributeSource],
        operator: Occur,
    ) -> Result<Clause> {
        let graph =
            GraphTokenStreamFiniteStrings::new(&mut Replay::new(tokens)).map_err(analysis)?;
        let mut builder = BooleanQuery::new();
        let points = graph.articulation_points().map_err(analysis)?;
        let mut last_state = 0;
        for i in 0..=points.len() {
            let start = last_state;
            let end = points.get(i).copied().unwrap_or(-1);
            last_state = end;
            let positional = if graph.has_side_path(start) {
                let mut queries = Vec::new();
                for mut side_path in graph.finite_strings_between(start, end).map_err(analysis)? {
                    queries.push(self.create_field_query(
                        &mut side_path,
                        Occur::Must,
                        field,
                        self.auto_generate_multi_term_synonyms_phrase_query,
                        0,
                    )?);
                }
                new_graph_synonym_query(queries)
            } else {
                let terms: Vec<(Vec<u8>, f32)> = graph
                    .terms(start)
                    .iter()
                    .map(|a| (a.term_bytes().to_vec(), DEFAULT_BOOST))
                    .collect();
                match &terms[..] {
                    [(term, boost)] => Some(new_term_query(field, term.clone(), *boost)),
                    _ => Some(new_synonym_query(field, terms)?),
                }
            };
            if let Some(q) = positional {
                push(&mut builder, q, operator);
            }
        }
        Ok(Clause::Boolean(Box::new(builder)))
    }

    /// `analyzeGraphPhrase`: a phrase per path of the graph, any of which
    /// may match.
    fn analyze_graph_phrase(
        &self,
        tokens: &[AttributeSource],
        field: &str,
        phrase_slop: u32,
    ) -> Result<Clause> {
        let graph =
            GraphTokenStreamFiniteStrings::new(&mut Replay::new(tokens)).map_err(analysis)?;
        let mut builder = BooleanQuery::new();
        for mut path in graph.finite_strings().map_err(analysis)? {
            if let Some(q) =
                self.create_field_query(&mut path, Occur::Must, field, true, phrase_slop)?
            {
                builder.should.push(q);
            }
        }
        Ok(Clause::Boolean(Box::new(builder)))
    }
}

/// Every token's attributes, `source` reset, consumed and ended.
fn read_tokens(
    source: &mut dyn TokenStream,
) -> std::result::Result<Vec<AttributeSource>, AnalysisError> {
    source.reset()?;
    let mut tokens = Vec::new();
    while source.increment_token()? {
        tokens.push(source.attributes().clone());
    }
    source.end()?;
    Ok(tokens)
}

/// The cached tokens replayed as a stream (`CachingTokenFilter` after its
/// first pass).
struct Replay<'t> {
    atts: AttributeSource,
    tokens: &'t [AttributeSource],
    upto: usize,
}

impl<'t> Replay<'t> {
    fn new(tokens: &'t [AttributeSource]) -> Self {
        Self {
            atts: AttributeSource::new(),
            tokens,
            upto: 0,
        }
    }
}

impl TokenStream for Replay<'_> {
    /// A source, not a wrapper: no conditional wrapper below it.
    fn conditional_root(&mut self) -> Option<&mut dyn std::any::Any> {
        None
    }
    fn attributes(&self) -> &AttributeSource {
        &self.atts
    }
    fn attributes_mut(&mut self) -> &mut AttributeSource {
        &mut self.atts
    }
    fn increment_token(&mut self) -> std::result::Result<bool, AnalysisError> {
        let Some(t) = self.tokens.get(self.upto) else {
            return Ok(false);
        };
        self.atts = t.clone();
        self.upto += 1;
        Ok(true)
    }
    fn reset(&mut self) -> std::result::Result<(), AnalysisError> {
        self.upto = 0;
        Ok(())
    }
}

/// A clause under the builder's operator.
fn push(q: &mut BooleanQuery, clause: Clause, operator: Occur) {
    match operator {
        Occur::Must => q.must.push(clause),
        _ => q.should.push(clause),
    }
}

/// `newTermQuery(term, boost)`.
fn new_term_query(field: &str, term: Vec<u8>, boost: f32) -> Clause {
    let q = Clause::Term(TermQuery::new(field, term));
    if boost == DEFAULT_BOOST {
        return q;
    }
    Clause::Boost(Box::new(BoostQuery {
        inner: Box::new(q),
        boost,
    }))
}

/// `newSynonymQuery(field, terms)`.
fn new_synonym_query(field: &str, terms: Vec<(Vec<u8>, f32)>) -> Result<Clause> {
    Ok(Clause::from(SynonymQuery::new(field, terms)?))
}

/// `newGraphSynonymQuery(queries)`: a disjunction, or its one clause.
fn new_graph_synonym_query(queries: Vec<Option<Clause>>) -> Option<Clause> {
    // Java adds each query, a `null` one included (`BooleanClause` refuses
    // it); a path of this builder always makes one.
    let mut bq = BooleanQuery::new();
    bq.should.extend(queries.into_iter().flatten());
    if bq.should.len() == 1 {
        return bq.should.pop();
    }
    Some(Clause::Boolean(Box::new(bq)))
}

/// `Query.toString(field)` for the queries a [`QueryBuilder`] makes (terms,
/// boosts, booleans, synonym sets, phrases and multi-phrases): the field
/// named only where it differs from `field`. `None` is Java's `"null"`.
pub fn to_query_string(query: Option<&Clause>, field: &str) -> String {
    match query {
        None => "null".to_string(),
        Some(q) => clause_string(q, field),
    }
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// `Float.toString` for the boosts these queries carry.
fn java_float(v: f32) -> String {
    if v.fract() == 0.0 && v.abs() < 1e7 {
        format!("{v:.1}")
    } else {
        format!("{v}")
    }
}

fn clause_string(q: &Clause, field: &str) -> String {
    match q {
        Clause::Term(t) => {
            if t.field == field {
                text(&t.term)
            } else {
                format!("{}:{}", t.field, text(&t.term))
            }
        }
        Clause::Boost(b) => format!(
            "({})^{}",
            clause_string(&b.inner, field),
            java_float(b.boost)
        ),
        Clause::Boolean(b) => {
            let msm = b.minimum_should_match;
            let mut out = String::new();
            if msm > 0 {
                out.push('(');
            }
            let clauses: Vec<(&str, &Clause)> = b
                .must
                .iter()
                .map(|c| ("+", c))
                .chain(b.filter.iter().map(|c| ("#", c)))
                .chain(b.should.iter().map(|c| ("", c)))
                .chain(b.must_not.iter().map(|c| ("-", c)))
                .collect();
            for (i, (occur, c)) in clauses.iter().enumerate() {
                out.push_str(occur);
                if matches!(c, Clause::Boolean(_)) {
                    out.push('(');
                    out.push_str(&clause_string(c, field));
                    out.push(')');
                } else {
                    out.push_str(&clause_string(c, field));
                }
                if i + 1 != clauses.len() {
                    out.push(' ');
                }
            }
            if msm > 0 {
                out.push(')');
                out.push('~');
                out.push_str(&msm.to_string());
            }
            out
        }
        Clause::Phrase(p) => {
            let mut out = String::new();
            if p.field != field {
                out.push_str(&p.field);
                out.push(':');
            }
            out.push('"');
            let positions = p.positions();
            let max = positions.last().copied().unwrap_or(-1);
            let mut pieces: Vec<Option<String>> = vec![None; usize::try_from(max + 1).unwrap_or(0)];
            for (term, &pos) in p.terms.iter().zip(&positions) {
                let slot = &mut pieces[pos as usize];
                *slot = Some(match slot.take() {
                    None => text(term),
                    Some(s) => format!("{s}|{}", text(term)),
                });
            }
            let joined: Vec<String> = pieces
                .into_iter()
                .map(|p| p.unwrap_or_else(|| "?".to_string()))
                .collect();
            out.push_str(&joined.join(" "));
            out.push('"');
            if p.slop != 0 {
                out.push('~');
                out.push_str(&p.slop.to_string());
            }
            out
        }
        Clause::MultiPhrase(m) => {
            let mut out = String::new();
            if m.field != field {
                out.push_str(&m.field);
                out.push(':');
            }
            out.push('"');
            let positions = m.positions();
            let mut last_pos = -1;
            for (i, (terms, &position)) in m.term_arrays.iter().zip(&positions).enumerate() {
                if i != 0 {
                    out.push(' ');
                    for _ in 1..(position - last_pos) {
                        out.push_str("? ");
                    }
                }
                if terms.len() > 1 {
                    let words: Vec<String> = terms.iter().map(|t| text(t)).collect();
                    out.push('(');
                    out.push_str(&words.join(" "));
                    out.push(')');
                } else if let Some(t) = terms.first() {
                    out.push_str(&text(t));
                }
                last_pos = position;
            }
            out.push('"');
            if m.slop != 0 {
                out.push('~');
                out.push_str(&m.slop.to_string());
            }
            out
        }
        Clause::Extended(e) => match e.as_ref() {
            crate::extended_query::ExtendedQuery::Synonym(s) => {
                let terms: Vec<String> = s
                    .terms
                    .iter()
                    .map(|(t, boost)| {
                        let term = clause_string(
                            &Clause::Term(TermQuery::new(&s.field, t.clone())),
                            field,
                        );
                        if *boost != 1.0 {
                            format!("{term}^{}", java_float(*boost))
                        } else {
                            term
                        }
                    })
                    .collect();
                format!("Synonym({})", terms.join(" "))
            }
            other => format!("{other:?}"),
        },
        other => format!("{other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn standard() -> Analyzer {
        Analyzer::standard(None)
    }

    #[test]
    fn standard_text_makes_terms_booleans_and_phrases() {
        let a = standard();
        let b = QueryBuilder::new(&a);
        let s = |q: Result<Option<Clause>>| to_query_string(q.unwrap().as_ref(), "body");
        assert_eq!(s(b.create_boolean_query("body", "")), "null");
        assert_eq!(s(b.create_boolean_query("body", "Foo")), "foo");
        assert_eq!(s(b.create_boolean_query("body", "foo bar")), "foo bar");
        assert_eq!(
            s(b.create_boolean_query_with("body", "foo bar", Occur::Must)),
            "+foo +bar"
        );
        assert_eq!(s(b.create_phrase_query("body", "foo bar")), "\"foo bar\"");
        assert_eq!(
            s(b.create_phrase_query_with_slop("body", "foo bar baz", 2)),
            "\"foo bar baz\"~2"
        );
        assert_eq!(
            s(b.create_min_should_match_query("body", "a b c d", 0.5)),
            "(a b c d)~2"
        );
        assert_eq!(
            s(b.create_min_should_match_query("body", "a b", 1.0)),
            "+a +b"
        );
        assert_eq!(s(b.create_min_should_match_query("body", "a", 0.5)), "a");
        assert!(b.create_min_should_match_query("body", "a", 1.5).is_err());
        assert!(b
            .create_boolean_query_with("body", "a", Occur::MustNot)
            .is_err());
        let mut ts = a.token_stream("body", "a b").unwrap();
        let err = b
            .create_field_query(&mut ts, Occur::Filter, "body", false, 0)
            .err()
            .unwrap();
        assert!(err.to_string().contains("only SHOULD or MUST"));
        assert!(analysis(AnalysisError::IllegalState("x".into()))
            .to_string()
            .contains("Error analyzing query text"));
        assert_eq!(
            to_query_string(b.create_boolean_query("f", "x").unwrap().as_ref(), "body"),
            "f:x"
        );
    }

    #[test]
    fn settings_round_trip() {
        let a = standard();
        let other = standard();
        let mut b = QueryBuilder::new(&a);
        assert!(b.enable_position_increments());
        assert!(b.enable_graph_queries());
        assert!(!b.auto_generate_multi_term_synonyms_phrase_query());
        b.set_enable_position_increments(false);
        b.set_enable_graph_queries(false);
        b.set_auto_generate_multi_term_synonyms_phrase_query(true);
        b.set_analyzer(&other);
        assert!(!b.enable_position_increments());
        assert!(!b.enable_graph_queries());
        assert!(b.auto_generate_multi_term_synonyms_phrase_query());
        assert!(std::ptr::eq(b.analyzer(), &other));
    }

    #[test]
    fn strings_of_boosts_and_other_queries() {
        let boosted = new_term_query("body", b"x".to_vec(), 2.0);
        assert_eq!(to_query_string(Some(&boosted), "body"), "(x)^2.0");
        let boosted = new_term_query("body", b"x".to_vec(), 0.25);
        assert_eq!(to_query_string(Some(&boosted), "f"), "(body:x)^0.25");
        let syn =
            new_synonym_query("body", vec![(b"b".to_vec(), 0.5), (b"a".to_vec(), 1.0)]).unwrap();
        assert_eq!(to_query_string(Some(&syn), "body"), "Synonym(a b^0.5)");
        let mut bq = BooleanQuery::new();
        bq.filter
            .push(Clause::Term(TermQuery::new("body", b"f".to_vec())));
        bq.must_not
            .push(Clause::Boolean(Box::new(BooleanQuery::new())));
        assert_eq!(
            to_query_string(Some(&Clause::Boolean(Box::new(bq))), "body"),
            "#f -()"
        );
        let p = PhraseQuery::with_positions(
            "t",
            [(b"a".to_vec(), 0), (b"b".to_vec(), 0), (b"c".to_vec(), 2)],
        )
        .unwrap();
        assert_eq!(
            to_query_string(Some(&Clause::Phrase(p)), "body"),
            "t:\"a|b ? c\""
        );
        let m = MultiPhraseQuery::with_positions(
            "t",
            [
                (vec![b"a".to_vec()], 0),
                (vec![b"b".to_vec(), b"c".to_vec()], 2),
            ],
        )
        .unwrap();
        assert_eq!(
            to_query_string(Some(&Clause::MultiPhrase(m)), "body"),
            "t:\"a ? (b c)\""
        );
        let all = Clause::MatchAllDocs(crate::query::MatchAllDocsQuery::new(1));
        assert!(to_query_string(Some(&all), "body").contains("MatchAll"));
        assert!(new_graph_synonym_query(vec![None]).is_some());
    }
}
