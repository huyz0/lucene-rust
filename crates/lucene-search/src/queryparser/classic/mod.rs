//! `org.apache.lucene.queryparser.classic`: `QueryParser`, the classic
//! query syntax (`+a -b c:d "e f"~2 g* [h TO i]^2`), and its errors.
//!
//! The lexer ([`token_manager`]) and parser ([`parser`]) are the JavaCC 7
//! code Lucene generates from `QueryParser.jj`, ported state for state and
//! production for production, so a query parses to Lucene's tree and a bad
//! one fails with Lucene's message (token, column and expected tokens).
//! `QueryParserBase`'s query building is in [`parser`] too.
//!
//! Not ported: date ranges (`setDateResolution`, `setLocale`,
//! `setTimeZone`): with no resolution set -- the default -- Lucene's date
//! parse of a range bound fails and is ignored, which is what this parser
//! does; `MultiTermQuery.RewriteMethod` and `determinizeWorkLimit` are not
//! configurable: a wildcard or regular expression is determinized within
//! the default limit, as Lucene's constructors do. `QueryParserBase`'s
//! protected extension points (`newBooleanClause`, `newFieldQuery`,
//! `getBooleanQuery`, ...) cannot be overridden.

/// `IndexSearcher.getMaxClauseCount()`'s default.
pub(crate) const MAX_CLAUSE_COUNT: usize = 1024;

mod constants;
mod parser;
pub(crate) use parser::{check_regexp, float_to_edits};
mod token_manager;

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use lucene_analysis::{AnalysisError, Analyzer};

use crate::query_visitor::Occur;
use crate::queryparser::ext::Extensions;
use crate::queryparser::{ParsedBoolean, ParsedQuery};

/// `QueryParser.Operator`: what an absent conjunction means.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operator {
    Or,
    And,
}

/// An exception `parse` throws: `ParseException` (`class` is
/// `"ParseException"`) or a runtime exception a query constructor threw.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryParserError {
    /// The Java exception's simple class name.
    pub class: &'static str,
    pub message: String,
}

impl QueryParserError {
    /// A `ParseException` an extension throws.
    pub fn parse_exception(message: impl Into<String>) -> Self {
        Self::parse(message)
    }

    pub(crate) fn parse(message: impl Into<String>) -> Self {
        QueryParserError {
            class: "ParseException",
            message: message.into(),
        }
    }

    pub(crate) fn illegal_argument(message: impl Into<String>) -> Self {
        QueryParserError {
            class: "IllegalArgumentException",
            message: message.into(),
        }
    }

    /// A checked `IOException` subclass of the surround rewrite.
    pub(crate) fn io_exception(class: &'static str, message: impl Into<String>) -> Self {
        QueryParserError {
            class,
            message: message.into(),
        }
    }

    pub(crate) fn runtime_class(class: &'static str, message: impl Into<String>) -> Self {
        QueryParserError {
            class,
            message: message.into(),
        }
    }

    pub(crate) fn runtime(message: impl Into<String>) -> Self {
        QueryParserError {
            class: "RuntimeException",
            message: message.into(),
        }
    }

    pub(crate) fn too_many_clauses() -> Self {
        QueryParserError {
            class: "TooManyClauses",
            message: format!("maxClauseCount is set to {MAX_CLAUSE_COUNT}"),
        }
    }

    pub(crate) fn too_complex(message: impl Into<String>) -> Self {
        QueryParserError {
            class: "TooComplexToDeterminizeException",
            message: message.into(),
        }
    }

    pub(crate) fn from_search(e: crate::Error) -> Self {
        match e {
            crate::Error::IllegalArgument(m) => Self::illegal_argument(m),
            other => Self::runtime(other.to_string()),
        }
    }

    pub(crate) fn from_analysis(e: AnalysisError) -> Self {
        match e {
            AnalysisError::IllegalArgument(m) => Self::illegal_argument(m),
            other => Self::runtime(other.to_string()),
        }
    }

    /// Whether this is a `ParseException`.
    pub fn is_parse_exception(&self) -> bool {
        self.class == "ParseException"
    }
}

impl fmt::Display for QueryParserError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.class, self.message)
    }
}

impl std::error::Error for QueryParserError {}

/// `QueryParser`: a default field, an analyzer, and the settings of
/// `QueryParserBase` and `QueryBuilder`.
pub struct QueryParser {
    /// `None` for a `MultiFieldQueryParser`, whose clauses with no field of
    /// their own go to every one of `fields`.
    field: Option<String>,
    analyzer: Arc<Analyzer>,
    fields: Vec<String>,
    boosts: HashMap<String, f32>,
    /// `ExtendableQueryParser`'s extensions.
    extensions: Option<Arc<Extensions>>,
    /// `ComplexPhraseQueryParser`'s `inOrder`, when this is one.
    complex_phrase_in_order: Option<bool>,
    operator: Operator,
    allow_leading_wildcard: bool,
    phrase_slop: i32,
    fuzzy_min_sim: f32,
    fuzzy_prefix_length: i32,
    auto_generate_phrase_queries: bool,
    split_on_whitespace: bool,
    enable_position_increments: bool,
    enable_graph_queries: bool,
    auto_generate_multi_term_synonyms_phrase_query: bool,
}

impl QueryParser {
    /// `QueryParser(f, a)`.
    pub fn new(field: impl Into<String>, analyzer: impl Into<Arc<Analyzer>>) -> Self {
        Self::with_field(Some(field.into()), analyzer.into())
    }

    /// `MultiFieldQueryParser(fields, analyzer)`: a clause with no field of
    /// its own is asked of every field.
    pub fn multi_field(fields: &[&str], analyzer: impl Into<Arc<Analyzer>>) -> Self {
        let mut p = Self::with_field(None, analyzer.into());
        p.fields = fields.iter().map(|f| f.to_string()).collect();
        p
    }

    /// `MultiFieldQueryParser(fields, analyzer, boosts)`: a field's queries
    /// boosted by its boost, where it has one.
    pub fn multi_field_with_boosts(
        fields: &[&str],
        analyzer: impl Into<Arc<Analyzer>>,
        boosts: HashMap<String, f32>,
    ) -> Self {
        let mut p = Self::multi_field(fields, analyzer);
        p.boosts = boosts;
        p
    }

    fn with_field(field: Option<String>, analyzer: Arc<Analyzer>) -> Self {
        QueryParser {
            field,
            analyzer,
            fields: Vec::new(),
            boosts: HashMap::new(),
            extensions: None,
            complex_phrase_in_order: None,
            operator: Operator::Or,
            allow_leading_wildcard: false,
            phrase_slop: 0,
            // `FuzzyQuery.defaultMaxEdits`, as a float.
            fuzzy_min_sim: 2.0,
            fuzzy_prefix_length: 0,
            auto_generate_phrase_queries: false,
            // `DEFAULT_SPLIT_ON_WHITESPACE`.
            split_on_whitespace: false,
            enable_position_increments: true,
            enable_graph_queries: true,
            auto_generate_multi_term_synonyms_phrase_query: false,
        }
    }

    /// The parser as `ExtendableQueryParser` makes it: a field query
    /// whose field name carries the extension delimiter goes to the
    /// extension registered under the key after it.
    pub fn with_extensions(mut self, extensions: Arc<Extensions>) -> Self {
        self.extensions = Some(extensions);
        self
    }

    /// `ComplexPhraseQueryParser(f, a)`: a quoted phrase is a
    /// [`ComplexPhraseQuery`](crate::queryparser::complex_phrase::ComplexPhraseQuery)
    /// whose words may be wildcards, fuzzy terms, ranges or booleans.
    pub fn complex_phrase(field: impl Into<String>, analyzer: impl Into<Arc<Analyzer>>) -> Self {
        let mut p = Self::new(field, analyzer);
        p.complex_phrase_in_order = Some(true);
        p
    }

    /// `ComplexPhraseQueryParser.setInOrder(inOrder)`: whether a phrase's
    /// words must keep their order; no effect on another parser.
    pub fn set_in_order(&mut self, in_order: bool) {
        if self.complex_phrase_in_order.is_some() {
            self.complex_phrase_in_order = Some(in_order);
        }
    }

    /// `getField()`: `None` for a multi-field parser.
    pub fn field(&self) -> Option<&str> {
        self.field.as_deref()
    }

    /// `getAnalyzer()`.
    pub fn analyzer(&self) -> &Analyzer {
        &self.analyzer
    }

    /// `setDefaultOperator(op)`.
    pub fn set_default_operator(&mut self, op: Operator) {
        self.operator = op;
    }

    /// `getDefaultOperator()`.
    pub fn default_operator(&self) -> Operator {
        self.operator
    }

    /// `setAllowLeadingWildcard(b)`.
    pub fn set_allow_leading_wildcard(&mut self, allow: bool) {
        self.allow_leading_wildcard = allow;
    }

    /// `getAllowLeadingWildcard()`.
    pub fn allow_leading_wildcard(&self) -> bool {
        self.allow_leading_wildcard
    }

    /// `setPhraseSlop(slop)`.
    pub fn set_phrase_slop(&mut self, slop: i32) {
        self.phrase_slop = slop;
    }

    /// `getPhraseSlop()`.
    pub fn phrase_slop(&self) -> i32 {
        self.phrase_slop
    }

    /// `setFuzzyMinSim(minSim)`.
    pub fn set_fuzzy_min_sim(&mut self, min_sim: f32) {
        self.fuzzy_min_sim = min_sim;
    }

    /// `getFuzzyMinSim()`.
    pub fn fuzzy_min_sim(&self) -> f32 {
        self.fuzzy_min_sim
    }

    /// `setFuzzyPrefixLength(len)`.
    pub fn set_fuzzy_prefix_length(&mut self, len: i32) {
        self.fuzzy_prefix_length = len;
    }

    /// `getFuzzyPrefixLength()`.
    pub fn fuzzy_prefix_length(&self) -> i32 {
        self.fuzzy_prefix_length
    }

    /// `setAutoGeneratePhraseQueries(value)`.
    ///
    /// # Errors
    /// `IllegalArgumentException` for `true` while whitespace is not split.
    pub fn set_auto_generate_phrase_queries(
        &mut self,
        value: bool,
    ) -> Result<(), QueryParserError> {
        if !self.split_on_whitespace && value {
            return Err(QueryParserError::illegal_argument(
                "setAutoGeneratePhraseQueries(true) is disallowed when getSplitOnWhitespace() == false",
            ));
        }
        self.auto_generate_phrase_queries = value;
        Ok(())
    }

    /// `getAutoGeneratePhraseQueries()`.
    pub fn auto_generate_phrase_queries(&self) -> bool {
        self.auto_generate_phrase_queries
    }

    /// `setSplitOnWhitespace(value)`.
    ///
    /// # Errors
    /// `IllegalArgumentException` for `false` while phrase queries are
    /// generated automatically.
    pub fn set_split_on_whitespace(&mut self, value: bool) -> Result<(), QueryParserError> {
        if !value && self.auto_generate_phrase_queries {
            return Err(QueryParserError::illegal_argument(
                "setSplitOnWhitespace(false) is disallowed when getAutoGeneratePhraseQueries() == true",
            ));
        }
        self.split_on_whitespace = value;
        Ok(())
    }

    /// `getSplitOnWhitespace()`.
    pub fn split_on_whitespace(&self) -> bool {
        self.split_on_whitespace
    }

    /// `setEnablePositionIncrements(enable)` (`QueryBuilder`).
    pub fn set_enable_position_increments(&mut self, enable: bool) {
        self.enable_position_increments = enable;
    }

    /// `setEnableGraphQueries(enable)` (`QueryBuilder`).
    pub fn set_enable_graph_queries(&mut self, enable: bool) {
        self.enable_graph_queries = enable;
    }

    /// `setAutoGenerateMultiTermSynonymsPhraseQuery(enable)`
    /// (`QueryBuilder`).
    pub fn set_auto_generate_multi_term_synonyms_phrase_query(&mut self, enable: bool) {
        self.auto_generate_multi_term_synonyms_phrase_query = enable;
    }

    /// `parse(query)`: the query, or `ParseException("Cannot parse
    /// '<query>': ...")`; `null` (every word analyzed away) is an empty
    /// `BooleanQuery`.
    ///
    /// # Errors
    /// The `ParseException`, or a runtime exception a query constructor
    /// threw (an invalid regular expression, a negative slop).
    pub fn parse(&self, query: &str) -> Result<ParsedQuery, QueryParserError> {
        let mut p = parser::Parser::new(self, query);
        let mut q = self.parse_top(&mut p, self.field.as_deref(), query)?;
        if self.complex_phrase_in_order.is_some() {
            // `ComplexPhraseQueryParser.parse`'s second pass: each phrase's
            // text parsed on its field, in the order the phrases were made.
            self.resolve_phrases(&mut q)?;
        }
        Ok(q)
    }

    fn resolve_phrases(&self, q: &mut ParsedQuery) -> Result<(), QueryParserError> {
        match q {
            ParsedQuery::ComplexPhrase(c) => {
                let mut p = parser::Parser::in_phrase(self, &c.text, &c.field);
                c.contents = Some(self.parse_top(&mut p, Some(&c.field), &c.text)?);
                if p.nested_phrase() {
                    // the phrase added to the list being iterated
                    return Err(QueryParserError::runtime_class(
                        "ConcurrentModificationException",
                        "null",
                    ));
                }
                Ok(())
            }
            ParsedQuery::Boolean(b) => {
                for (_, c) in &mut b.clauses {
                    self.resolve_phrases(c)?;
                }
                Ok(())
            }
            ParsedQuery::Boost(inner, _) => self.resolve_phrases(inner),
            _ => Ok(()),
        }
    }

    /// `QueryParserBase.parse(query)`'s body: `TopLevelQuery(field)` and
    /// its exceptions.
    fn parse_top(
        &self,
        p: &mut parser::Parser<'_>,
        field: Option<&str>,
        query: &str,
    ) -> Result<ParsedQuery, QueryParserError> {
        match p.top_level_query(field) {
            Ok(q) => Ok(q.unwrap_or_else(|| ParsedQuery::Boolean(ParsedBoolean::default()))),
            Err(parser::Fail::Parse(msg)) => Err(QueryParserError::parse(format!(
                "Cannot parse '{query}': {msg}"
            ))),
            Err(parser::Fail::TooManyClauses) => Err(QueryParserError::parse(format!(
                "Cannot parse '{query}': too many boolean clauses"
            ))),
            Err(parser::Fail::Runtime(e)) => Err(e),
        }
    }

    /// `MultiFieldQueryParser.parse(queries, fields, analyzer)`: each query
    /// parsed on its field, the non-empty ones `SHOULD`.
    ///
    /// # Errors
    /// `IllegalArgumentException` when the lengths differ, and the
    /// `ParseException` of a query that does not parse.
    pub fn parse_fields(
        queries: &[&str],
        fields: &[&str],
        analyzer: impl Into<Arc<Analyzer>>,
    ) -> Result<ParsedQuery, QueryParserError> {
        if queries.len() != fields.len() {
            return Err(QueryParserError::illegal_argument(
                "queries.length != fields.length",
            ));
        }
        let flags = vec![Occur::Should; fields.len()];
        Self::join_fields(queries, fields, &flags, analyzer.into())
    }

    /// `MultiFieldQueryParser.parse(query, fields, flags, analyzer)`: the
    /// query parsed on each field, joined with the field's flag.
    ///
    /// # Errors
    /// As [`Self::parse_fields`].
    pub fn parse_with_flags(
        query: &str,
        fields: &[&str],
        flags: &[Occur],
        analyzer: impl Into<Arc<Analyzer>>,
    ) -> Result<ParsedQuery, QueryParserError> {
        if fields.len() != flags.len() {
            return Err(QueryParserError::illegal_argument(
                "fields.length != flags.length",
            ));
        }
        let queries = vec![query; fields.len()];
        Self::join_fields(&queries, fields, flags, analyzer.into())
    }

    /// `MultiFieldQueryParser.parse(queries, fields, flags, analyzer)`.
    ///
    /// # Errors
    /// As [`Self::parse_fields`].
    pub fn parse_fields_with_flags(
        queries: &[&str],
        fields: &[&str],
        flags: &[Occur],
        analyzer: impl Into<Arc<Analyzer>>,
    ) -> Result<ParsedQuery, QueryParserError> {
        if !(queries.len() == fields.len() && queries.len() == flags.len()) {
            return Err(QueryParserError::illegal_argument(
                "queries, fields, and flags array have have different length",
            ));
        }
        Self::join_fields(queries, fields, flags, analyzer.into())
    }

    fn join_fields(
        queries: &[&str],
        fields: &[&str],
        flags: &[Occur],
        analyzer: Arc<Analyzer>,
    ) -> Result<ParsedQuery, QueryParserError> {
        let mut b = ParsedBoolean::default();
        for ((query, field), occur) in queries.iter().zip(fields).zip(flags) {
            let q = QueryParser::new(*field, Arc::clone(&analyzer)).parse(query)?;
            // an empty `BooleanQuery` (every word analyzed away) is left out
            if !matches!(&q, ParsedQuery::Boolean(bq) if bq.clauses.is_empty()) {
                b.clauses.push((*occur, q));
            }
        }
        Ok(ParsedQuery::Boolean(b))
    }

    /// `QueryParserBase.escape(s)`: every syntax character backslashed.
    pub fn escape(s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        for c in s.chars() {
            if matches!(
                c,
                '\\' | '+'
                    | '-'
                    | '!'
                    | '('
                    | ')'
                    | ':'
                    | '^'
                    | '['
                    | ']'
                    | '"'
                    | '{'
                    | '}'
                    | '~'
                    | '*'
                    | '?'
                    | '|'
                    | '&'
                    | '/'
            ) {
                out.push('\\');
            }
            out.push(c);
        }
        out
    }
}

#[cfg(test)]
mod tests;
