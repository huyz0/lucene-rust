//! `org.apache.lucene.queryparser.flexible`: the flexible query parser
//! framework with its standard syntax, `StandardQueryParser`.
//!
//! A query string goes through three stages, as in Java: the syntax parser
//! (`StandardSyntaxParser`, a JavaCC grammar: the lexer in
//! `token_manager.rs` and the lookaheads in `lookahead.rs` are translated by
//! `crates/lucene-search/tools/javacc_*.py`, the productions in `parser.rs`
//! by hand) builds a node tree, the processor pipeline rewrites it, and the
//! builders turn it into a query.

pub(crate) mod builders;
pub(crate) mod constants;
pub(crate) mod lookahead;
pub(crate) mod nodes;
pub(crate) mod parser;
pub(crate) mod processors;
#[cfg(test)]
mod tests;
pub(crate) mod token_manager;

use std::sync::Arc;

use lucene_analysis::Analyzer;

use crate::extended_query::RewriteMethod;
use crate::queryparser::classic::QueryParserError;
use crate::queryparser::ParsedQuery;

/// `StandardQueryConfigHandler.Operator`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operator {
    And,
    Or,
}

/// `org.apache.lucene.queryparser.flexible.standard.StandardQueryParser`.
pub struct StandardQueryParser {
    pub(crate) analyzer: Option<Arc<Analyzer>>,
    pub(crate) default_operator: Operator,
    pub(crate) allow_leading_wildcard: bool,
    pub(crate) phrase_slop: i32,
    pub(crate) enable_position_increments: bool,
    pub(crate) multi_fields: Option<Vec<String>>,
    pub(crate) fuzzy_min_sim: f32,
    pub(crate) fuzzy_prefix_length: i32,
    /// `FIELD_BOOST_MAP`, in the order given.
    pub(crate) field_boosts: Vec<(String, f32)>,
    /// `PrecedenceQueryParser`: AND binds tighter than OR, booleans nest.
    pub(crate) precedence: bool,
    /// `MULTI_TERM_REWRITE_METHOD`, which `MultiTermRewriteMethodProcessor`
    /// hands every wildcard, prefix, regexp and range node's builder.
    pub(crate) multi_term_rewrite_method: RewriteMethod,
}

impl Default for StandardQueryParser {
    fn default() -> Self {
        Self::new()
    }
}

impl StandardQueryParser {
    /// `new StandardQueryParser()`: no analyzer, `OR`, position increments
    /// enabled.
    pub fn new() -> Self {
        StandardQueryParser {
            analyzer: None,
            default_operator: Operator::Or,
            allow_leading_wildcard: false,
            phrase_slop: 0,
            enable_position_increments: true,
            multi_fields: None,
            // `FuzzyConfig`: `FuzzyQuery.defaultMaxEdits`, `defaultPrefixLength`
            fuzzy_min_sim: 2.0,
            fuzzy_prefix_length: 0,
            field_boosts: Vec::new(),
            precedence: false,
            multi_term_rewrite_method: RewriteMethod::ConstantScoreBlended,
        }
    }

    /// `new PrecedenceQueryParser(analyzer)` (`analyzer` `None`: `new
    /// PrecedenceQueryParser()`): the standard parser whose pipeline leaves
    /// booleans nested (no `BooleanQuery2ModifierNodeProcessor`) and ends
    /// with `BooleanModifiersQueryNodeProcessor`.
    pub fn precedence(analyzer: Option<Arc<Analyzer>>) -> Self {
        let mut p = Self::new();
        p.analyzer = analyzer;
        p.precedence = true;
        p
    }

    /// `new StandardQueryParser(analyzer)`.
    pub fn with_analyzer(analyzer: impl Into<Arc<Analyzer>>) -> Self {
        let mut p = Self::new();
        p.analyzer = Some(analyzer.into());
        p
    }

    /// `setMultiTermRewriteMethod(method)`.
    pub fn set_multi_term_rewrite_method(&mut self, method: RewriteMethod) {
        self.multi_term_rewrite_method = method;
    }

    /// `getMultiTermRewriteMethod()`.
    pub fn multi_term_rewrite_method(&self) -> RewriteMethod {
        self.multi_term_rewrite_method
    }

    /// `setAnalyzer(analyzer)`.
    pub fn set_analyzer(&mut self, analyzer: Option<Arc<Analyzer>>) {
        self.analyzer = analyzer;
    }

    /// `setDefaultOperator(operator)`.
    pub fn set_default_operator(&mut self, operator: Operator) {
        self.default_operator = operator;
    }

    /// `setAllowLeadingWildcard(allow)`.
    pub fn set_allow_leading_wildcard(&mut self, allow: bool) {
        self.allow_leading_wildcard = allow;
    }

    /// `setPhraseSlop(slop)`.
    pub fn set_phrase_slop(&mut self, slop: i32) {
        self.phrase_slop = slop;
    }

    /// `setEnablePositionIncrements(enabled)`.
    pub fn set_enable_position_increments(&mut self, enabled: bool) {
        self.enable_position_increments = enabled;
    }

    /// `setMultiFields(fields)`.
    pub fn set_multi_fields(&mut self, fields: &[&str]) {
        self.multi_fields = Some(fields.iter().map(|f| f.to_string()).collect());
    }

    /// `setFuzzyMinSim(minSim)`.
    pub fn set_fuzzy_min_sim(&mut self, min_sim: f32) {
        self.fuzzy_min_sim = min_sim;
    }

    /// `setFuzzyPrefixLength(length)`.
    pub fn set_fuzzy_prefix_length(&mut self, length: i32) {
        self.fuzzy_prefix_length = length;
    }

    /// `setFieldsBoost(boosts)`.
    pub fn set_fields_boost(&mut self, boosts: &[(&str, f32)]) {
        self.field_boosts = boosts.iter().map(|(f, b)| (f.to_string(), *b)).collect();
    }

    /// `parse(query, defaultField)`.
    ///
    /// # Errors
    /// A syntax error (`ParseException`, or `QueryNodeParseException` for a
    /// lexical one) with Java's message, a processor's `QueryNodeException`,
    /// and the runtime exceptions Java lets through.
    pub fn parse(
        &self,
        query: &str,
        default_field: Option<&str>,
    ) -> Result<ParsedQuery, QueryParserError> {
        let tree = parser::Parser::new(query)
            .top_level_query(default_field)
            .map_err(|f| syntax_error(query, f))?;
        let tree = processors::run(self, tree).map_err(java_exception)?;
        builders::build(self, &tree)
            .map_err(java_exception)?
            .ok_or_else(|| QueryParserError::runtime("null query"))
    }
}

/// What `StandardSyntaxParser.parse` makes of a failure: a `ParseException`
/// whose message `setQuery` replaced with `INVALID_SYNTAX_CANNOT_PARSE(query,
/// "")`; a `TokenMgrError` wrapped with its message; a `NumberFormatException`
/// let through.
fn syntax_error(query: &str, f: parser::Fail) -> QueryParserError {
    match f {
        parser::Fail::Syntax(m) if m.starts_with("Lexical error") => {
            QueryParserError::runtime_class(
                "org.apache.lucene.queryparser.flexible.core.QueryNodeParseException",
                format!("Syntax Error, cannot parse {query}: {m} "),
            )
        }
        parser::Fail::Syntax(_) => QueryParserError::runtime_class(
            "org.apache.lucene.queryparser.flexible.standard.parser.ParseException",
            format!("Syntax Error, cannot parse {query}:  "),
        ),
        parser::Fail::NumberFormat(m) => {
            QueryParserError::runtime_class("java.lang.NumberFormatException", m)
        }
    }
}

/// The class a runtime exception carries in Java, for the ones the shared
/// helpers name by their simple name.
fn java_exception(mut e: QueryParserError) -> QueryParserError {
    e.class = match e.class {
        "IllegalArgumentException" => "java.lang.IllegalArgumentException",
        "RuntimeException" => "java.lang.RuntimeException",
        "TooComplexToDeterminizeException" => {
            "org.apache.lucene.util.automaton.TooComplexToDeterminizeException"
        }
        other => other,
    };
    e
}

/// `org.apache.lucene.queryparser.flexible.standard.QueryParserUtil`: one
/// boolean of the parses of several queries or fields, and `escape`.
pub mod query_parser_util {
    use std::sync::Arc;

    use lucene_analysis::Analyzer;

    use super::StandardQueryParser;
    use crate::query_visitor::Occur;
    use crate::queryparser::classic::{QueryParserError, MAX_CLAUSE_COUNT};
    use crate::queryparser::{ParsedBoolean, ParsedQuery};

    fn illegal(message: &str) -> QueryParserError {
        QueryParserError::runtime_class("java.lang.IllegalArgumentException", message)
    }

    /// The boolean of `(query, field, occur)` parses, each with its own field.
    fn combine(
        parts: impl Iterator<Item = (String, String, Occur)>,
        analyzer: Arc<Analyzer>,
    ) -> Result<ParsedQuery, QueryParserError> {
        let mut qp = StandardQueryParser::new();
        qp.set_analyzer(Some(analyzer));
        let mut clauses = Vec::new();
        for (query, field, occur) in parts {
            let q = qp.parse(&query, Some(&field))?;
            if clauses.len() >= MAX_CLAUSE_COUNT {
                return Err(QueryParserError::too_many_clauses());
            }
            clauses.push((occur, q));
        }
        Ok(ParsedQuery::Boolean(ParsedBoolean {
            clauses,
            minimum_should_match: 0,
        }))
    }

    /// `parse(queries, fields, analyzer)`: each query on its field, `SHOULD`.
    ///
    /// # Errors
    /// Unequal lengths, and each parse's errors.
    pub fn parse(
        queries: &[&str],
        fields: &[&str],
        analyzer: Arc<Analyzer>,
    ) -> Result<ParsedQuery, QueryParserError> {
        if queries.len() != fields.len() {
            return Err(illegal("queries.length != fields.length"));
        }
        let parts = queries
            .iter()
            .zip(fields)
            .map(|(q, f)| (q.to_string(), f.to_string(), Occur::Should));
        combine(parts, analyzer)
    }

    /// `parse(query, fields, flags, analyzer)`: the query on each field, with
    /// its flag.
    ///
    /// # Errors
    /// Unequal lengths, and each parse's errors.
    pub fn parse_fields(
        query: &str,
        fields: &[&str],
        flags: &[Occur],
        analyzer: Arc<Analyzer>,
    ) -> Result<ParsedQuery, QueryParserError> {
        if fields.len() != flags.len() {
            return Err(illegal("fields.length != flags.length"));
        }
        let parts = fields
            .iter()
            .zip(flags)
            .map(|(f, o)| (query.to_string(), f.to_string(), *o));
        combine(parts, analyzer)
    }

    /// `parse(queries, fields, flags, analyzer)`: each query on its field,
    /// with its flag.
    ///
    /// # Errors
    /// Unequal lengths, and each parse's errors.
    pub fn parse_all(
        queries: &[&str],
        fields: &[&str],
        flags: &[Occur],
        analyzer: Arc<Analyzer>,
    ) -> Result<ParsedQuery, QueryParserError> {
        if !(queries.len() == fields.len() && queries.len() == flags.len()) {
            return Err(illegal(
                "queries, fields, and flags array have have different length",
            ));
        }
        let parts = queries
            .iter()
            .zip(fields)
            .zip(flags)
            .map(|((q, f), o)| (q.to_string(), f.to_string(), *o));
        combine(parts, analyzer)
    }

    /// `escape(s)`: a backslash before each character the syntax reserves.
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
