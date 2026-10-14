//! `org.apache.lucene.queryparser.xml`: queries written as XML documents,
//! through `CoreParser` and its builders.
//!
//! [`CoreParser`] is the core set (`TermQuery`, `TermsQuery`,
//! `MatchAllDocsQuery`, `BooleanQuery`, `RangeQuery`, `PointRangeQuery`,
//! `DisjunctionMaxQuery`, `UserQuery`, `ConstantScoreQuery`, and the span
//! queries). Java's `CorePlusQueriesParser` (`LikeThisQuery`) and
//! `CorePlusExtensionsParser` (`FuzzyLikeThisQuery`) are not ported, and a
//! caller cannot register its own builders as Java's `addQueryBuilder` does.
//!
//! **Differs:** a document type declaration is refused (Java accepts an
//! internal subset and refuses external entities); the XML reader is
//! `quick-xml`, whose syntax-error text is not Xerces'. Java's
//! `ParserException` text for a syntax error is therefore the reader's,
//! prefixed as Java prefixes it.

pub(crate) mod builders;
pub(crate) mod dom;
#[cfg(test)]
mod tests;

use std::sync::Arc;

use lucene_analysis::Analyzer;

use crate::queryparser::classic::{QueryParser, QueryParserError};
use crate::queryparser::ParsedQuery;
use builders::{parser_error, Ancestors, Builders};

/// `org.apache.lucene.queryparser.xml.ParserException`, the class of every
/// error the parser reports as Java would.
pub const PARSER_EXCEPTION: &str = "org.apache.lucene.queryparser.xml.ParserException";

/// `CoreParser`: parses an XML query document.
pub struct CoreParser {
    builders: Builders,
}

impl CoreParser {
    /// `new CoreParser(defaultField, analyzer)`: a `UserQuery` makes its own
    /// classic parser on its field (or `default_field`).
    pub fn new(default_field: Option<&str>, analyzer: impl Into<Arc<Analyzer>>) -> Self {
        CoreParser {
            builders: Builders {
                default_field: default_field.map(str::to_string),
                analyzer: analyzer.into(),
                parser: None,
            },
        }
    }

    /// `new CoreParser(analyzer, parser)`: every `UserQuery` goes through
    /// `parser`.
    pub fn with_parser(analyzer: impl Into<Arc<Analyzer>>, parser: QueryParser) -> Self {
        CoreParser {
            builders: Builders {
                default_field: None,
                analyzer: analyzer.into(),
                parser: Some(parser),
            },
        }
    }

    /// `CoreParser.parse(xmlStream)`: the query of the document element.
    ///
    /// # Errors
    /// `ParserException` for a document that is not well-formed, refused, or
    /// not a query the builders make; the runtime exceptions a builder's
    /// parse or constructor throws, as Java lets them through.
    pub fn parse(&self, xml: &str) -> Result<ParsedQuery, QueryParserError> {
        let root =
            dom::parse(xml).map_err(|m| parser_error(format!("Error parsing XML stream: {m}")))?;
        let ancestors: Ancestors<'_> = Vec::new();
        self.builders
            .query(&root, &ancestors)
            .map_err(java_exception)
    }
}

/// The class a runtime exception carries in Java: `java.lang` for the
/// `IllegalArgumentException` and `NullPointerException` the builders throw
/// through the query constructors, `org.apache.lucene.search` for a
/// `TooManyClauses`.
fn java_exception(mut e: QueryParserError) -> QueryParserError {
    e.class = match e.class {
        "IllegalArgumentException" => "java.lang.IllegalArgumentException",
        "NullPointerException" => "java.lang.NullPointerException",
        "NumberFormatException" => "java.lang.NumberFormatException",
        "TooManyClauses" => "org.apache.lucene.search.TooManyClauses",
        other => other,
    };
    e
}
